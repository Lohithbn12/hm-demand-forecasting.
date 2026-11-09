import sys
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

# ----------------------------------------------------------------- settings
PLAN_WEEKS = 4                                   # lead time + review period (weeks), matches forecast horizon
SERVICE_LEVEL = {"A": 0.95, "B": 0.90, "C": 0.85}  # target cycle service level by ABC class
ABC_CUTOFFS = (0.70, 0.90)                       # cumulative revenue share: A <= 70%, B <= 90%
XYZ_CUTOFFS = (0.40, 0.70)                       # weekly demand CV: X <= 0.4, Y <= 0.7, else Z
ERROR_BUFFER = 1.4                               # safety-stock multiplier; set from the printed calibration
BUFFER_GRID = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]  # safety-stock multipliers to sweep
REL_ERR_CLIP = (-1.0, 2.0)                       # clip relative errors (avoids blow-ups at tiny forecasts)
TARGET = "units"
# optional file: product_type,on_hand_units[,on_order_units]
INVENTORY_FILE = config.DATA_EXTERNAL / "current_inventory.csv"


def z_value(p):
    return NormalDist().inv_cdf(p)


# ----------------------------------------------------------------- segmentation
def abc_xyz(weekly):
    last_week = weekly["week_start"].max()
    recent = weekly[weekly["week_start"] > last_week - pd.Timedelta(weeks=52)]

    seg = recent.groupby("product_type", observed=True).agg(
        revenue=("revenue", "sum"),
        mean_units=(TARGET, "mean"),
        std_units=(TARGET, "std"),
    )
    seg["cv"] = seg["std_units"] / seg["mean_units"]
    seg["revenue_share_%"] = 100 * seg["revenue"] / seg["revenue"].sum()
    seg = seg.sort_values("revenue", ascending=False)
    cum = seg["revenue"].cumsum() / seg["revenue"].sum()
    # a category is class A if the cumulative share BEFORE it is still below the cutoff
    prev = cum.shift(1).fillna(0)
    seg["ABC"] = np.where(prev < ABC_CUTOFFS[0], "A",
                          np.where(prev < ABC_CUTOFFS[1], "B", "C"))
    seg["XYZ"] = np.where(seg["cv"] <= XYZ_CUTOFFS[0], "X",
                          np.where(seg["cv"] <= XYZ_CUTOFFS[1], "Y", "Z"))
    seg["class"] = seg["ABC"] + seg["XYZ"]
    seg["target_service"] = seg["ABC"].map(SERVICE_LEVEL)
    return seg


# ----------------------------------------------------------------- simulation
def rel_errors(df, col):
    """Relative forecast error per week: (actual - forecast) / forecast."""
    e = (df[TARGET] - df[col]) / df[col].clip(lower=1)
    return e.clip(*REL_ERR_CLIP)


def cycle_sigma(sigma_rel, weekly_forecasts, buffer):
    """Std of the 4-week demand error in units, assuming weekly errors are independent.
    Errors scale with the forecast level, so we use RELATIVE error x forecast.
    buffer > 1 inflates it when weekly errors are positively correlated."""
    return sigma_rel * np.sqrt(np.sum(np.square(weekly_forecasts))) * buffer


def simulate_policy(bt, col, seg, buffer=1.0):
    """4-week planning cycles. For each category and fold:
    order-up-to S = forecast(4 wks) + z * sigma_cycle  (sigma from OTHER folds' errors)
    then compare S with what was actually demanded."""
    d = bt[["product_type", "fold", "week_start", TARGET, col]].copy()
    d["eps"] = rel_errors(d, col)
    rows = []
    for ptype, g in d.groupby("product_type", observed=True):
        z = z_value(seg.loc[ptype, "target_service"])
        for f, gf in g.groupby("fold"):
            sigma_rel = g[g["fold"] != f]["eps"].std(ddof=1)
            ss = z * cycle_sigma(sigma_rel, gf[col].values, buffer)
            S = max(0.0, gf[col].sum() + ss)
            D = gf[TARGET].sum()
            rows.append({"product_type": ptype, "fold": f, "demand": D, "order_up_to": S,
                         "lost": max(0.0, D - S), "excess": max(0.0, S - D),
                         "stockout": int(D > S)})
    r = pd.DataFrame(rows)
    return {
        "fill_rate_%": 100 * (1 - r["lost"].sum() / r["demand"].sum()),
        "cycle_service_%": 100 * (1 - r["stockout"].mean()),
        "excess_stock_%_of_demand": 100 * r["excess"].sum() / r["demand"].sum(),
        "total_order_up_to_vs_demand_%": 100 * (r["order_up_to"].sum() / r["demand"].sum() - 1),
    }


def sweep_buffers(bt, col, seg):
    rows = []
    for b in BUFFER_GRID:
        r = simulate_policy(bt, col, seg, buffer=b)
        r["buffer"] = b
        rows.append(r)
    return pd.DataFrame(rows)


def interp_at(df, x_col, y_col, x_target):
    """Interpolate y at x_target along a (noisy) monotone curve; NaN if outside range."""
    d = df.sort_values(x_col)
    x = np.maximum.accumulate(d[x_col].values)     # enforce non-decreasing
    y = d[y_col].values
    if x_target < x.min() or x_target > x.max():
        return np.nan
    return float(np.interp(x_target, x, y))


# ----------------------------------------------------------------- main
def main():
    wf = config.DATA_PROCESSED / "weekly_features.csv"
    bp = config.REPORTS_DIR / "backtest_predictions.csv"
    mc = config.REPORTS_DIR / "model_comparison.csv"
    for p in (wf, bp, mc):
        if not p.exists():
            sys.exit(f"Missing {p}. Run the earlier steps first.")

    weekly = pd.read_csv(wf, parse_dates=["week_start"])
    bt = pd.read_csv(bp, parse_dates=["week_start"])
    comp = pd.read_csv(mc, index_col=0)

    pred_cols = [c for c in bt.columns if c.startswith("pred_")]
    lgbm_like = [c for c in comp.index if c.startswith(("lgbm", "ens_"))]
    best_name = lgbm_like[0]                       # best LightGBM/ensemble by WAPE (file is sorted)
    best_col = f"pred_{best_name}"
    print("Forecaster used for recommendations:", best_name)

    # ---------- 1. ABC-XYZ ----------
    seg = abc_xyz(weekly)
    print("\n=== ABC-XYZ segmentation (last 52 weeks) ===")
    print(seg[["revenue_share_%", "cv", "ABC", "XYZ", "class", "target_service"]].round(2).to_string())
    seg.round(3).to_csv(config.REPORTS_DIR / "abc_xyz_segments.csv")

    # ---------- 2. safety stock + order-up-to for the latest planning cycle ----------
    bt["eps"] = rel_errors(bt, best_col)
    sigma_rel = bt.groupby("product_type", observed=True)["eps"].std(ddof=1)
    latest = bt[bt["fold"] == bt["fold"].min()]     # fold 1 = most recent 4 weeks
    cycle_start, cycle_end = latest["week_start"].min().date(), latest["week_start"].max().date()
    fc = latest.groupby("product_type", observed=True)[best_col].sum()
    sigma_cyc = latest.groupby("product_type", observed=True)[best_col].apply(
        lambda x: np.sqrt(np.sum(np.square(x.values))))

    rec = pd.DataFrame({"forecast_4wk": fc, "rel_error_sigma_%": 100 * sigma_rel,
                        "_root_sumsq": sigma_cyc})
    rec = rec.join(seg[["ABC", "XYZ", "class", "target_service"]])
    rec["z"] = rec["target_service"].map(z_value)
    rec["safety_stock"] = rec["z"] * sigma_rel * rec["_root_sumsq"] * ERROR_BUFFER
    rec = rec.drop(columns="_root_sumsq")
    rec["order_up_to"] = rec["forecast_4wk"] + rec["safety_stock"]
    rec["safety_stock_%_of_forecast"] = 100 * rec["safety_stock"] / rec["forecast_4wk"]

    if INVENTORY_FILE.exists():
        inv = pd.read_csv(INVENTORY_FILE).set_index("product_type")
        rec["on_hand"] = inv["on_hand_units"]
        rec["on_order"] = inv["on_order_units"] if "on_order_units" in inv.columns else 0
        rec["order_qty"] = (rec["order_up_to"] - rec["on_hand"].fillna(0)
                            - rec["on_order"].fillna(0)).clip(lower=0)
        print(f"\nUsed current inventory from {INVENTORY_FILE}")
    else:
        print(f"\n(No inventory file at {INVENTORY_FILE}. Add one to get order quantities.)")

    rec = rec.sort_values("order_up_to", ascending=False)
    for c in ["forecast_4wk", "rel_error_sigma_%", "safety_stock", "safety_stock_%_of_forecast",
              "order_up_to", "on_hand", "on_order", "order_qty"]:
        if c in rec.columns:
            rec[c] = rec[c].round(0)
    print(f"\n=== Recommendations for the 4-week cycle {cycle_start} -> {cycle_end} ===")
    cols = ["class", "target_service", "forecast_4wk", "rel_error_sigma_%", "safety_stock",
            "safety_stock_%_of_forecast", "order_up_to"] + \
           (["on_hand", "order_qty"] if "order_qty" in rec.columns else [])
    print(rec[cols].to_string())
    rec.to_csv(config.REPORTS_DIR / "inventory_recommendations.csv")

    # ---------- 3. policy simulation across forecasters ----------
    candidates = [c for c in ["pred_naive_lag4", "pred_moving_avg_4", "pred_seasonal_naive",
                              "pred_lgbm_poisson_weather", best_col] if c in pred_cols]
    candidates = list(dict.fromkeys(candidates))
    sim = pd.DataFrame({c.replace("pred_", ""): simulate_policy(bt, c, seg, buffer=ERROR_BUFFER)
                        for c in candidates}).T.round(1)
    print("\n=== Policy simulation: 6 cycles x 15 categories, same service-level targets ===")
    print(sim.sort_values("excess_stock_%_of_demand").to_string())
    target_pct = 100 * seg["target_service"].mean()
    print(f"\nWeighted target cycle service: {target_pct:.1f}%  (A/B/C targets: {SERVICE_LEVEL})")
    print(f"Table above uses ERROR_BUFFER = {ERROR_BUFFER}.")
    sim.to_csv(config.REPORTS_DIR / "policy_simulation.csv")

    # ---------- 4. service-vs-cost frontier: compare forecasters at the SAME service ----------
    frontier = {c.replace("pred_", ""): sweep_buffers(bt, c, seg) for c in candidates}
    out_rows = []
    for name, df_ in frontier.items():
        out_rows.append({
            "forecaster": name,
            "excess_stock_%_at_target_service": interp_at(df_, "cycle_service_%", "excess_stock_%_of_demand", target_pct),
            "fill_rate_%_at_target_service": interp_at(df_, "cycle_service_%", "fill_rate_%", target_pct),
            "buffer_needed": interp_at(df_, "cycle_service_%", "buffer", target_pct),
            "max_service_reached_%": df_["cycle_service_%"].max(),
        })
    matched = pd.DataFrame(out_rows).set_index("forecaster").round(1)
    print(f"\n=== Same-service comparison: cost of reaching {target_pct:.0f}% cycle service ===")
    print(matched.sort_values("excess_stock_%_at_target_service").to_string())
    print("buffer_needed = ERROR_BUFFER value that reaches the target on this backtest "
          "(NaN = target not reached within the grid).")
    best_key = best_col.replace("pred_", "")
    bn = matched.loc[best_key, "buffer_needed"]
    if not np.isnan(bn):
        print(f"\n>>> Suggested ERROR_BUFFER for {best_key}: {bn:.2f}  "
              f"(set it at the top of this file and rerun to refresh the recommendations)")
    pd.concat(frontier, names=["forecaster"]).reset_index(level=0).to_csv(
        config.REPORTS_DIR / "service_cost_frontier.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df_ in frontier.items():
        d = df_.sort_values("buffer")
        ax.plot(d["excess_stock_%_of_demand"], d["cycle_service_%"], "-o", ms=3, label=name)
    ax.axhline(target_pct, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("Excess stock (% of demand)  -> more cost")
    ax.set_ylabel("Cycle service level (%)  -> better")
    ax.set_title("Service vs inventory cost (curves further up-left are better)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_service_cost_frontier.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"A": "tab:red", "B": "tab:orange", "C": "tab:blue"}
    for name, r in seg.iterrows():
        ax.scatter(r["cv"], r["revenue_share_%"], c=colors[r["ABC"]], s=60)
        ax.annotate(str(name), (r["cv"], r["revenue_share_%"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    for x in XYZ_CUTOFFS:
        ax.axvline(x, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("Demand variability (CV of weekly units)")
    ax.set_ylabel("Revenue share (%)")
    ax.set_title("ABC-XYZ map (colour = ABC class)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_abc_xyz_map.png", dpi=120)
    plt.close()

    print("\nReports in:", config.REPORTS_DIR)
    print("Charts in:", config.FIGURES_DIR)


if __name__ == "__main__":
    main()