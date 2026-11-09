import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load("train_mod", "03_train_forecast_models.py")
HORIZON, N_FOLDS, TARGET = train.HORIZON, train.N_FOLDS, train.TARGET
N_BOOT, BLOCK = 2000, 4          # circular block bootstrap over weeks

FAMILY_COLORS = {"baseline": "#98a2b3", "statistical": "#9b59b6", "linear": "#e67e22",
                 "bagged trees": "#27ae60", "boosted trees": "#2563eb", "ensemble": "#d92d20"}


def family(name):
    if name in ("naive_lag4", "moving_avg_4", "seasonal_naive", "seasonal_naive_smooth"):
        return "baseline"
    if name.startswith("ets"):
        return "statistical"
    if name.startswith("ridge"):
        return "linear"
    if name.startswith(("random_forest", "extra_trees")):
        return "bagged trees"
    if name.startswith(("lgbm", "xgboost")):
        return "boosted trees"
    return "ensemble"


# ------------------------------------------------------------------ features
def build_matrices(data, has_weather):
    """Ratio-feature matrix (for linear/tree models) and absolute-feature matrix (poisson)."""
    w = train.WEATHER_FEATURES if has_weather else []
    data = data.copy()
    data["has_last_year"] = data["lag_52"].notna().astype(float)
    ang = 2 * np.pi * data["week_of_year"] / 52.18
    for k in (1, 2, 3):
        data[f"fourier_sin{k}"] = np.sin(k * ang)
        data[f"fourier_cos{k}"] = np.cos(k * ang)
    fourier = [f"fourier_{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")]

    ratio_cols = (["product_type"] + train.RATIO_FEATURES + ["roll_mean_8"] + train.CAL_FEATURES
                  + train.MARKET_FEATURES + w + ["has_last_year"] + fourier)
    abs_cols = ["product_type"] + train.LAG_FEATURES + train.CAL_FEATURES + train.MARKET_FEATURES + w

    def onehot(cols):
        X = data[cols].copy()
        X["product_type"] = X["product_type"].astype(str)
        return pd.get_dummies(X, columns=["product_type"], dtype=float)

    ones_cols = [c for c in train.RATIO_FEATURES if c.startswith(("lag_5", "lag52"))] + \
                ["mkt_ly_idx", "mkt_ly_idx_max", "mkt_mom"]
    Xr = onehot(ratio_cols)
    for c in ones_cols:                                # no last-year info -> neutral ratio of 1
        if c in Xr.columns:
            Xr[c] = Xr[c].fillna(1.0)
    return data, Xr, onehot(abs_cols)


def fit_predict(model, Xtr, tr, Xte, te, mode):
    if mode == "ratio":
        med = Xtr.median()
        y = np.log(tr[TARGET].clip(lower=1) / tr["scale"])
        model.fit(Xtr.fillna(med), y)
        return np.clip(np.exp(model.predict(Xte.fillna(med))) * te["scale"].values, 0, None)
    model.fit(Xtr, tr[TARGET])                         # raw counts (poisson); xgboost handles NaN
    return np.clip(model.predict(Xte), 0, None)


def ets_forecast(series, steps):
    y = np.clip(np.asarray(series, float), 1, None)
    try:
        m = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        return np.clip(m.forecast(steps), 0, None)
    except Exception:
        return np.repeat(y[-HORIZON], steps)           # fall back to the last known value


# ------------------------------------------------------------------ bootstrap
def block_bootstrap_wape(bt, pred_cols, seed=7):
    """Resample 4-week blocks of weeks; returns WAPE draws per model."""
    wk = bt.groupby("week_start")
    actual = wk[TARGET].sum().values
    abserr = {c: wk.apply(lambda g: np.abs(g[TARGET] - g[c]).sum(), include_groups=False).values
              for c in pred_cols}
    n = len(actual)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BLOCK))
    idx = np.empty((N_BOOT, n_blocks * BLOCK), dtype=int)
    for b in range(N_BOOT):
        starts = rng.integers(0, n, n_blocks)
        idx[b] = ((starts[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()
    idx = idx[:, :n]
    den = actual[idx].sum(axis=1)
    return {c: 100 * abserr[c][idx].sum(axis=1) / den for c in pred_cols}


# ------------------------------------------------------------------ main
def main():
    wf = config.DATA_PROCESSED / "weekly_features.csv"
    bp = config.REPORTS_DIR / "backtest_predictions.csv"
    for p in (wf, bp):
        if not p.exists():
            sys.exit(f"Missing {p}. Run steps 1-3 first.")

    df = pd.read_csv(wf, parse_dates=["week_start"])
    df["product_type"] = df["product_type"].astype("category")
    has_weather = all(c in df.columns for c in train.WEATHER_FEATURES)
    full = train.add_features(df)
    data = full.dropna(subset=["lag_8", "roll_mean_4", "roll_mean_8"]).reset_index(drop=True)
    data, Xr, Xa = build_matrices(data, has_weather)
    weeks = sorted(data["week_start"].unique())
    pivot = df.pivot(index="week_start", columns="product_type", values=TARGET)
    print(f"Rows: {len(data)} | xgboost: {XGBRegressor is not None} | statsmodels: {ExponentialSmoothing is not None}")

    # models declared up front (not tuned on the backtest)
    def models():
        m = {
            "ridge_ratio_weather": ("ratio", Xr, lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
            "random_forest_ratio_weather": ("ratio", Xr, lambda: RandomForestRegressor(
                n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=42)),
            "extra_trees_ratio_weather": ("ratio", Xr, lambda: ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=42)),
        }
        if XGBRegressor is not None:
            m["xgboost_ratio_weather"] = ("ratio", Xr, lambda: XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=3, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0))
            m["xgboost_poisson_weather"] = ("raw", Xa, lambda: XGBRegressor(
                objective="count:poisson", n_estimators=500, learning_rate=0.03, max_depth=4,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0))
        return m

    new_rows = []
    for f in range(N_FOLDS):
        test_end = len(weeks) - f * HORIZON
        test_start = test_end - HORIZON
        origin, test_weeks = weeks[test_start - 1], weeks[test_start:test_end]
        tr_mask = (data["week_start"] <= origin).values
        te_mask = data["week_start"].isin(test_weeks).values
        tr, te = data[tr_mask], data[te_mask].copy()

        out = te[["product_type", "week_start"]].copy()
        out["product_type"] = out["product_type"].astype(str)
        for name, (mode, X, make) in models().items():
            out[f"pred_{name}"] = fit_predict(make(), X[tr_mask], tr, X[te_mask], te, mode)

        if ExponentialSmoothing is not None:
            ets = {}
            for ptype in pivot.columns:
                hist = pivot.loc[pivot.index <= origin, ptype].values
                ets[str(ptype)] = dict(zip(test_weeks, ets_forecast(hist, HORIZON)))
            out["pred_ets_damped"] = [ets[p][w] for p, w in zip(out["product_type"], out["week_start"])]
        new_rows.append(out)
        print(f"fold {f + 1}/{N_FOLDS} done (origin {pd.Timestamp(origin).date()})")

    new = pd.concat(new_rows)
    bt = pd.read_csv(bp, parse_dates=["week_start"])
    bt = bt.merge(new, on=["product_type", "week_start"], how="left")

    # pre-declared ensembles
    seas = bt["pred_seasonal_naive"]
    if "pred_xgboost_ratio_weather" in bt:
        bt["pred_ens_xgb_plus_seasonal"] = (bt["pred_xgboost_ratio_weather"] + seas) / 2
        bt["pred_ens_boosted_plus_seasonal"] = (bt["pred_lgbm_ratio_weather"]
                                                + bt["pred_xgboost_ratio_weather"] + seas) / 3
    bt["pred_ens_trees_plus_seasonal"] = (bt["pred_lgbm_ratio_weather"]
                                          + bt["pred_random_forest_ratio_weather"] + seas) / 3

    pred_cols = [c for c in bt.columns if c.startswith("pred_")]
    bt = bt.dropna(subset=pred_cols)

    # ---------------- leaderboard with bootstrap intervals ----------------
    boot = block_bootstrap_wape(bt, pred_cols)
    rows = []
    for c in pred_cols:
        y, p = bt[TARGET].values, bt[c].values
        diff = boot[c] - boot["pred_seasonal_naive"]
        rows.append({
            "model": c.replace("pred_", ""), "family": family(c.replace("pred_", "")),
            "WAPE_%": 100 * np.abs(y - p).sum() / y.sum(),
            "WAPE_ci90_low": np.percentile(boot[c], 5), "WAPE_ci90_high": np.percentile(boot[c], 95),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "Bias_%": 100 * (p.sum() - y.sum()) / y.sum(),
            "vs_seasonal_naive_pts": 100 * np.abs(y - p).sum() / y.sum()
                                     - 100 * np.abs(y - seas.loc[bt.index].values).sum() / y.sum(),
            "diff_ci90_low": np.percentile(diff, 5), "diff_ci90_high": np.percentile(diff, 95),
        })
    lb = pd.DataFrame(rows).set_index("model").sort_values("WAPE_%")
    lb["clearly_beats_seasonal_naive"] = (lb["diff_ci90_high"] < 0)
    lb = lb.round(2)
    show = lb[["family", "WAPE_%", "WAPE_ci90_low", "WAPE_ci90_high", "RMSE", "Bias_%",
               "vs_seasonal_naive_pts", "clearly_beats_seasonal_naive"]]
    print("\n=== MODEL LEADERBOARD: 24-week backtest, 4-week-ahead ===")
    print(show.to_string())
    print("\nWAPE_ci90_*: 90% interval from a block bootstrap over weeks (wide = few weeks of evidence).")
    print("clearly_beats_seasonal_naive = the 90% interval of the WAPE difference is entirely below 0.")
    print("NOTE: many models were compared on the same 24 weeks, so the top score is optimistic.")

    # ---------------- per-category view ----------------
    wape_cat = pd.DataFrame({
        c.replace("pred_", ""): {k: 100 * np.abs(g[TARGET] - g[c]).sum() / g[TARGET].sum()
                                 for k, g in bt.groupby("product_type")}
        for c in pred_cols}).round(1)
    best_by_cat = wape_cat.idxmin(axis=1)
    print("\nBest model per category (note: noisy with only 24 weeks):")
    print(pd.DataFrame({"best_model": best_by_cat, "its_WAPE_%": wape_cat.min(axis=1),
                        "seasonal_naive_WAPE_%": wape_cat["seasonal_naive"]}).to_string())
    print("\nCategories where each model is best:")
    print(best_by_cat.value_counts().to_string())

    lb.to_csv(config.REPORTS_DIR / "model_comparison_extended.csv")
    wape_cat.to_csv(config.REPORTS_DIR / "wape_by_category_extended.csv")
    bt.to_csv(config.REPORTS_DIR / "backtest_predictions_extended.csv", index=False)

    # ---------------- charts ----------------
    d = lb.sort_values("WAPE_%", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.6))
    colors = [FAMILY_COLORS[f] for f in d["family"]]
    ax.barh(d.index, d["WAPE_%"], color=colors,
            xerr=[d["WAPE_%"] - d["WAPE_ci90_low"], d["WAPE_ci90_high"] - d["WAPE_%"]],
            error_kw={"ecolor": "#444", "lw": 1, "capsize": 2})
    ax.axvline(lb.loc["seasonal_naive", "WAPE_%"], ls="--", c="k", lw=1)
    ax.text(lb.loc["seasonal_naive", "WAPE_%"], len(d) - 0.4, " seasonal naive", fontsize=8, va="bottom")
    ax.set_xlabel("WAPE % (lower is better), bars show 90% bootstrap interval")
    ax.set_title("Model comparison: 24-week rolling backtest, 4-week-ahead forecasts")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "09_model_leaderboard.png", dpi=130)
    plt.close()

    top = list(lb.index[:8])
    if "seasonal_naive" not in top:
        top.append("seasonal_naive")
    hm = wape_cat[top].T
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(top) + 2))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index, fontsize=8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("WAPE % by category and model (top models + seasonal naive)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "10_wape_by_category_heatmap.png", dpi=130)
    plt.close()
    print("\nSaved: model_comparison_extended.csv, wape_by_category_extended.csv, "
          "backtest_predictions_extended.csv")
    print("Charts: 09_model_leaderboard.png, 10_wape_by_category_heatmap.png")


if __name__ == "__main__":
    main()