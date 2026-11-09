import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

HORIZON = 4          # forecast up to 4 weeks ahead
N_FOLDS = 6          # rolling-origin backtest: 6 folds x 4 weeks = last 24 weeks
TARGET = "units"

CAL_FEATURES = ["week_of_year", "month", "holiday_days", "black_friday",
                "xmas_days", "days_to_holiday"]
WEATHER_FEATURES = ["temp_mean", "temp_max", "temp_min", "precip"]

# v1 features (absolute scale)
LAG_FEATURES = ["lag_4", "lag_5", "lag_6", "lag_8", "lag_51", "lag_52", "lag_53",
                "lag52_avg", "roll_mean_4", "roll_mean_8", "roll_std_4", "price_lag_4"]
# new: market-level "was the whole store spiking this week last year?"
MARKET_FEATURES = ["mkt_lag_4", "mkt_mom", "mkt_ly_idx", "mkt_ly_idx_max"]
# new: the same lags expressed relative to the product's recent level
RATIO_BASE = ["lag_4", "lag_5", "lag_6", "lag_8", "lag_51", "lag_52", "lag_53",
              "lag52_avg", "roll_mean_4", "roll_std_4"]
RATIO_FEATURES = [f"{c}_r" for c in RATIO_BASE]


# ------------------------------------------------------------------ features
def add_features(df):
    """Every lag is >= HORIZON, so each feature is known at forecast time."""
    df = df.sort_values(["product_type", "week_start"]).reset_index(drop=True)
    g = df.groupby("product_type", observed=True)
    for k in (4, 5, 6, 8, 51, 52, 53):
        df[f"lag_{k}"] = g[TARGET].shift(k)
    df["lag52_avg"] = df[["lag_51", "lag_52", "lag_53"]].mean(axis=1)

    shifted = g[TARGET].shift(HORIZON)
    by_type = df["product_type"]
    df["roll_mean_4"] = shifted.groupby(by_type, observed=True).transform(
        lambda s: s.rolling(4, min_periods=2).mean())
    df["roll_mean_8"] = shifted.groupby(by_type, observed=True).transform(
        lambda s: s.rolling(8, min_periods=4).mean())
    df["roll_std_4"] = shifted.groupby(by_type, observed=True).transform(
        lambda s: s.rolling(4, min_periods=2).std())
    df["price_lag_4"] = g["avg_price"].shift(HORIZON)

    # market-level features (all product types together)
    mkt = df.groupby("week_start")[TARGET].sum().sort_index()
    mkt_idx = mkt / mkt.rolling(9, center=True, min_periods=5).mean()
    mkt_feat = pd.DataFrame({
        "mkt_lag_4": mkt.shift(4),
        "mkt_mom": mkt.shift(4) / mkt.shift(4).rolling(8, min_periods=4).mean(),
        "mkt_ly_idx": mkt_idx.shift(52),
        "mkt_ly_idx_max": pd.concat([mkt_idx.shift(51), mkt_idx.shift(52),
                                     mkt_idx.shift(53)], axis=1).max(axis=1),
    }).reset_index()
    df = df.merge(mkt_feat, on="week_start", how="left")

    # ratio features
    df["scale"] = df["roll_mean_8"].clip(lower=1)
    for c in RATIO_BASE:
        df[f"{c}_r"] = df[c] / df["scale"]
    return df


# ------------------------------------------------------------------ modelling
def fit_model(train, features, mode, objective):
    if mode == "log1p":
        y = np.log1p(train[TARGET])
    elif mode == "ratio":
        y = np.log(train[TARGET].clip(lower=1) / train["scale"])
    else:                                   # raw counts (used with poisson)
        y = train[TARGET]
    model = lgb.LGBMRegressor(
        objective=objective, n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    model.fit(train[features], y)
    return model


def predict_model(model, test, features, mode):
    p = model.predict(test[features])
    if mode == "log1p":
        out = np.expm1(p)
    elif mode == "ratio":
        out = np.exp(p) * test["scale"].values
    else:
        out = p
    return np.clip(out, 0, None)


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "WAPE_%": 100 * np.abs(y - p).sum() / y.sum(),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "Bias_%": 100 * (p.sum() - y.sum()) / y.sum(),
    }


# ------------------------------------------------------------------ main
def main():
    src = config.DATA_PROCESSED / "weekly_features.csv"
    if not src.exists():
        sys.exit(f"Missing {src}. Run 02_build_features.py first.")

    df = pd.read_csv(src, parse_dates=["week_start"])
    df["product_type"] = df["product_type"].astype("category")

    gaps = df.groupby("product_type", observed=True)["week_start"].diff().dropna()
    if (gaps != pd.Timedelta(days=7)).any():
        sys.exit("Weekly series has gaps. Re-check 02_build_features.py output.")

    has_weather = all(c in df.columns for c in WEATHER_FEATURES)
    df = add_features(df)

    abs_base = ["product_type"] + LAG_FEATURES + CAL_FEATURES + MARKET_FEATURES
    ratio_base = ["product_type"] + RATIO_FEATURES + ["roll_mean_8"] + CAL_FEATURES + MARKET_FEATURES
    w = WEATHER_FEATURES if has_weather else []

    # the model variants we compare (declared up front, not tuned on the backtest)
    VARIANTS = {
        "lgbm_v1_weather": dict(
            features=["product_type", "lag_4", "lag_5", "lag_6", "lag_8", "lag_52",
                      "roll_mean_4", "roll_mean_8", "roll_std_4", "price_lag_4"]
                     + CAL_FEATURES + w,
            mode="log1p", objective="regression"),
        "lgbm_ratio_no_weather": dict(features=ratio_base, mode="ratio", objective="regression"),
        "lgbm_ratio_weather": dict(features=ratio_base + w, mode="ratio", objective="regression"),
        "lgbm_poisson_weather": dict(features=abs_base + w, mode="raw", objective="poisson"),
    }
    # ensembles: (lgbm members, include seasonal naive?)
    ENSEMBLES = {
        "ens_ratio_plus_seasonal": (["lgbm_ratio_weather"], True),
        "ens_all_lgbm": (["lgbm_v1_weather", "lgbm_ratio_weather", "lgbm_poisson_weather"], False),
    }

    data = df.dropna(subset=["lag_8", "roll_mean_4", "roll_mean_8"]).reset_index(drop=True)
    weeks = sorted(data["week_start"].unique())
    print(f"Modelling rows: {len(data)} | weeks: {weeks[0].date()} -> {weeks[-1].date()}")
    print(f"Weather features available: {has_weather}")

    # ---------------- rolling-origin backtest ----------------
    all_preds = []
    for f in range(N_FOLDS):
        test_end = len(weeks) - f * HORIZON
        test_start = test_end - HORIZON
        origin = weeks[test_start - 1]
        test_weeks = weeks[test_start:test_end]

        train = data[data["week_start"] <= origin]
        test = data[data["week_start"].isin(test_weeks)].copy()

        test["pred_naive_lag4"] = test["lag_4"]
        test["pred_seasonal_naive"] = test["lag_52"].fillna(test["lag_4"])
        test["pred_seasonal_naive_smooth"] = test["lag52_avg"].fillna(test["lag_4"])
        test["pred_moving_avg_4"] = test["roll_mean_4"]

        for name, v in VARIANTS.items():
            m = fit_model(train, v["features"], v["mode"], v["objective"])
            test[f"pred_{name}"] = predict_model(m, test, v["features"], v["mode"])

        for name, (members, with_seasonal) in ENSEMBLES.items():
            parts = [test[f"pred_{m}"] for m in members]
            if with_seasonal:
                parts.append(test["pred_seasonal_naive"])
            test[f"pred_{name}"] = np.mean(parts, axis=0)

        test["fold"] = f + 1
        all_preds.append(test)
        print(f"fold {f + 1}/{N_FOLDS} done | origin {pd.Timestamp(origin).date()} "
              f"| train rows {len(train)}")

    bt = pd.concat(all_preds).sort_values(["product_type", "week_start"])
    pred_cols = [c for c in bt.columns if c.startswith("pred_")]

    # ---------------- evaluation ----------------
    rows = []
    for c in pred_cols:
        s = score(bt[TARGET], bt[c])
        s["model"] = c.replace("pred_", "")
        rows.append(s)
    comparison = pd.DataFrame(rows).set_index("model").sort_values("WAPE_%").round(2)
    print("\n=== BACKTEST: last 24 weeks, 4-week-ahead forecasts ===")
    print(comparison.to_string())

    best = comparison.index[0]
    best_col = f"pred_{best}"
    base_name = comparison.loc[["seasonal_naive", "seasonal_naive_smooth"], "WAPE_%"].idxmin()
    base_wape = comparison.loc[base_name, "WAPE_%"]
    print(f"\nBest model: {best} | WAPE {comparison.loc[best, 'WAPE_%']:.1f}% "
          f"vs best baseline ({base_name}) {base_wape:.1f}% "
          f"({100 * (base_wape - comparison.loc[best, 'WAPE_%']) / base_wape:.0f}% better)")

    if has_weather:
        d = (comparison.loc["lgbm_ratio_no_weather", "WAPE_%"]
             - comparison.loc["lgbm_ratio_weather", "WAPE_%"])
        print(f"Weather effect on WAPE (ratio model): {d:+.2f} points (positive = weather helps).")
        print("NOTE: backtest uses ACTUAL weather. In production, weather beyond ~2 weeks "
              "is a forecast/climatology, so treat this gain as an upper bound.")
    print("NOTE: several variants were compared on the same 24 weeks, so the best score "
          "is slightly optimistic.")

    by_type = (bt.groupby("product_type", observed=True)
                 .apply(lambda g: pd.Series({
                     "units": g[TARGET].sum(),
                     "WAPE_best_%": score(g[TARGET], g[best_col])["WAPE_%"],
                     "WAPE_seasonal_naive_%": score(g[TARGET], g["pred_seasonal_naive"])["WAPE_%"],
                 }), include_groups=False)
                 .round(1).sort_values("units", ascending=False))
    print("\nBy product type:")
    print(by_type.to_string())

    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    by_type.to_csv(config.REPORTS_DIR / "metrics_by_product_type.csv")
    bt[["week_start", "product_type", "fold", TARGET] + pred_cols].to_csv(
        config.REPORTS_DIR / "backtest_predictions.csv", index=False)

    # ---------------- charts ----------------
    show = [p for p in ["Dress", "Sweater", "Bikini top", "Trousers"]
            if p in set(bt["product_type"].astype(str))]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 3 * len(show)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, show):
        g = bt[bt["product_type"] == p].sort_values("week_start")
        ax.plot(g["week_start"], g[TARGET], "k-o", ms=3, label="actual")
        ax.plot(g["week_start"], g[best_col], "-s", ms=3, label=best)
        ax.plot(g["week_start"], g["pred_seasonal_naive"], "--", label="seasonal naive")
        ax.set_title(p); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_backtest_actual_vs_predicted.png", dpi=120)
    plt.close()

    # ---------------- final models on all data ----------------
    final_models = {}
    for name, v in VARIANTS.items():
        final_models[name] = {"model": fit_model(data, v["features"], v["mode"], v["objective"]),
                              "features": v["features"], "mode": v["mode"]}
    joblib.dump({"models": final_models, "ensembles": ENSEMBLES, "best": best,
                 "product_types": list(df["product_type"].cat.categories),
                 "horizon": HORIZON},
                config.MODELS_DIR / "lgbm_demand_forecaster.joblib")
    print("\nSaved models:", config.MODELS_DIR / "lgbm_demand_forecaster.joblib")

    # importance + SHAP for the best single LightGBM variant
    lgbm_scores = comparison.loc[[n for n in VARIANTS if n in comparison.index], "WAPE_%"]
    best_single = lgbm_scores.idxmin()
    fm = final_models[best_single]
    imp = pd.Series(fm["model"].booster_.feature_importance("gain"),
                    index=fm["features"]).sort_values()
    plt.figure(figsize=(8, 8)); imp.plot(kind="barh")
    plt.title(f"Feature importance (gain): {best_single}"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "05_feature_importance.png", dpi=120)
    plt.close()

    try:
        import shap
        sample = data[fm["features"]].sample(min(500, len(data)), random_state=1)
        shap_values = shap.TreeExplainer(fm["model"]).shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "06_shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("Saved SHAP summary.")
    except Exception as e:
        print(f"SHAP skipped ({e}). Install with: python -m pip install shap")

    print("\nReports in:", config.REPORTS_DIR)
    print("Charts in:", config.FIGURES_DIR)


if __name__ == "__main__":
    main()