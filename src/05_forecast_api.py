import importlib.util
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

import config

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    """Import a script whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load("train_mod", "03_train_forecast_models.py")   # add_features, predict_model
inv = _load("inv_mod", "04_inventory_optimization.py")      # safety-stock helpers, ERROR_BUFFER

HORIZON = train.HORIZON
WEATHER_COLS = ["temp_mean", "temp_max", "temp_min", "precip"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)
STATE = {}


# ------------------------------------------------------------------ weather
def _daily_frame(d):
    return pd.DataFrame({
        "t_dat": pd.to_datetime(d["time"]),
        "temp_mean": d["temperature_2m_mean"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precip": d["precipitation_sum"],
    })


def fetch_daily_weather(start, end):
    """Past dates -> archive API. Recent/future dates -> live forecast API."""
    today = pd.Timestamp.today().normalize()
    base = {"latitude": config.LAT, "longitude": config.LON,
            "daily": config.WEATHER_VARS, "timezone": config.TIMEZONE}
    if end < today - pd.Timedelta(days=7):
        params = {**base, "start_date": str(start.date()), "end_date": str(end.date())}
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
        label = "open-meteo archive (observed)"
    else:
        params = {**base, "past_days": 14, "forecast_days": 16}
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        label = "open-meteo live forecast"
    r.raise_for_status()
    df = _daily_frame(r.json()["daily"])
    return df[(df["t_dat"] >= start) & (df["t_dat"] <= end)], label


def weekly_weather(future_weeks, weekly_hist):
    """One row per future week. Falls back to same-week-of-year averages (climatology)."""
    clim = weekly_hist.groupby("week_of_year")[WEATHER_COLS].mean()
    start, end = min(future_weeks), max(future_weeks) + pd.Timedelta(days=6)
    daily, label = None, None
    try:
        daily, label = fetch_daily_weather(start, end)
    except Exception as e:
        print(f"[weather] API unavailable ({e}). Using climatology.")

    rows = []
    for ws in future_weeks:
        got = None
        if daily is not None:
            d = daily[(daily["t_dat"] >= ws) & (daily["t_dat"] <= ws + pd.Timedelta(days=6))].dropna()
            if len(d) == 7:
                got = {"temp_mean": d["temp_mean"].mean(), "temp_max": d["temp_max"].max(),
                       "temp_min": d["temp_min"].min(), "precip": d["precip"].sum(),
                       "weather_source": label}
        if got is None:
            woy = int(ws.isocalendar()[1])
            c = clim.loc[woy] if woy in clim.index else clim.mean()
            got = {**{k: float(c[k]) for k in WEATHER_COLS}, "weather_source": "climatology"}
        got["week_start"] = ws
        rows.append(got)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calendar
def _black_friday(year):
    nov1 = pd.Timestamp(year, 11, 1)
    first_friday = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
    return first_friday + pd.Timedelta(weeks=3)


def calendar_features(week_starts):
    years = range(min(week_starts).year, max(week_starts).year + 2)
    hol = holidays.country_holidays(config.COUNTRY_CODE, years=years)
    hol_dates = pd.to_datetime(sorted(hol.keys()))
    hol_vals = hol_dates.values.astype("datetime64[ns]")
    rows = []
    for ws in week_starts:
        days = pd.date_range(ws, periods=7)
        idx = np.clip(np.searchsorted(hol_vals, days.values.astype("datetime64[ns]")),
                      0, len(hol_vals) - 1)
        dth = ((hol_vals[idx] - days.values.astype("datetime64[ns]"))
               / np.timedelta64(1, "D")).clip(0, 30)
        bf_days = {_black_friday(y) for y in {d.year for d in days}}
        rows.append({
            "week_start": ws,
            "holiday_days": int(days.isin(hol_dates).sum()),
            "black_friday": int(any(d in bf_days for d in days)),
            "xmas_days": int(((days.month == 12) & (days.day <= 24)).sum()),
            "days_to_holiday": float(dth.min()),
            "week_of_year": int(ws.isocalendar()[1]),
            "month": int(ws.month),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ forecasting
def build_forecast(weekly, bundle, bt, seg):
    types = bundle["product_types"]
    last = weekly["week_start"].max()
    future_weeks = [last + pd.Timedelta(weeks=i) for i in range(1, HORIZON + 1)]

    ctx = calendar_features(future_weeks).merge(weekly_weather(future_weeks, weekly),
                                                on="week_start")
    fut = pd.MultiIndex.from_product([types, future_weeks],
                                     names=["product_type", "week_start"]).to_frame(index=False)
    fut = fut.merge(ctx, on="week_start", how="left")
    for c in ("units", "revenue", "avg_price"):
        fut[c] = np.nan

    keep = [c for c in fut.columns if c != "weather_source"]
    allrows = pd.concat([weekly[[c for c in keep if c in weekly.columns]], fut[keep]],
                        ignore_index=True)
    allrows["product_type"] = pd.Categorical(allrows["product_type"], categories=types)
    feats = train.add_features(allrows)
    newf = feats[feats["week_start"].isin(future_weeks)].copy()

    # model predictions (same logic as the backtest)
    preds = {n: train.predict_model(m["model"], newf, m["features"], m["mode"])
             for n, m in bundle["models"].items()}
    seasonal = newf["lag_52"].fillna(newf["lag_4"]).values
    best = bundle["best"]
    if best in bundle["ensembles"]:
        members, with_seasonal = bundle["ensembles"][best]
        parts = [preds[m] for m in members] + ([seasonal] if with_seasonal else [])
        newf["forecast"] = np.mean(parts, axis=0)
    elif best == "seasonal_naive":
        newf["forecast"] = seasonal
    else:
        newf["forecast"] = preds[best]

    # safety stock (same method as step 4)
    best_col = f"pred_{best}"
    bt = bt.copy()
    bt["eps"] = inv.rel_errors(bt, best_col)
    sigma_rel = bt.groupby("product_type", observed=True)["eps"].std(ddof=1)

    src_by_week = ctx.set_index("week_start")["weather_source"]
    products = []
    for ptype, g in newf.groupby("product_type", observed=True):
        g = g.sort_values("week_start")
        f = g["forecast"].values
        tgt = float(seg.loc[ptype, "target_service"])
        z = inv.z_value(tgt)
        ss = z * inv.cycle_sigma(float(sigma_rel[ptype]), f, inv.ERROR_BUFFER)
        products.append({
            "product_type": str(ptype),
            "class": str(seg.loc[ptype, "class"]),
            "target_service": tgt,
            "weekly_forecast": [{"week_start": str(w.date()), "units": round(float(u))}
                                for w, u in zip(g["week_start"], f)],
            "forecast_4wk": round(float(f.sum())),
            "safety_stock": round(float(ss)),
            "order_up_to": round(float(f.sum() + ss)),
        })
    products.sort(key=lambda p: -p["order_up_to"])

    return {
        "model": best,
        "last_actual_week": str(last.date()),
        "forecast_weeks": [str(w.date()) for w in future_weeks],
        "weather_source_by_week": {str(w.date()): src_by_week[w] for w in future_weeks},
        "city": config.CITY,
        "error_buffer": inv.ERROR_BUFFER,
        "products": products,
    }


def get_state(refresh=False):
    if refresh or "forecast" not in STATE:
        weekly = pd.read_csv(config.DATA_PROCESSED / "weekly_features.csv",
                             parse_dates=["week_start"])
        bundle = joblib.load(config.MODELS_DIR / "lgbm_demand_forecaster.joblib")
        bt = pd.read_csv(config.REPORTS_DIR / "backtest_predictions.csv",
                         parse_dates=["week_start"])
        seg = pd.read_csv(config.REPORTS_DIR / "abc_xyz_segments.csv", index_col="product_type")
        STATE["forecast"] = build_forecast(weekly, bundle, bt, seg)
    return STATE["forecast"]


def _with_order_qty(prod, on_hand, on_order):
    p = dict(prod)
    if on_hand is not None:
        p["on_hand"] = on_hand
        p["on_order"] = on_order
        p["order_qty"] = max(0, round(p["order_up_to"] - on_hand - on_order))
    return p


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return jsonify({
        "service": "H&M demand forecasting + inventory API",
        "endpoints": {
            "/health": "service status",
            "/forecast": "all categories; add ?product_type=Dress[&on_hand=40000&on_order=0] "
                         "and ?refresh=1 to recompute",
            "/weather/live": "live 14-day weather forecast from Open-Meteo",
            "/dashboard": "visual dashboard",
        },
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/forecast")
def forecast():
    try:
        data = get_state(refresh=request.args.get("refresh") == "1")
    except FileNotFoundError as e:
        return jsonify({"error": f"Missing file: {e.filename}. Run steps 1-4 first."}), 500

    ptype = request.args.get("product_type")
    on_hand = request.args.get("on_hand", type=float)
    on_order = request.args.get("on_order", default=0.0, type=float)

    out = dict(data)
    if ptype:
        match = [p for p in data["products"] if p["product_type"].lower() == ptype.lower()]
        if not match:
            return jsonify({"error": f"Unknown product_type '{ptype}'",
                            "available": [p["product_type"] for p in data["products"]]}), 404
        out["products"] = [_with_order_qty(match[0], on_hand, on_order)]
    else:
        out["products"] = [_with_order_qty(p, None, 0) for p in data["products"]]
    return jsonify(out)


@app.route("/weather/live")
def weather_live():
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": config.LAT, "longitude": config.LON,
            "daily": config.WEATHER_VARS, "forecast_days": 14,
            "timezone": config.TIMEZONE}, timeout=30)
        r.raise_for_status()
        df = _daily_frame(r.json()["daily"])
        df["t_dat"] = df["t_dat"].dt.strftime("%Y-%m-%d")
        return jsonify({"city": config.CITY, "source": "open-meteo live forecast",
                        "daily": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": f"Weather API unavailable: {e}"}), 503


@app.route("/dashboard")
def dashboard():
    return send_from_directory(HERE / "static", "dashboard.html")


if __name__ == "__main__":
    print("Starting API on http://127.0.0.1:5000  (first /forecast call builds the forecast)")
    app.run(host="127.0.0.1", port=5000, debug=False)