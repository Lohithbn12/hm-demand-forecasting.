import sys
import numpy as np
import pandas as pd
import requests
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


# ------------------------------------------------------------- weather APIs
def _weather_frame(d):
    return pd.DataFrame({
        "t_dat": pd.to_datetime(d["time"]),
        "temp_mean": d["temperature_2m_mean"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precip": d["precipitation_sum"],
    })


def fetch_historical_weather(start, end):
    """Open-Meteo archive API (free, no key). Cached after first call."""
    cache = config.DATA_EXTERNAL / "weather_daily.csv"
    if cache.exists():
        print("Using cached weather:", cache)
        return pd.read_csv(cache, parse_dates=["t_dat"])
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": config.LAT, "longitude": config.LON,
                    "start_date": str(start.date()), "end_date": str(end.date()),
                    "daily": config.WEATHER_VARS, "timezone": config.TIMEZONE},
            timeout=60,
        )
        r.raise_for_status()
        df = _weather_frame(r.json()["daily"])
        df.to_csv(cache, index=False)
        print(f"Historical weather fetched: {len(df)} days")
        return df
    except Exception as e:
        print(f"WARNING: weather API failed ({e}). Continuing WITHOUT weather features.")
        return None


def fetch_live_forecast(days=14):
    """Live forecast from the Open-Meteo forecast API (used in Step 5)."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": config.LAT, "longitude": config.LON,
                    "daily": config.WEATHER_VARS, "forecast_days": days,
                    "timezone": config.TIMEZONE},
            timeout=60,
        )
        r.raise_for_status()
        df = _weather_frame(r.json()["daily"])
        df.to_csv(config.DATA_EXTERNAL / "live_weather_forecast.csv", index=False)
        print(f"\nLIVE forecast for {config.CITY} (next {days} days):")
        print(df.head(7).to_string(index=False))
        return df
    except Exception as e:
        print(f"WARNING: live forecast failed ({e}). We can proceed without it.")
        return None


# ------------------------------------------------------------- calendar
def black_friday(year):
    nov1 = pd.Timestamp(year, 11, 1)
    first_friday = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
    return first_friday + pd.Timedelta(weeks=3)


def build_calendar(start, end):
    cal = pd.DataFrame({"t_dat": pd.date_range(start, end, freq="D")})
    hol_dict = holidays.country_holidays(config.COUNTRY_CODE, years=range(start.year, end.year + 2))
    hol_dates = pd.to_datetime(list(hol_dict.keys()))
    hol = np.array(sorted(hol_dates), dtype="datetime64[ns]")
    d = cal["t_dat"].values.astype("datetime64[ns]")

    cal["is_holiday"] = cal["t_dat"].isin(hol_dates).astype(int)
    idx = np.clip(np.searchsorted(hol, d), 0, len(hol) - 1)
    cal["days_to_holiday"] = ((hol[idx] - d) / np.timedelta64(1, "D")).clip(0, 30)

    bf = {black_friday(y) for y in range(start.year, end.year + 1)}
    cal["is_black_friday"] = cal["t_dat"].isin(bf).astype(int)
    cal["is_xmas_season"] = ((cal["t_dat"].dt.month == 12) & (cal["t_dat"].dt.day <= 24)).astype(int)
    return cal


# ------------------------------------------------------------- main
def main():
    src = config.DATA_INTERIM / "daily_sales.csv"
    if not src.exists():
        sys.exit(f"Missing {src}. Run 01_prepare_sales_data.py first, "
                 f"or move daily_sales.csv into data/interim/.")

    sales = pd.read_csv(src, parse_dates=["t_dat"])
    start, end = sales["t_dat"].min(), sales["t_dat"].max()
    print("Sales:", sales.shape, start.date(), "->", end.date())

    weather = fetch_historical_weather(start, end)
    fetch_live_forecast(14)
    cal = build_calendar(start, end)

    # daily context table -> weekly roll-up
    ctx = cal.copy()
    if weather is not None:
        ctx = ctx.merge(weather, on="t_dat", how="left")
    ctx["week_start"] = ctx["t_dat"] - pd.to_timedelta(ctx["t_dat"].dt.weekday, unit="D")

    agg = {"holiday_days": ("is_holiday", "sum"),
           "black_friday": ("is_black_friday", "max"),
           "xmas_days": ("is_xmas_season", "sum"),
           "days_to_holiday": ("days_to_holiday", "min")}
    if weather is not None:
        agg.update({"temp_mean": ("temp_mean", "mean"), "temp_max": ("temp_max", "max"),
                    "temp_min": ("temp_min", "min"), "precip": ("precip", "sum")})
    ctx_weekly = ctx.groupby("week_start").agg(**agg).reset_index()

    # weekly sales per product type (complete 7-day weeks only)
    sales["week_start"] = sales["t_dat"] - pd.to_timedelta(sales["t_dat"].dt.weekday, unit="D")
    weekly = (sales.groupby(["week_start", "product_type"])
              .agg(units=("units", "sum"), revenue=("revenue", "sum"), n_days=("t_dat", "nunique"))
              .reset_index())
    weekly = weekly[weekly["n_days"] == 7].drop(columns="n_days")
    weekly["avg_price"] = weekly["revenue"] / weekly["units"]

    weekly = weekly.merge(ctx_weekly, on="week_start", how="left")
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month
    weekly = weekly.sort_values(["product_type", "week_start"]).reset_index(drop=True)

    out = config.DATA_PROCESSED / "weekly_features.csv"
    weekly.to_csv(out, index=False)
    print("\nSaved:", out, weekly.shape)
    print("Weeks per product type:", weekly.groupby("product_type").size().iloc[0])
    missing = weekly.isna().sum()
    print("Missing values:\n", missing[missing > 0] if (missing > 0).any() else "none")

    # ---------------- EDA charts ----------------
    tot = weekly.groupby("week_start")["units"].sum()
    plt.figure(figsize=(11, 4)); tot.plot()
    plt.title("Total weekly units (top product types)"); plt.ylabel("units"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "01_total_weekly_units.png", dpi=120); plt.close()

    plt.figure(figsize=(11, 4))
    for p in ["Bikini top", "Sweater", "Dress", "Leggings/Tights"]:
        g = weekly[weekly["product_type"] == p].set_index("week_start")["units"]
        if len(g):
            (g / g.mean()).plot(label=p)
    plt.title("Seasonality: weekly units / average"); plt.legend(); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "02_seasonality.png", dpi=120); plt.close()

    if "temp_mean" in weekly.columns:
        corr = pd.Series({p: g["units"].corr(g["temp_mean"])
                          for p, g in weekly.groupby("product_type")}).sort_values()
        plt.figure(figsize=(8, 5)); corr.plot(kind="barh")
        plt.title("Correlation: weekly units vs mean temperature"); plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "03_temperature_correlation.png", dpi=120); plt.close()
        print("\nCorrelation of weekly units with temperature:")
        print(corr.round(2).to_string())

    print("\nTotal units by product type:")
    print(weekly.groupby("product_type")["units"].sum().sort_values(ascending=False).to_string())
    print("\nCharts saved in:", config.FIGURES_DIR)


if __name__ == "__main__":
    main()