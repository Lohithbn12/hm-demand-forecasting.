# Apparel Demand Forecasting & Inventory Planner

Forecasts weekly demand for 15 apparel product categories, turns each forecast into safety stock and
order-up-to levels, and serves everything through a Flask API (with live weather from Open-Meteo) and an
interactive dashboard.

> **Status:** portfolio prototype built on public data (H&M, 2018–2020). It is not deployed on production data.
> It mirrors forecasting and inventory-planning problems in apparel retail and is designed so that a real
> sales and stock extract can replace the public data.

![Dashboard](docs/images/dashboard.png)

## The problem
Apparel demand moves with season, weather and promotions. Planners need two answers for the next 4 weeks:
**how much will we sell**, and **how much extra stock should we hold** so we rarely run out without overstocking.

## What this project does
1. Aggregates ~31M H&M transactions (Sep 2018 – Sep 2020) into weekly sales for the top 15 product types.
2. Adds weather (Open-Meteo API), Swedish holiday and Black Friday features, and market-level seasonality signals.
3. Benchmarks 19 forecasting approaches with a **rolling-origin backtest** (6 folds × 4 weeks, 4-week-ahead).
   All features are lagged at least 4 weeks, so nothing from the future leaks in.
4. Converts forecasts into **ABC-XYZ classes, safety stock and order-up-to levels**, and simulates the policy
   against actual demand.
5. Serves forecasts via a **Flask API** and a **dashboard** with an order calculator.

## Results

### Forecast accuracy (24-week backtest, 4-week-ahead, WAPE = lower is better)
| Model | Family | WAPE % | 90% interval | Bias % |
|---|---|---|---|---|
| **Ensemble: LightGBM (ratio target) + seasonal naive** | ensemble | **23.2** | 19.2 – 27.4 | +3.8 |
| Ensemble: LightGBM + Random Forest + seasonal naive | ensemble | 23.7 | 19.7 – 27.9 | +1.1 |
| LightGBM (Poisson) | boosted trees | 24.8 | 21.4 – 29.0 | −3.3 |
| XGBoost (Poisson) | boosted trees | 25.7 | 21.7 – 30.6 | −3.2 |
| Extra Trees | bagged trees | 25.8 | 21.4 – 31.0 | −3.4 |
| Seasonal naive (same week last year) | baseline | 25.9 | 19.9 – 32.2 | +12.7 |
| Random Forest | bagged trees | 27.5 | 22.9 – 32.8 | −4.2 |
| Ridge regression | linear | 31.2 | 27.4 – 35.3 | −4.5 |
| ETS (damped trend) | statistical | 36.6 | 28.5 – 45.4 | +5.1 |
| 4-week moving average | baseline | 37.7 | 29.9 – 47.3 | −0.2 |
| Last known value | baseline | 39.1 | 30.6 – 49.3 | −0.05 |

(Full table of 19 models: `outputs/reports/model_comparison_extended.csv`.)

![Leaderboard](docs/images/09_model_leaderboard.png)

**What I take from this**
- With about two years of history, "same week last year" is a strong baseline. **No model clearly beats it:**
  the 90% block-bootstrap intervals overlap. I report that rather than claim a win.
- The chosen ensemble has ~10% lower WAPE (23.2 vs 25.9) and much lower bias (+3.8% vs +12.7%).
  Over-forecasting means overstock, so bias matters for inventory.
- Simple methods (moving average, last value) and classical/linear models are clearly worse.
- By category, the ensemble helps most on Blouse (37% → 22%), Shorts (51% → 37%), Sweater (45% → 37%),
  Vest top (37% → 30%) and T-shirt (18.5% → 14.3%), and does not help on Dress, Trousers, Skirt or Bikini top.
- Weather adds value: removing it raises WAPE by about 1.6–2.0 points. Caveat: the backtest uses *observed*
  weather, so this is an upper bound.

![Seasonality](docs/images/02_seasonality.png)

### Inventory policy
- **ABC-XYZ** classes by revenue share and demand variability (e.g. Trousers AX, Dress AY, Sweater AZ).
- Safety stock = z(service goal) × relative forecast error × forecast level × a calibrated multiplier (1.4).
  Using *relative* instead of absolute error fixed inflated off-season buffers
  (Shorts: 100% → 35% of forecast before calibration).
- To compare forecasters fairly, I measured the **excess stock needed to reach the same 91% service level**:

| Forecaster | Excess stock at 91% service |
|---|---|
| Ensemble | 35.9% of demand |
| Seasonal naive | 36.7% |
| LightGBM (Poisson) | 43.5% |
| Last known value | 79.4% |
| 4-week moving average | could not reach 91% |

The ensemble and seasonal naive are effectively tied here, and both are far better than the simple methods.

![Service vs cost](docs/images/07_service_cost_frontier.png)

## Limitations
- Only ~2 years of data, so each seasonal peak is seen once or twice. The 24-week test window includes COVID-19.
- Many models were compared on the same 24 weeks, so the top score is somewhat optimistic.
- Prices are anonymized, there is no store location (Stockholm is assumed for weather) and no inventory data.
  Stock numbers in the dashboard calculator are typed in by the user.
- Promotions are not labeled; the model infers them only indirectly.
- The simulation is single-cycle with no carry-over, and the safety multiplier is calibrated on the same backtest.
- "Replay mode": the API forecasts the 4 weeks after the data ends (21 Sep – 12 Oct 2020) using observed
  weather. With current sales data it would use Open-Meteo's live forecast (and climatology beyond ~16 days).

## Run it
```bash
pip install -r requirements.txt
# download articles.csv and transactions_train.csv from the Kaggle competition into data/raw/
python run_pipeline.py                 # steps 1-4 (+ benchmark)
python src/05_forecast_api.py          # then open http://127.0.0.1:5000/dashboard
```
Endpoints: `/forecast`, `/forecast?product_type=Dress&on_hand=40000&on_order=5000`, `/weather/live`, `/dashboard`.

## Repo structure
```
src/
  config.py                    paths and settings
  01_prepare_sales_data.py     transactions -> daily sales by product type
  02_build_features.py         weather API, holidays, weekly features, EDA charts
  03_train_forecast_models.py  rolling backtest, models, ensemble, SHAP
  03b_model_benchmark.py       extra models + bootstrap intervals
  04_inventory_optimization.py ABC-XYZ, safety stock, policy simulation
  05_forecast_api.py           Flask API
  static/dashboard.html        dashboard
outputs/   reports (CSV) and figures
models/    trained model bundle
```

## Data and attribution
- Sales data: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
  (Kaggle). Subject to the competition rules, so the data is not included in this repo.
- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

## What I would do next
Add labeled promotion data, forecast at SKU level, use a longer history, add a multi-period inventory
simulation, and validate on a real sales and stock extract.