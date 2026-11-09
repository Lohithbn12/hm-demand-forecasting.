from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"

for _p in (DATA_RAW, DATA_INTERIM, DATA_EXTERNAL, DATA_PROCESSED,
           MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---- settings ----
TOP_N_PRODUCT_TYPES = 15
CHUNK_SIZE = 2_000_000          # lower to 1_000_000 if memory is tight

# H&M data has no store location, so we assume Stockholm
CITY, LAT, LON, TIMEZONE = "Stockholm", 59.3293, 18.0686, "Europe/Stockholm"
COUNTRY_CODE = "SE"
WEATHER_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"