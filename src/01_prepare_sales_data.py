import sys
import pandas as pd
from pathlib import Path

TOP_N = 15

# ---------- locate data (works locally AND inside a Kaggle notebook) ----------
KAGGLE_DIR = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")

try:
    BASE = Path(__file__).resolve().parent      # folder containing this script
except NameError:                               # running in a notebook
    BASE = Path.cwd()

if KAGGLE_DIR.exists():
    RAW = KAGGLE_DIR
    OUT = Path("/kaggle/working")
else:
    RAW = BASE / "data" / "raw"
    OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def find_file(name):
    """Return the path to name or name.zip, or None if missing."""
    for candidate in (RAW / name, RAW / f"{name}.zip"):
        if candidate.exists():
            return candidate
    return None


articles_path = find_file("articles.csv")
trans_path = find_file("transactions_train.csv")

print("Looking in:", RAW)
print("articles.csv            ->", articles_path)
print("transactions_train.csv  ->", trans_path)

if articles_path is None or trans_path is None:
    sys.exit(
        "\nMISSING FILES. Download articles.csv and transactions_train.csv "
        f"from Kaggle and place them in:\n  {RAW}\n"
        "(or run this script inside a Kaggle notebook instead)."
    )

# ---------- 1. article -> product type mapping ----------
articles = pd.read_csv(articles_path, usecols=["article_id", "product_type_name"])
type_map = articles.set_index("article_id")["product_type_name"]

# ---------- 2. aggregate transactions in chunks ----------
parts = []
reader = pd.read_csv(
    trans_path,
    usecols=["t_dat", "article_id", "price"],
    dtype={"article_id": "int32", "price": "float32"},
    parse_dates=["t_dat"],
    chunksize=2_000_000,
)
for i, chunk in enumerate(reader):
    chunk["product_type"] = chunk["article_id"].map(type_map)
    agg = (
        chunk.groupby(["t_dat", "product_type"])
        .agg(units=("article_id", "size"), revenue=("price", "sum"))
        .reset_index()
    )
    parts.append(agg)
    print(f"chunk {i} done")

daily = pd.concat(parts).groupby(["t_dat", "product_type"], as_index=False).sum()
daily["avg_price"] = daily["revenue"] / daily["units"]

# ---------- 3. keep top N product types ----------
top = daily.groupby("product_type")["units"].sum().nlargest(TOP_N).index
daily = daily[daily["product_type"].isin(top)].sort_values(["product_type", "t_dat"])

daily.to_csv(OUT / "daily_sales.csv", index=False)
print("\nSaved:", OUT / "daily_sales.csv")
print(daily.shape)
print(daily["product_type"].value_counts())
print(daily["t_dat"].min(), daily["t_dat"].max())