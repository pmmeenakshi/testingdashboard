import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
CSV_DEFAULT = BASE_DIR / "standardized_wide_fy2024_25.csv"
WIDE_PARQUET = BASE_DIR / "wide.parquet"
LONG_PARQUET = BASE_DIR / "long.parquet"

METRIC_COL_REGEX = re.compile(
    r"^(Impact|Tonnage|CO2_Kgs_Averted|Households_Participating|Segregation_Compliance_Pct)_(\d{4}-\d{2})$"
)
ID_COLS_REQUIRED = ["City", "Community", "Pincode"]
ID_COLS_OPTIONAL = ["Lat", "Lon"]

def _detect_metric_month_cols(columns):
    cols, months = [], set()
    for c in columns:
        m = METRIC_COL_REGEX.match(c)
        if m:
            cols.append(c); months.add(m.group(2))
    return cols, sorted(months)

def build_wide_and_long(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # numeric coercion for monthly metric columns
    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # long format (what your app uses for fast filtering/plotting)
    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"])

    # tidy types (helps memory + speed)
    for c in ["City","Community","Pincode"]:
        if c in df.columns:       df[c] = df[c].astype("string")
        if c in long_df.columns:  long_df[c] = long_df[c].astype("string")

    return df, long_df

if __name__ == "__main__":
    if not CSV_DEFAULT.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_DEFAULT}")
    df_wide, df_long = build_wide_and_long(CSV_DEFAULT)
    # write Parquet (snappy compression by default via pyarrow)
    df_wide.to_parquet(WIDE_PARQUET, index=False)
    df_long.to_parquet(LONG_PARQUET, index=False)
    print(f"✅ Wrote {WIDE_PARQUET.name} and {LONG_PARQUET.name}")
