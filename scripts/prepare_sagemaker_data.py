"""Prepare chronological train/test CSVs for the optional SageMaker pipeline.

Usage: python scripts/prepare_sagemaker_data.py --input data/raw/market_daily.csv
The default local file must contain Date, Close and optionally Volume columns.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

FEATURES = ["close_lag_1", "close_lag_2", "close_lag_5", "return_1d", "return_5d", "rolling_mean_5", "rolling_std_5", "rolling_mean_20", "rolling_std_20", "volume_lag_1"]

def build_frame(source: Path) -> pd.DataFrame:
    """Document the purpose and behavior of build frame. """
    df = pd.read_csv(source)
    required = {"Date", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input needs columns {sorted(required)}; found {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" not in df: df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Date", "Close"]).query("Close > 0").sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    close = df["Close"]
    out = pd.DataFrame({"Date": df["Date"].dt.strftime("%Y-%m-%d"), "gold_close": close, "target_price": close.shift(-1), "target_return": close.shift(-1) / close - 1,
        "close_lag_1": close, "close_lag_2": close.shift(1), "close_lag_5": close.shift(4), "return_1d": close.pct_change(), "return_5d": close.pct_change(5),
        "rolling_mean_5": close.rolling(5).mean(), "rolling_std_5": close.rolling(5).std(), "rolling_mean_20": close.rolling(20).mean(), "rolling_std_20": close.rolling(20).std(), "volume_lag_1": df["Volume"].shift(1)})
    out = out.dropna().replace([float("inf"), float("-inf")], pd.NA).dropna().reset_index(drop=True)
    return out

def main():
    """Parse command-line options and execute the script entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/market_daily.csv")
    p.add_argument("--output-dir", default="data/processed/sagemaker")
    p.add_argument("--test-fraction", type=float, default=0.2)
    a = p.parse_args()
    if not 0.05 <= a.test_fraction <= 0.4: p.error("--test-fraction must be between 0.05 and 0.4")
    frame = build_frame(Path(a.input))
    if len(frame) < 100: raise ValueError(f"Need at least 100 usable chronological rows; got {len(frame)}")
    cut = int(len(frame) * (1 - a.test_fraction))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame.iloc[:cut].to_csv(out / "sagemaker_train.csv", index=False)
    frame.iloc[cut:].to_csv(out / "sagemaker_test.csv", index=False)
    print(f"Rows={len(frame)} train={cut} test={len(frame)-cut}; train_end={frame.iloc[cut-1]['Date']}; test_start={frame.iloc[cut]['Date']}")
    print(f"Wrote {out / 'sagemaker_train.csv'} and {out / 'sagemaker_test.csv'}")
    print("Note: holdout is chronological; the local research pipeline remains the primary nested walk-forward evaluation.")

if __name__ == "__main__": main()
