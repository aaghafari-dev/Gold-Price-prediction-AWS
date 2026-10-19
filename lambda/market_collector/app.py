"""Scheduled Lambda collector for the educational Gold Price Prediction project.

Fetches a recent window of Yahoo Finance daily bars and writes a normalized CSV
snapshot to the project's private S3 bucket. This is not an exchange-certified feed.
"""
from __future__ import annotations
import csv
import io
import json
import os
from datetime import datetime, timezone
import boto3
import yfinance as yf
import pandas as pd

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """Document the purpose and behavior of lambda handler. """
    bucket = os.environ["DATA_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "gold-price-prediction").strip("/")
    symbol = os.environ.get("GOLD_SYMBOL", "GC=F")
    period = os.environ.get("FETCH_PERIOD", "10d")
    ticker = yf.Ticker(symbol)
    frame = ticker.history(period=period, interval="1d", auto_adjust=False, actions=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {symbol}; check source availability and CloudWatch logs.")
    frame = frame.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    normalized = io.StringIO()
    writer = csv.DictWriter(normalized, fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Symbol", "RetrievedAtUTC"])
    writer.writeheader()
    retrieved = datetime.now(timezone.utc).isoformat()
    rows = 0
    for _, row in frame.iterrows():
        close = row.get("Close")
        if close is None or not float(close) > 0:
            continue
        date_value = row[date_col]
        date_text = date_value.date().isoformat() if hasattr(date_value, "date") else str(date_value)[:10]
        writer.writerow({"Date": date_text, "Open": row.get("Open", ""), "High": row.get("High", ""), "Low": row.get("Low", ""), "Close": close, "Adj Close": row.get("Adj Close", close), "Volume": row.get("Volume", ""), "Symbol": symbol, "RetrievedAtUTC": retrieved})
        rows += 1
    if rows == 0:
        raise RuntimeError("No valid positive closing prices remained after normalization.")
    now = datetime.now(timezone.utc)
    key = f"{prefix}/raw/market/daily/snapshot_date={now:%Y-%m-%d}/{symbol.replace('=', '_').replace('^', '')}.csv"
    payload = normalized.getvalue().encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="text/csv", ServerSideEncryption="AES256", Metadata={"symbol": symbol, "rows": str(rows), "retrieved-at-utc": retrieved})
    # Idempotently merge the recent window into the bootstrap object used by SageMaker.
    history_key = f"{prefix}/raw/market/bootstrap_market_history.csv"
    recent = pd.read_csv(io.BytesIO(payload))
    try:
        old_obj = s3.get_object(Bucket=bucket, Key=history_key)
        previous = pd.read_csv(old_obj["Body"])
        merged = pd.concat([previous, recent], ignore_index=True, sort=False)
    except s3.exceptions.NoSuchKey:
        merged = recent
    except Exception as exc:
        # Missing object is normal before bootstrap; fail on other S3 errors rather than overwrite blindly.
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            merged = recent
        else:
            raise
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged["Close"] = pd.to_numeric(merged["Close"], errors="coerce")
    merged = merged.dropna(subset=["Date", "Close"]).query("Close > 0").sort_values("Date").drop_duplicates("Date", keep="last")
    history_bytes = merged.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=history_key, Body=history_bytes, ContentType="text/csv", ServerSideEncryption="AES256", Metadata={"symbol": symbol, "rows": str(len(merged)), "retrieved-at-utc": retrieved})
    result = {"status": "ok", "symbol": symbol, "rows_this_run": rows, "historical_rows_after_merge": int(len(merged)), "s3_uri": f"s3://{bucket}/{key}", "history_s3_uri": f"s3://{bucket}/{history_key}", "retrieved_at_utc": retrieved, "source": "Yahoo Finance via yfinance; educational prototype, not exchange-certified"}
    print(json.dumps(result))
    return result
