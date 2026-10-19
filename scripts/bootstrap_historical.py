"""Download a historical Yahoo Finance GC=F proxy snapshot and optionally upload it to S3.

This is a reproducible educational bootstrap, not an exchange-certified settlement feed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

def main():
    """Parse command-line options and execute the script entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', default='GC=F')
    p.add_argument('--period', default='max', help='Yahoo Finance period, e.g. 5y or max')
    p.add_argument('--output', default='data/raw/market_daily.csv')
    p.add_argument('--bucket', default='')
    p.add_argument('--region', default='eu-central-1')
    p.add_argument('--s3-key', default='gold-price-prediction/raw/market/bootstrap_market_history.csv')
    a = p.parse_args()
    data = yf.Ticker(a.symbol).history(period=a.period, interval='1d', auto_adjust=False, actions=False)
    if data is None or data.empty: raise RuntimeError(f'Yahoo Finance returned no history for {a.symbol}.')
    data = data.reset_index()
    date_col = 'Date' if 'Date' in data else data.columns[0]
    out = pd.DataFrame({'Date': pd.to_datetime(data[date_col], errors='coerce').dt.strftime('%Y-%m-%d'), 'Open': data.get('Open'), 'High': data.get('High'), 'Low': data.get('Low'), 'Close': pd.to_numeric(data.get('Close'), errors='coerce'), 'Adj Close': data.get('Adj Close', data.get('Close')), 'Volume': data.get('Volume', 0)})
    out = out.dropna(subset=['Date', 'Close']).query('Close > 0').drop_duplicates('Date').sort_values('Date')
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(path, index=False)
    print(f'Rows={len(out)} date_range={out.Date.min()}..{out.Date.max()} local_file={path.resolve()}')
    if a.bucket:
        import boto3
        boto3.client('s3', region_name=a.region).upload_file(str(path), a.bucket, a.s3_key)
        print(f's3://{a.bucket}/{a.s3_key}')
    print('Reminder: inspect missing dates and source semantics; GC=F is a Yahoo Finance futures proxy, not a guaranteed official settlement record.')
if __name__ == '__main__': main()
