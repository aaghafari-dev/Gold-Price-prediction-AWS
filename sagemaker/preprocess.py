"""SageMaker Processing entry point: validate raw bars, engineer lag features, split chronologically.

Input: /opt/ml/processing/input/market.csv
Outputs: /opt/ml/processing/train/train.csv and /opt/ml/processing/test/test.csv
"""
from pathlib import Path
import json
import pandas as pd

FEATURES = ["close_lag_1", "close_lag_2", "close_lag_5", "return_1d", "return_5d", "rolling_mean_5", "rolling_std_5", "rolling_mean_20", "rolling_std_20", "volume_lag_1"]

def main():
    """Parse command-line options and execute the script entry point."""
    input_path = Path('/opt/ml/processing/input/market.csv')
    train_dir = Path('/opt/ml/processing/train'); test_dir = Path('/opt/ml/processing/test')
    train_dir.mkdir(parents=True, exist_ok=True); test_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if not {'Date', 'Close'}.issubset(df.columns):
        raise ValueError(f"Raw input requires Date and Close columns; found {list(df.columns)}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if 'Volume' not in df: df['Volume'] = 0.0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Date', 'Close']).query('Close > 0').sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True)
    close = df['Close']
    out = pd.DataFrame({'Date': df['Date'].dt.strftime('%Y-%m-%d'), 'gold_close': close, 'target_price': close.shift(-1), 'target_return': close.shift(-1) / close - 1,
        'close_lag_1': close, 'close_lag_2': close.shift(1), 'close_lag_5': close.shift(4), 'return_1d': close.pct_change(), 'return_5d': close.pct_change(5),
        'rolling_mean_5': close.rolling(5).mean(), 'rolling_std_5': close.rolling(5).std(), 'rolling_mean_20': close.rolling(20).mean(), 'rolling_std_20': close.rolling(20).std(), 'volume_lag_1': df['Volume'].shift(1)})
    out = out.dropna().replace([float('inf'), float('-inf')], pd.NA).dropna().reset_index(drop=True)
    if len(out) < 100: raise ValueError(f'Need at least 100 usable rows after feature engineering; got {len(out)}')
    cut = int(len(out) * 0.8)
    out.iloc[:cut].to_csv(train_dir / 'train.csv', index=False)
    out.iloc[cut:].to_csv(test_dir / 'test.csv', index=False)
    report = {'input_rows_after_cleaning': int(len(df)), 'feature_rows': int(len(out)), 'train_rows': int(cut), 'test_rows': int(len(out)-cut), 'train_end': str(out.iloc[cut-1]['Date']), 'test_start': str(out.iloc[cut]['Date']), 'feature_columns': FEATURES, 'split': 'chronological 80/20; no shuffle'}
    Path('/opt/ml/processing/train/split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__': main()
