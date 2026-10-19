"""Create a next-trading-session research forecast from the latest Yahoo Finance market row.

This script requires a human-approved champion model. It refreshes market data,
constructs the same lagged feature schema used for training, and writes a JSON
forecast under artifacts/. It does not deploy to AWS and does not fetch news.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.project_config import CHAMPION_MANIFEST, CHAMPION_MODEL, RAW, ARTIFACTS
from src.data.market import fetch_market_bundle, save_market_bundle
from src.data.news import load_news
from src.features.market_features import make_features
from src.features.news_features import aggregate_news_daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', default='2y', help='Market lookback, e.g. 1y, 2y, 5y; use enough history for 60-session features')
    parser.add_argument('--output', default=str(ARTIFACTS / 'latest_forecast.json'))
    args = parser.parse_args()

    if not CHAMPION_MODEL.exists() or not CHAMPION_MANIFEST.exists():
        raise SystemExit('No approved champion found. Run training, review the holdout metrics, and approve only if appropriate.')
    manifest = json.loads(CHAMPION_MANIFEST.read_text(encoding='utf-8'))
    market = fetch_market_bundle(period=args.period)
    save_market_bundle(market)
    if market.empty:
        raise SystemExit('Yahoo Finance returned no market rows.')
    last_date = pd.to_datetime(market['Date']).max()
    latest_close = float(market.loc[pd.to_datetime(market['Date']).eq(last_date), 'gold_close'].iloc[-1])

    news_path = RAW / 'news.csv'
    if news_path.exists():
        news = load_news(news_path)
    else:
        news = pd.DataFrame(columns=['published_at', 'title', 'summary', 'source', 'url', 'retrieved_at'])
    daily_news = aggregate_news_daily(news, market_dates=market['Date'])

    # make_features() intentionally removes rows without a future target. Add a
    # temporary synthetic next row solely so the last real row has a target and
    # survives that function's supervised-data filter. The prediction uses only
    # predictors on the last real market date; the synthetic row is never a feature row.
    synthetic = market.iloc[[-1]].copy()
    synthetic['Date'] = pd.to_datetime(synthetic['Date']).max() + pd.Timedelta(days=1)
    extended = pd.concat([market, synthetic], ignore_index=True)
    feature_frame, _ = make_features(extended, daily_news)
    feature_frame['Date'] = pd.to_datetime(feature_frame['Date'])
    row = feature_frame.loc[feature_frame['Date'].eq(last_date)]
    if row.empty:
        raise SystemExit('Could not construct a valid latest feature row; download a longer period (recommended: 2y or 5y).')
    feature_names = manifest['features']
    missing = [name for name in feature_names if name not in row.columns]
    if missing:
        raise SystemExit(f'Champion schema does not match current feature builder. Missing features: {missing}')
    X = row[feature_names].astype(float)
    model = joblib.load(CHAMPION_MODEL)
    raw = float(model.predict(X)[0])
    if manifest.get('target_type') == 'next_session_return_reconstructed_to_price':
        prediction = latest_close * (1.0 + raw)
        predicted_return = raw
    else:
        prediction = raw
        predicted_return = None
    if not pd.notna(prediction) or prediction <= 0:
        raise SystemExit(f'Model produced an invalid price estimate: {prediction}')
    result = {
        'forecast_created_at_utc': datetime.now(timezone.utc).isoformat(),
        'last_observed_market_date': last_date.strftime('%Y-%m-%d'),
        'forecast_horizon': 'next available trading row (not necessarily tomorrow calendar day)',
        'gold_symbol': 'GC=F',
        'last_observed_gold_close_usd_per_oz': latest_close,
        'predicted_gold_price_usd_per_oz': float(prediction),
        'predicted_return': predicted_return,
        'model_name': manifest.get('model_name'),
        'feature_set': manifest.get('feature_set'),
        'news_rows_available': int(len(news)),
        'news_features_in_champion': any(str(f).startswith('news_') for f in feature_names),
        'holdout_rmse': manifest.get('metrics', {}).get('holdout_rmse') if isinstance(manifest.get('metrics'), dict) else None,
        'holdout_beats_persistence': manifest.get('metrics', {}).get('holdout_beats_persistence') if isinstance(manifest.get('metrics'), dict) else None,
        'disclaimer': 'Educational research estimate only; not financial advice. GC=F is a Yahoo Finance gold-futures proxy, not an official spot-price quote.'
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Wrote forecast: {out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
