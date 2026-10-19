"""Poll public RSS/API headlines, score with FinBERT, and maintain live sentiment snapshots.

This is a near-live polling monitor, not a push stream or an intraday price forecaster.
It records timestamped sentiment vectors and EWMA features so they can be inspected and
used in a future point-in-time validated intraday model.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except Exception:
    pass
from src.data.news import fetch_news_provider, save_news
from scripts.compare_sentiment import score_news


def main():
    """Parse command-line options and execute the script entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', default='google_news_rss', choices=['google_news_rss','newsapi','alphavantage'])
    parser.add_argument('--query', default='gold OR bullion OR "Federal Reserve" OR inflation OR geopolitics OR US President')
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--once', action='store_true', help='Fetch one cycle then exit')
    parser.add_argument('--finbert', action='store_true', help='Use ProsusAI/finbert; requires requirements-nlp.txt and model download')
    args = parser.parse_args()
    if args.poll_seconds < 60 and not args.once:
        parser.error('--poll-seconds must be at least 60 to avoid hammering providers')
    archive = ROOT / 'data/raw/news.csv'
    scored_path = ROOT / 'data/processed/live_news_sentiment.csv'
    while True:
        stamp = pd.Timestamp.now(tz='UTC')
        try:
            fresh = fetch_news_provider(args.provider, args.query, count=100)
            if fresh.empty:
                raise RuntimeError('Provider returned no usable timestamped headlines')
            # Persist each cycle; save_news deduplicates on publication timestamp/title/URL.
            save_news(fresh, archive, append=True, ingestion_metadata={'adapter': args.provider, 'status':'success', 'query':args.query, 'timespan':'live-poll'})
            scored = score_news(fresh, use_finbert=args.finbert)
            scored['observed_at_utc'] = stamp
            # Snapshot features are based only on articles published by the observation time.
            scored['published_at'] = pd.to_datetime(scored['published_at'].astype('string'), utc=True, errors='coerce')
            scored = scored.dropna(subset=['published_at']).copy()
            scored = scored.sort_values('published_at', kind='stable')
            now = pd.Timestamp.now(tz='UTC')
            for label, minutes in [('30m',30),('2h',120),('24h',1440)]:
                window = scored[scored['published_at'] >= now - pd.Timedelta(minutes=minutes)]
                scored[f'sentiment_ewma_{label}'] = float(window['sentiment_score'].ewm(halflife=3, adjust=False).mean().iloc[-1]) if not window.empty else 0.0
            scored_path.parent.mkdir(parents=True, exist_ok=True)
            if scored_path.exists():
                old = pd.read_csv(scored_path)
                scored = pd.concat([old, scored], ignore_index=True, sort=False)
            if 'url' in scored:
                scored = scored.drop_duplicates(subset=[c for c in ['published_at','title','url'] if c in scored], keep='last')
            scored.to_csv(scored_path, index=False)
            print(f'[{stamp.isoformat()}] OK provider={args.provider} new_headlines={len(fresh)} scored_rows={len(scored)} output={scored_path.relative_to(ROOT)}', flush=True)
        except Exception as exc:
            print(f'[{stamp.isoformat()}] ERROR provider={args.provider}: {type(exc).__name__}: {exc}', flush=True)
        if args.once:
            break
        time.sleep(args.poll_seconds)

if __name__ == '__main__':
    main()
