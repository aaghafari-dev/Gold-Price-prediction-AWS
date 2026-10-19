"""Fetch timestamped financial-news headlines from configurable public/API providers."""
import argparse, sys, os, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass
from src.data.news import fetch_news_provider, save_news

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', default='google_news_rss', help='Comma-separated providers: google_news_rss,newsapi,alphavantage; legacy gdelt,yfinance still supported')
parser.add_argument('--symbol', default='GC=F')
parser.add_argument('--query', default='gold OR bullion OR "Federal Reserve" OR inflation OR geopolitics OR US President')
parser.add_argument('--timespan', default='7d', help='Requested shared lookback, e.g. 1d, 30d, 180d, 1y, 5y; providers may return only recent headlines')
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--output', default='data/raw/news.csv')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
providers = [p.strip().lower() for p in args.source.split(',') if p.strip()]
if not providers:
    parser.error('--source must name at least one provider')
all_rows, failures = 0, 0
for provider in providers:
    now = datetime.now(timezone.utc)
    # Log the requested query window, not a hard-coded 30-day interval.
    match = re.fullmatch(r"\s*(\d+)\s*(h|d|w|mo|y)\s*", str(args.timespan).lower())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = timedelta(hours=amount) if unit == "h" else (timedelta(days=amount) if unit == "d" else (timedelta(weeks=amount) if unit == "w" else (timedelta(days=30 * amount) if unit == "mo" else timedelta(days=365 * amount))))
    else:
        delta = timedelta(days=7)
    metadata = {'adapter': provider, 'query': args.query, 'timespan': args.timespan,
                'query_window_start_utc': (now - delta).isoformat(), 'query_window_end_utc': now.isoformat()}
    try:
        frame = fetch_news_provider(provider, args.query, symbol=args.symbol, timespan=args.timespan, count=args.count)
        # Enforce schema and avoid an empty-success status for a provider that returned no valid rows.
        if frame.empty:
            raise RuntimeError('Provider returned zero usable records; archive was not cleared.')
        metadata['status'] = 'success'
        print(f'OK [{provider}]: {len(frame)} usable timestamped headlines.')
    except Exception as exc:
        frame = pd.DataFrame(columns=['published_at','retrieved_at','title','summary','source','url','provider'])
        metadata['status'] = 'error'; metadata['error'] = f'{type(exc).__name__}: {exc}'; failures += 1
        print(f'ERROR [{provider}]: {metadata["error"]}')
    # Never overwrite a populated archive with an empty result; in multi-provider mode,
    # append all successful results and preserve a prior archive if every provider fails.
    do_overwrite = bool(args.overwrite and all_rows == 0 and not frame.empty)
    out = save_news(frame, Path(args.output), append=not do_overwrite, ingestion_metadata=metadata)
    all_rows += len(frame)
    print(f'Logged [{provider}] rows_this_run={len(frame)} archive={out}')
print(f'SUMMARY: providers={",".join(providers)} rows_returned_total={all_rows} provider_errors={failures}')
print('Important: a successful fetch is not proof of complete historical coverage. See data/raw/news_ingestion_log.csv.')
if failures == len(providers):
    raise SystemExit(2)
