"""Validate raw inputs and write the engineered training table and feature manifest."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.market import load_or_fetch_market, fetch_market_bundle, save_market_bundle
from src.data.news import load_news
from src.features.news_features import aggregate_news_daily
from src.features.market_features import make_features
from config.project_config import RAW, PROCESSED, ARTIFACTS
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--news", default=str(RAW / "news.csv"), help="Normalized news CSV; optional if it does not exist")
parser.add_argument("--out", default=str(PROCESSED / "train.csv"))
parser.add_argument("--years", type=int, default=10, help="Legacy lookback in years")
parser.add_argument("--period", default=None, help="Shared market/news analysis lookback, e.g. 1d, 30d, 6mo, 1y, 5y")
parser.add_argument("--refresh-market", action="store_true")
args = parser.parse_args()
if args.refresh_market:
    market = fetch_market_bundle(years=args.years, period=args.period)
    save_market_bundle(market)
else:
    market = load_or_fetch_market(period=args.period)
news_path = Path(args.news)
if news_path.exists():
    news = load_news(news_path)
else:
    import pandas as pd
    print(f"WARNING: {news_path} not found; creating market-only features.")
    news = pd.DataFrame(columns=["published_at", "title", "summary", "source", "url", "retrieved_at"])
# Write a raw-ingestion quality audit before feature validation, so a failed validation
# still leaves diagnostics. Weekend/holiday calendar gaps are normal for daily bars.
import pandas as pd
market_dates = pd.to_datetime(market["Date"], errors="coerce")
market_quality = {
    "market_rows_received": int(len(market)),
    "invalid_date_rows_received": int(market_dates.isna().sum()),
    "duplicate_date_rows_received": int(market_dates.duplicated().sum()),
    "date_sorted_on_arrival": bool(market_dates.is_monotonic_increasing),
    "gold_close_missing_or_non_numeric": int(pd.to_numeric(market["gold_close"], errors="coerce").isna().sum()),
    "gold_close_non_positive": int((pd.to_numeric(market["gold_close"], errors="coerce") <= 0).sum()),
    "first_market_date_received": str(market_dates.min().date()) if market_dates.notna().any() else None,
    "last_market_date_received": str(market_dates.max().date()) if market_dates.notna().any() else None,
    "news_rows_loaded": int(len(news)),
    "requested_period": args.period or f"{args.years}y",
    "note": "Weekends/market holidays are not missing trading observations. News lookback requests do not guarantee historical source completeness."
}
ARTIFACTS.mkdir(parents=True, exist_ok=True)
(ARTIFACTS / "data_quality_report.json").write_text(json.dumps(market_quality, indent=2), encoding="utf-8")
print("INGESTION DATA QUALITY AUDIT:", json.dumps(market_quality, indent=2), flush=True)
daily = aggregate_news_daily(news, market_dates=market["Date"])
frame, features = make_features(market, daily)
market_quality.update({
    "feature_rows": int(len(frame)),
    "feature_rows_removed_by_warmup_target_or_validation": int(max(0, len(market) - len(frame))),
    "first_supervised_date": str(frame["Date"].min().date()) if len(frame) else None,
    "last_supervised_date": str(frame["Date"].max().date()) if len(frame) else None,
})
(ARTIFACTS / "data_quality_report.json").write_text(json.dumps(market_quality, indent=2), encoding="utf-8")
print("FINAL DATA QUALITY AUDIT:", json.dumps(market_quality, indent=2), flush=True)
if frame.empty:
    print("WARNING: no supervised rows remain after rolling-window warm-up and target construction. This window is usable for ingestion/plotting only, not ML evaluation.", flush=True)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(out, index=False)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
(ARTIFACTS / "prepared_feature_manifest.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
print(f"Wrote {len(frame)} rows and {len(features)} features to {out}")
print("Forecast target: next trading row's gold_close; news features shifted by one market row.")
