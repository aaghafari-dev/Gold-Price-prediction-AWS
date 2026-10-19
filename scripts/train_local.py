"""Train locally and print a concise run summary."""
import argparse, json, sys, os
# Windows/joblib: avoid fragile physical-core detection; set a conservative default.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.train import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--news", default=None, help="Normalized timestamped news CSV; defaults to data/raw/news.csv if present")
parser.add_argument("--refresh-market", action="store_true", help="Force market data refresh")
parser.add_argument("--models", nargs="+", default=None, help="Optional subset: XGBoost 'Random Forest' 'Extra Trees' HistGradientBoosting")
parser.add_argument("--min-improvement-pct", type=float, default=0.0, help="Minimum walk-forward RMSE improvement over the strongest configured baseline to create a candidate")
parser.add_argument("--approve-promotion", action="store_true", help="Explicitly approve promotion during this run; review-first workflow is preferred")
parser.add_argument("--min-news-coverage", type=float, default=0.8, help="Minimum fraction of training dates inside logged query windows (not proof of article completeness)")
parser.add_argument("--allow-unverified-news-experiment", action="store_true", help="Exploratory only: evaluate recent-feed news without a verified point-in-time archive; candidates using it cannot be promoted")
parser.add_argument("--robust-seeds", action="store_true", help="Slower sensitivity analysis: use three deterministic model seeds per walk-forward fold")
parser.add_argument("--period", default=None, help="Shared lookback window used by the GUI, e.g. 1d, 180d, 5y")
parser.add_argument("--test-fraction", type=float, default=0.20, help="Newest chronological holdout fraction (0.10–0.40)")
parser.add_argument("--use-news-features", action="store_true", help="Include news features in an explicitly requested exploratory experiment; unverified news models remain ineligible for promotion")
parser.add_argument("--disable-tuning", action="store_true", help="Disable nested time-series hyperparameter tuning and feature selection")
parser.add_argument("--compute-mode", choices=["cpu", "gpu", "auto"], default="cpu", help="Compute mode; GPU applies to XGBoost only and requires a CUDA-enabled XGBoost installation")
args = parser.parse_args()
result = run(news_path=args.news, models=args.models, refresh_market=args.refresh_market, min_improvement_pct=args.min_improvement_pct, approve_promotion=args.approve_promotion, min_news_coverage=args.min_news_coverage, allow_unverified_news=args.allow_unverified_news_experiment, robust_seeds=args.robust_seeds, test_fraction=args.test_fraction, use_news_features=args.use_news_features, nested_tuning=not args.disable_tuning, compute_mode=args.compute_mode, period=args.period)
print(json.dumps({"run_id": result["run_id"], "training_rows": result["training_rows"], "holdout_test_rows": result.get("holdout_test_rows"), "holdout_test_fraction": result.get("holdout_test_fraction"), "news_rows": result["news_rows"], "best": result["best"], "promotion": result["promotion"], "selection_status": result.get("selection_status"), "selection_reason": result.get("selection_reason")}, indent=2, default=str))
