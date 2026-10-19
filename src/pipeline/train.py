"""Reproducible training: market-only vs market+news, walk-forward and gated promotion."""
from __future__ import annotations
import hashlib, json, platform, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import sklearn
import xgboost
from config.project_config import TEST_FOLDS, ARTIFACTS, MODELS, TARGET_HORIZON, MODEL_NAMES, RAW
from src.data.market import load_or_fetch_market
from src.data.news import load_news
from src.data.validation import market_freshness_report
from src.data.time_alignment import load_verified_session_calendar, calendar_audit
from src.features.news_features import aggregate_news_daily
from src.features.market_features import make_features
from src.models.evaluation import walk_forward, fit_final
from src.models.governance import promote, save_candidate, CANDIDATE_MODEL, CANDIDATE_MANIFEST
from src.utils.experiment_tracking import log_to_mlflow


def _sha256(path: Path) -> str | None:
    """Compute a streaming SHA-256 digest for an existing file, returning None when it is absent."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(news_path=None, models=None, refresh_market=False, min_improvement_pct=0.0, approve_promotion=False, min_news_coverage=0.8, allow_unverified_news=False, robust_seeds=False, test_fraction=0.20, use_news_features=False, nested_tuning=True, compute_mode="cpu", period=None):
    """Run the complete local research pipeline and write auditable artifacts.

    Recent GDELT/Yahoo news is not considered a verified historical archive. The
    market-plus-news experiment therefore requires both the configured query-window
    coverage threshold and explicit ``allow_unverified_news=True`` for exploratory
    live-feed experiments. Exploratory unverified-news candidates are never eligible
    for promotion. Set ``NEWS_ARCHIVE_POINT_IN_TIME_VERIFIED=1`` only after manually
    auditing an immutable historical archive and its publication-time coverage.
    """
    # Invalidate old candidates so a failed/new run cannot accidentally approve stale results.
    for candidate_path in (CANDIDATE_MODEL, CANDIDATE_MANIFEST):
        if candidate_path.exists():
            candidate_path.unlink()
    print("[stage 1/6] Loading/refreshing market history...", flush=True)
    market = load_or_fetch_market(force=refresh_market, period=period)
    print(f"[stage 1/6] Market rows loaded: {len(market)}; requested period={period or 'configured default'}", flush=True)
    candidate_news_path = Path(news_path) if news_path else RAW / "news.csv"
    print("[stage 2/6] Loading selected news archive and checking timestamp schema...", flush=True)
    if candidate_news_path.exists():
        news = load_news(candidate_news_path)
    else:
        news = pd.DataFrame(columns=["published_at", "title", "summary", "source", "url", "retrieved_at"])
    if "sentiment_method" in news.columns and news["sentiment_method"].notna().any():
        sentiment_method_used = ", ".join(sorted(news["sentiment_method"].dropna().astype(str).unique()))
    elif "sentiment_score" in news.columns:
        sentiment_method_used = "precomputed sentiment_score (method not declared)"
    else:
        sentiment_method_used = "english_keyword_lexicon"
    print(f"[stage 2/6] News rows loaded: {len(news)}", flush=True)
    calendar_path = RAW / "gc_session_calendar.csv"
    session_calendar = load_verified_session_calendar(calendar_path)
    calendar_status = calendar_audit(session_calendar, calendar_path)
    news_daily = aggregate_news_daily(news, market_dates=market["Date"], session_cutoffs=session_calendar if calendar_status["verified"] else None)
    print("[stage 3/6] Building lagged market and sentiment features...", flush=True)
    frame, all_features = make_features(market, news_daily, TARGET_HORIZON)
    print(f"[stage 3/6] Supervised rows={len(frame)}; candidate features={len(all_features)}", flush=True)
    if len(frame) < TEST_FOLDS + 30:
        raise ValueError(
            f"Only {len(frame)} supervised rows remain after rolling-feature warm-up and target alignment. "
            "This lookback is too short for the configured walk-forward CV plus final chronological holdout. "
            "Use a short window for ingestion/plot/data-quality checks, and select at least about 6 months (preferably 1–5 years) for ML evaluation."
        )

    # A queried date window is not proof of complete source coverage, but it is a
    # minimum guard against treating a recent 7-day feed as ten years of history.
    coverage_log = candidate_news_path.with_name("news_ingestion_log.csv")
    if not coverage_log.exists() and (RAW / "news_ingestion_log.csv").exists():
        coverage_log = RAW / "news_ingestion_log.csv"
    covered = pd.Series(False, index=frame.index)
    coverage_log_rows = 0
    if coverage_log.exists():
        log = pd.read_csv(coverage_log)
        coverage_log_rows = len(log)
        for _, item in log.iterrows():
            start = pd.to_datetime(item.get("query_window_start_utc"), utc=True, errors="coerce")
            end = pd.to_datetime(item.get("query_window_end_utc"), utc=True, errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            start = start.tz_convert(None).normalize()
            end = end.tz_convert(None).normalize()
            covered |= frame["Date"].between(start, end)
    news_coverage_ratio = float(covered.mean()) if len(covered) else 0.0

    import os
    news_features = [c for c in all_features if c.startswith("news_")]
    feature_sets = {"market_only": [c for c in all_features if c not in news_features]}
    archive_declared_verified = os.getenv("NEWS_ARCHIVE_POINT_IN_TIME_VERIFIED", "0").strip().lower() in {"1", "true", "yes"} and bool(calendar_status["verified"])
    coverage_gate_passed = bool(not news.empty and news_features and news_coverage_ratio >= min_news_coverage)
    news_experiment_allowed = bool(not news.empty and news_features and ((coverage_gate_passed and archive_declared_verified) or allow_unverified_news))
    if news_experiment_allowed or (use_news_features and not news.empty and news_features):
        feature_sets["market_plus_news"] = all_features
    if compute_mode not in {"cpu", "gpu", "auto"}:
        raise ValueError("compute_mode must be cpu, gpu, or auto")
    import os
    os.environ["AURUM_COMPUTE_MODE"] = compute_mode

    if not 0.10 <= float(test_fraction) <= 0.40:
        raise ValueError("test_fraction must be between 0.10 and 0.40 to preserve a useful chronological holdout")
    # Reserve the newest observations as a final chronological holdout. Model selection and
    # walk-forward tuning use only the earlier development period; the holdout is scored once.
    holdout_size = max(10, int(round(len(frame) * float(test_fraction))))
    if len(frame) - holdout_size < TEST_FOLDS + 25:
        raise ValueError(f"Not enough rows ({len(frame)}) for {test_fraction:.0%} holdout plus walk-forward development; reduce test size or extend market history")
    split_index = len(frame) - holdout_size
    dev_frame, holdout_frame = frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()
    print(f"[stage 4/6] Chronological split: development={len(dev_frame)}; final holdout={len(holdout_frame)} ({test_fraction:.0%})", flush=True)
    print(f"[stage 5/6] Starting walk-forward evaluation; tuning={nested_tuning}; compute={compute_mode}; feature sets={list(feature_sets)}", flush=True)
    results = []
    target = dev_frame["target_price"].to_numpy(dtype=float)
    baseline = dev_frame["gold_close"].to_numpy(dtype=float)
    seasonal_baseline = dev_frame["seasonal_naive_price"].to_numpy(dtype=float)
    target_return = dev_frame["target_return"].to_numpy(dtype=float)
    for feature_set, feature_cols in feature_sets.items():
        X = dev_frame[feature_cols].astype(float)
        X_holdout = holdout_frame[feature_cols].astype(float)
        for name in (models or MODEL_NAMES):
            print(f"[stage 5/6] Evaluating model={name}; feature_set={feature_set} ...", flush=True)
            metrics = walk_forward(name, X, target, target_return, baseline, n_splits=TEST_FOLDS, gap=TARGET_HORIZON, seasonal_baseline_price=seasonal_baseline, robust_seeds=robust_seeds, nested_tuning=nested_tuning)
            # One final untouched chronological test set, distinct from the CV folds.
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np
            holdout_model = fit_final(name, X, target_return, nested_tuning=nested_tuning)
            predicted_return = np.asarray(holdout_model.predict(X_holdout), dtype=float)
            actual = holdout_frame["target_price"].to_numpy(dtype=float)
            persistence = holdout_frame["gold_close"].to_numpy(dtype=float)
            pred = persistence * (1.0 + predicted_return)
            metrics["holdout_test_fraction"] = float(test_fraction)
            metrics["holdout_test_rows"] = int(len(holdout_frame))
            metrics["holdout_start_date"] = str(holdout_frame["Date"].min())
            metrics["holdout_end_date"] = str(holdout_frame["Date"].max())
            metrics["holdout_rmse"] = float(np.sqrt(mean_squared_error(actual, pred)))
            metrics["holdout_mae"] = float(mean_absolute_error(actual, pred))
            metrics["holdout_directional_accuracy"] = float(np.mean(np.sign(pred - persistence) == np.sign(actual - persistence)))
            metrics["holdout_persistence_rmse"] = float(np.sqrt(mean_squared_error(actual, persistence)))
            metrics["holdout_beats_persistence"] = bool(metrics["holdout_rmse"] < metrics["holdout_persistence_rmse"])
            metrics["feature_set"] = feature_set
            metrics["target_type"] = "next_session_return_reconstructed_to_price"
            metrics["feature_count"] = len(feature_cols)
            metrics["features"] = feature_cols
            results.append(metrics)
            print(f"[stage 5/6] Complete model={name}; feature_set={feature_set}; CV_RMSE={metrics['rmse']:.5f}; beats_baselines={metrics['beats_baseline']}", flush=True)

    print("[stage 6/6] Applying benchmark gate and writing audit/model comparison artifacts...", flush=True)
    eligible = [r for r in results if r["beats_baseline"] and (r["feature_set"] != "market_plus_news" or archive_declared_verified)]
    # Prefer news only when it actually improves out-of-sample RMSE; otherwise market-only wins.
    best = min(eligible, key=lambda r: r["rmse"]) if eligible else None
    promotion = None
    unverified_news_passed = bool(any(r["feature_set"] == "market_plus_news" and r["beats_baseline"] for r in results) and not archive_declared_verified)
    selection_status = "CANDIDATE_ELIGIBLE" if best and best["improvement_pct"] >= min_improvement_pct else ("NEWS_MODEL_INELIGIBLE_UNVERIFIED_ARCHIVE" if best is None and unverified_news_passed else "NO_MODEL_BEAT_BASELINES" if best is None else "BELOW_MINIMUM_IMPROVEMENT")
    selection_reason = (
        f"Best eligible result: {best['model']} ({best['feature_set']}); pooled walk-forward RMSE={best['rmse']:.4f}, "
        f"persistence RMSE={best['baseline_rmse']:.4f}, seasonal-naive RMSE={best.get('seasonal_naive_rmse')}. "
        + ("It meets the configured benchmark and improvement gates." if best["improvement_pct"] >= min_improvement_pct else "It does not meet the configured minimum improvement threshold, so no candidate is created.")
        if best else
        ("A market-plus-news model passed the numerical benchmarks, but its archive is unverified; it is not eligible for selection or promotion. " if unverified_news_passed else "No tested model passed the configured benchmark gate (persistence and 5-session seasonal-naive where available). ")
        + "This is a valid evaluation outcome, not necessarily a runtime error; inspect artifacts/model_comparison.csv, "
        "fold metrics, feature quality, target horizon, and market regime before changing the acceptance rule."
    )
    if best and best["improvement_pct"] >= min_improvement_pct and (best["feature_set"] != "market_plus_news" or archive_declared_verified):
        selected_features = best["features"]
        final_model = fit_final(best["model"], dev_frame[selected_features].astype(float), target_return, nested_tuning=nested_tuning)
        save_candidate(final_model, best, selected_features, best["model"], best["feature_set"])
        if approve_promotion:
            promotion = promote(final_model, best, selected_features, best["model"], feature_set=best["feature_set"])
        else:
            promotion = {"status": "PENDING_HUMAN_APPROVAL", "candidate_manifest": "models/candidate_manifest.json", "next_step": "Review candidate metrics, then run python scripts/promote_candidate.py --approve"}

    serializable_results = [{k: v for k, v in row.items() if k != "features"} for row in results]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    freshness = market_freshness_report(market)
    # Immutable per-run copies of the exact input files make reruns auditable.
    snapshot_dir = ARTIFACTS / "runs" / run_id / "inputs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_files = {}
    for label, source in {"market_daily.csv": RAW / "market_daily.csv", "news.csv": candidate_news_path,
                          "news_ingestion_log.csv": coverage_log, "gc_session_calendar.csv": calendar_path}.items():
        if source.exists() and source.is_file():
            destination = snapshot_dir / label
            if source.resolve() != destination.resolve(): shutil.copy2(source, destination)
            snapshot_files[label] = {"snapshot_path": str(destination), "sha256": _sha256(destination), "bytes": destination.stat().st_size}
    requirements_path = Path(__file__).resolve().parents[2] / "requirements.txt"
    audit = {
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {"input_snapshots": snapshot_files, "requirements_sha256": _sha256(requirements_path),
                            "python_executable": sys.executable, "random_seed": 42, "code_version": "14.0.0"},
        "app_version": "14.0.0", "target_type": "next_session_return_reconstructed_to_price", "python_version": platform.python_version(),
        "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__, "market_rows": int(len(market)),
        "training_rows": int(len(dev_frame)), "holdout_test_rows": int(len(holdout_frame)), "holdout_test_fraction": float(test_fraction),
        "holdout_start_date": str(holdout_frame["Date"].min()), "holdout_end_date": str(holdout_frame["Date"].max()), "news_rows": int(len(news)),
        "news_file": str(candidate_news_path), "news_sha256": _sha256(candidate_news_path),
        "news_ingestion_log": str(coverage_log) if coverage_log.exists() else None,
        "news_ingestion_log_rows": coverage_log_rows,
        "news_query_window_coverage_ratio": news_coverage_ratio,
        "minimum_news_coverage_ratio": min_news_coverage,
        "news_query_window_coverage_gate_passed": coverage_gate_passed,
        "news_archive_declared_point_in_time_verified": archive_declared_verified,
        "allow_unverified_news_experiment": bool(allow_unverified_news),
        "news_experiment_allowed": bool(news_experiment_allowed),
        "news_coverage_gate_passed": bool(coverage_gate_passed and archive_declared_verified),
        "news_experiment_status": ("VERIFIED_ARCHIVE" if archive_declared_verified and news_experiment_allowed else "EXPLORATORY_UNVERIFIED" if allow_unverified_news and news_experiment_allowed else "BLOCKED_BY_COVERAGE_OR_ARCHIVE_GATE"),
        "market_sha256": _sha256(RAW / "market_daily.csv"), "target_horizon_sessions": TARGET_HORIZON,
        "gc_session_calendar_path": str(calendar_path), "gc_session_calendar_audit": calendar_status,
        "nested_chronological_tuning_enabled": bool(nested_tuning), "compute_mode_requested": compute_mode, "use_news_features_requested": bool(use_news_features), "feature_selection_scope": "training fold only; SelectKBest is fitted inside each inner chronological search",
        "walk_forward_folds": TEST_FOLDS, "fold_gap": TARGET_HORIZON, "holdout_protocol": "chronological tail split; excluded from model selection and walk-forward development", "robust_seeds_enabled": bool(robust_seeds),
        "feature_sets_tested": list(feature_sets),
        "market_latest_date": freshness["latest_date"],
        "market_staleness_days_utc": freshness["staleness_days"],
        "market_freshness_status": freshness["status"],
        "market_calendar_gaps_over_5_days": freshness.get("calendar_gaps_over_5_days"),
        "market_largest_calendar_gap_days": freshness.get("largest_calendar_gap_days"),
        "market_session_check_method": freshness.get("session_check_method"),
        "news_method": sentiment_method_used,
        "warnings": [
            "Yahoo Finance and GDELT responses are not guaranteed complete point-in-time historical archives.",
            "A model using recent news against a long market history is not a valid historical news backtest unless news coverage is point-in-time and sufficiently complete.",
            "The query-window ratio measures attempted-query date span, not article completeness.",
            "Recent RSS/API feeds are not point-in-time historical archives; market-plus-news can be explored when explicitly selected but unverified-news models are never eligible for promotion.",
            "Exploratory unverified-news candidates cannot be promoted.",
            "Promotion indicates beating persistence and the available seasonal-naive baseline on these walk-forward folds only; it does not establish trading profitability.",
            "A market_freshness_status of STALE_WARNING means the latest downloaded market date is more than five calendar days old; inspect provider status and refresh before using live predictions.",
            "Market calendar gap checks are heuristic and do not validate an official futures exchange holiday calendar.",
            "Point-in-time news eligibility requires data/raw/gc_session_calendar.csv with audited cutoff_utc rows and verified=true; no calendar is silently assumed."
        ],
        "results": serializable_results,
        "best": {k: v for k, v in best.items() if k != "features"} if best else None,
        "promotion": promotion,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"training_audit_{run_id}.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (ARTIFACTS / "training_result.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(serializable_results).to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    (ARTIFACTS / "feature_manifest.json").write_text(json.dumps({k: v for k, v in feature_sets.items()}, indent=2), encoding="utf-8")
    tracking = log_to_mlflow(audit, [ARTIFACTS / "training_result.json", ARTIFACTS / f"training_audit_{run_id}.json", ARTIFACTS / "model_comparison.csv", ARTIFACTS / "feature_manifest.json"])
    audit["experiment_tracking"] = tracking
    (ARTIFACTS / f"training_audit_{run_id}.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (ARTIFACTS / "training_result.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return audit
