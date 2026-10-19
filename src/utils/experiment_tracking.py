"""Optional MLflow tracking; JSON/CSV audit remains the no-service default."""
from __future__ import annotations
import os
from pathlib import Path


def log_to_mlflow(audit: dict, artifact_paths: list[Path]) -> dict:
    """Optionally log run metadata and audit artifacts to MLflow; remain a no-op when tracking is not configured."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return {"status": "SKIPPED", "reason": "MLFLOW_TRACKING_URI is not set; local audit files were written."}
    try:
        import mlflow
    except ImportError:
        return {"status": "SKIPPED", "reason": "Install requirements-mlflow.txt to enable optional MLflow tracking."}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "gold-price-prediction-v4"))
        with mlflow.start_run(run_name=str(audit.get("run_id", "gold-price-run"))) as active:
            mlflow.log_params({
                "app_version": audit.get("app_version", "4.0.0"),
                "target_horizon_sessions": audit.get("target_horizon_sessions"),
                "walk_forward_folds": audit.get("walk_forward_folds"),
                "news_method": audit.get("news_method"),
                "news_coverage_gate_passed": audit.get("news_coverage_gate_passed"),
            })
            for row in audit.get("results", []):
                prefix = f"{row.get('feature_set', 'unknown')}_{row.get('model', 'model')}".replace(" ", "_").lower()
                for metric in ("rmse", "mae", "baseline_rmse", "improvement_pct", "directional_accuracy"):
                    value = row.get(metric)
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}_{metric}", float(value))
            for path in artifact_paths:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            return {"status": "LOGGED", "tracking_uri": tracking_uri, "mlflow_run_id": active.info.run_id}
    except Exception as exc:  # tracking is optional; never discard a completed local run
        return {"status": "FAILED", "tracking_uri": tracking_uri, "reason": str(exc), "local_audit_preserved": True}
