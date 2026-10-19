"""Candidate/champion governance with an explicit human approval step."""
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
from config.project_config import MODELS, CHAMPION_MANIFEST, CHAMPION_MODEL

CANDIDATE_MODEL = MODELS / "candidate_model.joblib"
CANDIDATE_MANIFEST = MODELS / "candidate_manifest.json"


def evaluate_promotion(metrics: dict, min_improvement_pct: float = 0.0) -> bool:
    """Return whether metrics beat the required baseline and meet the minimum RMSE improvement threshold."""
    return bool(metrics.get("beats_baseline", False) and metrics.get("improvement_pct", -999) >= min_improvement_pct)


def save_candidate(model, metrics: dict, feature_cols: list[str], model_name: str, feature_set: str, params: dict | None = None):
    """Serialize an evaluated model and a manifest for explicit human review; do not update the champion."""
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL)
    manifest = {
        "model_name": model_name, "feature_set": feature_set,
        "target_type": metrics.get("target_type", "price"),
        "created_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
        "feature_count": len(feature_cols), "features": feature_cols, "params": params or {},
        "model_file": CANDIDATE_MODEL.name, "status": "PENDING_HUMAN_APPROVAL",
        "acceptance_gate": "Walk-forward RMSE must beat persistence baseline; review metrics and data coverage before approval.",
    }
    CANDIDATE_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"status": "PENDING_HUMAN_APPROVAL", "candidate_manifest": str(CANDIDATE_MANIFEST), "candidate_model": str(CANDIDATE_MODEL)}


def promote(model, metrics: dict, feature_cols: list[str], model_name: str, params: dict | None = None, feature_set: str = "market_plus_news"):
    """Write the approved champion model and manifest after the candidate passes the benchmark gate."""
    if not evaluate_promotion(metrics):
        return {"status": "REJECTED", "reason": "Model did not pass the persistence-baseline acceptance gate."}
    CHAMPION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHAMPION_MODEL)
    manifest = {
        "model_name": model_name, "feature_set": feature_set,
        "target_type": metrics.get("target_type", "price"),
        "promoted_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
        "feature_count": len(feature_cols), "features": feature_cols, "params": params or {},
        "governance": "HUMAN_APPROVED", "model_file": CHAMPION_MODEL.name,
        "acceptance_gate": "Walk-forward RMSE beat persistence and human approval was explicitly supplied.",
    }
    CHAMPION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if CANDIDATE_MANIFEST.exists():
        try:
            candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
            candidate_manifest["status"] = "HUMAN_APPROVED_AND_PROMOTED"
            candidate_manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
            CANDIDATE_MANIFEST.write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass
    return {"status": "PROMOTED", **manifest}


def approve_saved_candidate():
    """Validate and promote the saved pending candidate only after an explicit approval command."""
    if not CANDIDATE_MANIFEST.exists() or not CANDIDATE_MODEL.exists():
        raise FileNotFoundError("No candidate found. Run scripts/train_local.py first.")
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PENDING_HUMAN_APPROVAL":
        raise ValueError(f"Candidate is not awaiting approval (status={manifest.get('status')}). Run training to create a new candidate.")
    metrics = manifest["metrics"]
    if not evaluate_promotion(metrics):
        raise ValueError("Candidate failed the persistence-baseline gate and cannot be promoted.")
    model = joblib.load(CANDIDATE_MODEL)
    result = promote(model, metrics, manifest["features"], manifest["model_name"], manifest.get("params"), manifest["feature_set"])
    manifest["status"] = "HUMAN_APPROVED_AND_PROMOTED"
    manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
    CANDIDATE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
