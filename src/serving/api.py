"""Local FastAPI serving with manifest-driven feature validation."""
from functools import lru_cache
import json, math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from config.project_config import CHAMPION_MODEL, CHAMPION_MANIFEST
from src.monitoring.prediction_ledger import append_prediction

app = FastAPI(
    title="Gold Price Prediction API",
    version="15.0.0",
    description=(
        "Local educational/research API for the approved Gold Price Prediction champion. "
        "Use GET /health and GET /ready to check service status, GET /model to inspect "
        "the approved model manifest and required feature names, and POST /predict to "
        "submit those exact numeric features. Train, evaluate, and explicitly approve a "
        "candidate in the desktop workflow before prediction. This API does not fetch "
        "market data or automatically construct features. Research estimate only; not financial advice."
    ),
    openapi_tags=[
        {"name": "Service status", "description": "Check whether the local API process and approved model are available."},
        {"name": "Model information", "description": "Inspect the approved model manifest and feature schema."},
        {"name": "Prediction", "description": "Submit feature values matching the approved model manifest to obtain a research estimate."},
    ],
)

class PredictionRequest(BaseModel):
    """Represent the PredictionRequest component and its behavior."""
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(..., description="Numeric feature values matching models/champion_manifest.json")
    as_of_date: str | None = Field(default=None, description="Last market session represented by features, YYYY-MM-DD; needed for delayed realized-error evaluation")

@lru_cache(maxsize=1)
def _load_champion():
    """Load the approved champion artifact and manifest, rejecting missing or inconsistent registry files."""
    if not CHAMPION_MODEL.exists() or not CHAMPION_MANIFEST.exists():
        raise FileNotFoundError("Champion model or manifest is missing")
    manifest = json.loads(CHAMPION_MANIFEST.read_text(encoding="utf-8"))
    return joblib.load(CHAMPION_MODEL), manifest

@app.get("/health", tags=["Service status"], summary="Check API process health")
def health():
    """Return service health and whether an approved champion is available."""
    return {"status": "ok", "model_available": CHAMPION_MODEL.exists() and CHAMPION_MANIFEST.exists(), "version": "15.0.0"}

@app.get("/ready", tags=["Service status"], summary="Check prediction readiness")
def readiness():
    """Readiness probe: ready only when an approved champion and manifest exist."""
    ready = CHAMPION_MODEL.exists() and CHAMPION_MANIFEST.exists()
    return {"ready": ready, "status": "ready" if ready else "not_ready", "version": "15.0.0"}

@app.get("/model", tags=["Model information"], summary="Inspect approved model metadata")
def model_info():
    """Expose approved model identity, feature schema and recorded evaluation metadata without exposing the model object."""
    if not CHAMPION_MANIFEST.exists():
        raise HTTPException(404, "No human-approved champion model registered. Train, review, and approve a candidate first.")
    return json.loads(CHAMPION_MANIFEST.read_text(encoding="utf-8"))

@app.post("/predict", tags=["Prediction"], summary="Predict the next-session gold price")
def predict(req: PredictionRequest):
    """Validate request feature names and finite numeric values, then return the approved model prediction."""
    if req.as_of_date is not None:
        try:
            parsed_as_of = pd.to_datetime(req.as_of_date, format="%Y-%m-%d", errors="raise")
            if parsed_as_of.strftime("%Y-%m-%d") != req.as_of_date:
                raise ValueError("as_of_date must use YYYY-MM-DD")
        except Exception as exc:
            raise HTTPException(422, "as_of_date must be a valid YYYY-MM-DD market-session date.") from exc
    if not CHAMPION_MODEL.exists() or not CHAMPION_MANIFEST.exists():
        raise HTTPException(503, "No human-approved champion model is available.")
    try:
        model, manifest = _load_champion()
    except Exception as exc:
        raise HTTPException(503, f"Could not load champion model: {exc}") from exc
    cols = manifest["features"]
    missing = [col for col in cols if col not in req.features]
    extra = [col for col in req.features if col not in cols]
    if missing or extra:
        raise HTTPException(422, {"missing_features": missing, "unexpected_features": extra})
    values = [req.features[col] for col in cols]
    if not all(math.isfinite(float(value)) for value in values):
        raise HTTPException(422, "All feature values must be finite numeric values.")
    X = pd.DataFrame([values], columns=cols)
    raw_prediction = float(model.predict(X)[0])
    if manifest.get("target_type") == "next_session_return_reconstructed_to_price":
        if "gold_close" not in req.features or float(req.features["gold_close"]) <= 0:
            raise HTTPException(422, "A positive current gold_close is required to convert the predicted return into a price.")
        pred = float(req.features["gold_close"]) * (1.0 + raw_prediction)
        predicted_return = raw_prediction
    else:
        pred = raw_prediction
        predicted_return = None
    if not math.isfinite(pred) or pred <= 0:
        raise HTTPException(422, "Model produced a non-finite or non-positive price prediction.")
    append_prediction(forecast_date=req.as_of_date, predicted_price=pred, model_name=manifest["model_name"],
                      feature_set=manifest.get("feature_set", "unknown"), features=req.features)
    return {"prediction_usd_per_oz": pred, "predicted_return": predicted_return, "target_type": manifest.get("target_type", "price"),
            "model": manifest["model_name"], "feature_set": manifest.get("feature_set"),
            "as_of_date": req.as_of_date, "ledgered": True, "disclaimer": "Research estimate, not financial advice."}
