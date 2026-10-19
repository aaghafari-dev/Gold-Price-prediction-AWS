"""SageMaker inference handlers for the joblib model archive."""
import json, os, math, joblib, pandas as pd

MODEL = None
MANIFEST = None

def model_fn(model_dir):
    """Load the serialized estimator from the SageMaker model directory."""
    global MODEL, MANIFEST
    MODEL = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as handle:
        MANIFEST = json.load(handle)
    return MODEL

def input_fn(request_body, request_content_type):
    """Parse and validate a JSON or CSV inference request into the model input structure."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    if request_content_type in ("text/csv", "application/csv"):
        from io import StringIO
        return pd.read_csv(StringIO(request_body))
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run the loaded SageMaker estimator on validated input data."""
    if isinstance(input_data, dict):
        payload = input_data.get("features", input_data)
        cols = MANIFEST["features"]
        missing = [c for c in cols if c not in payload]
        extra = [c for c in payload if c not in cols]
        if missing or extra:
            raise ValueError(f"Feature schema mismatch; missing={missing}, unexpected={extra}")
        values = [float(payload[c]) for c in cols]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All feature values must be finite")
        X = pd.DataFrame([values], columns=cols)
    elif isinstance(input_data, list):
        X = pd.DataFrame(input_data, columns=MANIFEST["features"])
    else:
        X = input_data
    pred = model.predict(X)
    return {"prediction_usd_per_oz": float(pred[0]), "model": MANIFEST["model_name"]}

def output_fn(prediction, accept):
    """Serialize predictions into the response content type requested by SageMaker."""
    return json.dumps(prediction), "application/json"
