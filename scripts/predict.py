"""Load the approved champion and issue a local next-session gold-price prediction from a JSON feature payload."""
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json, joblib, pandas as pd
from config.project_config import CHAMPION_MODEL, CHAMPION_MANIFEST

p=argparse.ArgumentParser()
p.add_argument("--features-json", required=True)
a=p.parse_args()

manifest=json.loads(CHAMPION_MANIFEST.read_text())
x=json.loads(a.features_json) if hasattr(a,"features_json") else json.loads(a.features_json)
X=pd.DataFrame([[x[c] for c in manifest["features"]]], columns=manifest["features"])
model=joblib.load(CHAMPION_MODEL)
raw_prediction=float(model.predict(X)[0])
if manifest.get("target_type") == "next_session_return_reconstructed_to_price":
    if "gold_close" not in x or float(x["gold_close"]) <= 0:
        raise ValueError("A positive gold_close feature is required to convert predicted return to price")
    prediction=float(x["gold_close"])*(1.0+raw_prediction)
    predicted_return=raw_prediction
else:
    prediction=raw_prediction
    predicted_return=None
print({"prediction_usd_per_oz": prediction, "predicted_return": predicted_return, "target_type": manifest.get("target_type", "price"), "model": manifest["model_name"]})
