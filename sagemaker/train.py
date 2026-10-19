"""SageMaker training entry point.

Input: /opt/ml/input/data/train/train.csv
Expected columns: feature columns plus target_price.
Output: /opt/ml/model/model.joblib and manifest.json
"""
import argparse, json, os
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

def make_model(name, seed=42):
    """Build one supported regression estimator using project defaults, deterministic random seed and bounded CPU parallelism."""
    if name == "XGBoost":
        return XGBRegressor(n_estimators=500,max_depth=4,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,objective="reg:squarederror",random_state=seed,n_jobs=2,tree_method="hist")
    if name == "Random Forest": return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2)
    if name == "Extra Trees": return ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2)
    if name == "HistGradientBoosting": return HistGradientBoostingRegressor(max_iter=400,learning_rate=0.04,max_leaf_nodes=31,l2_regularization=0.5,random_state=seed)
    raise ValueError(name)

def main():
    """Parse SageMaker training arguments, load the training data, fit the estimator and save its artifact."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="XGBoost")
    p.add_argument("--target-column", default="target_price")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_path = Path("/opt/ml/input/data/train/train.csv")
    df = pd.read_csv(train_path)
    y = df.pop(args.target_column)
    # metadata columns must not enter the model
    for c in ["Date", "target_return", "gold_close"]:
        if c in df:
            df = df.drop(columns=[c])
    model = make_model(args.model_name, seed=args.seed)
    model.fit(df, y)

    out = Path("/opt/ml/model")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out/"model.joblib")
    (out/"manifest.json").write_text(json.dumps({
        "model_name": args.model_name,
        "features": list(df.columns),
    }, indent=2))

if __name__ == "__main__":
    main()
