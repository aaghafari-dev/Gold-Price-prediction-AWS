"""Construct configured scikit-learn/XGBoost regressors with deterministic seeds and bounded CPU use."""
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

def make_model(name: str, seed: int = 42, params: dict | None = None):
    """Build one supported regression estimator using project defaults, deterministic random seed and bounded CPU parallelism."""
    p = params or {}
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.03),
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=2,
            tree_method="hist",
            device=("cuda" if __import__("os").getenv("AURUM_COMPUTE_MODE") == "gpu" else "cpu"),
        )
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", 12),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            random_state=seed, n_jobs=2,
        )
    if name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", 12),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            random_state=seed, n_jobs=2,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=p.get("max_iter", 400),
            learning_rate=p.get("learning_rate", 0.04),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            l2_regularization=p.get("l2_regularization", 0.5),
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {name}")
