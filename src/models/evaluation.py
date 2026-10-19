"""Leakage-aware chronological model evaluation and benchmark comparison.

Uses expanding-window TimeSeriesSplit folds with a target-horizon gap. The same
held-out timestamps are scored against persistence and a 5-session seasonal-naive
benchmark. A moving-block bootstrap gives a descriptive RMSE interval that respects
some short-range serial dependence; it is not a guarantee of future performance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.factory import make_model


def _moving_block_rmse_ci(actual: np.ndarray, predicted: np.ndarray, *, n_bootstrap: int = 500,
                           block_length: int = 5, seed: int = 42) -> tuple[float, float]:
    """Estimate a 95% RMSE interval by resampling contiguous error blocks.

    Args:
        actual: Concatenated chronological held-out target values.
        predicted: Predictions for the same held-out timestamps.
        n_bootstrap: Number of bootstrap replicates.
        block_length: Length of each contiguous resampling block.
        seed: Random seed for reproducibility.

    Returns:
        Lower and upper percentile limits. The interval is descriptive and should
        not be interpreted as a trading-profitability interval.
    """
    errors_sq = (np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) ** 2
    n = len(errors_sq)
    if n < 10:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([errors_sq[s:s + block] for s in starts])[:n]
        values[i] = np.sqrt(sample.mean())
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def walk_forward(name, X, y_price, y_return, baseline_price, n_splits=5, params=None,
                 gap=1, seasonal_baseline_price=None, robust_seeds=False, nested_tuning=True):
    """Evaluate a regressor on expanding chronological folds and common baselines.

    Args:
        name: Model identifier accepted by ``make_model``.
        X: Numeric features sorted by observation date.
        y_price: Future target price at the configured forecast horizon.
        y_return: Future target return, retained for reporting extensibility.
        baseline_price: Persistence forecast, equal to current close.
        n_splits: Number of chronological validation folds.
        params: Optional estimator parameters.
        gap: Number of rows omitted between train and validation to reduce leakage.
        seasonal_baseline_price: Optional same-weekday/5-session-naive price forecast.
        robust_seeds: If true, train three deterministic seeds per fold and report seed sensitivity.
        nested_tuning: Tune a small parameter grid and select features within each outer training fold only.

    Returns:
        Metrics pooled over held-out folds, fold-level metrics, baseline comparisons,
        and a moving-block-bootstrap RMSE interval. The regressor is fitted to the
        next-session return and predictions are reconstructed into price units using
        the known current close before they are scored.
    """
    if len(X) != len(y_price) or len(X) != len(baseline_price) or len(X) != len(y_return):
        raise ValueError("X, price target, return target, and baseline arrays must have the same length")
    if not np.isfinite(np.asarray(y_return, dtype=float)).all():
        raise ValueError("Return target contains missing or non-finite values")
    if len(X) < n_splits + 20:
        raise ValueError(f"Not enough rows ({len(X)}) for {n_splits} walk-forward folds")
    seasonal = None if seasonal_baseline_price is None else np.asarray(seasonal_baseline_price, dtype=float)
    if seasonal is not None and len(seasonal) != len(X):
        raise ValueError("seasonal baseline must have the same length as X")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=max(0, int(gap)))
    preds, actuals, base, seasonal_values = [], [], [], []
    seeds = [42, 2026, 7] if robust_seeds else [42]
    seed_predictions = {seed: [] for seed in seeds}
    fold_rows = []
    tuning_records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), 1):
        actual = np.asarray(y_price, dtype=float)[test_idx]
        persistence = np.asarray(baseline_price, dtype=float)[test_idx]
        fold_predictions = []
        for seed in seeds:
            base_model = make_model(name, seed=seed + fold, params=params)
            if nested_tuning and len(train_idx) >= 30 and X.shape[1] >= 2:
                # Feature selection and parameter selection are fitted only on the
                # outer fold's training rows; outer test rows never enter GridSearchCV.
                selector_k = sorted(set(k for k in (min(5, X.shape[1]), min(10, X.shape[1]), X.shape[1]) if k >= 1))
                pipe = Pipeline([("select", SelectKBest(score_func=f_regression)), ("model", base_model)])
                grid = {"select__k": selector_k}
                if name == "XGBoost":
                    grid.update({"model__max_depth": [3, 4], "model__learning_rate": [0.03, 0.07], "model__n_estimators": [150, 300]})
                elif name in {"Random Forest", "Extra Trees"}:
                    grid.update({"model__max_depth": [8, 12], "model__min_samples_leaf": [2, 4], "model__n_estimators": [150, 300]})
                elif name == "HistGradientBoosting":
                    grid.update({"model__max_leaf_nodes": [15, 31], "model__learning_rate": [0.03, 0.07], "model__max_iter": [150, 300]})
                inner_splits = min(3, max(2, len(train_idx) // 30))
                inner = TimeSeriesSplit(n_splits=inner_splits, gap=max(0, int(gap)))
                search = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=1, refit=True, error_score="raise", verbose=1)
                search.fit(X.iloc[train_idx], np.asarray(y_return)[train_idx])
                fitted = search.best_estimator_
                tuning_records.append({"fold": fold, "seed": seed, "best_params": search.best_params_, "inner_best_rmse": float(-search.best_score_), "selected_features": list(X.columns[fitted.named_steps["select"].get_support()])})
            else:
                fitted = base_model.fit(X.iloc[train_idx], np.asarray(y_return)[train_idx])
                tuning_records.append({"fold": fold, "seed": seed, "best_params": {}, "inner_best_rmse": None, "selected_features": list(X.columns)})
            # Learn the next-session return, then reconstruct a price forecast from
            # the currently observed close. This reduces the trend-extrapolation
            # burden for tree models and keeps evaluation in interpretable price units.
            predicted_return = np.asarray(fitted.predict(X.iloc[test_idx]), dtype=float)
            seed_pred = persistence * (1.0 + predicted_return)
            seed_predictions[seed].extend(seed_pred.tolist())
            fold_predictions.append(seed_pred)
        pred = np.mean(np.vstack(fold_predictions), axis=0)
        preds.extend(pred.tolist()); actuals.extend(actual.tolist()); base.extend(persistence.tolist())
        fold_row = {
            "fold": fold, "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "persistence_rmse": float(np.sqrt(mean_squared_error(actual, persistence))),
            "test_start_index": int(test_idx[0]), "test_end_index": int(test_idx[-1]),
        }
        if seasonal is not None and np.isfinite(seasonal[test_idx]).all():
            s_pred = seasonal[test_idx]
            seasonal_values.extend(s_pred.tolist())
            fold_row["seasonal_naive_rmse"] = float(np.sqrt(mean_squared_error(actual, s_pred)))
        # Fold-level volatility-regime diagnostics use the current known 20-session
        # volatility predictor, with the threshold estimated only from that fold's train rows.
        if "gold_vol_20" in X.columns:
            train_vol = pd.to_numeric(X.iloc[train_idx]["gold_vol_20"], errors="coerce").dropna()
            test_vol = pd.to_numeric(X.iloc[test_idx]["gold_vol_20"], errors="coerce")
            if len(train_vol) and test_vol.notna().any():
                threshold = float(train_vol.median())
                for regime, mask in (("low_volatility", test_vol <= threshold), ("high_volatility", test_vol > threshold)):
                    mask = np.asarray(mask.fillna(False), dtype=bool)
                    if mask.any(): fold_row[f"{regime}_rmse"] = float(np.sqrt(mean_squared_error(actual[mask], pred[mask])))
        fold_rows.append(fold_row)
    preds_a, actuals_a, base_a = map(np.asarray, (preds, actuals, base))
    rmse = float(np.sqrt(mean_squared_error(actuals_a, preds_a)))
    mae = float(mean_absolute_error(actuals_a, preds_a))
    persistence_rmse = float(np.sqrt(mean_squared_error(actuals_a, base_a)))
    beats_persistence = bool(rmse < persistence_rmse)
    seasonal_rmse = float('nan')
    beats_seasonal = None
    if seasonal is not None and len(seasonal_values) == len(actuals_a):
        seasonal_a = np.asarray(seasonal_values, dtype=float)
        seasonal_rmse = float(np.sqrt(mean_squared_error(actuals_a, seasonal_a)))
        beats_seasonal = bool(rmse < seasonal_rmse)
    ci_low, ci_high = _moving_block_rmse_ci(actuals_a, preds_a, seed=42)
    per_seed_rmse = {str(seed): float(np.sqrt(mean_squared_error(actuals_a, np.asarray(seed_predictions[seed], dtype=float)))) for seed in seeds}
    seed_spread = float(max(per_seed_rmse.values()) - min(per_seed_rmse.values())) if len(per_seed_rmse) > 1 else 0.0
    all_baseline_rmse = [persistence_rmse]
    if np.isfinite(seasonal_rmse): all_baseline_rmse.append(seasonal_rmse)
    strongest_baseline_rmse = min(all_baseline_rmse)
    improvement = float((strongest_baseline_rmse - rmse) / strongest_baseline_rmse * 100) if strongest_baseline_rmse else 0.0
    beats_all = bool(beats_persistence and (beats_seasonal is not False))
    return {
        "model": name, "rmse": rmse, "rmse_ci95_low": ci_low, "rmse_ci95_high": ci_high,
        "mae": mae, "baseline_rmse": persistence_rmse,
        "seasonal_naive_rmse": seasonal_rmse if np.isfinite(seasonal_rmse) else None,
        "strongest_baseline_rmse": strongest_baseline_rmse,
        "improvement_pct": improvement, "directional_accuracy": float(np.mean(np.sign(preds_a-base_a) == np.sign(actuals_a-base_a))),
        "beats_persistence": beats_persistence, "beats_seasonal_naive": beats_seasonal,
        "beats_baseline": beats_all, "n_predictions": int(len(actuals_a)), "folds": fold_rows,
        "seed_rmse": per_seed_rmse, "seed_rmse_spread": seed_spread, "robust_seed_count": len(seeds),
        "nested_tuning": bool(nested_tuning), "tuning_records": tuning_records,
    }


def fit_final(name, X, y, params=None, nested_tuning=True):
    """Tune on development data only, then fit a feature-selection/model pipeline.

    The outer walk-forward scores remain the reported generalization estimate. This
    final inner search is only for producing the candidate artifact after evaluation;
    it never sees the held-out target rows outside the development matrix supplied.
    """
    if nested_tuning and len(X) >= 30 and X.shape[1] >= 2:
        base_model = make_model(name, params=params)
        ks = sorted(set(k for k in (min(5, X.shape[1]), min(10, X.shape[1]), X.shape[1]) if k >= 1))
        pipe = Pipeline([("select", SelectKBest(score_func=f_regression)), ("model", base_model)])
        grid = {"select__k": ks}
        if name == "XGBoost":
            grid.update({"model__max_depth": [3, 4], "model__learning_rate": [0.03, 0.07], "model__n_estimators": [150, 300]})
        elif name in {"Random Forest", "Extra Trees"}:
            grid.update({"model__max_depth": [8, 12], "model__min_samples_leaf": [2, 4], "model__n_estimators": [150, 300]})
        elif name == "HistGradientBoosting":
            grid.update({"model__max_leaf_nodes": [15, 31], "model__learning_rate": [0.03, 0.07], "model__max_iter": [150, 300]})
        cv = TimeSeriesSplit(n_splits=3, gap=1)
        search = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1, refit=True, error_score="raise", verbose=1)
        search.fit(X, y)
        return search.best_estimator_
    model = make_model(name, params=params)
    model.fit(X, y)
    return model
