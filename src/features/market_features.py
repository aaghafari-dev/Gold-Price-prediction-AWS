"""Build market predictors, lagged news predictors, and the next-session target.

Forecast convention: row t represents a forecast made after the UTC calendar day
associated with market session t has completed. Article aggregates are assigned to
UTC publication dates and then lagged by one available market row. Consequently,
row t only uses news assigned to an earlier market row. This is deliberately
conservative, but it is not a substitute for exchange-calendar point-in-time archives.
"""
from __future__ import annotations
import pandas as pd
from src.features.news_features import NEWS_FEATURE_COLUMNS
from src.data.validation import validate_market_frame


def make_features(market: pd.DataFrame, news_daily: pd.DataFrame | None = None, horizon: int = 1):
    """Create a leakage-aware supervised feature table.

    Args:
        market: Daily market frame containing Date and gold_close.
        news_daily: Daily article aggregates from ``aggregate_news_daily``.
        horizon: Number of market rows ahead for the target close.

    Returns:
        A tuple of (clean feature/target DataFrame, predictor column names). The
        output retains benchmark-only columns, but those columns are excluded from
        the predictor list to avoid target leakage.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = validate_market_frame(market).sort_values("Date").reset_index(drop=True)
    if news_daily is not None and not news_daily.empty:
        n = news_daily.copy()
        n["Date"] = pd.to_datetime(n["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        n = n.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")
        df = df.merge(n, on="Date", how="left", validate="one_to_one")
    for col in NEWS_FEATURE_COLUMNS:
        if col not in df: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    p = df["gold_close"]
    for lag in [1, 2, 3, 5, 10, 20]: df[f"gold_lag_{lag}"] = p.shift(lag)
    for window in [5, 10, 20, 60]:
        df[f"gold_ma_{window}"] = p.rolling(window, min_periods=window).mean()
        df[f"gold_vol_{window}"] = p.pct_change().rolling(window, min_periods=window).std()
    for col in ["dxy_close", "silver_close", "oil_close", "sp500_close", "vix_close", "us10y_close", "eurusd_close", "usdjpy_close"]:
        if col in df: df[f"{col}_ret1"] = df[col].pct_change()
    df["gold_ret1"] = p.pct_change()
    df["gold_ret5"] = p.pct_change(5)
    df["gold_momentum20"] = p / p.shift(20) - 1.0

    # Benchmarks are retained for evaluation but never supplied as predictors.
    seasonal_lag = 5 - horizon
    # For horizon=1, use close t-4 (the prior same-weekday close). If the target
    # horizon is longer than five sessions, the previous same-weekday target would
    # be in the future at forecast time, so seasonal naive is marked unavailable.
    df["seasonal_naive_price"] = p.shift(seasonal_lag) if seasonal_lag >= 0 else float("nan")
    for col in NEWS_FEATURE_COLUMNS: df[f"{col}_lag1"] = df[col].shift(1)
    df["target_price"] = p.shift(-horizon)
    df["target_return"] = p.shift(-horizon) / p - 1.0

    excluded = {"Date", "target_price", "target_return", "seasonal_naive_price", *NEWS_FEATURE_COLUMNS}
    feature_cols = [c for c in df.columns if c not in excluded and not c.endswith("_volume")]
    # Drop only rows whose required model inputs or target are missing/infinite.
    # The current observed gold close is a valid predictor at forecast time; excluding it
    # makes tree models needlessly extrapolate from older lags and can explain huge RMSE.
    out = df.replace([float("inf"), float("-inf")], float("nan"))
    out = out.dropna(subset=feature_cols + ["target_price", "target_return"]).reset_index(drop=True)
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric: raise TypeError(f"Non-numeric features detected: {non_numeric}")
    return out, feature_cols
