"""Unit tests for target alignment, feature leakage, news timing, lexicon behavior and data validation."""
import pandas as pd
import numpy as np
from src.features.market_features import make_features
from src.features.news_features import aggregate_news_daily, simple_sentiment


def synthetic_market(n=160):
    """Create deterministic synthetic market rows for unit tests without network calls."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    idx = np.arange(n, dtype=float)
    frame = pd.DataFrame({"Date": dates, "gold_close": 100 + idx * 0.25 + np.sin(idx / 5)})
    for name, base in [("dxy", 100), ("silver", 20), ("oil", 60), ("sp500", 3000), ("vix", 15), ("us10y", 4), ("eurusd", 1.1), ("usdjpy", 140)]:
        frame[f"{name}_close"] = base + idx * (0.001 if base < 100 else 0.1)
        frame[f"{name}_volume"] = 1000
    return frame


def test_target_is_future_price_and_current_close_is_valid_predictor():
    """Assert that next-session price is the label and the known current close is included as a valid predictor."""
    market = synthetic_market()
    frame, features = make_features(market)
    assert "gold_close" in features
    assert "target_price" not in features
    assert "target_return" not in features
    date = frame.iloc[0]["Date"]
    market_idx = market.index[market["Date"].eq(date)][0]
    assert frame.iloc[0]["target_price"] == market.iloc[market_idx + 1]["gold_close"]
    assert len(frame) > 0


def test_news_is_lagged_one_market_row():
    """Assert that a news aggregate is unavailable to its same-date feature row and enters only after the one-row lag."""
    market = synthetic_market(130)
    news_date = market.iloc[75]["Date"]
    article = pd.DataFrame({
        "published_at": [news_date.strftime("%Y-%m-%d") + "T12:00:00Z"], "title": ["Gold faces crisis and uncertainty"],
        "summary": [""], "source": ["test"], "url": ["https://example.invalid/1"]
    })
    daily = aggregate_news_daily(article, market_dates=market["Date"])
    frame, _ = make_features(market, daily)
    day_index = market.index[market["Date"].eq(news_date)][0]
    row_day = frame[frame["Date"].eq(news_date)]
    next_day = frame[frame["Date"].eq(market.iloc[day_index + 1]["Date"])]
    if not row_day.empty:
        assert row_day.iloc[0]["news_count_lag1"] == 0
    if not next_day.empty:
        assert next_day.iloc[0]["news_count_lag1"] == 1


def test_sentiment_lexicon_is_transparent_baseline():
    """Check positive, negative and unmatched text produce expected lexicon scores."""
    assert simple_sentiment("stable growth and recovery") > 0
    assert simple_sentiment("war crisis and uncertainty") < 0
    assert simple_sentiment("the central bank met on Tuesday") == 0


def test_market_data_sort_dedup_and_invalid_price_rejected():
    """Check market data sorting and rejection of non-positive gold prices."""
    market = synthetic_market(120).iloc[::-1].copy()
    frame, _ = make_features(market)
    assert frame["Date"].is_monotonic_increasing
    broken = synthetic_market(120)
    broken.loc[5, "gold_close"] = -1
    import pytest
    with pytest.raises(ValueError):
        make_features(broken)


def test_news_published_on_session_date_is_not_used_on_same_feature_row():
    """Articles on session date t must only enter a later feature row because of lag1."""
    market = synthetic_market(140)
    idx = 90
    date = market.iloc[idx]["Date"]
    article = pd.DataFrame({"published_at": [pd.Timestamp(date).strftime("%Y-%m-%d") + "T23:59:00Z"],
                           "title": ["Gold rally and growth"], "summary": [""], "source": ["test"], "url": ["u"]})
    daily = aggregate_news_daily(article, market_dates=market["Date"])
    frame, _ = make_features(market, daily)
    current = frame.loc[frame["Date"].eq(date)]
    next_date = market.iloc[idx + 1]["Date"]
    next_row = frame.loc[frame["Date"].eq(next_date)]
    if not current.empty: assert current.iloc[0]["news_count_lag1"] == 0
    if not next_row.empty: assert next_row.iloc[0]["news_count_lag1"] == 1


def test_seasonal_naive_benchmark_is_not_a_predictor():
    """The seasonal-naive benchmark exists for evaluation but is excluded from X."""
    frame, features = make_features(synthetic_market(150))
    assert "seasonal_naive_price" in frame.columns
    assert "seasonal_naive_price" not in features
    assert frame["seasonal_naive_price"].notna().all()


def test_market_freshness_report_flags_stale_market_cache():
    """A deterministic stale-cache check should warn without discarding data."""
    from src.data.validation import market_freshness_report
    report = market_freshness_report(synthetic_market(120), as_of_utc="2026-10-15T00:00:00Z", stale_after_days=5)
    assert report["status"] == "STALE_WARNING"
    assert report["staleness_days"] > 5


def test_longer_horizon_keeps_rows_when_seasonal_naive_unavailable():
    """A horizon beyond five sessions must not drop the entire dataset for benchmark-only NaNs."""
    frame, features = make_features(synthetic_market(170), horizon=6)
    assert len(frame) > 0
    assert frame["seasonal_naive_price"].isna().all()
    assert "seasonal_naive_price" not in features


def test_walk_forward_reports_baselines_confidence_interval_and_volatility_regimes():
    """Evaluation output should include two baselines, uncertainty and regime diagnostics."""
    from src.models.evaluation import walk_forward
    market = synthetic_market(190)
    frame, features = make_features(market)
    metrics = walk_forward("HistGradientBoosting", frame[features], frame["target_price"].to_numpy(),
                           frame["target_return"].to_numpy(), frame["gold_close"].to_numpy(),
                           n_splits=3, gap=1, seasonal_baseline_price=frame["seasonal_naive_price"].to_numpy(), robust_seeds=True)
    assert "seasonal_naive_rmse" in metrics
    assert "rmse_ci95_low" in metrics and "rmse_ci95_high" in metrics
    assert metrics["robust_seed_count"] == 3
    assert len(metrics["seed_rmse"]) == 3 and metrics["seed_rmse_spread"] >= 0
    assert any("volatility_rmse" in key for fold in metrics["folds"] for key in fold)


def test_utc_day_boundary_keeps_next_day_article_out_of_previous_session_row():
    """An article after the UTC date boundary must not leak into the earlier feature row."""
    market = synthetic_market(145)
    idx = 92
    day0 = pd.Timestamp(market.iloc[idx]["Date"])
    day1 = pd.Timestamp(market.iloc[idx + 1]["Date"])
    news = pd.DataFrame({
        "published_at": [day0.strftime("%Y-%m-%d") + "T23:59:00Z", day1.strftime("%Y-%m-%d") + "T00:01:00Z"],
        "title": ["Gold rally", "Gold crisis"], "summary": ["", ""],
        "source": ["test", "test"], "url": ["late-day", "next-day"]
    })
    daily = aggregate_news_daily(news, market_dates=market["Date"])
    frame, _ = make_features(market, daily)
    row_day1 = frame.loc[frame["Date"].eq(day1)]
    row_day2 = frame.loc[frame["Date"].eq(market.iloc[idx + 2]["Date"])]
    if not row_day1.empty:
        assert row_day1.iloc[0]["news_count_lag1"] == 1
    if not row_day2.empty:
        assert row_day2.iloc[0]["news_count_lag1"] == 1


def test_weekend_news_is_mapped_to_next_market_session_then_lagged():
    """Weekend article metadata maps to the next session and is lagged one more row."""
    market = synthetic_market(145)
    idx = next(i for i in range(80, 120) if pd.Timestamp(market.iloc[i]["Date"]).weekday() == 4)
    friday = pd.Timestamp(market.iloc[idx]["Date"])
    monday = pd.Timestamp(market.iloc[idx + 1]["Date"])
    tuesday = pd.Timestamp(market.iloc[idx + 2]["Date"])
    news = pd.DataFrame({"published_at":[(friday + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + "T10:00:00Z"],
                         "title":["Gold rally"], "summary":[""], "source":["test"], "url":["weekend"]})
    daily = aggregate_news_daily(news, market_dates=market["Date"])
    assert pd.Timestamp(daily.iloc[0]["Date"]) == monday
    frame, _ = make_features(market, daily)
    monday_row = frame.loc[frame["Date"].eq(monday)]
    tuesday_row = frame.loc[frame["Date"].eq(tuesday)]
    if not monday_row.empty: assert monday_row.iloc[0]["news_count_lag1"] == 0
    if not tuesday_row.empty: assert tuesday_row.iloc[0]["news_count_lag1"] == 1


def test_return_model_prediction_is_reconstructed_to_price(monkeypatch):
    """A zero predicted return must reconstruct exactly the persistence forecast in price units."""
    from src.models import evaluation

    class ZeroReturnModel:
        """Small deterministic estimator that always predicts a zero return."""
        def fit(self, X, y):
            """Accept training data and return the fitted estimator."""
            return self
        def predict(self, X):
            """Return a zero next-session return for every requested row."""
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(evaluation, "make_model", lambda name, seed=42, params=None: ZeroReturnModel())
    market = synthetic_market(150)
    frame, features = make_features(market)
    metrics = evaluation.walk_forward(
        "ZeroReturn", frame[features].astype(float),
        frame["target_price"].to_numpy(float), frame["target_return"].to_numpy(float),
        frame["gold_close"].to_numpy(float), n_splits=3, gap=1,
        seasonal_baseline_price=frame["seasonal_naive_price"].to_numpy(float),
        nested_tuning=False,
    )
    assert np.isclose(metrics["rmse"], metrics["baseline_rmse"])
    assert metrics["beats_persistence"] is False
