"""Transparent lexicon sentiment and daily news aggregates.

This is an explainable baseline, not a finance-trained NLP model. A daily aggregate
is shifted by one market row in market_features.py, so a forecast made after close t
cannot consume articles from calendar date t unless the caller explicitly changes
that conservative policy and validates the exact cutoff.
"""
from __future__ import annotations
import re
import pandas as pd
from src.data.validation import validate_news_frame

POSITIVE = {
    "safe", "stability", "stable", "recovery", "growth", "strong", "easing", "support",
    "dovish", "peace", "demand", "bullish", "gain", "gains", "rise", "rises", "rising",
    "surge", "surges", "upbeat", "optimism", "optimistic", "rally", "record-high"
}
NEGATIVE = {
    "war", "attack", "crisis", "uncertainty", "risk", "inflation", "hawkish", "recession",
    "sanction", "conflict", "collapse", "default", "fear", "volatile", "volatility", "weak",
    "fall", "falls", "falling", "drop", "drops", "plunge", "plunges", "selloff", "sell-off",
    "downturn", "loss", "losses", "tensions", "threat"
}
NEWS_FEATURE_COLUMNS = [
    "news_count", "news_sentiment_mean", "news_sentiment_std",
    "news_negative_share", "news_positive_share", "news_source_count", "news_observed"
]


def simple_sentiment(text: str) -> float:
    """Score text with the fixed English keyword lexicon; each matching token is counted once and the result is bounded from -1 to +1."""
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", str(text).lower()))
    positive = len(tokens & POSITIVE)
    negative = len(tokens & NEGATIVE)
    return float((positive - negative) / max(1, positive + negative))


def aggregate_news_daily(news: pd.DataFrame, market_dates=None, session_cutoffs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate articles by UTC publication date while retaining source coverage.

    A date absent from the returned table means no article was returned by the source;
    it does not prove that no relevant news existed that day.
    """
    if news is None or news.empty:
        return pd.DataFrame(columns=["Date", *NEWS_FEATURE_COLUMNS])
    n = validate_news_frame(news)
    n["text"] = (n["title"].fillna("") + " " + n["summary"].fillna("")).str.strip()
    # A precomputed sentiment_score is accepted only when a separate, explicitly
    # generated comparison file is supplied (e.g. FinBERT experiment output).
    if "sentiment_score" in n.columns and pd.to_numeric(n["sentiment_score"], errors="coerce").notna().any():
        n["sentiment"] = pd.to_numeric(n["sentiment_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        n["sentiment"] = n["text"].map(simple_sentiment)
    n["Date"] = n["published_at"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    if market_dates is not None:
        # Assign weekend/holiday articles to the next available market session.
        # Features are still shifted one additional market row downstream.
        sessions = pd.DatetimeIndex(pd.to_datetime(pd.Series(market_dates), utc=True).dt.tz_convert(None).dt.normalize().dropna().unique()).sort_values()
        if len(sessions) == 0:
            return pd.DataFrame(columns=["Date", *NEWS_FEATURE_COLUMNS])
        positions = sessions.searchsorted(pd.DatetimeIndex(n["Date"]), side="left")
        valid = positions < len(sessions)
        n = n.loc[valid].copy()
        positions = positions[valid]
        n["Date"] = sessions.take(positions).to_numpy()
    if session_cutoffs is not None and not session_cutoffs.empty:
        # Calendar file must contain session_date and cutoff_utc. Filter each article
        # against the declared forecast cutoff for its assigned next session.
        cuts = session_cutoffs.copy()
        cuts["session_date"] = pd.to_datetime(cuts["session_date"], errors="coerce").dt.normalize()
        cuts["cutoff_utc"] = pd.to_datetime(cuts["cutoff_utc"], utc=True, errors="coerce")
        n = n.merge(cuts[["session_date", "cutoff_utc"]].dropna(), left_on="Date", right_on="session_date", how="inner")
        n = n[n["published_at"] <= n["cutoff_utc"]].copy()
        if n.empty:
            return pd.DataFrame(columns=["Date", *NEWS_FEATURE_COLUMNS])
    daily = n.groupby("Date", as_index=False).agg(
        news_count=("sentiment", "size"),
        news_sentiment_mean=("sentiment", "mean"),
        news_sentiment_std=("sentiment", "std"),
        news_negative_share=("sentiment", lambda s: float((s < 0).mean())),
        news_positive_share=("sentiment", lambda s: float((s > 0).mean())),
        news_source_count=("source", "nunique"),
    )
    daily["news_observed"] = 1.0
    return daily.fillna(0.0)
