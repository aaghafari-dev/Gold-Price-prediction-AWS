"""Data contracts and validation helpers for market and news inputs."""
from __future__ import annotations
import pandas as pd

MARKET_REQUIRED = {"Date", "gold_close"}
NEWS_REQUIRED = {"published_at", "title"}


def validate_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate market schema, dates, duplicate rows, positive gold prices and numeric columns."""
    missing = MARKET_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Market data missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if out["Date"].isna().any():
        raise ValueError("Market data contains invalid dates")
    out = out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    out["gold_close"] = pd.to_numeric(out["gold_close"], errors="coerce")
    if out["gold_close"].isna().any() or (out["gold_close"] <= 0).any():
        raise ValueError("gold_close must contain finite positive numeric values")
    for col in out.columns:
        if col != "Date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Ingestion and plotting are allowed for very short windows. Training applies a
    # stricter observation-count check after feature warm-up and chronological splitting.
    if not out["gold_close"].map(lambda value: pd.notna(value) and value != float("inf") and value != float("-inf")).all():
        raise ValueError("gold_close contains non-finite values")
    return out


def validate_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required news fields and normalize publication/retrieval timestamps and text columns."""
    missing = NEWS_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"News data missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["published_at"])
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out = out[out["title"].ne("")]
    for col, default in (("summary", ""), ("source", "unknown"), ("url", ""), ("retrieved_at", pd.NaT)):
        if col not in out:
            out[col] = default
    out["retrieved_at"] = pd.to_datetime(out["retrieved_at"], errors="coerce", utc=True)
    return out.drop_duplicates(["published_at", "title", "url"]).reset_index(drop=True)


def market_freshness_report(frame: pd.DataFrame, *, as_of_utc=None, stale_after_days: int = 5) -> dict:
    """Report the latest market date and flag stale data without hiding historical data.

    Args:
        frame: Validated or raw market frame with a Date column.
        as_of_utc: Optional deterministic timestamp for tests; defaults to current UTC.
        stale_after_days: Calendar-day threshold after which a warning is raised.

    Returns:
        A dictionary containing the latest date, staleness in calendar days and status.
    """
    if frame is None or frame.empty or "Date" not in frame.columns:
        return {"latest_date": None, "staleness_days": None, "status": "NO_MARKET_DATA"}
    now = pd.Timestamp(as_of_utc) if as_of_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is not None: now = now.tz_convert("UTC").tz_localize(None)
    latest = pd.to_datetime(frame["Date"], utc=True, errors="coerce").dropna().max()
    if pd.isna(latest):
        return {"latest_date": None, "staleness_days": None, "status": "INVALID_MARKET_DATES"}
    latest = latest.tz_convert(None).normalize()
    days = max(0, int((now.normalize() - latest).days))
    dates = pd.to_datetime(frame["Date"], utc=True, errors="coerce").dropna().dt.tz_convert(None).dt.normalize().drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()
    long_gaps = int((gaps > 5).sum())
    largest_gap = int(gaps.max()) if len(gaps) else 0
    return {"latest_date": latest.date().isoformat(), "staleness_days": days,
            "status": "STALE_WARNING" if days > stale_after_days else "FRESH_ENOUGH",
            "calendar_gaps_over_5_days": long_gaps, "largest_calendar_gap_days": largest_gap,
            "session_check_method": "calendar-gap heuristic only; not exchange-calendar validation"}
