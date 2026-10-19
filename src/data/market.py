"""Market-data ingestion with Yahoo Finance and deterministic caching."""
from pathlib import Path
import time
import warnings
import re
from datetime import datetime, timedelta, timezone
import pandas as pd

from config.project_config import RAW, MARKET_SYMBOLS, HISTORY_YEARS, GOLD_SYMBOL

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to a simple column index when necessary."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _period_start(period: str) -> pd.Timestamp:
    """Convert a supported lookback such as ``1d``, ``6mo`` or ``5y`` to a UTC-naive start date."""
    match = re.fullmatch(r"\s*(\d+)\s*(d|w|mo|y)\s*", str(period).lower())
    if not match:
        raise ValueError(f"Unsupported lookback period {period!r}; use a number followed by d, w, mo, or y")
    amount, unit = int(match.group(1)), match.group(2)
    if amount < 1:
        raise ValueError("Lookback period must be at least one unit")
    now = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    if unit == "d": return now - pd.Timedelta(days=amount)
    if unit == "w": return now - pd.Timedelta(weeks=amount)
    if unit == "mo": return now - pd.DateOffset(months=amount)
    return now - pd.DateOffset(years=amount)


def fetch_symbol(symbol: str, years: int = HISTORY_YEARS, retries: int = 3, period: str | None = None) -> pd.DataFrame:
    """Download one daily market series for a shared lookback period, normalize dates, and return numeric columns.

    ``period`` supports short and long lookbacks (for example ``1d``, ``30d``, ``6mo``, ``1y`` and ``5y``).
    The legacy ``years`` argument remains supported for existing scripts.
    """
    import yfinance as yf
    last_error = None
    df = None
    for attempt in range(max(1, retries)):
        try:
            lookback = period or f"{max(1, years)}y"
            start = _period_start(lookback).date().isoformat()
            end = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)).date().isoformat()
            df = yf.download(
                symbol, start=start, end=end, interval="1d", auto_adjust=False,
                progress=False, threads=False, timeout=30,
            )
            if df is not None and not df.empty:
                break
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {symbol} after {retries} attempts: {last_error}")
    df = _flatten(df).reset_index()
    date_col = "Date"
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None).dt.normalize()
    wanted = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[[date_col] + wanted].copy()
    df = df.rename(columns={"Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("Date").sort_values("Date")

def fetch_market_bundle(years: int = HISTORY_YEARS, period: str | None = None) -> pd.DataFrame:
    """Download and date-join gold and context markets using one common requested lookback."""
    frames = []
    for name, symbol in MARKET_SYMBOLS.items():
        try:
            df = fetch_symbol(symbol, years, period=period)
        except Exception as exc:
            if name == "gold":
                raise RuntimeError(f"Required gold target series {symbol} could not be downloaded: {exc}") from exc
            warnings.warn(f"Optional context series {name} ({symbol}) unavailable; continuing without it: {exc}", RuntimeWarning)
            continue
        df = df[["Date", "close", "volume"]].rename(
            columns={"close": f"{name}_close", "volume": f"{name}_volume"}
        )
        frames.append(df)
    if not frames:
        raise RuntimeError("No market series could be downloaded; gold target data is required.")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Date", how="outer")
    out = out.sort_values("Date").reset_index(drop=True)
    # Market instruments have different holidays; forward-fill only predictors,
    # never the gold target itself.
    predictor_cols = [c for c in out.columns if c != "Date" and not c.startswith("gold_")]
    out[predictor_cols] = out[predictor_cols].ffill()
    out = out.dropna(subset=["gold_close"]).reset_index(drop=True)
    return out

def save_market_bundle(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the normalized market panel to the local CSV cache."""
    path = path or RAW / "market_daily.csv"
    df.to_csv(path, index=False)
    return path

def load_or_fetch_market(force: bool = False, period: str | None = None) -> pd.DataFrame:
    """Reuse a cache only when it covers the requested period; otherwise download and save the requested panel."""
    path = RAW / "market_daily.csv"
    requested_start = _period_start(period or f"{HISTORY_YEARS}y")
    if path.exists() and not force:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h <= 24:
            cached = pd.read_csv(path, parse_dates=["Date"])
            if not cached.empty and pd.to_datetime(cached["Date"], errors="coerce").min() <= requested_start:
                return cached.loc[pd.to_datetime(cached["Date"]) >= requested_start].reset_index(drop=True)
    df = fetch_market_bundle(period=period)
    save_market_bundle(df, path)
    return df
