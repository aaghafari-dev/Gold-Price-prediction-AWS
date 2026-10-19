"""Normalized news ingestion adapters for a no-licensed-data prototype.

Sources: yfinance current ticker news and the public GDELT DOC API. Neither adapter
should be assumed to provide a complete, immutable historical archive. Always retain
published_at and retrieved_at and enforce point-in-time cutoffs downstream.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
import time
import email.utils
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from config.project_config import RAW, NEWS_LOOKBACK_DAYS

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def _as_utc(value):
    """Parse provider-specific numeric, compact-string or ISO timestamps into timezone-aware UTC pandas timestamps."""
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    if isinstance(value, str):
        compact = value.strip()
        # GDELT often uses YYYYMMDDTHHMMSSZ or YYYYMMDDHHMMSS timestamps.
        if len(compact) == 16 and compact[8] == "T" and compact.endswith("Z"):
            return pd.to_datetime(compact, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if len(compact) == 15 and compact[8] == "T":
            return pd.to_datetime(compact, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
        if len(compact) == 14 and compact.isdigit():
            return pd.to_datetime(compact, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _normalise(rows: list[dict], source_default: str = "unknown") -> pd.DataFrame:
    """Map provider-specific news records into a common timestamped metadata schema and deduplicate them."""
    retrieved_at = pd.Timestamp.now(tz="UTC")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Yahoo's response has changed shape across yfinance releases. Accept
        # both the newer {"content": {...}} records and legacy flat records.
        content = row.get("content") if isinstance(row.get("content"), dict) else row
        title = content.get("title") or row.get("title") or content.get("headline") or ""
        summary = (content.get("summary") or content.get("description")
                   or content.get("snippet") or row.get("summary") or "")
        pub = (content.get("pubDate") or content.get("seendate") or content.get("seenDate")
               or content.get("providerPublishTime") or content.get("published_at")
               or content.get("publishedAt") or content.get("time_published") or content.get("displayTime")
               or row.get("published_at") or row.get("publishedAt")
               or row.get("providerPublishTime"))
        published_at = _as_utc(pub)
        if pd.isna(published_at) or not str(title).strip():
            continue
        provider = content.get("provider")
        if isinstance(provider, dict):
            source = provider.get("displayName", source_default)
        else:
            source = content.get("source") or content.get("domain") or source_default
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url", "")
        else:
            url = content.get("url") or row.get("url") or ""
        records.append({
            "published_at": published_at,
            "retrieved_at": retrieved_at,
            "title": str(title).strip(),
            "summary": str(summary).strip(),
            "source": str(source),
            "url": str(url),
        })
    cols = ["published_at", "retrieved_at", "title", "summary", "source", "url"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records, columns=cols).drop_duplicates(subset=["published_at", "title", "url"])


def _fetch_yahoo_rss(symbol: str, timeout: int = 20) -> list[dict]:
    """Fallback to Yahoo Finance's public RSS feed when yfinance's JSON feed is empty."""
    from urllib.parse import quote
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urlencode({
        "s": symbol, "region": "US", "lang": "en-US"
    })
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; timeout=timeout)"}, timeout=timeout)
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError(f"Yahoo RSS returned invalid XML: {exc}") from exc
    rows = []
    for item in root.findall(".//item"):
        pub = item.findtext("pubDate")
        parsed = email.utils.parsedate_to_datetime(pub) if pub else None
        rows.append({
            "title": item.findtext("title") or "",
            "summary": item.findtext("description") or "",
            "pubDate": parsed.isoformat() if parsed else pub,
            "url": item.findtext("link") or "",
            "source": "Yahoo Finance RSS",
        })
    return rows


def fetch_live_yfinance_news(symbol: str = "GC=F", count: int = 100, lookback_days: int | None = None) -> pd.DataFrame:
    """Fetch ticker news using yfinance, then Yahoo RSS fallback; report usable-record counts.

    Yahoo's feed is recent rather than a complete historical archive. A wider configurable
    cutoff avoids discarding valid articles merely because the default was only three days.
    """
    import yfinance as yf
    diagnostics = []
    rows = []
    try:
        ticker = yf.Ticker(symbol)
        try:
            result = ticker.get_news(count=count, tab="news")
        except TypeError:
            result = ticker.news
        if isinstance(result, dict):
            result = result.get("news") or result.get("items") or []
        rows = result if isinstance(result, list) else []
        diagnostics.append(f"yfinance payload items={len(rows)}")
    except Exception as exc:
        diagnostics.append(f"yfinance JSON feed failed: {type(exc).__name__}: {exc}")
    df = _normalise(rows, source_default="Yahoo Finance")
    if df.empty:
        try:
            rss_rows = _fetch_yahoo_rss(symbol)
            diagnostics.append(f"Yahoo RSS items={len(rss_rows)}")
            df = _normalise(rss_rows, source_default="Yahoo Finance RSS")
        except Exception as exc:
            diagnostics.append(f"Yahoo RSS fallback failed: {type(exc).__name__}: {exc}")
    if df.empty:
        raise RuntimeError("No usable Yahoo news records. " + " | ".join(diagnostics) +
                           ". Check network/proxy restrictions and the provider feed; no articles were written.")
    days = NEWS_LOOKBACK_DAYS if lookback_days is None else max(1, int(lookback_days))
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    recent = df[df["published_at"] >= cutoff].reset_index(drop=True)
    # Some Yahoo feeds omit/incorrectly format publication timestamps. Keep only timestamped
    # records in the canonical archive, but explain why all valid records were filtered.
    if recent.empty:
        newest = df["published_at"].max()
        raise RuntimeError(f"Yahoo returned {len(df)} usable timestamped articles, but none fell within {days} days. "
                           f"Newest timestamp={newest}; increase --yahoo-lookback-days if appropriate. " + " | ".join(diagnostics))
    recent.attrs["diagnostics"] = diagnostics
    return recent

def fetch_gdelt_news(query: str, timespan: str = "7d", max_records: int = 100,
                     timeout: int = 30, retries: int = 4) -> pd.DataFrame:
    """Fetch GDELT article metadata with rate-limit-aware retries and query fallbacks.

    HTTP 429 is treated specially: honour Retry-After when supplied, then use exponential
    backoff. If the original request is still rate-limited, try a simpler query and fewer
    records once. This avoids hammering GDELT; it cannot bypass a sustained rate limit.
    """
    query_variants = [(query, max(1, min(int(max_records), 250)))]
    if query.strip().lower() not in {"gold", '"gold"'}:
        query_variants.append(("gold", min(50, max(1, int(max_records)))))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; educational use)"}
    errors = []
    for variant_query, variant_count in query_variants:
        params = {"query": variant_query, "mode": "ArtList", "format": "json",
                  "sort": "DateDesc", "maxrecords": variant_count, "timespan": timespan}
        for attempt in range(max(1, retries)):
            try:
                response = requests.get(GDELT_ENDPOINT, params=params, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        wait = min(60.0, max(1.0, float(retry_after))) if retry_after else min(30.0, 2.0 ** attempt)
                    except ValueError:
                        wait = min(30.0, 2.0 ** attempt)
                    errors.append(f"HTTP 429 query={variant_query!r}; wait={wait:g}s")
                    if attempt < retries - 1:
                        time.sleep(wait)
                        continue
                    break
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    excerpt = response.text[:240].replace("\n", " ")
                    raise RuntimeError(f"GDELT returned non-JSON (HTTP {response.status_code}): {excerpt}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"GDELT returned unexpected JSON type: {type(payload).__name__}")
                articles = payload.get("articles", [])
                if not isinstance(articles, list):
                    raise RuntimeError("GDELT JSON has no usable 'articles' list")
                df = _normalise(articles, source_default="GDELT")
                if not df.empty:
                    df["query"] = variant_query
                    df["timespan"] = timespan
                if df.empty:
                    errors.append(f"GDELT returned zero usable records for query={variant_query!r}")
                return df
            except requests.RequestException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < retries - 1:
                    time.sleep(min(30.0, 2.0 ** attempt))
            except RuntimeError as exc:
                errors.append(str(exc))
                break
    raise RuntimeError("GDELT could not return usable results after rate-aware retries/fallbacks: " + " | ".join(errors[-8:]))


def fetch_google_news_rss(query: str, timeout: int = 20, max_records: int = 100) -> pd.DataFrame:
    """Fetch public Google News RSS search results; no API key required.

    This is a headline discovery source, not a complete archive or guaranteed SLA.
    The RSS timestamp is preserved as publication time and retrieval time is logged.
    """
    from urllib.parse import quote_plus
    url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AurumLab/10.0; educational research)"}, timeout=timeout)
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError(f"Google News RSS returned invalid XML: {exc}") from exc
    rows = []
    for item in root.findall('.//item')[:max(1, min(int(max_records), 100))]:
        pub = item.findtext('pubDate')
        try:
            parsed = email.utils.parsedate_to_datetime(pub) if pub else None
            stamp = parsed.isoformat() if parsed else pub
        except (TypeError, ValueError, OverflowError):
            stamp = pub
        rows.append({"title": item.findtext('title') or "", "summary": item.findtext('description') or "",
                     "publishedAt": stamp, "url": item.findtext('link') or "", "source": "Google News RSS"})
    frame = _normalise(rows, source_default="Google News RSS")
    if frame.empty:
        raise RuntimeError("Google News RSS responded successfully but contained no usable timestamped headlines")
    frame["provider"] = "google_news_rss"
    return frame


def fetch_newsapi(query: str, api_key: str | None = None, timeout: int = 20, max_records: int = 100) -> pd.DataFrame:
    """Fetch NewsAPI.org recent headlines when NEWSAPI_API_KEY is configured."""
    import os
    key = api_key or os.getenv("NEWSAPI_API_KEY")
    if not key:
        raise RuntimeError("NEWSAPI_API_KEY is not configured; set it in the process environment or .env and load it securely")
    response = requests.get("https://newsapi.org/v2/everything", params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": min(100, max(1, int(max_records))), "apiKey": key}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {payload.get('code', 'unknown')} — {payload.get('message', 'no message')}")
    frame = _normalise(payload.get("articles", []), source_default="NewsAPI")
    if frame.empty:
        raise RuntimeError("NewsAPI returned no usable timestamped articles for this query")
    frame["provider"] = "newsapi"
    return frame


def fetch_alpha_vantage_news(query: str, api_key: str | None = None, timeout: int = 20, max_records: int = 100) -> pd.DataFrame:
    """Fetch Alpha Vantage NEWS_SENTIMENT results when ALPHAVANTAGE_API_KEY is configured."""
    import os
    key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured; set it in the process environment or .env")
    response = requests.get("https://www.alphavantage.co/query", params={"function": "NEWS_SENTIMENT", "tickers": "GC=F", "topics": "financial_markets,economy_monetary", "limit": min(1000, max(1, int(max_records))), "apikey": key}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("Note") or payload.get("Information") or payload.get("Error Message"):
        raise RuntimeError("Alpha Vantage response: " + str(payload.get("Note") or payload.get("Information") or payload.get("Error Message")))
    frame = _normalise(payload.get("feed", []), source_default="Alpha Vantage")
    if frame.empty:
        raise RuntimeError("Alpha Vantage returned no usable timestamped news items")
    frame["provider"] = "alphavantage"
    return frame


def fetch_news_provider(provider: str, query: str, symbol: str = "GC=F", timespan: str = "7d", count: int = 100) -> pd.DataFrame:
    """Provider registry used by the GUI/CLI; provider failures are isolated and logged."""
    key = provider.strip().lower()
    if key in {"google", "google_news", "google_news_rss", "rss"}:
        frame = fetch_google_news_rss(query, max_records=count)
    elif key == "newsapi":
        frame = fetch_newsapi(query, max_records=count)
    elif key in {"alphavantage", "alpha_vantage"}:
        frame = fetch_alpha_vantage_news(query, max_records=count)
    elif key == "yfinance":
        frame = fetch_live_yfinance_news(symbol, count=count, lookback_days=30)
        frame["provider"] = "yfinance"
    elif key == "gdelt":
        frame = fetch_gdelt_news(query, timespan=timespan, max_records=count)
        frame["provider"] = "gdelt"
    else:
        raise ValueError(f"Unknown news provider: {provider}")
    # Apply a consistent local cutoff where APIs do not expose equivalent time-window parameters.
    # This filters returned records only; it cannot expand a provider's historical coverage.
    import re
    match = re.fullmatch(r"\s*(\d+)\s*(h|d|y)\s*", str(timespan).lower())
    if match and not frame.empty and "published_at" in frame:
        amount, unit = int(match.group(1)), match.group(2)
        delta = pd.Timedelta(hours=amount) if unit == "h" else (pd.Timedelta(days=amount) if unit == "d" else pd.Timedelta(days=365 * amount))
        cutoff = pd.Timestamp.now(tz="UTC") - delta
        published = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        frame = frame.loc[published >= cutoff].copy()
        if frame.empty:
            raise RuntimeError(f"Provider returned no usable headlines within requested lookback {timespan}; its feed may not cover that period")
    return frame

def load_historical_news_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize a supplied timestamped news CSV without assuming provider completeness."""
    df = pd.read_csv(path)
    if not {"published_at", "title"}.issubset(df.columns):
        raise ValueError("News CSV must contain published_at and title columns")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if "retrieved_at" in df:
        df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
    else:
        df["retrieved_at"] = pd.NaT
    for col, default in [("summary", ""), ("source", "unknown"), ("url", "")]:
        if col not in df:
            df[col] = default
    return df.dropna(subset=["published_at"]).drop_duplicates(["published_at", "title", "url"])


def save_news(df: pd.DataFrame, path: str | Path | None = None, append: bool = True, ingestion_metadata: dict | None = None) -> Path:
    """Persist normalized news, deduplicating against the existing local archive.

    Append is useful for building a timestamped research archive over time. It cannot
    reconstruct missing historical articles that the provider never returned.
    """
    path = Path(path or RAW / "news.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = df.copy()
    if not append and incoming.empty:
        raise ValueError("Refusing to overwrite the news archive with an empty fetch result.")
    if append and path.exists():
        existing = pd.read_csv(path)
        incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if not incoming.empty and {"published_at", "title"}.issubset(incoming.columns):
        if "url" not in incoming:
            incoming["url"] = ""
        incoming["published_at"] = pd.to_datetime(incoming["published_at"], utc=True, errors="coerce")
        incoming = incoming.dropna(subset=["published_at"])
        incoming = incoming.drop_duplicates(["published_at", "title", "url"], keep="last")
        incoming = incoming.sort_values("published_at", kind="stable")
    if incoming.empty:
        incoming = pd.DataFrame(columns=["published_at", "retrieved_at", "title", "summary", "source", "url"])
    incoming.to_csv(path, index=False)
    # Append-only ingestion log: records when data was queried, not completeness guarantees.
    log_path = path.with_name("news_ingestion_log.csv")
    metadata = ingestion_metadata or {}
    log_row = pd.DataFrame([{
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_returned_this_run": int(len(df)),
        "archive_rows_after_write": int(len(incoming)),
        "archive_path": str(path),
        "adapter": metadata.get("adapter", "unknown"),
        "status": metadata.get("status", "success" if len(df) else "empty"),
        "error": metadata.get("error", ""),
        "query": metadata.get("query", ""),
        "timespan": metadata.get("timespan", ""),
        "query_window_start_utc": metadata.get("query_window_start_utc", ""),
        "query_window_end_utc": metadata.get("query_window_end_utc", ""),
    }])
    if append and log_path.exists():
        log_row = pd.concat([pd.read_csv(log_path), log_row], ignore_index=True)
    log_row.to_csv(log_path, index=False)

    # Append a retrieval-time snapshot of exactly the metadata returned by the
    # adapter. This preserves successive versions without pretending the API gave
    # us the full article body or a complete historical archive.
    snapshot_path = path.with_name("news_snapshots.jsonl")
    snapshot_time = datetime.now(timezone.utc).isoformat()
    with snapshot_path.open("a", encoding="utf-8") as snapshot_file:
        for record in df.to_dict(orient="records"):
            snapshot = {key: (value.isoformat() if hasattr(value, "isoformat") else value)
                        for key, value in record.items()}
            snapshot["snapshot_retrieved_at_utc"] = snapshot_time
            snapshot["ingestion_metadata"] = metadata
            snapshot_file.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
    return path


def load_news(path: str | Path | None = None) -> pd.DataFrame:
    """Load the configured local news archive and fail clearly if it does not exist."""
    path = Path(path or RAW / "news.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Fetch news first or provide a timestamped news CSV.")
    return load_historical_news_csv(path)
