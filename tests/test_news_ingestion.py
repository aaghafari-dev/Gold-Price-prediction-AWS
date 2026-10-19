"""Unit tests for provider timestamp normalization, archive deduplication, ingestion logging and sentiment score overrides."""
import pandas as pd
from src.data.news import _as_utc, save_news


def test_gdelt_compact_timestamp_formats():
    """Check GDELT compact timestamp variants normalize to UTC."""
    assert _as_utc('20261012T070000Z') == pd.Timestamp('2026-10-12T07:00:00Z')
    assert _as_utc('20261012T070000') == pd.Timestamp('2026-10-12T07:00:00Z')
    assert _as_utc('20261012070000') == pd.Timestamp('2026-10-12T07:00:00Z')


def test_news_archive_appends_deduplicates_and_logs_window(tmp_path):
    """Check repeated news fetches deduplicate the archive while retaining separate query-attempt log rows."""
    archive = tmp_path / 'news.csv'
    row = pd.DataFrame([{
        'published_at': '2026-10-12T07:00:00Z', 'retrieved_at': '2026-10-12T08:00:00Z',
        'title': 'Gold market article', 'summary': '', 'source': 'Example', 'url': 'https://example.invalid/1'
    }])
    metadata = {'adapter': 'gdelt', 'query': 'gold', 'timespan': '7d',
                'query_window_start_utc': '2026-10-05T08:00:00Z', 'query_window_end_utc': '2026-10-12T08:00:00Z'}
    save_news(row, archive, append=True, ingestion_metadata=metadata)
    save_news(row, archive, append=True, ingestion_metadata=metadata)
    saved = pd.read_csv(archive)
    log = pd.read_csv(tmp_path / 'news_ingestion_log.csv')
    assert len(saved) == 1
    assert len(log) == 2
    assert log.iloc[0]['adapter'] == 'gdelt'
    snapshot_path = tmp_path / 'news_snapshots.jsonl'
    assert snapshot_path.exists() and len(snapshot_path.read_text(encoding='utf-8').splitlines()) == 2


def test_empty_overwrite_does_not_destroy_archive(tmp_path):
    """Check an empty overwrite cannot erase an existing archive."""
    archive = tmp_path / 'news.csv'
    row = pd.DataFrame([{'published_at': '2026-10-12T07:00:00Z', 'title': 'Keep me', 'url': 'x'}])
    save_news(row, archive)
    import pytest
    with pytest.raises(ValueError):
        save_news(pd.DataFrame(), archive, append=False)
    assert len(pd.read_csv(archive)) == 1


def test_sentiment_override_is_used_for_downstream_comparison():
    """A scored archive lets lexicon/FinBERT be compared through the same feature pipeline."""
    from src.features.news_features import aggregate_news_daily
    news = pd.DataFrame([{
        "published_at": "2026-10-12T07:00:00Z", "title": "neutral headline", "summary": "",
        "source": "test", "url": "u", "sentiment_score": -0.7
    }])
    daily = aggregate_news_daily(news, market_dates=pd.to_datetime(["2026-10-12", "2026-10-13"]))
    assert daily.iloc[0]["news_sentiment_mean"] == -0.7


def test_news_archive_audit_reports_query_window_gaps(tmp_path):
    """Archive audit should report attempted-window gaps without claiming completeness."""
    from scripts.audit_news_archive import audit_archive
    news_path=tmp_path/"news.csv"; log_path=tmp_path/"news_ingestion_log.csv"; out=tmp_path/"audit.json"
    pd.DataFrame([{"published_at":"2026-01-01T10:00:00Z","retrieved_at":"2026-01-01T11:00:00Z","title":"Gold","source":"GDELT"}]).to_csv(news_path,index=False)
    pd.DataFrame([
      {"query_window_start_utc":"2026-01-01T00:00:00Z","query_window_end_utc":"2026-01-02T00:00:00Z"},
      {"query_window_start_utc":"2026-01-06T00:00:00Z","query_window_end_utc":"2026-01-07T00:00:00Z"}
    ]).to_csv(log_path,index=False)
    report=audit_archive(news_path,log_path,out)
    assert report["query_window_gaps"][0]["missing_calendar_days"] == 3
    assert "does not prove complete" in report["interpretation"]


def test_provider_payload_normalization_handles_gdelt_and_yfinance_shapes():
    """Both GDELT ArtList and current yfinance content records normalize to the shared schema."""
    from src.data.news import _normalise
    gdelt = _normalise([{
        "title": "Gold rises", "seendate": "20261016T090000Z",
        "domain": "example.org", "url": "https://example.org/a"
    }], source_default="GDELT")
    assert len(gdelt) == 1
    assert gdelt.iloc[0]["source"] == "example.org"
    yf = _normalise([{
        "content": {
            "title": "Gold outlook", "pubDate": "2026-10-16T09:15:00Z",
            "summary": "Market update", "provider": {"displayName": "Example News"},
            "canonicalUrl": {"url": "https://example.org/b"}
        }
    }], source_default="Yahoo Finance")
    assert len(yf) == 1
    assert yf.iloc[0]["source"] == "Example News"
    assert yf.iloc[0]["url"] == "https://example.org/b"


def test_news_ingestion_log_records_empty_and_error_status(tmp_path):
    """An empty or failed provider attempt must be visible in the ingestion log."""
    archive = tmp_path / "news.csv"
    import pandas as pd
    empty = pd.DataFrame(columns=["published_at", "retrieved_at", "title", "summary", "source", "url"])
    save_news(empty, archive, ingestion_metadata={"adapter": "gdelt", "status": "empty"})
    save_news(empty, archive, ingestion_metadata={"adapter": "yfinance", "status": "error", "error": "network timeout"})
    log = pd.read_csv(tmp_path / "news_ingestion_log.csv")
    assert list(log["status"]) == ["empty", "error"]
    assert log.iloc[1]["error"] == "network timeout"


def test_gdelt_rate_limit_uses_backoff_and_simpler_query(monkeypatch):
    """A persistent 429 for the full query should trigger a low-volume fallback, not rapid retries."""
    from src.data import news as news_module
    class Response:
        """Represent the Response component and its behavior."""
        def __init__(self, status, payload=None):
            """Initialize object state and configure its dependencies."""
            self.status_code = status
            self.headers = {"Retry-After": "1"} if status == 429 else {}
            self._payload = payload or {}
            self.text = "rate limited"
        def raise_for_status(self):
            """Document the purpose and behavior of raise for status. """
            if self.status_code >= 400:
                import requests
                raise requests.HTTPError(f"HTTP {self.status_code}")
        def json(self):
            """Document the purpose and behavior of json. """
            return self._payload
    calls = []
    def fake_get(url, params, headers, timeout):
        """Document the purpose and behavior of fake get. """
        calls.append(params["query"])
        if params["query"] != "gold":
            return Response(429)
        return Response(200, {"articles": [{"title": "Gold news", "seendate": "20261016T090000Z", "url": "https://example.org/a"}]})
    monkeypatch.setattr(news_module.requests, "get", fake_get)
    monkeypatch.setattr(news_module.time, "sleep", lambda _: None)
    df = news_module.fetch_gdelt_news('gold OR bullion OR inflation', timespan='7d', retries=2)
    assert len(df) == 1
    assert calls[-1] == "gold"


def test_yahoo_rss_fallback_normalizes_feed_records(monkeypatch):
    """RSS fallback records should use the same normalized archive schema."""
    from src.data import news as news_module
    monkeypatch.setattr(news_module, "_fetch_yahoo_rss", lambda symbol: [{
        "title": "Gold rises", "summary": "Market report", "pubDate": "2026-10-16T09:00:00+00:00",
        "url": "https://example.org/rss", "source": "Yahoo Finance RSS"
    }])
    df = news_module._normalise(news_module._fetch_yahoo_rss("GC=F"), source_default="Yahoo Finance RSS")
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Gold rises"


def test_google_news_rss_normalizes_timestamped_headlines(monkeypatch):
    """Test the expected behavior of the corresponding project component."""
    from src.data.news import fetch_google_news_rss
    class Response:
        """Represent the Response component and its behavior."""
        content = b'''<rss><channel><item><title>Gold rises after policy remarks</title><description>Market summary</description><pubDate>Sat, 17 Oct 2026 06:00:00 GMT</pubDate><link>https://example.test/story</link></item></channel></rss>'''
        def raise_for_status(self):
            """Document the purpose and behavior of raise for status. """
            return None
    monkeypatch.setattr('src.data.news.requests.get', lambda *a, **k: Response())
    frame = fetch_google_news_rss('gold')
    assert len(frame) == 1
    assert frame.iloc[0]['title'] == 'Gold rises after policy remarks'
    assert frame.iloc[0]['source'] == 'Google News RSS'
    assert 'published_at' in frame.columns


def test_newsapi_requires_api_key(monkeypatch):
    """Test the expected behavior of the corresponding project component."""
    from src.data.news import fetch_newsapi
    monkeypatch.delenv('NEWSAPI_API_KEY', raising=False)
    import pytest
    with pytest.raises(RuntimeError, match='NEWSAPI_API_KEY'):
        fetch_newsapi('gold')


def test_news_archive_sort_handles_mixed_timestamp_strings(tmp_path):
    """Mixed CSV string timestamps and incoming pandas timestamps must not crash live polling."""
    archive = tmp_path / "news.csv"
    pd.DataFrame([{"published_at": "2026-10-16T09:00:00Z", "title": "Older", "url": "a"}]).to_csv(archive, index=False)
    fresh = pd.DataFrame([{"published_at": pd.Timestamp("2026-10-17T09:00:00Z"), "title": "Newer", "url": "b"}])
    save_news(fresh, archive, append=True)
    saved = pd.read_csv(archive)
    assert list(saved["title"]) == ["Older", "Newer"]


def test_provider_registry_applies_lookback_to_returned_rows(monkeypatch):
    """Provider-independent lookback filters old articles but does not invent missing coverage."""
    from src.data import news as news_module
    now = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame([
        {"published_at": now - pd.Timedelta(minutes=20), "title": "Recent", "url": "r"},
        {"published_at": now - pd.Timedelta(days=3), "title": "Old", "url": "o"},
    ])
    monkeypatch.setattr(news_module, "fetch_google_news_rss", lambda *a, **k: frame.copy())
    got = news_module.fetch_news_provider("google_news_rss", "gold", timespan="1h")
    assert list(got["title"]) == ["Recent"]
