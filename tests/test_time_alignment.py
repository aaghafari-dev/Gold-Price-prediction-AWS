"""tests/test_time_alignment.py: project component; see README.md and docs/ for workflow and configuration details."""
import pandas as pd
from src.data.time_alignment import calendar_audit, load_verified_session_calendar
from src.features.news_features import aggregate_news_daily


def test_calendar_requires_explicit_verified_source(tmp_path):
    """Test the expected behavior of the corresponding project component."""
    p = tmp_path / 'gc_session_calendar.csv'
    p.write_text('session_date,cutoff_utc,calendar_source,verified\n2026-01-05,2026-01-05T22:00:00Z,example,false\n')
    frame = load_verified_session_calendar(p)
    assert not calendar_audit(frame, p)['verified']


def test_news_after_cutoff_is_excluded():
    """Test the expected behavior of the corresponding project component."""
    news = pd.DataFrame([
        {'published_at': '2026-01-05T20:00:00Z', 'title': 'gold stability', 'summary': '', 'source': 'a', 'url': 'u'},
        {'published_at': '2026-01-05T23:00:00Z', 'title': 'gold crisis', 'summary': '', 'source': 'b', 'url': 'v'},
    ])
    cuts = pd.DataFrame([{'session_date': '2026-01-05', 'cutoff_utc': '2026-01-05T22:00:00Z'}])
    out = aggregate_news_daily(news, market_dates=['2026-01-05'], session_cutoffs=cuts)
    assert len(out) == 1
    assert out.iloc[0]['news_count'] == 1
