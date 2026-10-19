"""Audit timestamped news metadata and logged query windows for coverage gaps.

This tool reports archive size, unique publication dates, source mix, query-window
span and gaps. It cannot prove that a provider returned every relevant article and
must not be presented as a completeness certification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def audit_archive(news_path: Path, log_path: Path, output_path: Path) -> dict:
    """Summarize article metadata and query-window gaps and write a JSON audit."""
    news = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not news.empty:
        news["published_at"] = pd.to_datetime(news.get("published_at"), utc=True, errors="coerce")
        if "retrieved_at" in news:
            news["retrieved_at"] = pd.to_datetime(news["retrieved_at"], utc=True, errors="coerce")
        valid = news.dropna(subset=["published_at"])
        publication_days = sorted(valid["published_at"].dt.strftime("%Y-%m-%d").unique().tolist())
        source_counts = valid.get("source", pd.Series("unknown", index=valid.index)).fillna("unknown").value_counts().to_dict()
        first_pub = valid["published_at"].min().isoformat() if len(valid) else None
        last_pub = valid["published_at"].max().isoformat() if len(valid) else None
        invalid_timestamps = int(news["published_at"].isna().sum())
        if "retrieved_at" in news:
            retrieved_before_publication = int((news["retrieved_at"] < news["published_at"]).fillna(False).sum())
        else:
            retrieved_before_publication = None
    else:
        publication_days, source_counts, first_pub, last_pub, invalid_timestamps, retrieved_before_publication = [], {}, None, None, 0, None
    windows=[]
    if not log.empty:
        for _, row in log.iterrows():
            start=pd.to_datetime(row.get("query_window_start_utc"), utc=True, errors="coerce")
            end=pd.to_datetime(row.get("query_window_end_utc"), utc=True, errors="coerce")
            if pd.notna(start) and pd.notna(end) and start <= end:
                windows.append((start.normalize(), end.normalize()))
    windows.sort(key=lambda x:x[0])
    merged=[]
    for start,end in windows:
        if not merged or start > merged[-1][1] + pd.Timedelta(days=1): merged.append([start,end])
        else: merged[-1][1]=max(merged[-1][1],end)
    gaps=[]
    for left,right in zip(merged,merged[1:]):
        gap_days=max(0,int((right[0]-left[1]).days)-1)
        if gap_days: gaps.append({"from_utc":left[1].date().isoformat(),"to_utc":right[0].date().isoformat(),"missing_calendar_days":gap_days})
    report={
        "news_path":str(news_path),"query_log_path":str(log_path),"audited_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),
        "article_rows":int(len(news)),"valid_publication_timestamps":int(news["published_at"].notna().sum()) if not news.empty else 0,
        "unique_publication_days":int(len(publication_days)),"first_publication_utc":first_pub,"last_publication_utc":last_pub,
        "invalid_publication_timestamps":invalid_timestamps,"source_row_counts":source_counts,
        "query_log_rows":int(len(log)),"valid_query_windows":int(len(windows)),"merged_query_windows":int(len(merged)),
        "query_window_gaps":gaps,"retrieved_before_publication_count":retrieved_before_publication,
        "interpretation":"This is a metadata and query-window audit only. It does not prove complete article recall or point-in-time archive validity."
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    return report


def main() -> None:
    """Parse CLI options and print a compact archive audit summary."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news",default="data/raw/news.csv")
    parser.add_argument("--log",default="data/raw/news_ingestion_log.csv")
    parser.add_argument("--out",default="artifacts/news_archive_audit.json")
    args=parser.parse_args()
    report=audit_archive(Path(args.news),Path(args.log),Path(args.out))
    print(json.dumps({k:report[k] for k in ["article_rows","unique_publication_days","query_log_rows","valid_query_windows","query_window_gaps","interpretation"]},indent=2))

if __name__ == "__main__": main()
