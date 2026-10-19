"""Explicit, auditable forecast-cutoff calendar for GC futures news alignment.

The project deliberately does not invent an exchange cutoff. The supplied calendar
must be verified against the chosen CME/COMEX GC contract/session convention by the
project owner before point-in-time news backtests are declared valid.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = {"session_date", "cutoff_utc", "calendar_source", "verified"}

def load_verified_session_calendar(path: Path) -> pd.DataFrame:
    """Load the declared session calendar; return empty when absent or malformed."""
    if not path.exists():
        return pd.DataFrame(columns=["session_date", "cutoff_utc", "calendar_source", "verified"])
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["session_date", "cutoff_utc", "calendar_source", "verified"])
    if not REQUIRED.issubset(frame.columns):
        return pd.DataFrame(columns=["session_date", "cutoff_utc", "calendar_source", "verified"])
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.normalize()
    frame["cutoff_utc"] = pd.to_datetime(frame["cutoff_utc"], utc=True, errors="coerce")
    frame["verified"] = frame["verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame = frame.dropna(subset=["session_date", "cutoff_utc"])
    return frame.sort_values("session_date").drop_duplicates("session_date", keep="last")

def calendar_audit(frame: pd.DataFrame, path: Path) -> dict:
    """Return conservative verification status and a SHA-256 for provenance."""
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    valid = bool(not frame.empty and frame["verified"].all() and frame["calendar_source"].astype(str).str.strip().ne("").all())
    return {"path": str(path), "exists": path.exists(), "rows": int(len(frame)), "verified": valid,
            "all_rows_marked_verified": bool(not frame.empty and frame["verified"].all()), "sha256": digest,
            "required_columns": sorted(REQUIRED),
            "note": "A verified flag is a user attestation, not automatic CME certification. Record the authoritative calendar source and audit the GC session convention."}
