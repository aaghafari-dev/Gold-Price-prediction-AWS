"""Append local prediction records and evaluate forecasts after labels mature.

This is a lightweight research ledger, not a transaction system. Forecast dates should
be supplied explicitly as the last market session used to build the request features.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config.project_config import ROOT, RAW, ARTIFACTS, TARGET_HORIZON

LEDGER_PATH = ROOT / "logs" / "prediction_ledger.csv"


def append_prediction(*, forecast_date: str | None, predicted_price: float, model_name: str,
                      feature_set: str, features: dict) -> Path:
    """Append a prediction and its feature payload to the local audit ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "forecasted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "forecast_date": forecast_date or "",
        "predicted_price": float(predicted_price),
        "model_name": model_name,
        "feature_set": feature_set,
        "features_json": json.dumps(features, sort_keys=True, default=str),
        "target_date": "", "actual_price": np.nan, "absolute_error": np.nan,
        "squared_error": np.nan, "direction_correct": np.nan,
    }])
    if LEDGER_PATH.exists():
        row = pd.concat([pd.read_csv(LEDGER_PATH), row], ignore_index=True)
    row.to_csv(LEDGER_PATH, index=False)
    return LEDGER_PATH


def evaluate_matured_predictions(ledger_path: Path = LEDGER_PATH, market_path: Path = RAW / "market_daily.csv",
                                 output_path: Path = ARTIFACTS / "prediction_quality_report.json") -> dict:
    """Join matured ledger forecasts to realized market closes and report forecast errors.

    ``forecast_date`` must identify the last market session represented by request
    features. The realized target is ``TARGET_HORIZON`` market rows ahead. Rows whose
    target date has not appeared in the downloaded market history remain pending.
    """
    if not ledger_path.exists(): raise FileNotFoundError(f"Prediction ledger not found: {ledger_path}")
    if not market_path.exists(): raise FileNotFoundError(f"Market cache not found: {market_path}")
    ledger = pd.read_csv(ledger_path)
    market = pd.read_csv(market_path, parse_dates=["Date"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    market_dates = pd.to_datetime(market["Date"]).dt.normalize()
    market_close = pd.to_numeric(market["gold_close"], errors="coerce")
    if "actual_price" not in ledger: ledger["actual_price"] = np.nan
    if "target_date" not in ledger: ledger["target_date"] = ""
    if "absolute_error" not in ledger: ledger["absolute_error"] = np.nan
    if "squared_error" not in ledger: ledger["squared_error"] = np.nan
    if "direction_correct" not in ledger: ledger["direction_correct"] = np.nan
    for i, row in ledger.iterrows():
        if pd.notna(row.get("actual_price")) and str(row.get("actual_price")) not in ("", "nan"):
            continue
        asof = pd.to_datetime(row.get("forecast_date"), errors="coerce")
        if pd.isna(asof): continue
        asof = asof.normalize()
        base_idx = int(market_dates.searchsorted(asof, side="right") - 1)
        target_idx = base_idx + int(TARGET_HORIZON)
        if base_idx < 0 or target_idx >= len(market): continue
        target_price = float(market_close.iloc[target_idx])
        current_price = float(market_close.iloc[base_idx])
        predicted = float(row["predicted_price"])
        ledger.loc[i, "target_date"] = market_dates.iloc[target_idx].date().isoformat()
        ledger.loc[i, "actual_price"] = target_price
        ledger.loc[i, "absolute_error"] = abs(predicted - target_price)
        ledger.loc[i, "squared_error"] = (predicted - target_price) ** 2
        ledger.loc[i, "direction_correct"] = float(np.sign(predicted-current_price) == np.sign(target_price-current_price))
    ledger.to_csv(ledger_path, index=False)
    matured = ledger.dropna(subset=["actual_price", "absolute_error", "squared_error"])
    report = {
        "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "ledger_path": str(ledger_path), "matured_predictions": int(len(matured)),
        "pending_predictions": int(len(ledger)-len(matured)), "target_horizon_sessions": int(TARGET_HORIZON),
        "mae": float(matured["absolute_error"].mean()) if len(matured) else None,
        "rmse": float(np.sqrt(matured["squared_error"].mean())) if len(matured) else None,
        "directional_accuracy": float(matured["direction_correct"].mean()) if len(matured) and matured["direction_correct"].notna().any() else None,
        "interpretation": "Only matured predictions with a supplied forecast_date are scored; refresh market data before evaluating recent forecasts."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
