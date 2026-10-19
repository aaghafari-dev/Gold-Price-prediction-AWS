"""Evaluate predictions in the local ledger after the target market session has matured."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring.prediction_ledger import evaluate_matured_predictions, LEDGER_PATH


def main() -> None:
    """Parse ledger paths, score matured forecasts and print the quality summary."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger",default=str(LEDGER_PATH))
    parser.add_argument("--market",default="data/raw/market_daily.csv")
    parser.add_argument("--out",default="artifacts/prediction_quality_report.json")
    args=parser.parse_args()
    report=evaluate_matured_predictions(Path(args.ledger),Path(args.market),Path(args.out))
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
