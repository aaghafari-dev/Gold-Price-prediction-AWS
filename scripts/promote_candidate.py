"""Explicitly approve the latest candidate and copy it to the champion registry."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.governance import approve_saved_candidate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--approve", action="store_true", help="Required explicit human approval to promote candidate")
args = parser.parse_args()
if not args.approve:
    parser.error("Promotion is gated. Review models/candidate_manifest.json, then rerun with --approve.")
print(json.dumps(approve_saved_candidate(), indent=2, default=str))
