"""Create a chronological train/test split for the optional SageMaker workflow."""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', default='data/processed/train.csv')
parser.add_argument('--train-out', default='data/processed/sagemaker_train.csv')
parser.add_argument('--test-out', default='data/processed/sagemaker_test.csv')
parser.add_argument('--test-fraction', type=float, default=0.2)
args = parser.parse_args()
if not 0.05 <= args.test_fraction <= 0.4:
    parser.error('--test-fraction must be between 0.05 and 0.4')
frame = pd.read_csv(args.input, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
if len(frame) < 100:
    raise SystemExit('At least 100 engineered rows are required.')
split_at = int(len(frame) * (1 - args.test_fraction))
train, test = frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()
for target in (args.train_out, args.test_out):
    Path(target).parent.mkdir(parents=True, exist_ok=True)
train.to_csv(args.train_out, index=False)
test.to_csv(args.test_out, index=False)
print(f'Chronological split: train={len(train)} rows ({train.Date.min()} to {train.Date.max()}); test={len(test)} rows ({test.Date.min()} to {test.Date.max()})')
print('The test period is later than the training period; do not tune repeatedly on this final holdout.')
