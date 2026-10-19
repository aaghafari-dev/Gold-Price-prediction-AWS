"""SageMaker Processing evaluation step: compare candidate with persistence baseline."""
import argparse, json, tarfile, subprocess, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--test-data', default='/opt/ml/processing/test/test.csv')
parser.add_argument('--model-artifact', default='/opt/ml/processing/model/model.tar.gz')
parser.add_argument('--output-dir', default='/opt/ml/processing/evaluation')
args = parser.parse_args()
requirements_file = Path(__file__).resolve().parent / 'requirements.txt'
if requirements_file.exists():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
model_archive = Path(args.model_artifact)
model_dir = model_archive.parent / 'extracted_model'
model_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open(model_archive, 'r:gz') as archive:
    root = model_dir.resolve()
    for member in archive.getmembers():
        destination = (model_dir / member.name).resolve()
        if destination != root and root not in destination.parents:
            raise ValueError(f'Unsafe model archive member: {member.name}')
    archive.extractall(model_dir)
manifest_path = model_dir / 'manifest.json'
model_path = model_dir / 'model.joblib'
if not manifest_path.exists() or not model_path.exists():
    raise FileNotFoundError('Model archive must contain model.joblib and manifest.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
model = joblib.load(model_path)
test = pd.read_csv(args.test_data)
features = manifest['features']
missing = [col for col in features if col not in test.columns]
if missing:
    raise ValueError(f'Test dataset is missing model features: {missing}')
if 'target_price' not in test or 'gold_close' not in test:
    raise ValueError('Test dataset must include target_price and gold_close for evaluation')
X = test[features].astype(float)
y = test['target_price'].to_numpy(dtype=float)
baseline = test['gold_close'].to_numpy(dtype=float)
pred = np.asarray(model.predict(X), dtype=float)
rmse = float(np.sqrt(mean_squared_error(y, pred)))
mae = float(mean_absolute_error(y, pred))
base_rmse = float(np.sqrt(mean_squared_error(y, baseline)))
improvement = float((base_rmse - rmse) / base_rmse * 100) if base_rmse else 0.0
report = {
    'model_name': manifest['model_name'], 'test_rows': int(len(test)),
    'rmse': rmse, 'mae': mae, 'baseline_rmse': base_rmse,
    'improvement_pct': improvement, 'beats_baseline': bool(rmse < base_rmse),
    'evaluation_gate_passed': bool(rmse < base_rmse and improvement >= 0.0),
    'test_start': str(test['Date'].min()) if 'Date' in test else None,
    'test_end': str(test['Date'].max()) if 'Date' in test else None,
    'limitations': 'Single chronological holdout is a gate, not evidence of trading profitability.'
}
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
(out / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
