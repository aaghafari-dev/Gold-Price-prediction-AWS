"""tests/test_sagemaker_prep.py: project component; see README.md and docs/ for workflow and configuration details."""
import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_sagemaker_data', ROOT / 'scripts' / 'prepare_sagemaker_data.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

def test_cloud_preparation_is_chronological_and_has_required_columns(tmp_path):
    """Test the expected behavior of the corresponding project component."""
    source = tmp_path / 'market.csv'
    dates = pd.date_range('2020-01-01', periods=90, freq='D')
    pd.DataFrame({'Date': dates, 'Close': [1500 + i for i in range(90)], 'Volume': [1000 + i for i in range(90)]}).to_csv(source, index=False)
    frame = MODULE.build_frame(source)
    assert {'Date', 'gold_close', 'target_price', *MODULE.FEATURES}.issubset(frame.columns)
    assert frame['Date'].is_monotonic_increasing
    assert frame['target_price'].iloc[0] == frame['gold_close'].iloc[1]

def test_cloud_preparation_deduplicates_dates_and_drops_invalid_prices(tmp_path):
    """Test the expected behavior of the corresponding project component."""
    source = tmp_path / 'market.csv'
    pd.DataFrame({'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03'], 'Close': [100, 101, -1, 102], 'Volume': [1, 2, 3, 4]}).to_csv(source, index=False)
    frame = MODULE.build_frame(source)
    assert frame['Date'].is_unique
    assert (frame['gold_close'] > 0).all()
