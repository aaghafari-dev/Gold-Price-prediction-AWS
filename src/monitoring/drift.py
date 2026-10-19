"""Lightweight production drift checks; can run before/alongside SageMaker monitoring."""
import numpy as np
import pandas as pd

def psi(expected, actual, bins=10):
    """Calculate a Population Stability Index estimate using reference-distribution quantile bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)
    e = np.clip(e / max(1, len(expected)), 1e-6, None)
    a = np.clip(a / max(1, len(actual)), 1e-6, None)
    return float(np.sum((a-e)*np.log(a/e)))

def drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Compute numeric feature PSI values for reference and current datasets with matching columns."""
    common = [c for c in reference.columns if c in current.columns]
    out = {}
    for c in common:
        if pd.api.types.is_numeric_dtype(reference[c]):
            out[c] = psi(reference[c].dropna(), current[c].dropna())
    return out
