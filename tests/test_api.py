"""Contract tests for the FastAPI health and prediction endpoints."""
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

def test_health_endpoint_shape():
    """Assert that the API health endpoint returns a structured service status."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_prediction_ledger_scores_matured_forecast(tmp_path):
    """A forecast with a known as-of date is joined to the target close after horizon."""
    import pandas as pd
    from src.monitoring.prediction_ledger import evaluate_matured_predictions
    ledger=tmp_path/"ledger.csv"; market=tmp_path/"market.csv"; report_path=tmp_path/"report.json"
    pd.DataFrame([{"forecast_date":"2026-01-05","predicted_price":102.0,"actual_price":float("nan"),
                   "absolute_error":float("nan"),"squared_error":float("nan"),"direction_correct":float("nan")}]).to_csv(ledger,index=False)
    pd.DataFrame([{"Date":"2026-01-05","gold_close":100.0},{"Date":"2026-01-06","gold_close":103.0},
                  {"Date":"2026-01-07","gold_close":104.0}]).to_csv(market,index=False)
    report=evaluate_matured_predictions(ledger,market,report_path)
    assert report["matured_predictions"] == 1
    assert report["mae"] == 1.0
    assert report_path.exists()
