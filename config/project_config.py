"""Central configuration for market tickers, news source parameters, paths, training folds, model registry paths and optional AWS resources."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

for p in (RAW, PROCESSED, ARTIFACTS, MODELS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

APP_VERSION = os.getenv("APP_VERSION", "14.0.0")
GOLD_SYMBOL = os.getenv("GOLD_SYMBOL", "GC=F")
MARKET_SYMBOLS = {
    "gold": GOLD_SYMBOL,
    "dxy": os.getenv("DXY_SYMBOL", "DX-Y.NYB"),
    "silver": os.getenv("SILVER_SYMBOL", "SI=F"),
    "oil": os.getenv("OIL_SYMBOL", "CL=F"),
    "sp500": os.getenv("SP500_SYMBOL", "^GSPC"),
    "vix": os.getenv("VIX_SYMBOL", "^VIX"),
    "us10y": os.getenv("US10Y_SYMBOL", "^TNX"),
    "eurusd": os.getenv("EURUSD_SYMBOL", "EURUSD=X"),
    "usdjpy": os.getenv("USDJPY_SYMBOL", "JPY=X"),
}
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "10"))
TARGET_HORIZON = int(os.getenv("TARGET_HORIZON", "1"))
TEST_FOLDS = int(os.getenv("TEST_FOLDS", "5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# News
NEWS_PROVIDER = os.getenv("NEWS_PROVIDER", "gdelt")
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "30"))
NEWS_MIN_RELEVANCE = float(os.getenv("NEWS_MIN_RELEVANCE", "0.0"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_QUERY = os.getenv(
    "NEWS_QUERY",
    '"gold" OR bullion OR "precious metals" OR "Federal Reserve" OR inflation OR geopolitics'
)

# AWS
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "gold-price-prediction")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "gold-price-predictor")
SAGEMAKER_MODEL_PACKAGE_GROUP = os.getenv(
    "SAGEMAKER_MODEL_PACKAGE_GROUP", "gold-price-prediction"
)

MODEL_NAMES = ["XGBoost", "Random Forest", "Extra Trees", "HistGradientBoosting"]
CHAMPION_MANIFEST = MODELS / "champion_manifest.json"
CHAMPION_MODEL = MODELS / "champion_model.joblib"
FEATURE_MANIFEST = ARTIFACTS / "feature_manifest.json"
