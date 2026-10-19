# Project Structure

This document describes the main files and directories in the Gold Price Prediction Workbench. Runtime data, logs, trained model binaries, and generated artifacts are intentionally excluded from source control by default; the application creates them locally when needed.


Gold-Price-Prediction-Workbench/
├── .github/
│   └── workflows/
│       └── ci.yml                       # Automated checks in GitHub Actions
├── config/
│   └── project_config.py                # Central paths and environment configuration
├── data/
│   ├── raw/                             # Downloaded market/news data (local, ignored)
│   ├── processed/                       # Cleaned data and engineered features (ignored)
│   └── sample_inference.json             # Example inference payload
├── docs/
│   ├── ARCHITECTURE.md                  # System architecture notes
│   ├── ARCHITECTURE.gv            # AWS deployment and operations guide
|
├── gui/
│   └── app.py                           # Main Tkinter desktop application
├── lambda/
│   └── market_collector/
│       ├── app.py                       # AWS Lambda market-data collector
│       └── requirements.txt             # Lambda-specific deployment dependencies
├── logs/                                # Runtime logs (ignored)
├── models/
│   ├── .gitkeep                         # Keeps the empty directory in Git
│   └── *.joblib / *manifest.json        # Local model files (ignored)
├── presentation/
│   ├── Gold_Price_Prediction_Presentation.pptx         # Presentation materials
│                           
├── scripts/
│   ├── fetch_news.py                    # Collect news from configured providers
│   ├── audit_news_archive.py            # Inspect news archive quality and coverage
│   ├── compare_sentiment.py             # Optional sentiment scoring/comparison
│   ├── live_news_monitor.py             # Periodic headline/sentiment snapshots
│   ├── prepare_training_data.py         # Prepare market data and model features
│   ├── train_local.py                   # Local model training/evaluation entry point
│   ├── predict.py                       # Prediction utility
│   ├── predict_next_session.py          # Next-session forecast entry point
│   ├── promote_candidate.py              # Candidate review/promotion workflow
│   ├── evaluate_prediction_ledger.py    # Evaluate recorded predictions
│   ├── prepare_sagemaker_data.py        # Prepare data for cloud ML
│   ├── upload_to_s3.py                  # Upload selected files to S3
│   ├── bootstrap_historical.py          # Historical data bootstrap helper
│   ├── start_pipeline.py                # Start a SageMaker pipeline execution
│   ├── deploy_approved_model.py         # Optional deployment helper
│   ├── invoke_sagemaker_endpoint.py     # Invoke a configured endpoint
│   ├── create_cloudwatch_alarm.py       # Create optional CloudWatch alarm
│   └── aws_preflight.ps1                # AWS environment preflight checks
├── sagemaker/
│   ├── preprocess.py                   # Cloud-side preprocessing
│   ├── train.py                         # SageMaker training entry point
│   ├── evaluate.py                      # Model evaluation and quality checks
│   ├── register.py                      # Register eligible model artifacts
│   ├── pipeline.py                      # SageMaker pipeline definition
│   ├── submit_training.py               # Standalone training submission helper
│   ├── inference.py                     # Model inference handler
│   ├── deploy.py                        # Optional endpoint deployment helper
│   └── requirements.txt                 # SageMaker SDK/runtime dependencies
├── src/
│   ├── data/
│   │   ├── market.py                    # Market data retrieval and normalization
│   │   ├── news.py                      # News provider and archive logic
│   │   ├── time_alignment.py            # Timestamp alignment and leakage safeguards
│   │   └── validation.py                # Data validation helpers
│   ├── features/
│   │   ├── market_features.py           # Market-derived features and targets
│   │   └── news_features.py             # News/sentiment aggregation features
│   ├── models/
│   │   ├── factory.py                   # Candidate model construction
│   │   ├── evaluation.py                # Metrics, baselines and validation
│   │   └── governance.py                # Candidate/champion approval logic
│   ├── pipeline/
│   │   └── train.py                     # Reusable training pipeline
│   ├── serving/
│   │   └── api.py                       # Optional FastAPI prediction service
│   ├── monitoring/
│   │   ├── drift.py                     # Monitoring and drift utilities
│   │   └── prediction_ledger.py         # Prediction history and evaluation
│   └── utils/
│       └── experiment_tracking.py       # Experiment metadata helpers
├── tests/
│   ├── test_features.py                 # Feature engineering tests
│   ├── test_news_ingestion.py           # News ingestion tests
│   ├── test_time_alignment.py           # Time alignment tests
│   ├── test_api.py                      # API request/response tests
│   └── test_sagemaker_prep.py           # SageMaker input preparation tests
├── artifacts/                           # Reports, metrics, forecasts (ignored)
├── .env.example                         # Safe configuration template; no secrets
├── .gitignore                           # Excludes secrets and generated files
|
├── Dockerfile                           # Container build definition
├── Project Structure.md                 # This document
├── README.md                             # Project overview and quick start
├── requirements.txt                     # Main Python dependencies
├── requirements-aws.txt                 # AWS-related dependencies
├── requirements-mlflow.txt              # Optional experiment tracking dependencies
├── requirements-nlp.txt                 # Optional NLP/sentiment dependencies
└── template.yaml                         # AWS SAM/CloudFormation infrastructure template



## Main responsibilities

| Component | Responsibility |
|---|---|
| `gui/app.py` | Desktop user interface, workflow controls, Forecast Center, AWS Control Center, and live console output |
| `config/project_config.py` | Central configuration, paths, model symbols, and optional AWS settings |
| `src/data/` | Market/news ingestion, normalization, validation, and time alignment |
| `src/features/` | Market and news feature engineering |
| `src/models/` | Model construction, evaluation, baselines, and model governance |
| `src/pipeline/` | Reusable local training workflow |
| `src/serving/` | Optional FastAPI inference interface; not required for the desktop GUI |
| `src/monitoring/` | Prediction ledger and monitoring utilities |
| `scripts/` | Runnable workflows for data collection, training, forecasting, AWS operations, and diagnostics |
| `lambda/market_collector/` | Scheduled AWS market-data ingestion function |
| `sagemaker/` | Cloud preprocessing, training, evaluation, registration, pipeline, and optional deployment code |
| `template.yaml` | Infrastructure resources managed by AWS SAM/CloudFormation |
| `tests/` | Automated tests for core data, feature, API, and cloud-preparation behavior |
| `docs/` and `architectures/` | Operational guides, design rationale, and diagrams |

## Generated files and source control

The following locations are used at runtime and are normally excluded from Git:

- `data/raw/` — downloaded market data and news archives
- `data/processed/` — normalized datasets and engineered features
- `models/` — serialized model files and champion/candidate manifests
- `artifacts/` — evaluation metrics, data-quality reports, manifests, and forecasts
- `logs/` — runtime and ingestion logs
- `.env` — private environment settings and credentials

The `.gitkeep` files preserve empty directories in the repository. Do not commit API keys, AWS credentials, private data, or generated model binaries unless there is a deliberate, reviewed reason to publish them.

## Execution boundaries

- **Local workflow:** runs ingestion, feature preparation, model training/evaluation, and desktop forecasting on the local computer.
- **AWS foundation:** `template.yaml` provisions the S3 bucket, IAM role, Lambda collector, and scheduled EventBridge trigger.
- **SageMaker workflow:** code under `sagemaker/` and the associated scripts can run cloud-side processing, training, evaluation, and registration after the required role, S3 inputs, and settings are supplied.
- **Optional serving:** `src/serving/api.py` provides a separate FastAPI interface. It is not required for the desktop GUI's Forecast Center.

The presence of a script or infrastructure definition does not mean that its cloud resources have already been deployed or that an endpoint is currently running. Confirm execution status in AWS and review the relevant logs before treating a cloud step as complete.
