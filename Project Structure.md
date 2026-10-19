# Project Structure — Gold Price Prediction



Gold-Price-Prediction/
├── gui/app.py                         # Tkinter desktop workbench
├── config/project\_config.py           # paths, symbols, fold settings, AWS config
├── data/
│   ├── sample\_inference.json            # illustrative endpoint request payload
│   ├── raw/                            # local source snapshots; not committed by default
│   ├── processed/                      # engineered features and news scores
│   └── raw/gc\_session\_calendar\_TEMPLATE.csv
├── src/
│   ├── data/                           # market/news ingestion, validation, time alignment
│   ├── features/                       # market and optional news feature engineering
│   ├── models/                         # model factory, evaluation, governance
│   ├── pipeline/train.py               # local end-to-end research pipeline
│   ├── serving/api.py                  # local FastAPI inference service
│   ├── monitoring/                     # drift checks and prediction ledger
│   └── utils/experiment\_tracking.py    # optional run tracking
├── scripts/
│   ├── fetch\_news.py                   # optional recent news metadata
│   ├── prepare\_training\_data.py        # local market feature preparation
│   ├── train\_local.py                  # local nested walk-forward training
│   ├── audit\_news\_archive.py           # news query-window audit
│   ├── prepare\_sagemaker\_data.py       # chronological cloud train/test CSVs
│   ├── bootstrap\_historical.py         # market history download + optional S3 upload
│   ├── upload\_to\_s3.py                 # explicit data upload helper
│   └── start\_pipeline.py               # upsert/start SageMaker Pipeline
├── lambda/market\_collector/            # scheduled AWS Lambda collector
├── template.yaml                       # AWS SAM S3/Lambda/EventBridge infrastructure
├── sagemaker/                          # preprocessing, training, evaluation, pipeline, inference
├── architectures/                      # editable .gv + rendered .png/.svg diagrams
├── artifacts/                          # run audits, snapshots, comparison CSVs
├── models/                             # candidate/champion model files and manifests
├── tests/                              # local tests
├── docs

│   ├── ARCHITECTURE.gv			# architecture explanation

│   └── ARCHITECTURE.png		# architecture explanation

├── README.md                           # quick start



**Main workflow**

GUI → source ingestion → validation/time alignment → feature engineering → nested chronological evaluation → baseline comparison → run provenance → human approval → local API



**AWS path:**

EventBridge → Lambda → S3 → SageMaker Processing/Training/Evaluation → quality gate → Model Registry (manual approval) → endpoint → CloudWatch/Model Monitor

The GUI is not an AWS console and local actions do not start cloud resources. AWS deployment is an explicit, separately documented phase.



## functional notes

* gui/app.py: GDELT and Yahoo Finance news-provider checkboxes; one or both can be selected without changing the existing tab layout.
* scripts/fetch\_news.py: provider-isolated fetching, per-provider status/error logging, archive append/deduplication, and summary counts.
* src/data/news.py: common news schema, provider timestamp normalization, deduplication, retrieval log and snapshots.
* data/raw/news.csv: normalized article metadata returned by the selected providers.
* data/raw/news\_ingestion\_log.csv: one row per provider attempt, including status, error, rows\_returned\_this\_run, query/timespan and attempted UTC window.
* data/raw/news\_snapshots.jsonl: retrieval-time snapshots for returned article metadata.
* data/processed/news\_lexicon\_scored.csv or news\_finbert\_scored.csv: scored outputs created by Score News.
* docs/AWS\_IRONHACK\_RUNBOOK.md: console-vs-CLI credential distinction, IAM permissions, SAM deployment and SageMaker pipeline walkthrough.



## Code map: what each file does and why it exists



|File|What it does|Why it is needed|
|-|-|-|
|config/project\\\_config.py|Centralizes tickers, paths, target horizon, folds and optional AWS settings.|Prevents hidden settings scattered across scripts and makes runs easier to reproduce.|
|src/data/market.py|Downloads daily market series, retries failed requests, joins dates and caches the result.|Supplies the target and context variables while isolating provider-specific code.|
|src/data/news.py|Calls the GDELT or Yahoo ticker-news adapter, normalizes records, appends the archive, query log and metadata snapshots.|Preserves source provenance and separates data collection from modelling.|
|src/data/validation.py|Enforces required columns, valid timestamps, positive gold prices and reports stale market data.|Stops malformed inputs from silently contaminating training.|
|src/features/news\\\_features.py|Defines the exact fixed lexicons, scores article title/summary text and aggregates daily sentiment.|Makes the baseline transparent and lets a separately scored FinBERT file use the same pipeline.|
|src/features/market\\\_features.py|Builds market lags/rolling features, lags news by one market row, creates the future target and seasonal-naive benchmark column.|Converts raw observations into a supervised dataset and separates predictor columns from evaluation-only columns.|
|src/models/factory.py|Constructs the four configured regressors with deterministic seeds and bounded CPU parallelism.|Gives all experiments a consistent model configuration.|
|src/models/evaluation.py|Runs expanding chronological folds, computes baseline comparisons and a moving-block-bootstrap RMSE interval.|Prevents random-split leakage and checks whether models add value beyond simple forecasts.|
|src/pipeline/train.py|Coordinates ingestion/cache, feature creation, news gate, feature-set ablation, evaluation, audit outputs and candidate creation.|Provides one reproducible orchestration entry point for CLI and GUI.|
|src/models/governance.py|Saves eligible candidates and promotes only after explicit approval.|Keeps evaluation separate from deployment and prevents silent champion replacement.|
|src/serving/api.py|Serves the approved champion through FastAPI and validates feature schema.|Provides a testable prediction interface independent of the GUI.|
|src/monitoring/drift.py|Computes PSI summaries for numeric feature distributions.|Provides an initial signal for distribution changes; PSI alone is not proof of failure.|
|src/monitoring/prediction\\\_ledger.py|Appends API forecasts and scores them after the target close becomes available.|Tracks realized forecast quality instead of relying on drift alone.|
|scripts/evaluate\\\_prediction\\\_ledger.py|Updates matured prediction outcomes and writes a quality report.|Separates forecast issuance from later outcome evaluation.|
|scripts/fetch\\\_news.py|Exposes source, query, timespan/count and output path through CLI.|Allows repeatable ingestion outside the GUI.|
|scripts/audit\\\_news\\\_archive.py|Reports article metadata counts and gaps between logged query windows.|Helps detect gaps without falsely certifying completeness.|
|scripts/prepare\\\_training\\\_data.py|Refreshes market data and writes a prepared feature table/manifest.|Separates data preparation from model fitting.|
|scripts/compare\\\_sentiment.py|Writes lexicon scores or optional FinBERT scores to a separate compatible CSV.|Enables controlled sentiment-method experiments on the same downstream code.|
|scripts/train\\\_local.py|Sets the Windows/joblib CPU cap before importing ML libraries, parses training options and prints a run summary.|Provides a reliable Python 3.11 CLI and avoids the physical-core detection warning where possible.|
|scripts/promote\\\_candidate.py`|Requires an explicit `--approve` flag before promotion.|Adds a deliberate human approval boundary.|
|gui/app.py`|Runs scripts asynchronously, displays logs and metrics, and collects user settings.|Provides a user-friendly desktop workflow without Streamlit.|
|sagemaker/|Contains optional cloud training, evaluation, registration and deployment entry points.|Demonstrates the cloud extension without requiring AWS for local development.|
|tests/|Tests feature timing, target alignment, archive behavior, sentiment override and API contracts.|Catches regressions in the most important data and governance rules.|





## 

