# Gold Price Prediction Project

## Highlights

* Google News RSS is the default key-free headline source; optional NewsAPI and Alpha Vantage connectors can be enabled with your own API keys. Legacy GDELT/Yahoo adapters remain available.
* Provider failures are isolated and logged. Empty responses do not silently overwrite an existing archive; if all selected providers fail, the command exits with an error.
* Optional pretrained ProsusAI/finbert sentiment produces positive/negative/neutral probabilities, signed score, confidence and magnitude.
* A five-minute live headline polling monitor writes timestamped sentiment snapshots and rolling-window features.
* GUI exposes optional NewsAPI and Alpha Vantage key fields (masked input) and an explicit save-to-.env action; selected providers with missing keys are blocked before fetch. The ML experiment has a **Use news features** checkbox (market-only remains the default), a nested tuning/feature-selection checkbox, and CPU / GPU-for-XGBoost / auto CPU fallback options. Unverified recent news may be analyzed but cannot be promoted.
* One shared analysis-window control (1d through 30y) is passed to both market and news ingestion, so the requested time spans match. This does not guarantee that a news provider can return historical articles for the entire span. One-day windows are for ingestion/plot/data-quality checks; supervised model evaluation needs substantially more observations after rolling-feature warm-up.
* GUI setting **Final chronological test fraction** reserves the newest 10–40% of observations for holdout evaluation (default 20%). Development-period model selection uses chronological walk-forward folds.
* Architecture diagrams, a detailed manual, AWS runbook, news pipeline guide and automated tests are included.



## Install and launch (Windows 11 / VS Code / Python 3.11)

powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe gui\\app.py
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-nlp.txt



The first FinBERT run may download model weights and require additional disk space and memory. API keys are saved to .env; do not commit it. For AWS, prefer an AWS CLI named profile or SSO rather than long-lived AWS keys in .env.

## 

## Recommended GUI workflow

1. Select **Google News RSS** (default) and optionally enable API providers whose keys you have configured.
2. Click **Fetch News** and inspect the provider-specific status/errors in the live console.
3. Verify data/raw/news.csv and data/raw/news\_ingestion\_log.csv; run **Audit News Archive**.
4. If using FinBERT, select FinBERT and click **Score News**. The five-minute monitor is optional and can be stopped using **Stop Current Task**.
5. Choose whether to include news in the model experiment. Start with market-only, then compare market-only versus market-plus-news on the same chronological protocol.
6. Click **Prepare Market + Features**.
7. Set the final chronological test fraction (default 0.20) and click **Train \& Evaluate**.
8. Review artifacts/model\_comparison.csv and artifacts/training\_result.json; only approve a candidate after reviewing the metrics and data-quality warnings.
9. Start the local FastAPI only after a valid candidate has been approved.

## 

## CLI examples

Fetch public RSS headlines without API keys:

powershell
.\\.venv\\Scripts\\python.exe scripts\\fetch\_news.py --source google\_news\_rss --query "gold OR bullion OR Federal Reserve OR inflation" --count 100



Fetch from multiple sources independently (API keys are required for the two optional API providers):

powershell
.\\.venv\\Scripts\\python.exe scripts\\fetch\_news.py --source google\_news\_rss,newsapi,alphavantage --query "gold OR Federal Reserve"



Score with FinBERT:

powershell
.\\.venv\\Scripts\\python.exe scripts\\compare\_sentiment.py --news data/raw/news.csv --out data/processed/news\_finbert\_scored.csv --finbert



Train with a 25% chronological holdout:

powershell
.\\.venv\\Scripts\\python.exe scripts\\train\_local.py --test-fraction 0.25



Run automated tests:

powershell
$env:PYTHONPATH = '.'
.\\.venv\\Scripts\\python.exe -m pytest -q



## Data and evaluation limitations

* GC=F is Yahoo Finance's continuous gold-futures proxy, not spot gold or a guaranteed official settlement feed.
* Public headline feeds can be rate-limited, incomplete, delayed, or changed by providers. news\\\_ingestion\\\_log.csv is the diagnostic source of truth for fetch attempts.
* Live-feed data cannot be treated as a historical point-in-time archive. Recent headlines joined to many years of prices do not create a valid historical news backtest.
* FinBERT sentiment is an input feature, not a guarantee of directional relevance to gold. Evaluate market-only and market-plus-news models on identical chronological splits.
* AWS deployment is optional, not performed automatically, and may incur charges. Follow docs/AWS\_IRONHACK\_RUNBOOK.md and delete endpoints when finished.

## 

## Project files

See the  Project Structure.md file for a complete description of the project structure.

This is an educational research project, not financial advice or evidence of a profitable trading strategy.



# GUI reference and end-to-end operating guide

This section is intentionally written so that a teammate who did not build the app can understand each visible control. The desktop app is a **local research workbench**; pressing a local pipeline button does not automatically deploy anything to AWS.

## 

## A. Data Settings controls

* **News search query** — terms sent to the selected news provider(s). Provider query syntax and coverage differ; a broad query can retrieve irrelevant stories, while a narrow query can miss important stories.
* **Shared analysis window — news + market** — the requested lookback passed to both market and news ingestion (1d through 30y). This aligns requested windows, not actual coverage: a free RSS provider may return only recent headlines even when 5y is selected. Use 1d–90d mainly for ingestion, plotting, and data-quality checks. Use 6mo or longer for ML; 1y–5y is a sensible starting range.
* **Minimum logged news-window overlap (0–1)** — minimum fraction of supervised market dates that fall within at least one *logged query window*. 0.80 means 80% of the training dates overlap a logged query interval. It does **not** mean that 80% of relevant articles were captured or that the archive is point-in-time complete. Keep news disabled for the first market-only baseline experiment.
* **Final chronological test fraction (0.10–0.40)** — fraction of the newest supervised rows reserved for a final holdout. `0.20` means the newest 20% is held back. It should be evaluated once after model selection, not repeatedly used to tune the model.
* **Google News RSS** — public RSS-based headline retrieval; no API key, but historical coverage and completeness are not guaranteed.
* **NewsAPI (API key)** — use only if you have a valid key and have checked the provider's plan/limits. Missing keys should be fixed before selecting this provider.
* **Alpha Vantage (API key)** — optional provider that requires a valid key and is subject to provider-specific limits.
* **Legacy GDELT / Legacy Yahoo** — compatibility adapters. They may return empty results, rate-limit, or change their response formats; review the Live Run Console.
* **NewsAPI key / Alpha Vantage key** — masked input fields. **Save API keys to .env** stores the entered values locally. Keep `.env` private and never commit or send it to colleagues.
* **Sentiment file: lexicon / FinBERT** — choose the lightweight keyword/lexicon sentiment method or the optional pretrained FinBERT model. FinBERT may need requirements-nlp.txt, a first-run model download, extra disk space and memory. Sentiment is a feature, not proof of price impact.
* **Score News** — creates a scored article file for the selected sentiment method. It does not train the gold model.
* **Use news features in ML experiment** — adds available lagged news/sentiment features to the model comparison. First run market-only; then compare market-only with market-plus-news using the same dates and split.
* **Exploratory unverified news (no promotion)** — permits a clearly exploratory market-plus-recent-news run when a verified historical archive is unavailable. Such a run is not a valid historical news backtest and must not be promoted as if its historical coverage were proven.
* **Nested time-series tuning + feature selection (slower)** — performs parameter search/feature selection inside chronological training folds. It is slower but helps keep feature selection inside training data. Turn it off for a smoke test, then enable it for the final documented experiment.
* **3-seed sensitivity check (slower)** — repeats fits using three fixed random seeds to see whether results depend on estimator randomness. It does not create three independent datasets, improve the data, or replace walk-forward validation / the final holdout.
* **Compute: CPU (recommended)** — run supported estimators on CPU; safest option for reproducibility.
* **GPU + CPU (XGBoost GPU only)** — requests GPU for XGBoost only when the installed XGBoost build and hardware support it. Other estimators still run on CPU. If the console reports no compatible GPU, choose CPU.
* **Auto (CPU fallback)** — allow the configured automatic mode with CPU fallback. Inspect the console to see what was actually used.

## 

## B. Pipeline buttons

* **Fetch News** — queries selected provider(s), saves returned records and a provider/query-window log. A successful command can still return zero articles; inspect the row count and provider messages.
* **Prepare Market + Features** — fetches/refreshes market history for the shared window, validates dates/prices, creates market and lagged news features, constructs the next-session target, and writes the processed training table plus artifacts/data\\\_quality\\\_report.json. A short period can legitimately yield zero supervised rows after rolling-feature warm-up and target construction. Exit code 0 from preparation means the preparation command completed; it does not mean that enough rows exist for ML.
* **Train \& Evaluate** — runs chronological walk-forward validation, optional nested tuning, baseline comparisons, and a final chronological holdout. The GUI now warns before attempting training with very short windows. For an ML experiment, choose 6mo or longer, prepare data, inspect the audit, then train.
* **Approve Candidate** — asks for human confirmation and invokes the local candidate promotion process. Review artifacts/model\\\_comparison.csv and models/candidate\\\_manifest.json first. Approval is not the same as SageMaker Model Registry registration.
* **Start Local API** — starts Uvicorn/FastAPI bound to `127.0.0.1:8000`. It serves only on the local machine, not the public internet. Keep the console open while using the API.
* **Open API Docs** — opens Swagger UI at http://127.0.0.1:8000/docs. Start the API first. This page lets you inspect endpoints and send manual test requests; it does not train a model or deploy to AWS.
* **Refresh Results** — reloads the saved run summary and comparison table from artifacts.
* **Audit News Archive** — checks the archive and logged query windows for coverage/quality problems. It cannot prove that a news provider returned every relevant historical article.
* **Start Live News Monitor** — polls a selected provider periodically and writes near-live headline/sentiment snapshots. It is not a validated intraday gold forecasting pipeline.
* **Run Tests** — runs the local automated test suite using the Python interpreter that launched the GUI.
* **Plot Market Data** — opens a local market-history plot; hover near a plotted point to see its date and price.
* **AWS: Check Identity** — runs AWS STS caller-identity lookup. It identifies the configured account/role and checks whether the credentials are accepted; it does not create resources.
* **AWS: Preflight** — runs preliminary PowerShell checks and displays results in the console. A passing preflight does not mean the SAM template is valid or deployed.
* **SAM Validate** — runs sam validate --lint --template-file template.yaml; local template/lint check only, no deployment.
* **SAM Build** — runs sam build --template-file template.yaml to prepare local build artifacts, commonly in .aws-sam/build; it does not deploy resources.
* **AWS Runbook** — See part E
* **Open Project Folder** — opens the project root in Windows Explorer.
* **Stop Current Task** — requests termination of the current subprocess. Check the console to determine whether it stopped cleanly and whether partial files were written.

## 

## C. Model Results tab

* **Saved run / Load selected run** — select and load a prior recorded experiment, where available, rather than assuming the table always represents the most recent run.
* **Run summary** — shows run ID, row counts, selection status/reason, news experiment status, query-window overlap, and latest market date.
* **Model / Features** — estimator and whether it used market-only or market-plus-news features.
* **RMSE / MAE** — price-error metrics; lower is better. Compare only results evaluated on the same dates and target definition.
* **RMSE CI low/high** — lower/upper limits of the reported uncertainty interval. It is not a guarantee of future performance.
* **Holdout RMSE / MAE** — errors on the final newest chronological block. This is the closest available estimate of out-of-sample performance in this experiment; it is not a guarantee of live performance.
* **Persistence RMSE** — error from predicting the next close as the current close.
* **Seasonal RMSE** — error from the configured seasonal-naive baseline (the prior five-session reference in this project).
* **vs strongest %** — percentage improvement against the strongest baseline (the lower-error benchmark). Positive is better; negative means worse than that benchmark.
* **Directional Acc.** — fraction of observations where the predicted price change has the same sign as the actual price change. It does not measure profitability after transaction costs.
* **Beats persistence / Beats seasonal / Beats both baselines** — true only when the model RMSE beats the respective benchmark on the defined evaluation. `False` is a legitimate result. Never force a pass by weakening or changing the benchmark after seeing the result.
* **Open artifacts folder** — opens the local `artifacts` directory, where CSV/JSON reports and run records are stored.

## 

## E. AWS workflow 

The AWS steps are deliberately separate from local ML training. Run them in order and stop when a step fails.

1. **Install/verify tools.** In a new VS Code PowerShell terminal run aws --version, sam --version, where.exe aws, and where.exe sam. If the commands work in the terminal but not in the GUI, close all VS Code windows and reopen VS Code so the GUI inherits the current PATH.

2. **Set up credentials through the course-approved method.** Prefer a named AWS CLI profile or AWS SSO/Identity Center if your instructor supports it. For a profile, use the profile name your instructor specifies; aws configure sso is one common setup flow. Do not paste credentials into source code or share .env. Static environment credentials can override profile credentials, so remove stale AWS\\\_ACCESS\\\_KEY\\\_ID / AWS\\\_SECRET\\\_ACCESS\\\_KEY values from `.env` when switching to profile authentication.
3. **Click AWS: Check Identity.** The underlying command is aws sts get-caller-identity. Confirm the account ID and ARN are the expected training account. InvalidClientTokenId indicates the credentials being sent are invalid/expired/mismatched; it is not a SAM-template error.
4. **Click AWS: Preflight.** Read every line in the Live Run Console. Preflight checks prerequisites/configuration only; it does not prove deployment readiness in every respect and creates no stack by itself.
5. **Click SAM Validate.** It checks template.yaml locally with linting. Fix errors before continuing.
6. **Click SAM Build.** It builds the application locally. Inspect .aws-sam/build and the console. It still does not create cloud resources.
7. **Review the template before deployment.** Inspect template.yaml, docs/AWS\\\_IRONHACK\\\_RUNBOOK.md, region, permissions and resource types. In the current SAM template the core serverless flow is a private S3 bucket plus a scheduled Lambda market collector, an IAM execution role, EventBridge schedule and CloudWatch logs. The SAM template does **not** by itself deploy the full SageMaker training/registry/real-time endpoint/API Gateway workflow. SageMaker helper scripts are a separate optional path and need their own configuration.
8. **Review cost and permissions.** Confirm the account and region; use the least-privilege permissions your course requires; set a budget alert. S3 storage/requests, Lambda, logs, and any separate SageMaker endpoint/training jobs can incur charges.
9. **Deploy only after instructor/account review.** The usual interactive command is sam deploy --guided. It creates/updates a CloudFormation stack. Carefully review stack name, region, IAM capabilities and prompts. Never assume that validation/build is deployment.
10. **Verify.** Check CloudFormation stack status, Lambda function, EventBridge schedule, S3 bucket and CloudWatch logs. Invoke/test the Lambda only as described in the runbook and verify the expected output object.
11. **Clean up.** Remove the stack when finished and verify that no billable endpoints, training jobs, buckets/objects, log groups, or other separately created resources remain. Some S3 buckets must be emptied before stack deletion.

### 

## 

