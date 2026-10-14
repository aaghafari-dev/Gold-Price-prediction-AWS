# Gold Price Prediction V8 — Graphviz architecture pack

All diagrams use top-to-bottom layout (`rankdir=TB`). Source `.gv` files are editable and paired with PNG/SVG renders.

- `system_architecture.gv`: joined end-to-end AWS solution
- `data_ingestion.gv`: EventBridge, Lambda A collector, S3 landing, Lambda B normalizer, normalized archive
- `news_pipeline.gv`: Yahoo ticker-news adapter, UTC timestamps, title/summary and sentiment
- `feature_engineering.gv`: SageMaker Processing, quality contracts, leakage checks, time split and feature selection
- `model_training_evaluation.gv`: baseline, walk-forward validation, tuning, metrics and registration gate
- `sagemaker_mlops.gv`: Pipelines, Experiments, Registry, human approval
- `serving_monitoring.gv`: endpoint, CloudWatch, Model Monitor and retraining trigger
- `deployment_architecture.gv`: local development boundary versus cloud production execution

Render with Graphviz on Windows:

```powershell
dot -Tpng architectures\system_architecture.gv -o architectures\system_architecture.png
dot -Tsvg architectures\system_architecture.gv -o architectures\system_architecture.svg
```
