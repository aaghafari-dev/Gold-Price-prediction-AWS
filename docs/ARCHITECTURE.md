# Architecture

## A. Academic / Ironhack architecture

```text
Yahoo Finance
     │
     ├── Gold
     └── Cross-assets
     │
Historical News
     │
     ▼
S3 / Local Raw Data
     │
     ▼
Data Validation
     │
     ▼
Point-in-Time Feature Engineering
     │
     ▼
Walk-Forward Evaluation
     │
     ├── Persistence
     ├── XGBoost
     ├── Random Forest
     ├── Extra Trees
     └── HistGradientBoosting
     │
     ▼
Governance
     │
     ▼
SageMaker Model Registry
     │
     ▼
SageMaker Endpoint
     │
     ▼
FastAPI / Dashboard
     │
     ▼
CloudWatch + Drift Jobs
     │
     ▼
Retraining
```

## B. Commercial architecture

```text
                DATA PROVIDERS
 ┌──────────────────────┬──────────────────────┐
 │ Market / Macro       │ News / Events        │
 │ licensed, PIT        │ licensed, PIT        │
 └──────────┬───────────┴──────────┬───────────┘
            │                      │
            └──────────┬───────────┘
                       ▼
                 S3 DATA LAKE
          raw / bronze / silver / gold
                       │
                       ▼
              DATA QUALITY LAYER
       schema + freshness + validity + lineage
                       │
                       ▼
             FEATURE ENGINEERING
     market + macro + NLP + event + regime
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Offline Store       Online Store
             │                   │
             └─────────┬─────────┘
                       ▼
                MODEL TRAINING
       baselines + ML + ensemble candidates
                       │
                       ▼
             WALK-FORWARD BACKTEST
                       │
                       ▼
                 MODEL RISK GATE
                       │
                       ▼
                MODEL REGISTRY
                       │
                       ▼
             APPROVAL / DEPLOYMENT
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Batch Forecast     Real-time API
              │                 │
              └────────┬────────┘
                       ▼
                MONITORING
        data + drift + quality + latency
                       │
                       ▼
              ALERT / RETRAINING
                       │
                       └───────────────►
```

## C. Recommended prediction design

Use three outputs:

1. point estimate,
2. prediction interval,
3. direction/probability.

Example:

```text
Next-day gold price:
$2,850/oz

80% interval:
$2,805–$2,900

Probability of positive return:
61%
```


## D. News design

```text
Article
  │
  ├── timestamp
  ├── source
  ├── title/body
  ├── entities
  ├── topics
  └── event type
        │
        ▼
     NLP model
        │
        ├── sentiment
        ├── relevance
        ├── event probability
        └── novelty
        │
        ▼
Recency-weighted event features
        │
        ▼
Forecast feature vector
```

```

## F. Failure handling

### Market data unavailable

Use last validated dataset only for monitoring; do not silently fabricate new
observations.

### News provider unavailable

Mark news features as unavailable and either:

- fail closed, or
- route to a model version explicitly trained without news.

Do not silently replace missing news with zero if the production model depends on
news.

### Model unavailable

Return an explicit service-unavailable state rather than silently claiming a
prediction from an unknown model.

### Drift detected

Do not immediately deploy a new model. Trigger a challenger workflow.

