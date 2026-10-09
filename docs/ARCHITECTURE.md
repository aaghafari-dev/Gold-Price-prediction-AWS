# Architecture

## &#x20;                MARKET DATA

## &#x20;     ┌────────────────────────────┐

## &#x20;     │ Gold                       			│

## &#x20;     │ DXY                       			│

## &#x20;     │ Silver                     			│

## &#x20;     │ Oil                        			│

## &#x20;     │ S\&P 500                    			│

## &#x20;     │ VIX                        			│

## &#x20;     │ US 10Y                     			│

## &#x20;     │ EUR/USD                    			│

## &#x20;     │ USD/JPY                    			│

## &#x20;     └─────────────┬──────────────┘

## &#x20;                   	│

## &#x20;                   	▼

## &#x20;            ┌─────────────┐

## &#x20;            │     S3          │

## &#x20;            │  Data Lake      │

## &#x20;            └──────┬──────┘

## &#x20;                     │

## &#x20;                     │

## &#x20;     ┌───────────▼──────────────┐

## &#x20;     │      NEWS DATA             		  │

## &#x20;     │                                  │

## &#x20;     │ Historical articles              │

## &#x20;     │ Publication timestamp            │

## &#x20;     │ Source                           │

## &#x20;     │ Gold relevance                   │

## &#x20;     │ Sentiment                        │

## &#x20;     │ Events                           │

## &#x20;     └─────────────┬────────────┘

## &#x20;                       │

## &#x20;                       ▼

## &#x20;      ┌─────────────────────────────┐

## &#x20;      │ Data Quality + Point-in-Time        │

## &#x20;      │ Alignment                           │

## &#x20;      └──────────────┬──────────────┘

## &#x20;                         │

## &#x20;                         ▼

## &#x20;      ┌─────────────────────────────┐

## &#x20;      │ Feature Engineering                 │

## &#x20;      │                                     │

## &#x20;      │ Market                              │

## &#x20;      │ Macro                               │

## &#x20;      │ News sentiment                      │

## &#x20;      │ News events                         │

## &#x20;      │ Volatility                          │

## &#x20;      │ Momentum                            │

## &#x20;      └──────────────┬──────────────┘

## &#x20;                         │

## &#x20;                         ▼

## &#x20;         ┌──────────────────────┐

## &#x20;         │ Walk-Forward Testing       │

## &#x20;         └──────────┬───────────┘

## &#x20;                       │

## &#x20;       ┌───────────┼────────────┐

## &#x20;       ▼              ▼               ▼

## &#x20;    XGBoost           RF/ET       Baselines

## &#x20;       │               │               │

## &#x20;       └────────────┼────────────┘

## &#x20;                       ▼

## &#x20;            Model Governance

## &#x20;                      │

## &#x20;            ┌───────┴────────┐

## &#x20;            │                    │

## &#x20;         REJECT               ACCEPT

## &#x20;            │                    │

## &#x20;            │                    ▼

## &#x20;            │              SageMaker Model

## &#x20;            │                Registry

## &#x20;            │                    │

## &#x20;            │                    ▼

## &#x20;            │              Approval Gate

## &#x20;            │                    │

## &#x20;            │                    ▼

## &#x20;            │             SageMaker Endpoint

## &#x20;            │                    │

## &#x20;            │                    ▼

## &#x20;            │             FastAPI / Client

## &#x20;            │                    │

## &#x20;            └───────┐          ▼

## &#x20;                      │   Monitoring

## &#x20;                      │        │

## &#x20;                      │        ▼

## &#x20;                      │   Drift / Error

## &#x20;                      │        │

## &#x20;                      │        ▼

## &#x20;                      └── Retraining

