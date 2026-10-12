# Project Structure


Gold-Price-Prediction-Professional/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── config/
│   ├── \_\_init\_\_.py
│   └── project\_config.py
│
├── data/
│   ├── raw/
│   │   ├── market\_daily.csv
│   │   └── news.csv
│   └── processed/
│       └── train.csv
│
├── models/
│   ├── champion\_model.joblib
│   └── champion\_manifest.json
│
├── artifacts/
│   ├── feature\_manifest.json
│   └── training\_result.json
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── COMMERCIAL\_IMPROVEMENTS.md
│   └── PROJECT\_STEPS.md
│
├── scripts/
│   ├── fetch\_news.py
│   ├── prepare\_training\_data.py
│   ├── predict.py
│   ├── train\_local.py
│   └── upload\_to\_s3.py
│
├── sagemaker/
│   ├── train.py
│   ├── inference.py
│   ├── requirements.txt
│   ├── pipeline.py
│   ├── submit\_training.py
│   ├── register.py
│   └── deploy.py
│
├── src/
│   ├── data/
│   │   ├── market.py
│   │   └── news.py
│   ├── features/
│   │   ├── market\_features.py
│   │   └── news\_features.py
│   ├── models/
│   │   ├── factory.py
│   │   ├── evaluation.py
│   │   └── governance.py
│   ├── monitoring/
│   │   └── drift.py
│   ├── pipeline/
│   │   └── train.py
│   └── serving/
│       └── api.py
│
├── tests/
│   ├── test\_features.py
│   └── test\_api.py
│
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── Mermaid Diagram.md
└── Project Structure.md

