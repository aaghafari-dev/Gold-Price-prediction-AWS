# Gold Price Prediction V7 — Graphviz architectures

All eight `.gv` files use `rankdir=TB` (top-to-bottom). Render on Windows after installing Graphviz:

```powershell
dot -Tpng architectures\system_architecture.gv -o architectures\system_architecture.png
dot -Tsvg architectures\system_architecture.gv -o architectures\system_architecture.svg
```

News is Yahoo Finance ticker-news metadata only via `yfinance.Ticker(...).get_news(...)`; no API key and no scraping. Production processing/training/serving are designed to run on AWS. The local GUI is a developer/operator interface and submits AWS actions using configured CLI/role credentials.
