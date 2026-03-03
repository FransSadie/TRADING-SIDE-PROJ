# Market News Bot (Ingestion + Baseline Model)

This project starts with a data-ingestion foundation for:
- News ingestion from News API
- Market price ingestion from Yahoo Finance
- Storage of raw records for downstream NLP/features/modeling
- NLP signal extraction + feature snapshots + labels for baseline modeling
- Baseline model training + prediction endpoint
- Data quality checks + walk-forward/baseline evaluation metadata
- React + Vite + Tailwind operator dashboard

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `NEWS_API_KEY` in `.env`.

Recommended coverage settings in `.env` for better model training:
- `MARKET_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,SPY,QQQ,IWM,DIA`
- `MARKET_HISTORY_PERIOD=1y`
- `MARKET_HISTORY_INTERVAL=1d`
- `NEWS_PAGE_SIZE=100`
- `NEWS_MAX_PAGES=3`
- `NEWS_LOOKBACK_DAYS=7`
- `PREDICT_HOLD_THRESHOLD=0.6`

## 2) Run API + scheduler

```bash
uvicorn app.main:app --reload
```

PowerShell helper:

```powershell
.\scripts\start_api.ps1
```

On startup, tables are created and the scheduler starts.

## 3) Run frontend dashboard (React + Vite + Tailwind)

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://127.0.0.1:5173`

The dashboard calls the API through Vite proxy and exposes:
- pipeline control buttons
- model training/predict
- data quality/status views
- prediction log table
- embedded `README.md` and `PROJECT_OVERVIEW.txt` text

## 4) Trigger ingestion manually

```bash
python -m app.ingestion.run_once
```

Or test API endpoints (run in a second terminal while API is running):

```powershell
.\scripts\smoke_api.ps1
```

## 5) Health check

`GET http://127.0.0.1:8000/health`


## 6) Run NLP + feature + label pipeline

```bash
python -m app.features.run_once
```

PowerShell helper:

```powershell
.\scripts\run_feature_pipeline.ps1
```

API option (while server is running):

- `POST /pipeline/run`
- `GET /pipeline/status`
- `GET /data/status`
- `GET /data/quality`

For larger historical backfills, run:

```bash
python -m app.ingestion.run_once
python -m app.features.run_once
```

## 7) Train baseline model

```bash
python -m app.models.train_baseline
```

PowerShell helper:

```powershell
.\scripts\run_model_training.ps1
```

Artifacts are saved in `./artifacts` by default:
- `baseline_model.joblib` (latest pointer)
- `baseline_metadata.json` (latest pointer)
- `baseline_model_<version>.joblib` (versioned)
- `baseline_metadata_<version>.json` (versioned)
- `current_model.json` (active version manifest)

Note: if training data has only one class, training falls back to a `DummyClassifier` until more varied data is collected.

API options:

- `POST /model/train`
- `GET /model/status`
- `GET /predict?ticker=NVDA`
- `GET /prediction/logs?limit=100`
- `GET /docs/text`
- `POST /run/full?train_model=true`

Prediction behavior:
- If confidence is below `PREDICT_HOLD_THRESHOLD`, endpoint returns `prediction=hold`.
- All API predictions are logged in `prediction_logs`.

## 8) Runbook Commands (PowerShell)

- Full refresh + train:
  - `.\scripts\run_full_refresh.ps1`
- Data status/quality report:
  - `.\scripts\show_data_checks.ps1`
- Local one-off prediction:
  - `.\scripts\predict_ticker.ps1 -Ticker SPY`

## Data model

- `news_articles`: raw article records + extracted tickers
- `market_prices`: OHLCV snapshots per ticker/interval
- `ingestion_runs`: job-level audit trail
- `news_signals`: per article/ticker sentiment + relevance rows
- `feature_snapshots`: rolling aggregates per ticker/window, including price volatility and momentum features
- `market_labels`: horizon-based direction labels from future returns
- `prediction_logs`: API prediction audit history (ticker, prediction, confidence, model version, timestamp)

## Next step after ingestion

Integrate historical non-paid news source (for example GDELT subset) and retrain on larger balanced data.
