# Market News Bot (Ingestion + Baseline Model)

This project starts with a data-ingestion foundation for:
- News ingestion from News API
- Market price ingestion from Yahoo Finance
- Storage of raw records for downstream NLP/features/modeling
- NLP signal extraction + feature snapshots + labels for baseline modeling
- Baseline model training + prediction endpoint

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `NEWS_API_KEY` in `.env`.

## 2) Run API + scheduler

```bash
uvicorn app.main:app --reload
```

PowerShell helper:

```powershell
.\scripts\start_api.ps1
```

On startup, tables are created and the scheduler starts.

## 3) Trigger ingestion manually

```bash
python -m app.ingestion.run_once
```

Or test API endpoints (run in a second terminal while API is running):

```powershell
.\scripts\smoke_api.ps1
```

## 4) Health check

.\.venv\Scripts\python.exe -m app.ingestion.run_once
.\.venv\Scripts\python.exe -m app.features.run_once


## 5) Run NLP + feature + label pipeline

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

## 6) Train baseline model

```bash
python -m app.models.train_baseline
```

PowerShell helper:

```powershell
.\scripts\run_model_training.ps1
```

Artifacts are saved in `./artifacts` by default:
- `baseline_model.joblib`
- `baseline_metadata.json`

Note: if training data has only one class, training falls back to a `DummyClassifier` until more varied data is collected.

API options:

- `POST /model/train`
- `GET /model/status`
- `GET /predict?ticker=NVDA`

## Data model

- `news_articles`: raw article records + extracted tickers
- `market_prices`: OHLCV snapshots per ticker/interval
- `ingestion_runs`: job-level audit trail
- `news_signals`: per article/ticker sentiment + relevance rows
- `feature_snapshots`: rolling aggregates per ticker/window
- `market_labels`: horizon-based direction labels from future returns

## Next step after ingestion

Improve model quality (more tickers/features, better NLP, larger history, walk-forward backtests).
