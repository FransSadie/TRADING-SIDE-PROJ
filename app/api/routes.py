import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import FeatureSnapshot, IngestionRun, MarketLabel, MarketPrice, NewsArticle, NewsSignal
from app.db.session import get_db_session
from app.features.pipeline import run_feature_generation, run_label_generation
from app.ingestion.pipeline import run_all_ingestion
from app.models.inference import predict_for_ticker
from app.models.train_baseline import train_and_save_baseline
from app.nlp.pipeline import run_news_nlp

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest/run")
def ingest_run() -> dict:
    return run_all_ingestion()


@router.get("/ingest/status")
def ingest_status() -> dict:
    session = get_db_session()
    try:
        latest_run = session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)).scalar_one_or_none()
        article_count = session.execute(select(func.count(NewsArticle.id))).scalar_one()
        price_count = session.execute(select(func.count(MarketPrice.id))).scalar_one()
        return {
            "latest_run": {
                "job_name": latest_run.job_name if latest_run else None,
                "status": latest_run.status if latest_run else None,
                "rows_inserted": latest_run.rows_inserted if latest_run else None,
                "started_at": latest_run.started_at.isoformat() if latest_run else None,
            },
            "totals": {"news_articles": article_count, "market_prices": price_count},
        }
    finally:
        session.close()


@router.post("/pipeline/run")
def pipeline_run() -> dict:
    news_signal_count = run_news_nlp()
    feature_count = run_feature_generation(window_hours=24)
    label_count = run_label_generation(horizon_days=1)
    return {
        "news_signals_inserted": news_signal_count,
        "feature_rows_inserted": feature_count,
        "labels_inserted": label_count,
    }


@router.get("/pipeline/status")
def pipeline_status() -> dict:
    session = get_db_session()
    try:
        signal_count = session.execute(select(func.count(NewsSignal.id))).scalar_one()
        feature_count = session.execute(select(func.count(FeatureSnapshot.id))).scalar_one()
        label_count = session.execute(select(func.count(MarketLabel.id))).scalar_one()
        latest_run = session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)).scalar_one_or_none()
        return {
            "latest_run": {
                "job_name": latest_run.job_name if latest_run else None,
                "status": latest_run.status if latest_run else None,
                "rows_inserted": latest_run.rows_inserted if latest_run else None,
                "started_at": latest_run.started_at.isoformat() if latest_run else None,
            },
            "totals": {
                "news_signals": signal_count,
                "feature_snapshots": feature_count,
                "market_labels": label_count,
            },
        }
    finally:
        session.close()


@router.post("/model/train")
def model_train() -> dict:
    try:
        return train_and_save_baseline()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model/status")
def model_status() -> dict:
    settings = get_settings()
    metadata_path = Path(settings.model_artifacts_dir) / "baseline_metadata.json"
    if not metadata_path.exists():
        return {"trained": False, "metadata": None}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"trained": True, "metadata": metadata}


@router.get("/predict")
def predict(ticker: str) -> dict:
    try:
        return predict_for_ticker(ticker=ticker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
