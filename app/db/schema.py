import logging

from sqlalchemy import text

from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)


def _add_column_if_missing(table_name: str, column_name: str, definition: str) -> None:
    statement = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(statement))
            trans.commit()
            logger.info("Added column %s.%s", table_name, column_name)
        except Exception as exc:
            trans.rollback()
            message = str(exc).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise


def ensure_schema() -> None:
    Base.metadata.create_all(bind=engine)

    # Lightweight runtime migration path for existing DBs.
    _add_column_if_missing("news_articles", "source_type", "VARCHAR(64) DEFAULT 'live_api'")
    _add_column_if_missing("news_articles", "external_id", "VARCHAR(255)")
    _add_column_if_missing("news_articles", "import_batch", "VARCHAR(255)")
    _add_column_if_missing("news_articles", "metadata_text", "TEXT")
    _add_column_if_missing("news_articles", "source_hash", "VARCHAR(64)")
    _add_column_if_missing("news_articles", "nlp_source_hash", "VARCHAR(64)")
    _add_column_if_missing("news_articles", "nlp_processed_at", "TIMESTAMP")

    _add_column_if_missing("news_signals", "source_weight", "DOUBLE PRECISION DEFAULT 1.0")
    _add_column_if_missing("news_signals", "event_tags", "VARCHAR(256)")

    _add_column_if_missing("feature_snapshots", "source_weight_mean", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "event_intensity", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "news_count_change_24h", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "sentiment_momentum_24h", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "price_return_5d", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "rolling_volatility_20d", "DOUBLE PRECISION")
    _add_column_if_missing("feature_snapshots", "volume_zscore_20d", "DOUBLE PRECISION")
