import logging
import re
from datetime import datetime

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import IngestionRun, NewsArticle, NewsSignal
from app.db.session import get_db_session

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
WORD_PATTERN = re.compile(r"[A-Za-z]+")

POSITIVE_WORDS = {
    "beat",
    "bullish",
    "growth",
    "gain",
    "gains",
    "positive",
    "rally",
    "record",
    "surge",
    "upside",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "drop",
    "downgrade",
    "fall",
    "loss",
    "miss",
    "negative",
    "recession",
    "risk",
}

DEFAULT_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "iphone"),
    "MSFT": ("microsoft", "azure"),
    "NVDA": ("nvidia", "geforce"),
    "SPY": ("s&p", "s&p 500", "sp500", "wall street", "stocks", "stock market", "equities", "market"),
    "QQQ": ("nasdaq 100", "nasdaq", "tech stocks", "big tech"),
}


def _start_run(job_name: str) -> IngestionRun:
    session = get_db_session()
    try:
        run = IngestionRun(job_name=job_name, status="running", rows_inserted=0, started_at=datetime.utcnow())
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    finally:
        session.close()


def _finish_run(run_id: int, status: str, rows_inserted: int, message: str | None = None) -> None:
    session = get_db_session()
    try:
        run = session.get(IngestionRun, run_id)
        if not run:
            return
        run.status = status
        run.rows_inserted = rows_inserted
        run.message = message[:1000] if message else None
        run.completed_at = datetime.utcnow()
        session.commit()
    finally:
        session.close()


def _sentiment_score(text: str) -> float:
    tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    score = (positive - negative) / max(len(tokens), 1)
    return max(min(score * 10, 1.0), -1.0)


def _extract_tickers(article: NewsArticle, universe: set[str], text: str) -> set[str]:
    tickers = set()
    if article.tickers:
        tickers.update([t.strip().upper() for t in article.tickers.split(",") if t.strip()])
    tickers.update([m for m in TICKER_PATTERN.findall(text.upper()) if m in universe])
    lower_text = text.lower()
    for ticker in universe:
        if ticker.lower() in lower_text:
            tickers.add(ticker)
        aliases = DEFAULT_TICKER_ALIASES.get(ticker, ())
        if any(alias in lower_text for alias in aliases):
            tickers.add(ticker)
    return {ticker for ticker in tickers if ticker in universe}


def _relevance_score(text: str, ticker: str) -> float:
    if not text:
        return 0.0
    mentions = text.upper().count(ticker.upper())
    tokens = max(len(WORD_PATTERN.findall(text)), 1)
    return max(min((mentions / tokens) * 50, 1.0), 0.0)


def run_news_nlp() -> int:
    settings = get_settings()
    run = _start_run("news_nlp")
    inserted = 0
    universe = set(settings.market_ticker_list)
    try:
        session = get_db_session()
        try:
            articles = session.execute(select(NewsArticle).order_by(NewsArticle.id.asc())).scalars().all()
            for article in articles:
                text = f"{article.title or ''} {article.description or ''} {article.content or ''}".strip()
                tickers = _extract_tickers(article, universe, text)
                if not tickers:
                    continue

                sentiment = _sentiment_score(text)
                for ticker in tickers:
                    exists = session.execute(
                        select(NewsSignal.id).where(NewsSignal.article_id == article.id, NewsSignal.ticker == ticker)
                    ).scalar_one_or_none()
                    if exists:
                        continue
                    signal = NewsSignal(
                        article_id=article.id,
                        ticker=ticker,
                        sentiment_score=sentiment,
                        relevance_score=_relevance_score(text, ticker),
                        published_at=article.published_at,
                    )
                    session.add(signal)
                    inserted += 1
            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} news signals")
        logger.info("News NLP complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("News NLP failed")
        return inserted
