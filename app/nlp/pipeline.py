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

SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.3,
    "bloomberg": 1.3,
    "wall street journal": 1.25,
    "financial times": 1.25,
    "cnbc": 1.15,
    "marketwatch": 1.1,
    "yahoo finance": 1.05,
}

EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "guidance"),
    "rates": ("fed", "rate", "interest rate", "cut", "hike"),
    "inflation": ("inflation", "cpi", "ppi"),
    "m_and_a": ("acquire", "merger", "buyout"),
    "regulation": ("regulation", "antitrust", "lawsuit", "fine"),
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


def _source_weight(source_name: str) -> float:
    lowered = (source_name or "").lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in lowered:
            return weight
    return 1.0


def _event_tags(text: str) -> str | None:
    lowered = (text or "").lower()
    tags = [tag for tag, keywords in EVENT_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    if not tags:
        return None
    return ",".join(sorted(tags))


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
                source_weight = _source_weight(article.source_name)
                event_tags = _event_tags(text)
                for ticker in tickers:
                    exists = session.execute(
                        select(NewsSignal.id).where(NewsSignal.article_id == article.id, NewsSignal.ticker == ticker)
                    ).scalar_one_or_none()
                    if exists:
                        continue
                    signal = NewsSignal(
                        article_id=article.id,
                        ticker=ticker,
                        sentiment_score=max(min(sentiment * source_weight, 1.0), -1.0),
                        relevance_score=_relevance_score(text, ticker),
                        source_weight=source_weight,
                        event_tags=event_tags,
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
