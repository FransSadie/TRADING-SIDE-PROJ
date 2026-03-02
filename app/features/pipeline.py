import logging
from datetime import datetime, timedelta

from sqlalchemy import select

from app.db.models import FeatureSnapshot, IngestionRun, MarketLabel, MarketPrice, NewsSignal
from app.db.session import get_db_session

logger = logging.getLogger(__name__)


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


def run_feature_generation(window_hours: int = 24) -> int:
    run = _start_run("feature_generation")
    inserted = 0
    try:
        session = get_db_session()
        try:
            signals = session.execute(select(NewsSignal).order_by(NewsSignal.ticker, NewsSignal.published_at)).scalars().all()
            prices = session.execute(select(MarketPrice).where(MarketPrice.interval == "1d")).scalars().all()
            price_index = _build_price_index(prices)
            existing_keys = set(
                session.execute(
                    select(FeatureSnapshot.ticker, FeatureSnapshot.window_end, FeatureSnapshot.window_hours).where(
                        FeatureSnapshot.window_hours == window_hours
                    )
                ).all()
            )
            pending_keys: set[tuple[str, datetime, int]] = set()

            for idx, signal in enumerate(signals):
                if not signal.published_at:
                    continue
                window_end = signal.published_at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                feature_key = (signal.ticker, window_end, window_hours)
                if feature_key in existing_keys or feature_key in pending_keys:
                    continue

                window_start = window_end - timedelta(hours=window_hours)
                in_window = _window_signals(signals, signal.ticker, window_start, window_end, idx)
                if not in_window:
                    continue

                sentiments = [item.sentiment_score for item in in_window]
                relevances = [item.relevance_score for item in in_window]
                close_price, return_1d = _price_features(price_index.get(signal.ticker, []), window_end)

                row = FeatureSnapshot(
                    ticker=signal.ticker,
                    window_end=window_end,
                    window_hours=window_hours,
                    news_count=len(in_window),
                    sentiment_mean=_mean(sentiments),
                    sentiment_sum=sum(sentiments),
                    sentiment_std=_std(sentiments),
                    relevance_mean=_mean(relevances),
                    price_close=close_price,
                    return_1d=return_1d,
                )
                session.add(row)
                pending_keys.add(feature_key)
                inserted += 1
            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} feature rows")
        logger.info("Feature generation complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("Feature generation failed")
        return inserted


def run_label_generation(horizon_days: int = 1) -> int:
    run = _start_run("label_generation")
    inserted = 0
    try:
        session = get_db_session()
        try:
            prices = session.execute(select(MarketPrice).where(MarketPrice.interval == "1d")).scalars().all()
            by_ticker = _build_price_index(prices)
            for ticker, rows in by_ticker.items():
                for index in range(len(rows) - horizon_days):
                    current = rows[index]
                    future = rows[index + horizon_days]
                    if current.close is None or future.close is None or current.close == 0:
                        continue

                    exists = session.execute(
                        select(MarketLabel.id).where(
                            MarketLabel.ticker == ticker,
                            MarketLabel.timestamp == current.timestamp,
                            MarketLabel.horizon_days == horizon_days,
                        )
                    ).scalar_one_or_none()
                    if exists:
                        continue

                    future_return = (future.close - current.close) / current.close
                    row = MarketLabel(
                        ticker=ticker,
                        timestamp=current.timestamp,
                        horizon_days=horizon_days,
                        target_up=1 if future_return > 0 else 0,
                        future_return=future_return,
                    )
                    session.add(row)
                    inserted += 1
            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} labels")
        logger.info("Label generation complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("Label generation failed")
        return inserted


def _window_signals(signals: list[NewsSignal], ticker: str, start: datetime, end: datetime, stop_index: int) -> list[NewsSignal]:
    items: list[NewsSignal] = []
    for idx in range(stop_index, -1, -1):
        signal = signals[idx]
        if signal.ticker != ticker or not signal.published_at:
            continue
        if signal.published_at < start:
            if signal.ticker == ticker:
                break
        if start <= signal.published_at <= end:
            items.append(signal)
    return items


def _build_price_index(prices: list[MarketPrice]) -> dict[str, list[MarketPrice]]:
    index: dict[str, list[MarketPrice]] = {}
    for row in prices:
        index.setdefault(row.ticker, []).append(row)
    for ticker, rows in index.items():
        rows.sort(key=lambda x: x.timestamp)
        index[ticker] = rows
    return index


def _price_features(price_rows: list[MarketPrice], when: datetime) -> tuple[float | None, float | None]:
    if not price_rows:
        return None, None
    closest_idx = -1
    for idx, row in enumerate(price_rows):
        if row.timestamp <= when:
            closest_idx = idx
        else:
            break
    if closest_idx == -1:
        return None, None
    current = price_rows[closest_idx]
    close_price = current.close
    if close_price is None or close_price == 0:
        return close_price, None
    next_idx = closest_idx + 1
    if next_idx >= len(price_rows) or price_rows[next_idx].close is None:
        return close_price, None
    next_close = price_rows[next_idx].close
    if next_close is None:
        return close_price, None
    return close_price, (next_close - close_price) / close_price


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5
