from app.core.logging import setup_logging
from app.db.schema import ensure_schema
from app.features.pipeline import run_feature_generation, run_label_generation
from app.nlp.pipeline import run_news_nlp


def main() -> None:
    setup_logging()
    ensure_schema()

    news_signal_count = run_news_nlp()
    feature_count = run_feature_generation(window_hours=24)
    label_count = run_label_generation(horizon_days=1)

    print(
        {
            "news_signals_inserted": news_signal_count,
            "feature_rows_inserted": feature_count,
            "labels_inserted": label_count,
        }
    )


if __name__ == "__main__":
    main()
