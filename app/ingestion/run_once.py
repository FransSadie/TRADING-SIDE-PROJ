from app.core.logging import setup_logging
from app.db.base import Base
from app.db.session import engine
from app.ingestion.pipeline import run_all_ingestion


def main() -> None:
    setup_logging()
    Base.metadata.create_all(bind=engine)
    result = run_all_ingestion()
    print(result)


if __name__ == "__main__":
    main()

