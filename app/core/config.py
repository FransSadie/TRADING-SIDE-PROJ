from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="market-news-bot", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(default="sqlite:///./market_bot.db", alias="DATABASE_URL")

    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    news_api_base_url: str = Field(default="https://newsapi.org/v2", alias="NEWS_API_BASE_URL")
    news_query: str = Field(
        default="(stock OR market OR earnings OR inflation OR fed)", alias="NEWS_QUERY"
    )
    news_language: str = Field(default="en", alias="NEWS_LANGUAGE")
    news_page_size: int = Field(default=50, alias="NEWS_PAGE_SIZE")

    market_tickers: str = Field(default="AAPL,MSFT,NVDA,SPY,QQQ", alias="MARKET_TICKERS")
    ingest_interval_minutes: int = Field(default=15, alias="INGEST_INTERVAL_MINUTES")
    model_artifacts_dir: str = Field(default="./artifacts", alias="MODEL_ARTIFACTS_DIR")

    @property
    def market_ticker_list(self) -> List[str]:
        return [ticker.strip().upper() for ticker in self.market_tickers.split(",") if ticker.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
