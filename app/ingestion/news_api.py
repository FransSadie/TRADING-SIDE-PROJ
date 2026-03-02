import logging
from datetime import datetime
from typing import Any, Dict, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class NewsApiClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.news_api_base_url.rstrip("/")

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def fetch_latest(self) -> List[Dict[str, Any]]:
        if not self.settings.news_api_key:
            raise ValueError("NEWS_API_KEY is missing. Set it in your .env file.")

        url = f"{self.base_url}/everything"
        params = {
            "q": self.settings.news_query,
            "language": self.settings.news_language,
            "sortBy": "publishedAt",
            "pageSize": self.settings.news_page_size,
            "apiKey": self.settings.news_api_key,
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            raise RuntimeError(f"News API error: {payload}")

        articles = payload.get("articles", [])
        logger.info("Fetched %s articles from News API", len(articles))
        return articles


def parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None

