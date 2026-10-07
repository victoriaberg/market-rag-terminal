"""Fetch prices and news from yfinance and normalize them into our schemas."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from src.config import settings
from src.schemas import NewsArticle, StockSnapshot

logger = logging.getLogger(__name__)


def article_id(article: NewsArticle) -> str:
    """Unique ID for an article (used for dedup here and as the Chroma ID later)."""
    raw = f"{article.ticker}|{article.link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _parse_time(value: Any) -> datetime | None:
    """Converts various date formats from Yahoo into a standard, timezone-aware UTC datetime object."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def parse_news_item(ticker: str, item: dict[str, Any]) -> NewsArticle | None:
    """Convert one raw yfinance news dict to a NewsArticle.

    Handles both yfinance formats: the newer nested ``content`` layout and the
    older flat layout (``providerPublishTime``). Returns None if the item lacks
    the minimum fields (title, link, time).
    """
    content = item.get("content")
    if isinstance(content, dict):  # newer format
        title = content.get("title")
        summary = content.get("summary") or content.get("description") or ""
        publisher = (content.get("provider") or {}).get("displayName", "Unknown")
        link = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
        )
        published = _parse_time(content.get("pubDate") or content.get("displayTime"))
    else:  # older flat format
        title = item.get("title")
        summary = item.get("summary", "")
        publisher = item.get("publisher", "Unknown")
        link = item.get("link")
        published = _parse_time(item.get("providerPublishTime"))

    if not (title and link and published):
        return None

    return NewsArticle(
        ticker=ticker.upper(),
        title=title,
        publisher=publisher,
        link=link,
        publish_time=published,
        summary=summary,
    )


def _raw_news(ticker: str, limit: int) -> list[dict[str, Any]]:
    """Get raw news dicts. ``Search`` is primary because ``Ticker.news`` can come
    back empty on some yfinance/Yahoo setups; ``Ticker.news`` is the fallback."""
    try:
        items = yf.Search(ticker, news_count=limit).news
        if items:
            return items
    except Exception:
        logger.warning("yf.Search news failed for %s, falling back", ticker, exc_info=True)
    return yf.Ticker(ticker).news or []


def is_about_ticker(article: NewsArticle, aliases: list[str] | None = None) -> bool:
    """True if the article's title/summary mentions the ticker or a company alias.

    Yahoo's ``relatedTickers`` lists the searched ticker on nearly every
    result, even for articles that are really about other companies, so it
    can't be trusted for relevance. The ticker symbol is matched
    case-sensitively (avoids false hits for short symbols); aliases such as
    "Apple" are matched case-insensitively on word boundaries.
    """
    text = f"{article.title} {article.summary}"
    if re.search(rf"\b{re.escape(article.ticker)}\b", text):
        return True
    for alias in aliases or []:
        if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            return True
    return False


def fetch_news(ticker: str, limit: int = 10, aliases: list[str] | None = None) -> list[NewsArticle]:
    """Fetch the latest news that is actually about ``ticker``.

    Over-fetches (3x) because many search results are about other companies,
    drops malformed items and off-topic articles, then caps at ``limit``.
    """
    if aliases is None:
        aliases = settings.ticker_aliases.get(ticker.upper(), [])
    articles: list[NewsArticle] = []
    for item in _raw_news(ticker, limit * 3):
        article = parse_news_item(ticker, item)
        if article is None:
            logger.debug("Skipping malformed news item for %s", ticker)
            continue
        if not is_about_ticker(article, aliases):
            logger.debug("Skipping off-topic article for %s: %s", ticker, article.title)
            continue
        articles.append(article)
        if len(articles) >= limit:
            break
    return articles


def fetch_snapshot(ticker: str) -> StockSnapshot:
    """Fetch the current price snapshot for a ticker."""
    info = yf.Ticker(ticker).fast_info
    return StockSnapshot(
        ticker=ticker.upper(),
        current_price=float(info["lastPrice"]),
        day_high=float(info["dayHigh"]),
        day_low=float(info["dayLow"]),
        volume=int(info["lastVolume"]),
        timestamp=datetime.now(timezone.utc),
    )
