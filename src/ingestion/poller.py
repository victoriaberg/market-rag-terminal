"""Background poller: periodically fetches prices and news for a list of tickers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from src.config import settings
from src.ingestion import fetcher
from src.schemas import NewsArticle, StockSnapshot

logger = logging.getLogger(__name__)


class MarketPoller:
    """Polls yfinance on a background thread and keeps the latest state.

    Runs continuously in the background to fetch the latest stock prices 
    and news without freezing the main application. It safely stores the 
    latest prices and sends newly discovered news to a callback function.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        interval_seconds: int | None = None,
        news_limit: int | None = None,
        on_new_articles: Callable[[list[NewsArticle]], None] | None = None,
    ) -> None:
        self.tickers = [t.upper() for t in (tickers or settings.default_tickers)]
        self.interval_seconds = interval_seconds or settings.poll_interval_seconds
        self.news_limit = news_limit or settings.news_per_ticker
        self.on_new_articles = on_new_articles

        self.snapshots: dict[str, StockSnapshot] = {}
        self._seen_ids: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> list[NewsArticle]:
        """Run one polling cycle. Returns the newly seen articles."""
        new_articles: list[NewsArticle] = []
        for ticker in self.tickers:
            try:
                snapshot = fetcher.fetch_snapshot(ticker)
                with self._lock:
                    self.snapshots[ticker] = snapshot
            except Exception:
                logger.exception("Price fetch failed for %s", ticker)

            try:
                for article in fetcher.fetch_news(ticker, self.news_limit):
                    aid = fetcher.article_id(article)
                    with self._lock:
                        if aid in self._seen_ids:
                            continue
                        self._seen_ids.add(aid)
                    new_articles.append(article)
            except Exception:
                logger.exception("News fetch failed for %s", ticker)

        if new_articles and self.on_new_articles:
            try:
                self.on_new_articles(new_articles)
            except Exception:
                logger.exception("on_new_articles callback failed")
        return new_articles

    def get_snapshots(self) -> dict[str, StockSnapshot]:
        with self._lock:
            return dict(self.snapshots)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="market-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            