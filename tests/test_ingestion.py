from datetime import datetime, timezone
from unittest.mock import patch

from src.ingestion import fetcher
from src.ingestion.poller import MarketPoller
from src.schemas import NewsArticle, StockSnapshot

NEW_FORMAT_ITEM = {
    "id": "abc",
    "content": {
        "title": "Apple unveils new chip",
        "summary": "Apple announced a faster chip.",
        "pubDate": "2026-10-07T12:30:00Z",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://example.com/apple-chip"},
    },
}

OLD_FORMAT_ITEM = {
    "title": "Microsoft beats earnings",
    "publisher": "Bloomberg",
    "link": "https://example.com/msft-earnings",
    "providerPublishTime": 1791376200,
}


def _article(ticker: str = "AAPL", link: str = "https://example.com/a") -> NewsArticle:
    return NewsArticle(
        ticker=ticker,
        title="t",
        publisher="p",
        link=link,
        publish_time=datetime(2026, 10, 7, tzinfo=timezone.utc),
        summary="s",
    )


def _snapshot(ticker: str) -> StockSnapshot:
    return StockSnapshot(
        ticker=ticker,
        current_price=100.0,
        day_high=101.0,
        day_low=99.0,
        volume=1000,
        timestamp=datetime.now(timezone.utc),
    )


def test_parse_new_format_item():
    article = fetcher.parse_news_item("aapl", NEW_FORMAT_ITEM)
    assert article is not None
    assert article.ticker == "AAPL"
    assert article.publisher == "Reuters"
    assert article.publish_time == datetime(2026, 10, 7, 12, 30, tzinfo=timezone.utc)
    assert article.link == "https://example.com/apple-chip"


def test_parse_old_format_item_is_utc():
    article = fetcher.parse_news_item("MSFT", OLD_FORMAT_ITEM)
    assert article is not None
    assert article.publish_time.tzinfo == timezone.utc
    assert article.summary == ""


def test_parse_malformed_item_returns_none():
    assert fetcher.parse_news_item("AAPL", {"content": {"title": "no link or time"}}) is None
    assert fetcher.parse_news_item("AAPL", {}) is None


def test_article_id_is_stable_and_ticker_specific():
    a = _article("AAPL")
    assert fetcher.article_id(a) == fetcher.article_id(_article("AAPL"))
    assert fetcher.article_id(a) != fetcher.article_id(_article("MSFT"))


def test_fetch_news_uses_search_and_falls_back():
    class FakeSearch:
        def __init__(self, ticker, news_count=10):
            self.news = [OLD_FORMAT_ITEM]

    with patch.object(fetcher.yf, "Search", FakeSearch):
        articles = fetcher.fetch_news("MSFT")
    assert len(articles) == 1 and articles[0].ticker == "MSFT"

    class EmptySearch:
        def __init__(self, ticker, news_count=10):
            self.news = []

    class FakeTicker:
        def __init__(self, ticker):
            self.news = [NEW_FORMAT_ITEM]

    with patch.object(fetcher.yf, "Search", EmptySearch), patch.object(
        fetcher.yf, "Ticker", FakeTicker
    ):
        articles = fetcher.fetch_news("AAPL")
    assert len(articles) == 1 and articles[0].publisher == "Reuters"


def test_is_about_ticker():
    def art(title: str, summary: str = "") -> NewsArticle:
        return NewsArticle(
            ticker="AAPL",
            title=title,
            publisher="p",
            link="https://example.com/x",
            publish_time=datetime(2026, 10, 7, tzinfo=timezone.utc),
            summary=summary,
        )

    assert fetcher.is_about_ticker(art("Apple ordered to pay Masimo"), ["Apple"])
    assert fetcher.is_about_ticker(art("Why AAPL is moving"), [])
    assert fetcher.is_about_ticker(art("Chip news", "Apple supplier rallies"), ["Apple"])
    assert not fetcher.is_about_ticker(art("Netflix Stock Plunges 26.8%"), ["Apple"])
    assert not fetcher.is_about_ticker(art("Pineapple prices rise"), ["Apple"])


def test_fetch_news_drops_off_topic_articles():
    off_topic = {
        "title": "Netflix Stock Plunges 26.8% Year to Date",
        "publisher": "Zacks",
        "link": "https://example.com/nflx",
        "providerPublishTime": 1791376200,
        "relatedTickers": ["NFLX", "AAPL"],  # Yahoo tags AAPL on everything
    }
    on_topic = {
        "title": "Apple ordered to pay Masimo $184M",
        "publisher": "MedTech Dive",
        "link": "https://example.com/aapl",
        "providerPublishTime": 1791376200,
        "relatedTickers": ["AAPL", "DHR"],
    }

    class FakeSearch:
        def __init__(self, ticker, news_count=10):
            self.news = [off_topic, on_topic]

    with patch.object(fetcher.yf, "Search", FakeSearch):
        articles = fetcher.fetch_news("AAPL", aliases=["Apple"])
    assert [a.link for a in articles] == ["https://example.com/aapl"]


def test_poller_deduplicates_across_cycles():
    collected: list[NewsArticle] = []
    poller = MarketPoller(tickers=["AAPL"], on_new_articles=collected.extend)
    with patch.object(fetcher, "fetch_snapshot", side_effect=_snapshot), patch.object(
        fetcher, "fetch_news", return_value=[_article()]
    ):
        first = poller.poll_once()
        second = poller.poll_once()
    assert len(first) == 1
    assert second == []
    assert len(collected) == 1


def test_poller_survives_a_failing_ticker():
    def flaky_snapshot(ticker: str) -> StockSnapshot:
        if ticker == "BAD":
            raise RuntimeError("yfinance exploded")
        return _snapshot(ticker)

    poller = MarketPoller(tickers=["BAD", "AAPL"])
    with patch.object(fetcher, "fetch_snapshot", side_effect=flaky_snapshot), patch.object(
        fetcher, "fetch_news", return_value=[]
    ):
        poller.poll_once()
    assert "AAPL" in poller.get_snapshots()
    assert "BAD" not in poller.get_snapshots()