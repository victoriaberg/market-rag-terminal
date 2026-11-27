from datetime import datetime, timezone
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings

from src.rag.vector_store import NewsVectorStore
from src.schemas import NewsArticle


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fast, deterministic 3D embeddings for unit tests without loading PyTorch."""

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def name() -> str:
        return "fake-embedding"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "apple" in lower or "chip" in lower else 0.0,
                    1.0 if "nvidia" in lower or "gpu" in lower else 0.0,
                    1.0,
                ]
            )
        return vectors


def _make_article(
    ticker: str = "AAPL",
    title: str = "Apple unveils new chip",
    link: str = "https://example.com/aapl-1",
    summary: str = "Faster silicon announced.",
) -> NewsArticle:
    return NewsArticle(
        ticker=ticker,
        title=title,
        publisher="Reuters",
        link=link,
        publish_time=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
        summary=summary,
    )


def test_upsert_and_deduplicate(tmp_path):
    store = NewsVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
    )
    article = _make_article()

    assert store.upsert_articles([article, article]) == 1
    assert store.count() == 1

    # Re-indexing the same article should overwrite, not duplicate
    assert store.upsert_articles([article]) == 1
    assert store.count() == 1


def test_query_filters_by_ticker_and_returns_metadata(tmp_path):
    store = NewsVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
    )
    aapl = _make_article(ticker="AAPL", title="Apple unveils new chip", link="https://example.com/1")
    nvda = _make_article(ticker="NVDA", title="Nvidia launches new GPU", link="https://example.com/2")
    store.upsert_articles([aapl, nvda])

    results = store.query("Apple chip", ticker="AAPL", n_results=5)
    assert len(results) == 1
    assert results[0]["metadata"]["ticker"] == "AAPL"
    assert results[0]["metadata"]["publisher"] == "Reuters"
    assert isinstance(results[0]["metadata"]["publish_timestamp"], float)


def test_query_empty_store_returns_empty_list(tmp_path):
    store = NewsVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
    )
    assert store.query("anything") == []
