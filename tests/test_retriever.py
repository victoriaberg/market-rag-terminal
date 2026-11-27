from datetime import datetime, timedelta, timezone

import pytest

from src.rag.retriever import (
    TimeDecayRetriever,
    compute_relevance_score,
    cosine_distance_to_similarity,
    recency_decay,
)
from src.rag.vector_store import NewsVectorStore
from src.schemas import NewsArticle
from tests.test_rag import FakeEmbeddingFunction


def test_cosine_distance_to_similarity():
    assert cosine_distance_to_similarity(0.0) == pytest.approx(1.0)
    assert cosine_distance_to_similarity(0.4) == pytest.approx(0.6)
    assert cosine_distance_to_similarity(1.5) == pytest.approx(0.0)


def test_recency_decay_half_life():
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)

    assert recency_decay(now.timestamp(), now=now, half_life_hours=24.0) == pytest.approx(1.0)
    assert recency_decay(
        (now - timedelta(hours=24)).timestamp(), now=now, half_life_hours=24.0
    ) == pytest.approx(0.5)
    assert recency_decay(
        (now - timedelta(hours=48)).timestamp(), now=now, half_life_hours=24.0
    ) == pytest.approx(0.25)
    # Future timestamps (clock skew) should clamp cleanly to 1.0
    assert recency_decay(
        (now + timedelta(hours=2)).timestamp(), now=now, half_life_hours=24.0
    ) == pytest.approx(1.0)


def test_compute_relevance_score_blends_semantic_and_recency():
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
    one_day_ago = (now - timedelta(hours=24)).timestamp()

    semantic, recency, final = compute_relevance_score(
        distance=0.2,
        publish_timestamp=one_day_ago,
        now=now,
        half_life_hours=24.0,
        recency_weight=0.3,
    )
    assert semantic == pytest.approx(0.8)
    assert recency == pytest.approx(0.5)
    assert final == pytest.approx(0.7 * 0.8 + 0.3 * 0.5)


def test_retriever_prefers_newer_article_over_stale_article(tmp_path):
    store = NewsVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
    )
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)

    stale = NewsArticle(
        ticker="AAPL",
        title="Apple unveils new chip",
        publisher="Reuters",
        link="https://example.com/old",
        publish_time=now - timedelta(days=10),
        summary="Old chip news.",
    )
    fresh = NewsArticle(
        ticker="AAPL",
        title="Apple unveils new chip today",
        publisher="Bloomberg",
        link="https://example.com/new",
        publish_time=now - timedelta(hours=1),
        summary="Breaking chip news.",
    )
    store.upsert_articles([stale, fresh])

    retriever = TimeDecayRetriever(store, half_life_hours=24.0, recency_weight=0.3)
    results = retriever.retrieve("Apple chip", ticker="AAPL", top_k=2, now=now)

    assert len(results) == 2
    assert results[0].link == "https://example.com/new"
    assert results[0].final_score > results[1].final_score


def test_retriever_empty_store_returns_empty_list(tmp_path):
    store = NewsVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
    )
    retriever = TimeDecayRetriever(store)
    assert retriever.retrieve("Apple earnings") == []
