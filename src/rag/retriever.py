"""Time-decay retriever: re-ranks vector search results by combining semantic similarity and recency."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import settings
from src.rag.vector_store import NewsVectorStore
from src.schemas import RetrievedArticle


def cosine_distance_to_similarity(distance: float) -> float:
    """Converts ChromaDB cosine distance (0 = identical) into a 0.0 to 1.0 similarity score."""
    return max(0.0, min(1.0, 1.0 - distance))


def recency_decay(
    publish_timestamp: float,
    now: datetime | None = None,
    half_life_hours: float | None = None,
) -> float:
    """Calculates a 0.0 to 1.0 freshness score using exponential half-life decay.

    An article published right now scores 1.0; after one half-life it scores 0.5;
    after two half-lives it scores 0.25.
    """
    current_time = now or datetime.now(timezone.utc)
    half_life = half_life_hours if half_life_hours is not None else settings.decay_half_life_hours
    if half_life <= 0:
        raise ValueError("half_life_hours must be positive")

    age_hours = max(0.0, (current_time.timestamp() - publish_timestamp) / 3600.0)
    return 0.5 ** (age_hours / half_life)


def compute_relevance_score(
    distance: float,
    publish_timestamp: float,
    now: datetime | None = None,
    half_life_hours: float | None = None,
    recency_weight: float | None = None,
) -> tuple[float, float, float]:
    """Combines semantic similarity and recency decay into (semantic_score, recency_score, final_score)."""
    weight = recency_weight if recency_weight is not None else settings.recency_weight
    if not 0.0 <= weight <= 1.0:
        raise ValueError("recency_weight must be between 0.0 and 1.0")

    semantic = cosine_distance_to_similarity(distance)
    recency = recency_decay(publish_timestamp, now=now, half_life_hours=half_life_hours)
    final = (1.0 - weight) * semantic + weight * recency
    return semantic, recency, final


class TimeDecayRetriever:
    """Fetches candidate articles from ChromaDB and re-ranks them using time-decay relevance scoring."""

    def __init__(
        self,
        vector_store: NewsVectorStore,
        half_life_hours: float | None = None,
        recency_weight: float | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        """Connects the retriever to a NewsVectorStore and configures the decay parameters."""
        self.vector_store = vector_store
        self.half_life_hours = (
            half_life_hours if half_life_hours is not None else settings.decay_half_life_hours
        )
        self.recency_weight = (
            recency_weight if recency_weight is not None else settings.recency_weight
        )
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query_text: str,
        ticker: str | None = None,
        top_k: int = 5,
        now: datetime | None = None,
    ) -> list[RetrievedArticle]:
        """Searches the vector store, scores each candidate for relevance + freshness, and returns the top_k results."""
        if top_k <= 0:
            return []

        raw_candidates = self.vector_store.query(
            query_text=query_text,
            ticker=ticker,
            n_results=top_k * self.candidate_multiplier,
        )
        if not raw_candidates:
            return []

        current_time = now or datetime.now(timezone.utc)
        scored: list[RetrievedArticle] = []

        for item in raw_candidates:
            meta = item["metadata"]
            publish_ts = float(meta["publish_timestamp"])
            semantic, recency, final = compute_relevance_score(
                distance=float(item["distance"]),
                publish_timestamp=publish_ts,
                now=current_time,
                half_life_hours=self.half_life_hours,
                recency_weight=self.recency_weight,
            )
            scored.append(
                RetrievedArticle(
                    article_id=item["id"],
                    ticker=meta["ticker"],
                    title=meta["title"],
                    publisher=meta["publisher"],
                    link=meta["link"],
                    publish_time=datetime.fromtimestamp(publish_ts, tz=timezone.utc),
                    document=item["document"],
                    semantic_score=round(semantic, 4),
                    recency_score=round(recency, 4),
                    final_score=round(final, 4),
                )
            )

        scored.sort(key=lambda article: article.final_score, reverse=True)
        return scored[:top_k]
