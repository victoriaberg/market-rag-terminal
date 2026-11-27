"""Local ChromaDB vector store for indexing and querying financial news."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings
from src.ingestion.fetcher import article_id
from src.schemas import NewsArticle


class NewsVectorStore:
    """Stores news articles as vector embeddings in a local ChromaDB database on disk."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        """Connects to the local ChromaDB folder and opens (or creates) the news collection."""
        self.persist_dir = persist_dir or settings.chroma_db_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        if embedding_function is None:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_articles(self, articles: list[NewsArticle]) -> int:
        """Embeds and saves a list of news articles to ChromaDB. Uses deterministic IDs so duplicates are safely overwritten instead of creating double entries."""
        if not articles:
            return 0

        # Deduplicate within the incoming batch so ChromaDB never receives the same ID twice in one call
        unique: dict[str, NewsArticle] = {article_id(a): a for a in articles}

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for aid, article in unique.items():
            ids.append(aid)
            # Combine title and summary so articles with empty summaries still have rich text to embed
            text = f"{article.title}\n\n{article.summary}".strip()
            documents.append(text)
            metadatas.append(
                {
                    "ticker": article.ticker.upper(),
                    "title": article.title,
                    "publisher": article.publisher,
                    "link": article.link,
                    "publish_timestamp": article.publish_time.timestamp(),
                    "publish_time_iso": article.publish_time.isoformat(),
                }
            )

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def query(
        self,
        query_text: str,
        ticker: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Searches the database for the most semantically similar news articles, optionally filtered to a single stock ticker."""
        total = self.count()
        if total == 0:
            return []

        where = {"ticker": ticker.upper()} if ticker else None
        raw = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, total),
            where=where,
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for aid, doc, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                {
                    "id": aid,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return results

    def count(self) -> int:
        """Returns the total number of indexed articles currently stored in the database."""
        return self._collection.count()
