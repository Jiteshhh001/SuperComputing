"""
OpenAI embeddings wrapper for RAG pipeline.
"""

from __future__ import annotations

from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.utils.logger import logger


class EmbeddingManager:
    """Manage OpenAI embeddings for the RAG pipeline."""

    def __init__(self):
        self._embeddings = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("Initialized embeddings: %s", settings.embedding_model)
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch."""
        return self.embeddings.embed_documents(texts)


# Singleton instance
embedding_manager = EmbeddingManager()
