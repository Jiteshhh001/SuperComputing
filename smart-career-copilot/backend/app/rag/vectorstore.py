"""
ChromaDB vector store operations for RAG retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import embedding_manager
from app.utils.logger import logger


class VectorStoreManager:
    """Manage ChromaDB collections for document storage and retrieval."""

    def __init__(self):
        self._stores: Dict[str, Chroma] = {}

    def _get_store(self, collection_name: str) -> Chroma:
        """Get or create a ChromaDB collection."""
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_manager.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            logger.info("Initialized vector store: %s", collection_name)
        return self._stores[collection_name]

    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
    ) -> List[str]:
        """Add documents to a collection."""
        store = self._get_store(collection_name)
        ids = store.add_documents(documents)
        logger.info(
            "Added %d documents to collection '%s'",
            len(documents),
            collection_name,
        )
        return ids

    def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents in a collection."""
        store = self._get_store(collection_name)
        kwargs = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results = store.similarity_search(query, **kwargs)
        logger.info(
            "Search '%s' in '%s': %d results",
            query[:50],
            collection_name,
            len(results),
        )
        return results

    def similarity_search_with_score(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
        store = self._get_store(collection_name)
        return store.similarity_search_with_score(query, k=k)

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""
        store = self._get_store(collection_name)
        store.delete_collection()
        self._stores.pop(collection_name, None)
        logger.info("Deleted collection: %s", collection_name)

    def get_retriever(self, collection_name: str, k: int = 5):
        """Get a LangChain retriever for a collection."""
        store = self._get_store(collection_name)
        return store.as_retriever(search_kwargs={"k": k})


# Singleton instance
vectorstore_manager = VectorStoreManager()
