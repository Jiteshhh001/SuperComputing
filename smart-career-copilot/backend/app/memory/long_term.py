"""
Long-term memory — semantic memory storage using ChromaDB.
Stores important facts, user preferences, and past insights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.rag.vectorstore import vectorstore_manager
from app.utils.helpers import generate_id, utc_now
from app.utils.logger import logger

MEMORY_COLLECTION = "long_term_memory"


class LongTermMemory:
    """ChromaDB-backed semantic memory for persistent knowledge."""

    def store_memory(
        self,
        content: str,
        memory_type: str = "fact",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a piece of information in long-term memory."""
        mem_id = generate_id()
        meta = {
            "memory_id": mem_id,
            "memory_type": memory_type,
            "session_id": session_id or "",
            "stored_at": str(utc_now()),
            **(metadata or {}),
        }

        doc = Document(page_content=content, metadata=meta)
        vectorstore_manager.add_documents(MEMORY_COLLECTION, [doc])
        logger.info("Stored memory: %s (type=%s)", mem_id, memory_type)
        return mem_id

    def recall(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories based on a query."""
        filter_meta = None
        if memory_type:
            filter_meta = {"memory_type": memory_type}

        docs = vectorstore_manager.similarity_search(
            MEMORY_COLLECTION,
            query=query,
            k=k,
            filter_metadata=filter_meta,
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def recall_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recall memories with relevance scores."""
        results = vectorstore_manager.similarity_search_with_score(
            MEMORY_COLLECTION,
            query=query,
            k=k,
        )

        return [
            {
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]

    def clear_all(self) -> None:
        """Clear all long-term memories."""
        vectorstore_manager.delete_collection(MEMORY_COLLECTION)
        logger.info("Cleared all long-term memories")


# Singleton
long_term_memory = LongTermMemory()
