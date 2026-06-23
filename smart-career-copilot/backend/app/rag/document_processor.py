"""
Document processing pipeline: loading, splitting, and metadata injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.logger import logger


class DocumentProcessor:
    """Process documents for RAG ingestion."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and parse a PDF file into documents."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            documents = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": i + 1,
                            "total_pages": len(reader.pages),
                            "type": "pdf",
                        },
                    )
                    documents.append(doc)

            logger.info("Loaded PDF: %s (%d pages)", file_path, len(reader.pages))
            return documents

        except Exception as e:
            logger.error("Failed to load PDF %s: %s", file_path, str(e))
            return []

    def load_text(self, file_path: str) -> List[Document]:
        """Load a text or markdown file into documents."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "type": Path(file_path).suffix.lstrip("."),
                },
            )
            return [doc]
        except Exception as e:
            logger.error("Failed to load text file %s: %s", file_path, str(e))
            return []

    def load_from_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Create documents directly from text content."""
        doc = Document(
            page_content=text,
            metadata=metadata or {"source": "direct_input", "type": "text"},
        )
        return [doc]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding."""
        chunks = self.splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
        return chunks

    def process_file(self, file_path: str) -> List[Document]:
        """Load and split a file into chunks ready for embedding."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            docs = self.load_pdf(file_path)
        elif ext in (".txt", ".md", ".rst"):
            docs = self.load_text(file_path)
        else:
            logger.warning("Unsupported file type: %s", ext)
            return []

        return self.split_documents(docs)

    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Process raw text into chunks ready for embedding."""
        docs = self.load_from_text(text, metadata)
        return self.split_documents(docs)
