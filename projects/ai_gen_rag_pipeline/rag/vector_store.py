"""
Vector store module for the RAG pipeline.
Handles document embedding and similarity search.
"""
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """Manages vector storage and similarity search for documents."""

    def __init__(
        self,
        embedding_model: Embeddings | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            embedding_model: Embedding model to use.
            persist_dir: Directory to persist the vector store. If None, the
                vector store will be kept in memory.
        """
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.vector_store: FAISS | None = None

    def create_from_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> None:
        """Create a vector store from a list of documents.

        Args:
            documents: List of documents to add to the vector store.
            **kwargs: Additional arguments to pass to the FAISS vector store.
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            **kwargs,
        )
        if self.persist_dir:
            self._save()

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Search for similar documents to the query.

        Args:
            query: The query string.
            k: Number of similar documents to return.
            **kwargs: Additional arguments to pass to the similarity search.

        Returns:
            List of documents most similar to the query.
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call create_from_documents first."
            )
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def _save(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save.")
        if not self.persist_dir:
            raise ValueError("No persist directory specified.")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.persist_dir))

    @classmethod
    def load(
        cls,
        persist_dir: str | Path,
        embedding_model: Embeddings | None = None,
    ) -> 'VectorStore':
        """Load a vector store from disk.

        Args:
            persist_dir: Directory containing the persisted vector store.
            embedding_model: Embedding model to use.

        Returns:
            Loaded VectorStore instance.
        """
        persist_dir = Path(persist_dir)
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store directory not found: {persist_dir}"
            )

        embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vector_store = cls(embedding_model=embedding_model, persist_dir=persist_dir)
        vector_store.vector_store = FAISS.load_local(
            str(persist_dir),
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store