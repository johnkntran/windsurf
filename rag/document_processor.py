"""
Document processing module for the RAG pipeline.
Handles loading and preprocessing of documents.
"""
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles loading and processing of documents for the RAG pipeline."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor.

        Args:
            chunk_size: Maximum size of chunks to split documents into.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str | Path) -> list[Document]:
        """Load a document from a file.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Document objects.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = None
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path))
        elif file_path.suffix.lower() in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return loader.load()

    def process_documents(
        self, file_paths: list[str | Path]
    ) -> list[Document]:
        """Process multiple documents and split them into chunks.

        Args:
            file_paths: List of paths to document files.

        Returns:
            List of processed document chunks.
        """
        all_docs = []
        for file_path in file_paths:
            docs = self.load_document(file_path)
            chunks = self.text_splitter.split_documents(docs)
            all_docs.extend(chunks)
        return all_docs