"""
Main RAG pipeline implementation.
"""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .document_processor import DocumentProcessor
from .vector_store import VectorStore


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel | None = None,
        prompt_template: str | None = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            vector_store: Initialized vector store for document retrieval.
            llm: Language model for generation. Defaults to GPT-3.5-turbo.
            prompt_template: Template for the RAG prompt. If None, a default
                template will be used.
        """
        self.vector_store = vector_store
        self.llm = llm or ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

        self.prompt_template = prompt_template or """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Helpful Answer:
        """

        self.rag_chain = self._create_rag_chain()

    def _format_docs(self, docs: list[Document]) -> str:
        """Format documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_rag_chain(self):
        """Create the RAG chain with retrieval and generation."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # Create the chain
        rag_chain = (
            {"context": self.vector_store.similarity_search | self._format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def query(self, question: str, **kwargs: Any) -> str:
        """Query the RAG pipeline with a question.

        Args:
            question: The question to ask.
            **kwargs: Additional arguments to pass to the similarity search.

        Returns:
            The generated answer.
        """
        return self.rag_chain.invoke(question, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        vector_store: VectorStore | None = None,
        persist_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> 'RAGPipeline':
        """Create a RAG pipeline from a list of documents.

        Args:
            documents: List of documents to index.
            vector_store: Optional pre-initialized vector store. If None, a new one will be created.
            persist_dir: Directory to persist the vector store. If None, it will be kept in memory.
            **kwargs: Additional arguments to pass to the RAGPipeline constructor.

        Returns:
            Initialized RAGPipeline instance.
        """
        if vector_store is None:
            vector_store = VectorStore(persist_dir=persist_dir)

        vector_store.create_from_documents(documents)
        return cls(vector_store=vector_store, **kwargs)