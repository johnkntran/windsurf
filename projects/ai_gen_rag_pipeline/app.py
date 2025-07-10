"""
FastAPI application for the RAG pipeline and tools demonstration.
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import RAGPipeline
from rag.vector_store import VectorStore

# Import tools demo
from tools.router import router as tools_router


# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline & Tools API",
    description="API for RAG pipeline and tools demonstration",
    version="1.0.0"
)

# Include the tools router
app.include_router(tools_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the RAG pipeline
rag_pipeline: RAGPipeline | None = None


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str
    num_results: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    message: str
    num_documents: int


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the RAG pipeline on startup."""
    global rag_pipeline

    # Create directories if they don't exist
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)

    # Try to load existing vector store
    try:
        vector_store = VectorStore.load("data/vector_store")
        rag_pipeline = RAGPipeline(vector_store=vector_store)
        print("Loaded existing vector store")
    except FileNotFoundError:
        print("No existing vector store found. Please upload documents first.")
        rag_pipeline = None


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload and process documents."""
    global rag_pipeline

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files
    saved_files = []
    for file in files:
        file_path = f"data/documents/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_path)

    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_documents(saved_files)

    # Create or update the RAG pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline.from_documents(
            documents=documents,
            persist_dir="data/vector_store"
        )
    else:
        # In a production system, you'd want to update the existing vector store
        # For simplicity, we'll just recreate it here
        rag_pipeline = RAGPipeline.from_documents(
            documents=documents,
            persist_dir="data/vector_store"
        )

    return UploadResponse(
        message=f"Successfully processed {len(documents)} document chunks",
        num_documents=len(documents)
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG pipeline."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload documents first."
        )

    try:
        # Get the answer
        answer = rag_pipeline.query(
            request.question,
            k=request.num_results
        )

        # Get the sources (retrieved documents)
        sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in rag_pipeline.vector_store.similarity_search(
                request.question,
                k=request.num_results
            )
        ]

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}