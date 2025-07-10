# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for document-based question answering. This project implements a complete RAG system with document processing, vector storage, and a FastAPI-based web interface.

## Features

- Document processing for PDF, TXT, and Markdown files
- Vector storage using FAISS for efficient similarity search
- Integration with OpenAI's language models for generation
- Simple REST API for uploading documents and querying the system
- Support for both local embedding models (via HuggingFace) and OpenAI embeddings

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or pip
- OpenAI API key (for using GPT models)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-pipeline
   ```

2. Copy the example environment file and update with your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your OpenAI API key.

3. Install dependencies:
   ```bash
   # Using poetry (recommended)
   poetry install

   # Or using pip
   pip install -r requirements.txt
   ```

## Usage

### Starting the API Server

```bash
# Using poetry
poetry run uvicorn app:app --reload

# Or using pip
uvicorn app:app --reload
```

This will start the FastAPI server at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### Uploading Documents

Use the `/upload` endpoint to upload documents. You can upload multiple files at once.

Example using curl:
```bash
curl -X 'POST' \
  'http://localhost:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@document1.pdf;type=application/pdf' \
  -F 'files=@document2.txt;type=text/plain'
```

### Querying the RAG Pipeline

Use the `/query` endpoint to ask questions about your documents.

Example using curl:
```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is the main topic of the document?",
    "num_results": 3
  }'
```

## Project Structure

- `rag/`: Core RAG implementation
  - `document_processor.py`: Handles document loading and processing
  - `vector_store.py`: Manages vector storage and similarity search
  - `rag_pipeline.py`: Main RAG pipeline implementation
- `app.py`: FastAPI application with REST endpoints
- `requirements.txt`: Python dependencies

## Configuration

You can configure the RAG pipeline by setting environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `HUGGINGFACEHUB_API_KEY`: HuggingFace Hub API key (optional, for private models)
- `FAISS_MAX_INNER_PRODUCT`: Set to `True` to use max inner product for similarity (default: `False`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `info`)

## License

MIT
