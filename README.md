## Docs Q&A RAG

This project lets you upload PDF documents, index them with a vector store, and ask natural-language questions about their contents using retrieval-augmented generation (RAG).

### Features

- PDF extraction with `pypdf`
- Recursive text chunking for embeddings
- Qdrant vector search for retrieval
- Groq-hosted LLM answers grounded in document snippets
- FastAPI backend for chat and ingestion
- Next.js frontend for browser-based interaction

### Project structure

- `src/docs_qna_rag/__init__.py` — core indexing and question-answer logic
- `backend/main.py` — FastAPI API endpoints
- `frontend/` — Next.js web app
- `knowledge_base/` — PDFs to index

### Setup

1. Copy the environment template:

	cp .env.example .env

2. Fill in your credentials for Qdrant and Groq in `.env`.

3. Install Python dependencies:

	uv sync

4. Ensure the PDF collection exists in `knowledge_base` and then ingest it:

	uv run --no-editable docs-qna-rag ingest --knowledge-base ./knowledge_base

### Run the app locally

Start the backend:

uv run --no-editable uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Start the frontend in a second terminal:

cd frontend
npm run dev

Then open http://localhost:3000.

### API endpoints

- `GET /api/health` — health check
- `GET /api/documents` — list indexed PDF files
- `POST /api/ask` — ask a question using the collection
- `POST /api/ingest` — re-index PDFs from the configured directory

### Notes

- The project uses `uv` to manage the Python environment.
- If you hit editable-install issues with uv, prefer `uv run --no-editable ...` during local execution.
