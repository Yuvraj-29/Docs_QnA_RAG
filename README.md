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

2. Fill in your credentials for Qdrant and Groq in `.env`. `HF_TOKEN` is only required when using hosted embeddings.

3. Install Python dependencies, including the local embedding runtime:

	uv sync --extra local-embeddings

4. Ensure the PDF collection exists in `knowledge_base` and then ingest it locally:

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
- `POST /api/ingest` — disabled by default; local CLI ingestion is preferred

### Notes

- The project uses `uv` to manage the Python environment.
- If you hit editable-install issues with uv, prefer `uv run --no-editable ...` during local execution.

### Render deployment (free web service)

The free Render service has 512 MB RAM, so leave the query backend on the hosted embedding path. SentenceTransformer is an optional local dependency and is not installed by Render's build command below.

1. Create a Render **Web Service** from this repository. Set the root directory to the repository root, runtime to Python 3.14 (the version in `.python-version`), build command to `pip install uv && uv sync --no-dev --no-editable`, and start command to `uv run --no-sync uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
2. Add these Render environment variables: `EMBEDDING_BACKEND=hosted`, `HF_TOKEN`, `HF_INFERENCE_PROVIDER=hf-inference`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `GROQ_API_KEY`, and `CORS_ORIGINS=https://your-frontend.onrender.com`. Set `ENABLE_INGEST_API=false`. Keep `HF_TOKEN` in Render's secret environment settings; never commit it.
3. Deploy the Next.js frontend separately and set its `NEXT_PUBLIC_API_URL` to the deployed backend URL. Update `CORS_ORIGINS` to the frontend's exact deployed origin (scheme and hostname, no path).

The app's local SentenceTransformer path remains the default for development and document ingestion. Install it with `uv sync --extra local-embeddings`. Hosted inference uses Hugging Face `InferenceClient.feature_extraction`; the model's published Sentence Transformers configuration specifies mean-token pooling, 384 dimensions, and a Normalize module. If the endpoint returns token vectors, this app mean-pools the one input's returned token vectors and L2-normalizes them. The endpoint may instead return a pooled vector, which is normalized directly.

Matching dimension and pooling configuration is necessary but does not by itself prove that a third-party inference provider produces vectors numerically identical to the locally indexed vectors. The app therefore rejects hosted queries against legacy/unmarked or local-indexed points, and refuses hosted ingestion into a non-empty collection. Create a **new empty** Qdrant collection, set `EMBEDDING_BACKEND=hosted`, `HF_INFERENCE_PROVIDER=hf-inference`, and `QDRANT_COLLECTION` to that new name in your local `.env`, then run `uv run --no-editable docs-qna-rag ingest --knowledge-base ./knowledge_base` to re-index all documents with hosted embeddings. Only after ingestion finishes, deploy with the same collection name, provider, and hosted backend. Ingestion on Render remains disabled by default; the service filesystem is not a durable document store.

Hosted re-indexing sends one inference request per chunk and can be slow or subject to provider quotas. The model's Sentence Transformers max sequence length is 256 tokens; keep document chunks within that limit (the current chunk size is 225 tokens). The hosted backend is intentionally restricted to this exact model because its dimension and pooling assumptions are model-specific.

To explicitly enable the protected ingestion API, set `ENABLE_INGEST_API=true` and provide a long random `INGEST_API_KEY` in the service's secret environment. Requests must send it in the `X-Ingest-Token` header. The endpoint only ingests the server-configured `KNOWLEDGE_BASE`; it does not accept an arbitrary path from callers.
