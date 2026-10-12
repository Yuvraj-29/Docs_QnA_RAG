from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docs_qna_rag import answer_question, ingest, retrieve_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Docs Q&A RAG",
    description="Question-answering over uploaded PDF documents using vector search and Groq.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    knowledge_base: str | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> dict[str, list[str]]:
    knowledge_base = Path(os.getenv("KNOWLEDGE_BASE", str(PROJECT_ROOT / "knowledge_base")))
    if not knowledge_base.exists():
        return {"documents": []}
    return {"documents": sorted(file.name for file in knowledge_base.glob("*.pdf"))}


@app.post("/api/ingest")
def ingest_documents(payload: IngestRequest | None = None) -> dict[str, int | str]:
    knowledge_base_value = (payload.knowledge_base if payload and payload.knowledge_base else os.getenv("KNOWLEDGE_BASE", str(PROJECT_ROOT / "knowledge_base")))
    try:
        index_count = ingest(Path(knowledge_base_value))
    except Exception as exc:  # pragma: no cover - surfaced as API error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"indexed": index_count, "knowledge_base": knowledge_base_value}


@app.post("/api/ask")
def ask_question(payload: QuestionRequest) -> dict[str, str | list[dict[str, object]]]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        sources = retrieve_context(question)
        answer = answer_question(question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "answer": answer,
        "question": question,
        "sources": [
            {
                "source": source["source"],
                "page": source["page"],
                "text": source["text"],
            }
            for source in sources
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
