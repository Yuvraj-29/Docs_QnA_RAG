from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DEFAULT_KNOWLEDGE_BASE = Path.cwd() / "knowledge_base"
DEFAULT_COLLECTION = "docs_qna"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 225
CHUNK_OVERLAP = 45
TOP_K = 5
QDRANT_TIMEOUT_SECONDS = 60
UPSERT_BATCH_SIZE = 256


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value


def _qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=_required_env("QDRANT_URL"),
        api_key=_required_env("QDRANT_API_KEY"),
        timeout=QDRANT_TIMEOUT_SECONDS,
    )


def _embedder() -> SentenceTransformer:
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def _splitter(tokenizer: Any) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    return [
        (page_number, page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def ingest(knowledge_base: Path) -> int:
    pdf_paths = sorted(knowledge_base.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {knowledge_base}.")

    embedder = _embedder()
    splitter = _splitter(embedder.tokenizer)
    chunk_texts: list[str] = []
    payloads: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        for page_number, page_text in _pdf_pages(pdf_path):
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                chunk_texts.append(chunk)
                payloads.append(
                    {
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_number,
                    }
                )

    if not chunk_texts:
        raise RuntimeError("The PDFs contain no extractable text to index.")

    vectors = embedder.encode(chunk_texts, normalize_embeddings=True)
    client = _qdrant_client()
    _ensure_collection(client, int(vectors.shape[1]))
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client.create_payload_index(
        collection_name=collection,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    for pdf_path in pdf_paths:
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=pdf_path.name),
                        )
                    ]
                )
            ),
            wait=True,
        )

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload=payload,
        )
        for vector, payload in zip(vectors, payloads, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=collection,
            points=points[start : start + UPSERT_BATCH_SIZE],
            wait=True,
        )
    print(f"Indexed {len(points)} chunks from {len(pdf_paths)} PDF(s).")
    return len(points)


def answer_question(question: str) -> str:
    context_entries = retrieve_context(question)
    if not context_entries:
        raise RuntimeError("No matching document chunks were found.")

    context = "\n\n".join(
        f"[Source: {entry['source']}, page {entry['page']}]\n{entry['text']}"
        for entry in context_entries
    )
    groq_client = Groq(api_key=_required_env("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided document context. "
                    "If the context does not contain the answer, say you could not find it "
                    "in the supplied documents. Cite supporting sources by filename and page."
                ),
            },
            {
                "role": "user",
                "content": f"Document context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content or "No answer was returned."


def retrieve_context(question: str, limit: int = TOP_K) -> list[dict[str, Any]]:
    embedder = _embedder()
    query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    if not client.collection_exists(collection):
        raise RuntimeError("The Qdrant collection is empty. Run `docs-qna-rag ingest` first.")

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points
    if not results:
        raise RuntimeError("No matching document chunks were found.")

    entries: list[dict[str, Any]] = []
    for point in results:
        payload = point.payload or {}
        if not payload.get("text"):
            continue
        entries.append(
            {
                "source": payload.get("source", "unknown"),
                "page": payload.get("page", 0),
                "text": payload.get("text", ""),
            }
        )
    return entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions over PDFs with RAG.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Index PDFs from the knowledge base.")
    ingest_parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=DEFAULT_KNOWLEDGE_BASE,
        help="Directory containing source PDFs (default: %(default)s).",
    )

    ask_parser = commands.add_parser("ask", help="Ask a question about indexed PDFs.")
    ask_parser.add_argument("question", help="Question to answer from the documents.")
    return parser


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "ingest":
            ingest(arguments.knowledge_base)
        else:
            print(answer_question(arguments.question))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
