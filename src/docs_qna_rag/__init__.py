from __future__ import annotations

import argparse
from functools import lru_cache
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

DEFAULT_KNOWLEDGE_BASE = Path.cwd() / "knowledge_base"
DEFAULT_COLLECTION = "docs_qna"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_BACKEND = "local"
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


def _embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def _embedding_backend() -> str:
    backend = os.getenv("EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if backend not in {"local", "hosted"}:
        raise RuntimeError("EMBEDDING_BACKEND must be either 'local' or 'hosted'.")
    return backend


@lru_cache(maxsize=1)
def _hosted_inference_client() -> Any:
    from huggingface_hub import InferenceClient

    token = _required_env("HF_TOKEN")
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if model_name != DEFAULT_EMBEDDING_MODEL:
        raise RuntimeError(
            "The hosted embedding backend is configured for "
            f"{DEFAULT_EMBEDDING_MODEL}; EMBEDDING_MODEL must match that model."
        )
    provider = os.getenv("HF_INFERENCE_PROVIDER", "hf-inference")
    return InferenceClient(
        model=model_name,
        provider=provider,
        token=token,
        timeout=30,
    )


def _sentence_embedding(raw_output: Any) -> np.ndarray:
    try:
        output = np.asarray(raw_output, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Embedding output must be a numeric, rectangular array with a final "
            f"dimension of {EMBEDDING_DIMENSION}."
        ) from error

    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim == 2:
        if output.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Embedding endpoint returned token vectors with dimension "
                f"{output.shape[1]}; expected {EMBEDDING_DIMENSION}."
            )
        if output.shape[0] == 0:
            raise RuntimeError("Embedding endpoint returned no token vectors.")
        output = output.mean(axis=0)
    elif output.ndim == 1:
        if output.shape[0] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding endpoint returned {output.shape[0]} values; "
                f"expected {EMBEDDING_DIMENSION}."
            )
    else:
        raise RuntimeError(
            "Embedding endpoint returned an unsupported shape "
            f"{output.shape}; expected (384,), (tokens, 384), or (1, tokens, 384)."
        )

    if output.shape != (EMBEDDING_DIMENSION,):
        raise RuntimeError(
            f"Sentence embedding has shape {output.shape}; "
            f"expected ({EMBEDDING_DIMENSION},)."
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Embedding endpoint returned NaN or infinite values.")

    norm = float(np.linalg.norm(output))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("Embedding endpoint returned a zero-length vector.")
    return output / norm


def _hosted_embedding(text: str) -> np.ndarray:
    try:
        raw_output = _hosted_inference_client().feature_extraction(text)
    except Exception as error:
        raise RuntimeError(
            "Hugging Face embedding request failed. Check HF_TOKEN, provider access, "
            "and model availability."
        ) from error
    return _sentence_embedding(raw_output)


def _query_embedding(question: str) -> np.ndarray:
    if _embedding_backend() == "hosted":
        return _hosted_embedding(question)
    return _sentence_embedding(
        _embedder().encode(question, normalize_embeddings=True)
    )


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


def _collection_vector_dimension(client: QdrantClient, collection: str) -> int | None:
    vector_config = client.get_collection(collection).config.params.vectors
    if isinstance(vector_config, dict):
        if len(vector_config) != 1:
            raise RuntimeError(
                "The configured Qdrant collection uses named vectors; "
                "this application expects one unnamed vector."
            )
        vector_config = next(iter(vector_config.values()))
    return getattr(vector_config, "size", None)


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

    backend = _embedding_backend()
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
                        "embedding_backend": backend,
                        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                        "embedding_provider": (
                            os.getenv("HF_INFERENCE_PROVIDER", "hf-inference")
                            if backend == "hosted"
                            else "local"
                        ),
                    }
                )

    if not chunk_texts:
        raise RuntimeError("The PDFs contain no extractable text to index.")

    if backend == "hosted":
        vectors = np.stack([_hosted_embedding(text) for text in chunk_texts])
    else:
        encoded = embedder.encode(chunk_texts, normalize_embeddings=True)
        encoded = np.asarray(encoded, dtype=np.float32)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        vectors = np.stack([_sentence_embedding(vector) for vector in encoded])

    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Document embeddings have shape {vectors.shape}; expected "
            f"(number of chunks, {EMBEDDING_DIMENSION})."
        )
    client = _qdrant_client()
    _ensure_collection(client, int(vectors.shape[1]))
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    collection_dimension = _collection_vector_dimension(client, collection)
    if collection_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Qdrant collection '{collection}' uses {collection_dimension} dimensions, "
            f"but document embeddings use {EMBEDDING_DIMENSION}. No points were deleted."
        )
    if backend == "hosted" and client.count(collection_name=collection, exact=True).count:
        raise RuntimeError(
            f"Refusing to mix hosted embeddings into non-empty collection '{collection}'. "
            "Choose a new QDRANT_COLLECTION and re-index all documents with hosted embeddings."
        )
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


def answer_question(
    question: str,
    context_entries: list[dict[str, Any]] | None = None,
) -> str:
    if context_entries is None:
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
    backend = _embedding_backend()
    query_vector = _query_embedding(question)
    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    if not client.collection_exists(collection):
        raise RuntimeError("The Qdrant collection is empty. Run `docs-qna-rag ingest` first.")

    collection_dimension = _collection_vector_dimension(client, collection)
    if collection_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Qdrant collection '{collection}' uses {collection_dimension} dimensions, "
            f"but {DEFAULT_EMBEDDING_MODEL} produces {EMBEDDING_DIMENSION}. "
            "Use a collection indexed with the configured embedding model."
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    ).points
    if not results:
        raise RuntimeError("No matching document chunks were found.")

    entries: list[dict[str, Any]] = []
    for point in results:
        payload = point.payload or {}
        indexed_backend = payload.get("embedding_backend")
        indexed_model = payload.get("embedding_model")
        indexed_provider = payload.get("embedding_provider")
        expected_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        expected_provider = (
            os.getenv("HF_INFERENCE_PROVIDER", "hf-inference")
            if backend == "hosted"
            else "local"
        )
        if backend == "hosted" and (
            indexed_backend != "hosted"
            or indexed_model != expected_model
            or indexed_provider != expected_provider
        ):
            raise RuntimeError(
                "Hosted query embeddings cannot be used with these indexed vectors. "
                "Re-index all documents into a new empty Qdrant collection with "
                "EMBEDDING_BACKEND=hosted and the same HF_INFERENCE_PROVIDER."
            )
        if indexed_backend and (
            indexed_backend != backend
            or indexed_model != expected_model
            or indexed_provider != expected_provider
        ):
            raise RuntimeError(
                "The retrieved Qdrant vectors were indexed with a different embedding "
                "backend or model. Use a collection indexed with the active configuration."
            )
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
