from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from backend.main import ingest_documents
import docs_qna_rag


class SentenceEmbeddingTests(unittest.TestCase):
    def test_hosted_client_uses_token_and_provider_from_environment(self) -> None:
        from huggingface_hub import InferenceClient

        docs_qna_rag._hosted_inference_client.cache_clear()
        with (
            patch.dict(
                os.environ,
                {"HF_TOKEN": "test-token", "HF_INFERENCE_PROVIDER": "hf-inference"},
            ),
            patch("huggingface_hub.InferenceClient", wraps=InferenceClient) as client_factory,
        ):
            docs_qna_rag._hosted_inference_client()

        client_factory.assert_called_once_with(
            model=docs_qna_rag.DEFAULT_EMBEDDING_MODEL,
            provider="hf-inference",
            token="test-token",
            timeout=30,
        )
        docs_qna_rag._hosted_inference_client.cache_clear()

    def test_mean_pools_token_vectors_and_l2_normalizes(self) -> None:
        token_vectors = np.zeros((2, docs_qna_rag.EMBEDDING_DIMENSION), dtype=np.float32)
        token_vectors[0, 0] = 3.0
        token_vectors[1, 1] = 4.0

        embedding = docs_qna_rag._sentence_embedding(token_vectors)

        self.assertEqual(embedding.shape, (384,))
        np.testing.assert_allclose(embedding[:2], [0.6, 0.8], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_normalizes_sentence_vector(self) -> None:
        vector = np.zeros(docs_qna_rag.EMBEDDING_DIMENSION, dtype=np.float32)
        vector[0] = 5.0
        vector[1] = 12.0

        embedding = docs_qna_rag._sentence_embedding(vector)

        np.testing.assert_allclose(embedding[:2], [5 / 13, 12 / 13], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_rejects_wrong_embedding_dimension(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 384"):
            docs_qna_rag._sentence_embedding(np.ones(12, dtype=np.float32))

    def test_rejects_zero_vector(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "zero-length"):
            docs_qna_rag._sentence_embedding(np.zeros(384, dtype=np.float32))

    def test_hosted_retrieval_uses_mean_pooled_query_without_local_model(self) -> None:
        token_vectors = np.zeros((1, 2, docs_qna_rag.EMBEDDING_DIMENSION), dtype=np.float32)
        token_vectors[0, 0, 0] = 3.0
        token_vectors[0, 1, 1] = 4.0
        inference_client = SimpleNamespace(
            feature_extraction=lambda text: token_vectors
        )
        qdrant_client = _FakeQdrantClient()

        with (
            patch.dict(os.environ, {"EMBEDDING_BACKEND": "hosted"}),
            patch.object(docs_qna_rag, "_hosted_inference_client", return_value=inference_client),
            patch.object(docs_qna_rag, "_qdrant_client", return_value=qdrant_client),
            patch.object(
                docs_qna_rag,
                "_embedder",
                side_effect=AssertionError("hosted queries must not load SentenceTransformer"),
            ),
        ):
            entries = docs_qna_rag.retrieve_context("what is a network?")

        self.assertEqual(entries[0]["source"], "networking.pdf")
        query_vector = qdrant_client.query_arguments["query"]
        self.assertEqual(len(query_vector), 384)
        np.testing.assert_allclose(query_vector[:2], [0.6, 0.8], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(query_vector)), 1.0, places=6)

    def test_hosted_retrieval_rejects_legacy_local_vectors(self) -> None:
        inference_client = SimpleNamespace(
            feature_extraction=lambda text: np.ones(docs_qna_rag.EMBEDDING_DIMENSION)
        )
        qdrant_client = _FakeQdrantClient(
            payload={"source": "networking.pdf", "page": 3, "text": "Legacy local vector."}
        )

        with (
            patch.dict(os.environ, {"EMBEDDING_BACKEND": "hosted"}),
            patch.object(docs_qna_rag, "_hosted_inference_client", return_value=inference_client),
            patch.object(docs_qna_rag, "_qdrant_client", return_value=qdrant_client),
        ):
            with self.assertRaisesRegex(RuntimeError, "Re-index all documents"):
                docs_qna_rag.retrieve_context("what is a network?")


class IngestionApiSecurityTests(unittest.TestCase):
    def test_ingestion_api_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {"ENABLE_INGEST_API": "false"}):
            with self.assertRaises(HTTPException) as raised:
                ingest_documents()

        self.assertEqual(raised.exception.status_code, 404)

    def test_ingestion_api_requires_its_secret_token(self) -> None:
        with patch.dict(
            os.environ,
            {"ENABLE_INGEST_API": "true", "INGEST_API_KEY": "unit-test-secret"},
        ):
            with self.assertRaises(HTTPException) as raised:
                ingest_documents(x_ingest_token="wrong-token")

        self.assertEqual(raised.exception.status_code, 401)


class _FakeQdrantClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.query_arguments: dict[str, object] = {}
        self.payload = payload or {
            "source": "networking.pdf",
            "page": 3,
            "text": "A network connects devices.",
            "embedding_backend": "hosted",
            "embedding_model": docs_qna_rag.DEFAULT_EMBEDDING_MODEL,
            "embedding_provider": "hf-inference",
        }

    def collection_exists(self, collection: str) -> bool:
        return True

    def get_collection(self, collection: str) -> SimpleNamespace:
        vectors = SimpleNamespace(size=docs_qna_rag.EMBEDDING_DIMENSION)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.query_arguments = kwargs
        point = SimpleNamespace(payload=self.payload)
        return SimpleNamespace(points=[point])


if __name__ == "__main__":
    unittest.main()
