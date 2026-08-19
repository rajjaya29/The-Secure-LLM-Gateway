"""Unit tests for ChromaDB Semantic Vector Caching and all-MiniLM-L6-v2 embeddings."""

import pytest
import numpy as np
import asyncio
from app.cache.embeddings import EmbeddingEngine
from app.cache.chroma_store import ChromaVectorStore
from app.cache.semantic_cache import SemanticCacheManager
from app.schemas.openai import ChatCompletionRequest, ChatMessage


def test_embedding_engine_normalization():
    engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v1 = engine.embed_query("What is the capital of France?")
    
    assert isinstance(v1, np.ndarray)
    assert len(v1) == engine.vector_dim
    norm = np.linalg.norm(v1)
    assert pytest.approx(norm, 0.01) == 1.0


def test_chroma_store_exact_and_similarity_search():
    store = ChromaVectorStore(collection_name="test_collection", persist_directory=None)
    store.clear()
    engine = EmbeddingEngine()

    p1 = "What is the capital of France?"
    v1 = engine.embed_query(p1)
    resp_payload = {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}

    # Upsert item into ChromaDB
    doc_id = store.upsert(
        prompt=p1,
        embedding=v1,
        response_payload=resp_payload,
        model="gpt-4o",
        system_hash="sys_default",
    )
    assert doc_id.startswith("chroma-")

    # 1. Exact match lookup
    exact = store.get_by_exact_match(p1, model="gpt-4o", system_hash="sys_default")
    assert exact is not None
    assert exact["response_payload"]["choices"][0]["message"]["content"] == "Paris"

    # 2. ChromaDB search with cosine similarity
    matches = store.search(v1, model="gpt-4o", system_hash="sys_default", threshold=0.90)
    assert len(matches) == 1
    assert matches[0][1] >= 0.95


@pytest.mark.asyncio
async def test_semantic_cache_manager_chroma_hit_and_miss():
    store = ChromaVectorStore(collection_name="test_cache_mgr", persist_directory=None)
    store.clear()
    engine = EmbeddingEngine()
    manager = SemanticCacheManager(
        embedding_engine=engine,
        vector_store=store,
        similarity_threshold=0.90,
        enabled=True,
    )

    req1 = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="What is the capital city of France?")],
    )

    # First lookup should be MISS
    res1 = await manager.lookup(req1)
    assert res1.hit is False

    # Store response
    mock_resp = {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "The capital is Paris."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    await manager.store(req1, mock_resp, token_count=20)

    # Second lookup with exact prompt should be HIT
    res2 = await manager.lookup(req1)
    assert res2.hit is True
    assert res2.similarity >= 0.95
    assert res2.cached_response["choices"][0]["message"]["content"] == "The capital is Paris."
    assert res2.lookup_latency_ms < 25.0  # < 25ms requirement
