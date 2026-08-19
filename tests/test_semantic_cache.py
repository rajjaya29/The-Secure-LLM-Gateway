"""Unit tests for Semantic Vector Caching, embeddings, and similarity matching."""

import pytest
import numpy as np
import asyncio
from app.cache.embeddings import EmbeddingEngine
from app.cache.vector_store import InMemoryVectorStore
from app.cache.semantic_cache import SemanticCacheManager
from app.schemas.openai import ChatCompletionRequest, ChatMessage


def test_embedding_engine_normalization():
    engine = EmbeddingEngine()
    v1 = engine.embed_query("What is the capital of France?")
    
    assert isinstance(v1, np.ndarray)
    assert len(v1) == engine.vector_dim
    norm = np.linalg.norm(v1)
    assert pytest.approx(norm, 0.01) == 1.0


def test_vector_store_exact_and_similarity_search():
    store = InMemoryVectorStore(max_entries=100)
    engine = EmbeddingEngine()

    p1 = "What is the capital of France?"
    v1 = engine.embed_query(p1)
    resp_payload = {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}

    store.upsert(
        prompt=p1,
        embedding=v1,
        response_payload=resp_payload,
        model="gpt-4o",
        system_hash="sys_default",
    )

    exact = store.get_by_exact_match(p1, model="gpt-4o", system_hash="sys_default")
    assert exact is not None
    assert exact.response_payload["choices"][0]["message"]["content"] == "Paris"

    matches = store.search(v1, model="gpt-4o", system_hash="sys_default", threshold=0.92)
    assert len(matches) == 1
    assert matches[0][1] >= 0.99


@pytest.mark.asyncio
async def test_semantic_cache_manager_hit_and_miss():
    store = InMemoryVectorStore(max_entries=100)
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

    res1 = await manager.lookup(req1)
    assert res1.hit is False

    mock_resp = {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "The capital is Paris."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    await manager.store(req1, mock_resp, token_count=20)

    res2 = await manager.lookup(req1)
    assert res2.hit is True
    assert res2.similarity >= 0.99
    assert res2.cached_response["choices"][0]["message"]["content"] == "The capital is Paris."
    assert res2.lookup_latency_ms < 50.0
