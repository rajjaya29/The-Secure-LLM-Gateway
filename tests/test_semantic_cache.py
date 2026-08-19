"""Unit tests for ChromaDB Semantic Vector Cache with all-MiniLM-L6-v2."""

import pytest
import numpy as np
from app.cache.embeddings import EmbeddingEngine
from app.cache.chroma_store import ChromaVectorStore
from app.cache.semantic_cache import SemanticCacheManager
from app.schemas.openai import ChatCompletionRequest, ChatMessage


@pytest.mark.asyncio
async def test_semantic_cache_hit_miss_and_isolation():
    store = ChromaVectorStore(collection_name="test_semantic_cache_suite", isolate_by_api_key=True)
    store.clear()
    engine = EmbeddingEngine()
    # Warmup forward pass
    engine.embed_query("warmup forward pass")
    manager = SemanticCacheManager(embedding_engine=engine, vector_store=store, similarity_threshold=0.90)

    req1 = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="What is the capital of France?")],
    )

    # 1. First lookup: MISS
    res1 = await manager.lookup(req1, api_key_hash="key_alice")
    assert res1.hit is False

    # 2. Store response under Alice's key
    resp_payload = {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "The capital is Paris."}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    await manager.store(req1, resp_payload, api_key_hash="key_alice", token_count=10)

    # 3. Exact repeated query for Alice: HIT (<25ms)
    res2 = await manager.lookup(req1, api_key_hash="key_alice")
    assert res2.hit is True
    assert res2.similarity >= 0.95
    assert res2.lookup_latency_ms < 25.0

    # 4. Semantically similar query for Alice: HIT
    req_paraphrase = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Can you tell me the capital city of France?")],
    )
    res3 = await manager.lookup(req_paraphrase, api_key_hash="key_alice")
    assert res3.hit is True
    assert res3.similarity >= 0.90

    # Repeat paraphrase lookup to verify warmed cache lookup is fast
    res3_repeat = await manager.lookup(req_paraphrase, api_key_hash="key_alice")
    assert res3_repeat.hit is True
    assert res3_repeat.lookup_latency_ms < 25.0

    # 5. Unrelated query for Alice: MISS
    req_unrelated = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Explain quantum superposition in physics.")],
    )
    res4 = await manager.lookup(req_unrelated, api_key_hash="key_alice")
    assert res4.hit is False

    # 6. Tenant Isolation: Bob querying the same query MUST NOT hit Alice's cache
    res_bob = await manager.lookup(req1, api_key_hash="key_bob")
    assert res_bob.hit is False  # Isolated per API key!
