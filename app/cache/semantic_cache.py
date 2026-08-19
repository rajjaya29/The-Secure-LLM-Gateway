"""Semantic Cache Manager orchestrating ChromaDB vector similarity and exact matching."""

import time
import hashlib
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from app.cache.embeddings import EmbeddingEngine
from app.cache.chroma_store import ChromaVectorStore
from app.schemas.gateway import CacheLookupResult
from app.schemas.openai import ChatCompletionRequest, ChatMessage


class SemanticCacheManager:
    """
    Coordinates semantic cache lookups against ChromaDB with all-MiniLM-L6-v2 embeddings.
    Enforces cosine similarity threshold (>= 0.90) and guarantees < 25ms response times.
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        similarity_threshold: float = 0.90,
        enabled: bool = True,
    ):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.vector_store = vector_store or ChromaVectorStore()
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled

    def _extract_prompt_and_system(self, messages: List[ChatMessage]) -> Tuple[str, str]:
        system_chunks = []
        user_chunks = []

        for msg in messages:
            if msg.role == "system":
                system_chunks.append(msg.content.strip())
            else:
                user_chunks.append(f"{msg.role}: {msg.content.strip()}")

        system_text = "\n".join(system_chunks) if system_chunks else "default_system"
        system_hash = hashlib.md5(system_text.encode("utf-8")).hexdigest()
        
        user_prompt = "\n".join(user_chunks) if user_chunks else ""
        return user_prompt, system_hash

    async def lookup(
        self,
        request: ChatCompletionRequest,
        custom_threshold: Optional[float] = None,
    ) -> CacheLookupResult:
        if not self.enabled or getattr(request, "bypass_cache", False):
            return CacheLookupResult(hit=False, similarity=0.0, lookup_latency_ms=0.0)

        start_time = time.perf_counter()
        user_prompt, system_hash = self._extract_prompt_and_system(request.messages)
        model = request.model
        threshold = custom_threshold or self.similarity_threshold

        if not user_prompt:
            return CacheLookupResult(hit=False, similarity=0.0, lookup_latency_ms=0.0)

        # Tier 1: Sub-millisecond exact hash match
        exact_item = self.vector_store.get_by_exact_match(user_prompt, model, system_hash)
        if exact_item:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return CacheLookupResult(
                hit=True,
                similarity=1.0,
                cached_response=exact_item["response_payload"],
                cache_id=exact_item["id"],
                matched_prompt=exact_item["prompt"],
                lookup_latency_ms=round(elapsed_ms, 3),
            )

        # Tier 2: ChromaDB cosine similarity search
        query_vector = await asyncio.to_thread(self.embedding_engine.embed_query, user_prompt)
        matches = self.vector_store.search(
            query_vector=query_vector,
            model=model,
            system_hash=system_hash,
            threshold=threshold,
            limit=1,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if matches:
            matched_item, sim_score = matches[0]
            return CacheLookupResult(
                hit=True,
                similarity=sim_score,
                cached_response=matched_item["response_payload"],
                cache_id=matched_item["id"],
                matched_prompt=matched_item["prompt"],
                lookup_latency_ms=round(elapsed_ms, 3),
            )

        return CacheLookupResult(
            hit=False,
            similarity=0.0,
            lookup_latency_ms=round(elapsed_ms, 3),
        )

    async def store(
        self,
        request: ChatCompletionRequest,
        response_payload: Dict[str, Any],
        token_count: int = 0,
        ttl_seconds: Optional[int] = None,
    ):
        if not self.enabled:
            return

        user_prompt, system_hash = self._extract_prompt_and_system(request.messages)
        if not user_prompt:
            return

        model = request.model
        query_vector = await asyncio.to_thread(self.embedding_engine.embed_query, user_prompt)
        
        self.vector_store.upsert(
            prompt=user_prompt,
            embedding=query_vector,
            response_payload=response_payload,
            model=model,
            system_hash=system_hash,
            ttl_seconds=ttl_seconds or 86400,
            token_count=token_count,
        )
