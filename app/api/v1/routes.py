"""API v1 Endpoints for The Secure LLM Gateway."""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, Request, Response, HTTPException, Header, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelList,
    ModelCard,
    ModelPermission,
)
from app.schemas.gateway import (
    GuardrailTestRequest,
    GuardrailTestResponse,
)
from app.resilience.auth import verify_api_key, hash_api_key
from app.resilience.rate_limiter import SlidingWindowRateLimiter
from app.guardrails.prompt_validator import PromptValidator
from app.guardrails.pii_scrubber import PIIScrubber
from app.guardrails.output_guardrail import OutputGuardrail
from app.cache.semantic_cache import SemanticCacheManager
from app.cache.chroma_store import ChromaVectorStore
from app.cache.embeddings import EmbeddingEngine
from app.router.llm_router import LLMRouter
from app.router.providers import MockLLMProvider, OpenAIProvider, AnthropicProvider, OllamaProvider
from app.observability.metrics import metrics
from app.observability.logging import get_logger
from app.observability.database import sqlite_logger

logger = get_logger()
router = APIRouter()

# Singletons
vector_store = ChromaVectorStore(
    collection_name="semantic_cache",
    persist_directory=settings.CHROMA_PERSIST_DIR,
    max_entries=10000,
    isolate_by_api_key=settings.CACHE_ISOLATE_BY_API_KEY,
)
embedding_engine = EmbeddingEngine(model_name=settings.EMBEDDING_MODEL)
cache_manager = SemanticCacheManager(
    embedding_engine=embedding_engine,
    vector_store=vector_store,
    similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
    enabled=settings.ENABLE_SEMANTIC_CACHE,
)

prompt_validator = PromptValidator(enabled=settings.ENABLE_PROMPT_VALIDATION)
pii_scrubber = PIIScrubber(mask_style="tokenized")
output_guardrail = OutputGuardrail(enable_leak_prevention=True, pii_scrubber=pii_scrubber)

rate_limiter = SlidingWindowRateLimiter(
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    max_requests=settings.RATE_LIMIT_REQUESTS,
    enabled=settings.ENABLE_RATE_LIMITING,
)

providers_dict = {
    "mock": MockLLMProvider(name="mock", simulated_latency_ms=settings.MOCK_LLM_LATENCY_MS),
    "openai": OpenAIProvider(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL),
    "anthropic": AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_BASE_URL),
    "ollama": OllamaProvider(base_url=settings.OLLAMA_BASE_URL, default_model=settings.OLLAMA_MODEL),
}

llm_router = LLMRouter(
    providers=providers_dict,
    provider_priority=["mock", "openai", "ollama", "anthropic"],
)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    auth_info: Tuple[str, str] = Depends(verify_api_key),
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_similarity_threshold: Optional[float] = Header(default=None, alias="X-Similarity-Threshold"),
):
    """
    Authenticated, rate-limited LLM API Proxy with ChromaDB Semantic Caching:
    1. X-API-Key Authentication
    2. In-Memory Sliding-Window Rate Limiting
    3. Custom Prompt Validation (Security Filtering)
    4. ChromaDB Semantic Vector Cache (all-MiniLM-L6-v2, cosine sim >= 0.90 -> < 25ms HIT)
    5. Upstream Provider Routing with Fallback on Miss
    6. Structured SQLite Request Logging (Safe API Key Hash)
    """
    t_start = time.perf_counter()
    raw_api_key, api_key_hash = auth_info
    request_id = str(uuid.uuid4())

    # 1. In-Memory Sliding-Window Rate Limiting per API key
    allowed, retry_after, remaining = rate_limiter.check_limit(api_key_hash)
    response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(settings.RATE_LIMIT_WINDOW_SECONDS)

    if not allowed:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        await sqlite_logger.log_request(
            request_id=request_id,
            api_key_hash=api_key_hash,
            model=chat_request.model,
            prompt_length=0,
            response_length=0,
            latency_ms=elapsed_ms,
            cache_hit=False,
            similarity=0.0,
            provider="rate_limiter",
            status_code=429,
            error="Rate limit exceeded",
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.RATE_LIMIT_REQUESTS} req/{settings.RATE_LIMIT_WINDOW_SECONDS}s). Please retry after {retry_after}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    # 2. Extract & Validate Prompt
    user_prompt = ""
    for msg in chat_request.messages:
        if msg.role == "user":
            user_prompt = msg.content
            break

    prompt_length = len(user_prompt)

    # 3. Prompt Validation
    if settings.ENABLE_PROMPT_VALIDATION and user_prompt and not chat_request.bypass_guardrails:
        val_res = prompt_validator.validate(user_prompt)
        if val_res.blocked:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            metrics.record_guardrail_block(reason=val_res.threat_type or "prompt_injection")
            
            await sqlite_logger.log_request(
                request_id=request_id,
                api_key_hash=api_key_hash,
                model=chat_request.model,
                prompt_length=prompt_length,
                response_length=0,
                latency_ms=elapsed_ms,
                cache_hit=False,
                similarity=0.0,
                provider="prompt_validator",
                injection_blocked=True,
                status_code=400,
                error=val_res.reason,
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Prompt Rejected by Security Validator",
                    "reason": val_res.reason,
                    "threat_type": val_res.threat_type,
                    "score": val_res.score,
                },
            )

    # 4. PII Scrubbing
    scrubbed_messages = []
    pii_count = 0
    if settings.ENABLE_PII_SCRUBBING:
        for msg in chat_request.messages:
            if msg.role == "user":
                clean_text, detected_pii, _ = pii_scrubber.scrub(msg.content)
                pii_count += len(detected_pii)
                scrubbed_messages.append(ChatMessage(role=msg.role, content=clean_text, name=msg.name))
            else:
                scrubbed_messages.append(msg)
    else:
        scrubbed_messages = chat_request.messages

    sanitized_request = chat_request.model_copy(update={"messages": scrubbed_messages})

    # 5. ChromaDB Semantic Vector Cache Lookup (Cosine Similarity >= 0.90)
    cache_threshold = x_similarity_threshold or settings.SEMANTIC_SIMILARITY_THRESHOLD
    cache_res = await cache_manager.lookup(
        sanitized_request,
        api_key_hash=api_key_hash,
        custom_threshold=cache_threshold,
    )
    metrics.record_cache_lookup_latency(cache_res.lookup_latency_ms)

    if cache_res.hit and cache_res.cached_response:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        cached_dict = cache_res.cached_response

        completion_resp = ChatCompletionResponse(**cached_dict)
        completion_resp.id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"

        response.headers["X-Cache-Status"] = "HIT"
        response.headers["X-Cache-Similarity"] = str(cache_res.similarity)
        response.headers["X-Cache-Lookup-Ms"] = str(cache_res.lookup_latency_ms)
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Provider-Used"] = "chroma-semantic-cache"
        response.headers["X-PII-Entities-Scrubbed"] = str(pii_count)

        metrics.record_request(
            status="200",
            model=chat_request.model,
            cache_hit=True,
            e2e_latency_ms=elapsed_ms,
            prompt_tokens=completion_resp.usage.prompt_tokens,
            completion_tokens=completion_resp.usage.completion_tokens,
        )

        resp_length = len(completion_resp.choices[0].message.content)
        background_tasks.add_task(
            sqlite_logger.log_request,
            request_id=request_id,
            api_key_hash=api_key_hash,
            model=chat_request.model,
            prompt_length=prompt_length,
            response_length=resp_length,
            latency_ms=elapsed_ms,
            cache_hit=True,
            similarity=cache_res.similarity,
            provider="chroma-semantic-cache",
            prompt_tokens=completion_resp.usage.prompt_tokens,
            completion_tokens=completion_resp.usage.completion_tokens,
            pii_scrubbed_count=pii_count,
            status_code=200,
        )

        return completion_resp

    # 6. Cache MISS -> Call Upstream Provider
    preferred = x_preferred_provider or settings.DEFAULT_PROVIDER
    try:
        completion_resp, provider_used, attempted_chain = await llm_router.route_and_generate(
            sanitized_request, preferred_provider=preferred
        )
    except Exception as ex:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        background_tasks.add_task(
            sqlite_logger.log_request,
            request_id=request_id,
            api_key_hash=api_key_hash,
            model=chat_request.model,
            prompt_length=prompt_length,
            response_length=0,
            latency_ms=elapsed_ms,
            cache_hit=False,
            similarity=0.0,
            provider=preferred,
            status_code=502,
            error=str(ex),
        )
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {ex}")

    # 7. Output Guardrail
    if completion_resp.choices:
        orig = completion_resp.choices[0].message.content
        clean, _, _ = output_guardrail.verify_and_clean(orig)
        completion_resp.choices[0].message.content = clean

    # 8. Asynchronous Cache Storage & SQLite Audit Logging
    background_tasks.add_task(
        cache_manager.store,
        sanitized_request,
        completion_resp.model_dump(),
        api_key_hash=api_key_hash,
        token_count=completion_resp.usage.total_tokens,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    response.headers["X-Cache-Status"] = "MISS"
    response.headers["X-Cache-Similarity"] = "0.0"
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Provider-Used"] = provider_used
    response.headers["X-PII-Entities-Scrubbed"] = str(pii_count)

    metrics.record_request(
        status="200",
        model=chat_request.model,
        cache_hit=False,
        e2e_latency_ms=elapsed_ms,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
    )

    resp_length = len(completion_resp.choices[0].message.content)
    background_tasks.add_task(
        sqlite_logger.log_request,
        request_id=request_id,
        api_key_hash=api_key_hash,
        model=chat_request.model,
        prompt_length=prompt_length,
        response_length=resp_length,
        latency_ms=elapsed_ms,
        cache_hit=False,
        similarity=0.0,
        provider=provider_used,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
        pii_scrubbed_count=pii_count,
        status_code=200,
    )

    return completion_resp


@router.get("/models", response_model=ModelList)
async def list_models():
    model_ids = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "llama3", "mock-model"]
    cards = [
        ModelCard(
            id=m_id,
            permission=[ModelPermission(id=f"perm-{m_id}")],
        )
        for m_id in model_ids
    ]
    return ModelList(data=cards)


@router.get("/gateway/stats")
@router.get("/stats")
async def get_stats():
    """Live analytics metrics endpoint tracking cache-hit ratios, latency distributions, and per-key usage."""
    return await sqlite_logger.get_stats()


@router.delete("/gateway/cache")
async def clear_cache():
    vector_store.clear()
    return {"status": "success", "message": "Semantic cache successfully cleared", "current_size": vector_store.size}
