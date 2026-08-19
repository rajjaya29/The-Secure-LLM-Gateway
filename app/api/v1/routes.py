"""API v1 Endpoints for The Secure LLM Gateway."""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, Response, HTTPException, Header, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

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
    GatewayStatsResponse,
    GuardrailTestRequest,
    GuardrailTestResponse,
    GuardrailThreat,
)
from app.resilience.auth import verify_api_key
from app.resilience.rate_limiter import SlidingWindowRateLimiter
from app.guardrails.injection_detector import InjectionDetector
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
    persist_directory=None,  # In-memory ChromaDB for fast test/runtime execution
    max_entries=settings.CACHE_MAX_ENTRIES,
)
embedding_engine = EmbeddingEngine(model_name=settings.EMBEDDING_MODEL)
cache_manager = SemanticCacheManager(
    embedding_engine=embedding_engine,
    vector_store=vector_store,
    similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
    enabled=settings.ENABLE_SEMANTIC_CACHE,
)

injection_detector = InjectionDetector(
    confidence_threshold=settings.INJECTION_CONFIDENCE_THRESHOLD,
    block_on_detection=settings.GUARDRAIL_BLOCK_INJECTIONS,
)
pii_scrubber = PIIScrubber(mask_style=settings.PII_MASK_STYLE)
output_guardrail = OutputGuardrail(
    enable_leak_prevention=settings.OUTPUT_LEAK_PREVENTION,
    pii_scrubber=pii_scrubber,
)

rate_limiter = SlidingWindowRateLimiter(
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    enabled=settings.ENABLE_RATE_LIMITING,
)

providers_dict = {
    "mock": MockLLMProvider(name="mock", simulated_latency_ms=80.0),
    "openai": OpenAIProvider(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL),
    "anthropic": AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_BASE_URL),
    "ollama": OllamaProvider(base_url=settings.OLLAMA_BASE_URL, default_model=settings.OLLAMA_MODEL),
}

llm_router = LLMRouter(
    providers=providers_dict,
    provider_priority=settings.PROVIDER_PRIORITY,
    max_retries_per_provider=settings.MAX_PROVIDER_RETRIES,
    failure_threshold=settings.CIRCUIT_BREAKER_FAIL_THRESHOLD,
    recovery_seconds=settings.CIRCUIT_BREAKER_RECOVERY_SECONDS,
)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_similarity_threshold: Optional[float] = Header(default=None, alias="X-Similarity-Threshold"),
):
    """
    Authenticated, rate-limited LLM API Proxy with ChromaDB Semantic Caching:
    1. X-API-Key Authentication
    2. In-Memory Sliding-Window Rate Limiting per API Key
    3. Input Prompt Validation & Guardrails (Prompt Injection + PII Scrubbing)
    4. ChromaDB Semantic Vector Cache (all-MiniLM-L6-v2, cosine sim >= 0.90 -> < 25ms HIT)
    5. Upstream Provider Routing with Fallback
    6. Output Verification & Asynchronous SQLite Request Logging
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    # 1. In-Memory Sliding-Window Rate Limiting Check per API key
    allowed, wait_seconds = rate_limiter.check_limit(api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.RATE_LIMIT_MAX_REQUESTS} req/{settings.RATE_LIMIT_WINDOW_SECONDS}s). Please retry after {wait_seconds} seconds.",
            headers={"Retry-After": str(int(wait_seconds) + 1)},
        )

    # 2. Input Prompt Validation & Guardrails
    scrubbed_messages: List[ChatMessage] = []
    pii_entities_count = 0
    token_mapping_all: Dict[str, str] = {}
    raw_user_prompt = ""

    for msg in chat_request.messages:
        content = msg.content
        if msg.role == "user":
            raw_user_prompt = content
            if not chat_request.bypass_guardrails:
                # 2a. Prompt Injection Check
                if settings.ENABLE_INJECTION_GUARDRAIL:
                    guard_res = injection_detector.detect(content)
                    if guard_res.blocked:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        metrics.record_guardrail_block(reason="prompt_injection")
                        
                        # Asynchronously log blocked attack to SQLite
                        background_tasks.add_task(
                            sqlite_logger.log_request,
                            api_key=api_key,
                            model=chat_request.model,
                            prompt=raw_user_prompt,
                            response="BLOCKED_BY_GUARDRAIL",
                            latency_ms=elapsed_ms,
                            cache_hit=False,
                            similarity=0.0,
                            provider="guardrail",
                            injection_blocked=True,
                            status_code=400,
                        )

                        logger.warning(f"Blocked prompt injection from API key {api_key}: score={guard_res.injection_score}")
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "Prompt Rejected by Security Guardrail",
                                "reason": "Prompt Injection / Jailbreak vector detected",
                                "injection_score": guard_res.injection_score,
                                "threats": [t.model_dump() for t in guard_res.threats],
                            },
                        )

                # 2b. PII Scrubbing
                if settings.ENABLE_PII_SCRUBBING:
                    sanitized_text, detected_pii, t_map = pii_scrubber.scrub(content)
                    if detected_pii:
                        pii_entities_count += len(detected_pii)
                        token_mapping_all.update(t_map)
                        for entity in detected_pii:
                            metrics.record_pii_scrubbed(entity["type"])
                    content = sanitized_text

        scrubbed_messages.append(ChatMessage(role=msg.role, content=content, name=msg.name))

    sanitized_request = chat_request.model_copy(update={"messages": scrubbed_messages})

    # 3. ChromaDB Semantic Vector Cache Lookup (Cosine Similarity >= 0.90)
    cache_threshold = x_similarity_threshold or settings.CACHE_SIMILARITY_THRESHOLD
    cache_res = await cache_manager.lookup(sanitized_request, custom_threshold=cache_threshold)
    metrics.record_cache_lookup_latency(cache_res.lookup_latency_ms)

    if cache_res.hit and cache_res.cached_response:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        cached_dict = cache_res.cached_response
        
        completion_resp = ChatCompletionResponse(**cached_dict)
        completion_resp.id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"

        # Set Telemetry Headers
        response.headers["X-Cache-Status"] = "HIT"
        response.headers["X-Cache-Similarity"] = str(cache_res.similarity)
        response.headers["X-Cache-Lookup-Ms"] = str(cache_res.lookup_latency_ms)
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Provider-Used"] = "chroma-semantic-cache"
        response.headers["X-PII-Entities-Scrubbed"] = str(pii_entities_count)

        # Record Metrics & Structured SQLite Log
        metrics.record_request(
            status="200",
            model=chat_request.model,
            cache_hit=True,
            e2e_latency_ms=elapsed_ms,
            prompt_tokens=completion_resp.usage.prompt_tokens,
            completion_tokens=completion_resp.usage.completion_tokens,
        )

        background_tasks.add_task(
            sqlite_logger.log_request,
            api_key=api_key,
            model=chat_request.model,
            prompt=raw_user_prompt,
            response=completion_resp.choices[0].message.content,
            latency_ms=elapsed_ms,
            cache_hit=True,
            similarity=cache_res.similarity,
            provider="chroma-semantic-cache",
            prompt_tokens=completion_resp.usage.prompt_tokens,
            completion_tokens=completion_resp.usage.completion_tokens,
            pii_scrubbed_count=pii_entities_count,
            injection_blocked=False,
            status_code=200,
        )

        return completion_resp

    # 4. Cache MISS -> Fallback Multi-Provider LLM Routing
    preferred = x_preferred_provider or settings.DEFAULT_PROVIDER
    t_route_start = time.perf_counter()
    try:
        completion_resp, provider_used, attempted_chain = await llm_router.route_and_generate(
            sanitized_request, preferred_provider=preferred
        )
        route_latency_ms = (time.perf_counter() - t_route_start) * 1000
        metrics.record_provider_latency(route_latency_ms)
    except Exception as ex:
        logger.error(f"Provider routing failure: {ex}")
        raise HTTPException(status_code=502, detail=f"All upstream LLM providers failed: {ex}")

    # 5. Output Guardrail Verification
    if settings.ENABLE_OUTPUT_GUARDRAIL and completion_resp.choices:
        orig_reply = completion_resp.choices[0].message.content
        cleaned_reply, is_safe, violations = output_guardrail.verify_and_clean(orig_reply)
        completion_resp.choices[0].message.content = cleaned_reply

    # 6. Asynchronous Background ChromaDB Storage & SQLite Logging
    background_tasks.add_task(
        cache_manager.store,
        sanitized_request,
        completion_resp.model_dump(),
        token_count=completion_resp.usage.total_tokens,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Cache-Status"] = "MISS"
    response.headers["X-Cache-Similarity"] = "0.0"
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Provider-Used"] = provider_used
    response.headers["X-Fallback-Chain"] = " -> ".join(attempted_chain)
    response.headers["X-PII-Entities-Scrubbed"] = str(pii_entities_count)

    metrics.record_request(
        status="200",
        model=chat_request.model,
        cache_hit=False,
        e2e_latency_ms=elapsed_ms,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
    )

    background_tasks.add_task(
        sqlite_logger.log_request,
        api_key=api_key,
        model=chat_request.model,
        prompt=raw_user_prompt,
        response=completion_resp.choices[0].message.content,
        latency_ms=elapsed_ms,
        cache_hit=False,
        similarity=0.0,
        provider=provider_used,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
        pii_scrubbed_count=pii_entities_count,
        injection_blocked=False,
        status_code=200,
    )

    return completion_resp


@router.get("/models", response_model=ModelList)
async def list_models():
    model_ids = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "llama3", "mock-llm"]
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
async def get_gateway_stats():
    """Returns live cache-hit ratios, latency distributions, and per-key usage from SQLite."""
    sqlite_stats = await sqlite_logger.get_analytics()
    provider_statuses = llm_router.get_provider_statuses()

    return {
        "uptime_seconds": metrics.get_summary_stats()["uptime_seconds"],
        "total_requests": sqlite_stats["total_requests"],
        "cache_hits": sqlite_stats["cache_hits"],
        "cache_misses": sqlite_stats["cache_misses"],
        "cache_hit_ratio": sqlite_stats["cache_hit_ratio"],
        "total_injections_blocked": sqlite_stats["total_injections_blocked"],
        "total_pii_entities_scrubbed": sqlite_stats["total_pii_entities_scrubbed"],
        "total_tokens_served": sqlite_stats["total_tokens_served"],
        "estimated_tokens_saved": sqlite_stats["estimated_tokens_saved"],
        "avg_cached_latency_ms": sqlite_stats["avg_cached_latency_ms"],
        "avg_upstream_latency_ms": sqlite_stats["avg_upstream_latency_ms"],
        "latency_distribution": sqlite_stats["latency_distribution"],
        "per_key_usage": sqlite_stats["per_key_usage"],
        "providers": [p.model_dump() for p in provider_statuses],
        "cache_size": vector_store.size,
        "cache_max_size": vector_store.max_entries,
    }


@router.post("/gateway/guardrails/test", response_model=GuardrailTestResponse)
async def test_guardrails(request: GuardrailTestRequest):
    t_start = time.perf_counter()
    sanitized = request.prompt
    pii_entities = []
    threats = []
    injection_score = 0.0
    blocked = False
    is_safe = True

    if request.check_pii:
        sanitized, pii_entities, _ = pii_scrubber.scrub(sanitized, mask_style=request.mask_style)

    if request.check_injection:
        res = injection_detector.detect(request.prompt)
        threats = res.threats
        injection_score = res.injection_score
        blocked = res.blocked
        is_safe = res.is_safe

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return GuardrailTestResponse(
        original_prompt=request.prompt,
        sanitized_prompt=sanitized,
        is_safe=is_safe,
        blocked=blocked,
        injection_score=injection_score,
        threats=threats,
        pii_entities=pii_entities,
        processing_time_ms=round(elapsed_ms, 3),
    )


@router.delete("/gateway/cache")
async def clear_cache():
    vector_store.clear()
    return {"status": "success", "message": "Semantic cache successfully cleared", "current_size": vector_store.size}
