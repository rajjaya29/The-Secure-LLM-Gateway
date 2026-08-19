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
from app.guardrails.injection_detector import InjectionDetector
from app.guardrails.pii_scrubber import PIIScrubber
from app.guardrails.output_guardrail import OutputGuardrail
from app.cache.semantic_cache import SemanticCacheManager
from app.cache.vector_store import InMemoryVectorStore
from app.cache.embeddings import EmbeddingEngine
from app.resilience.rate_limiter import TokenBucketRateLimiter
from app.router.llm_router import LLMRouter
from app.router.providers import MockLLMProvider, OpenAIProvider, AnthropicProvider, OllamaProvider
from app.observability.metrics import metrics
from app.observability.logging import get_logger

logger = get_logger()
router = APIRouter()

vector_store = InMemoryVectorStore(
    max_entries=settings.CACHE_MAX_ENTRIES,
    default_ttl_seconds=settings.CACHE_TTL_SECONDS,
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

rate_limiter = TokenBucketRateLimiter(
    default_rpm=settings.RATE_LIMIT_RPM,
    default_tpm=settings.RATE_LIMIT_TPM,
    enabled=settings.ENABLE_RATE_LIMITING,
)

providers_dict = {
    "mock": MockLLMProvider(name="mock"),
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


def get_client_identifier(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and len(auth_header) > 7:
        return auth_header[7:].strip()
    return request.client.host if request.client else "127.0.0.1"


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_similarity_threshold: Optional[float] = Header(default=None, alias="X-Similarity-Threshold"),
):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    client_id = get_client_identifier(raw_request)

    # 1. Rate Limiting Check
    estimated_prompt_tokens = sum(len(m.content.split()) * 2 for m in chat_request.messages)
    allowed, wait_seconds, limit_type = rate_limiter.check_limit(client_id, estimated_tokens=estimated_prompt_tokens)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {limit_type}. Please retry after {wait_seconds} seconds.",
            headers={"Retry-After": str(int(wait_seconds) + 1)},
        )

    # 2. Input Guardrails
    scrubbed_messages: List[ChatMessage] = []
    pii_entities_count = 0
    token_mapping_all: Dict[str, str] = {}

    for msg in chat_request.messages:
        content = msg.content
        if msg.role == "user" and not chat_request.bypass_guardrails:
            if settings.ENABLE_INJECTION_GUARDRAIL:
                guard_res = injection_detector.detect(content)
                if guard_res.blocked:
                    metrics.record_guardrail_block(reason="prompt_injection")
                    logger.warning(f"Blocked prompt injection attempt from client {client_id}: score={guard_res.injection_score}")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Prompt Rejected by Security Guardrail",
                            "reason": "Prompt Injection / Jailbreak vector detected",
                            "injection_score": guard_res.injection_score,
                            "threats": [t.model_dump() for t in guard_res.threats],
                        },
                    )

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

    # 3. Semantic Vector Cache Lookup
    cache_threshold = x_similarity_threshold or settings.CACHE_SIMILARITY_THRESHOLD
    cache_res = await cache_manager.lookup(sanitized_request, custom_threshold=cache_threshold)
    metrics.record_cache_lookup_latency(cache_res.lookup_latency_ms)

    if cache_res.hit and cache_res.cached_response:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        cached_dict = cache_res.cached_response
        
        completion_resp = ChatCompletionResponse(**cached_dict)
        completion_resp.id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"

        response.headers["X-Cache-Status"] = "HIT"
        response.headers["X-Cache-Similarity"] = str(cache_res.similarity)
        response.headers["X-Cache-Lookup-Ms"] = str(cache_res.lookup_latency_ms)
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Provider-Used"] = "semantic-cache"
        response.headers["X-PII-Entities-Scrubbed"] = str(pii_entities_count)

        metrics.record_request(
            status="200",
            model=chat_request.model,
            cache_hit=True,
            e2e_latency_ms=elapsed_ms,
            prompt_tokens=completion_resp.usage.prompt_tokens,
            completion_tokens=completion_resp.usage.completion_tokens,
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

    # 5. Output Guardrail
    if settings.ENABLE_OUTPUT_GUARDRAIL and completion_resp.choices:
        orig_reply = completion_resp.choices[0].message.content
        cleaned_reply, is_safe, violations = output_guardrail.verify_and_clean(orig_reply)
        completion_resp.choices[0].message.content = cleaned_reply

    # 6. Asynchronous Background Cache Storage
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


@router.get("/gateway/stats", response_model=GatewayStatsResponse)
async def get_gateway_stats():
    stats = metrics.get_summary_stats()
    provider_statuses = llm_router.get_provider_statuses()

    return GatewayStatsResponse(
        uptime_seconds=stats["uptime_seconds"],
        total_requests=stats["total_requests"],
        cache_hits=stats["cache_hits"],
        cache_misses=stats["cache_misses"],
        cache_hit_ratio=stats["cache_hit_ratio"],
        total_injections_blocked=stats["total_injections_blocked"],
        total_pii_entities_scrubbed=stats["total_pii_entities_scrubbed"],
        total_tokens_served=stats["total_tokens_served"],
        estimated_tokens_saved=stats["estimated_tokens_saved"],
        avg_cached_latency_ms=stats["avg_cached_latency_ms"],
        avg_upstream_latency_ms=stats["avg_upstream_latency_ms"],
        providers=provider_statuses,
        cache_size=vector_store.size,
        cache_max_size=vector_store.max_entries,
    )


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
