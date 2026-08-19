"""Unit tests for Sliding-Window Rate Limiting, Circuit Breaking, and LLM Router."""

import pytest
import time
import asyncio
from app.resilience.rate_limiter import SlidingWindowRateLimiter
from app.resilience.circuit_breaker import CircuitBreaker
from app.router.providers import MockLLMProvider
from app.router.llm_router import LLMRouter
from app.schemas.openai import ChatCompletionRequest, ChatMessage


def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter(window_seconds=1, max_requests=3, enabled=True)
    key = "sk-client-test"

    # 3 allowed
    for _ in range(3):
        allowed, wait = limiter.check_limit(key)
        assert allowed is True

    # 4th in same window blocked
    allowed, wait = limiter.check_limit(key)
    assert allowed is False
    assert wait > 0

    # Wait for sliding window expiration
    time.sleep(1.1)
    allowed, wait = limiter.check_limit(key)
    assert allowed is True


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(name="test_provider", failure_threshold=2, recovery_timeout_seconds=0.1)

    assert cb.state == CircuitBreaker.STATE_CLOSED
    assert cb.can_execute() is True

    cb.record_failure(RuntimeError("fail 1"))
    assert cb.state == CircuitBreaker.STATE_CLOSED

    cb.record_failure(RuntimeError("fail 2"))
    assert cb.state == CircuitBreaker.STATE_OPEN
    assert cb.can_execute() is False

    time.sleep(0.12)
    assert cb.can_execute() is True
    assert cb.state == CircuitBreaker.STATE_HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitBreaker.STATE_CLOSED


@pytest.mark.asyncio
async def test_llm_router_fallback():
    failing_mock = MockLLMProvider(name="failing_provider", fail_mode=True)
    healthy_mock = MockLLMProvider(name="healthy_provider", fail_mode=False)

    router = LLMRouter(
        providers={"failing": failing_mock, "healthy": healthy_mock},
        provider_priority=["failing", "healthy"],
        max_retries_per_provider=1,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    resp, provider_used, chain = await router.route_and_generate(req)

    assert provider_used == "healthy"
    assert "failing" in chain
    assert "healthy" in chain
    assert resp.choices[0].message.content is not None
