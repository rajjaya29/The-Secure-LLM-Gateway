"""Resilient LLM Router with circuit breaking, exponential retries, and fallback pipelines."""

import time
import random
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from app.router.providers import BaseLLMProvider, MockLLMProvider, OpenAIProvider, AnthropicProvider, OllamaProvider
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse
from app.schemas.gateway import ProviderStatus

logger = logging.getLogger("secure_gateway.router")


class LLMRouter:
    def __init__(
        self,
        providers: Optional[Dict[str, BaseLLMProvider]] = None,
        provider_priority: Optional[List[str]] = None,
        max_retries_per_provider: int = 2,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
    ):
        self.providers: Dict[str, BaseLLMProvider] = providers or {
            "mock": MockLLMProvider(name="mock"),
            "openai": OpenAIProvider(api_key=None),
            "anthropic": AnthropicProvider(api_key=None),
            "ollama": OllamaProvider(),
        }
        self.provider_priority: List[str] = provider_priority or ["mock", "openai", "ollama", "anthropic"]
        self.max_retries = max_retries_per_provider

        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            name: CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_seconds,
            )
            for name in self.providers.keys()
        }

        self._provider_latencies: Dict[str, List[float]] = {name: [] for name in self.providers.keys()}

    async def route_and_generate(
        self,
        request: ChatCompletionRequest,
        preferred_provider: Optional[str] = None,
    ) -> Tuple[ChatCompletionResponse, str, List[str]]:
        execution_order = list(self.provider_priority)
        if preferred_provider and preferred_provider in self.providers:
            if preferred_provider in execution_order:
                execution_order.remove(preferred_provider)
            execution_order.insert(0, preferred_provider)

        attempted: List[str] = []
        last_exception: Optional[Exception] = None

        for provider_name in execution_order:
            provider = self.providers.get(provider_name)
            cb = self.circuit_breakers.get(provider_name)

            if not provider or not cb:
                continue

            if not cb.can_execute():
                logger.warning(f"Skipping provider '{provider_name}': Circuit Breaker is OPEN.")
                attempted.append(f"{provider_name} (circuit_open)")
                continue

            attempted.append(provider_name)
            logger.info(f"Attempting LLM generation with provider '{provider_name}'...")

            for retry in range(self.max_retries + 1):
                t_start = time.perf_counter()
                try:
                    response = await provider.generate(request)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    
                    cb.record_success()
                    self._record_latency(provider_name, elapsed_ms)
                    logger.info(f"Provider '{provider_name}' succeeded in {elapsed_ms:.1f}ms.")
                    return response, provider_name, attempted
                except Exception as ex:
                    last_exception = ex
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    logger.warning(
                        f"Provider '{provider_name}' attempt {retry + 1}/{self.max_retries + 1} failed ({elapsed_ms:.1f}ms): {ex}"
                    )
                    
                    if retry < self.max_retries:
                        backoff = (0.1 * (2 ** retry)) + random.uniform(0.01, 0.05)
                        await asyncio.sleep(backoff)
                    else:
                        cb.record_failure(ex)

        error_msg = f"All LLM providers in fallback chain failed: {attempted}. Last error: {last_exception}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _record_latency(self, provider_name: str, latency_ms: float):
        l_list = self._provider_latencies.setdefault(provider_name, [])
        l_list.append(latency_ms)
        if len(l_list) > 100:
            l_list.pop(0)

    def get_provider_statuses(self) -> List[ProviderStatus]:
        statuses: List[ProviderStatus] = []
        for name, cb in self.circuit_breakers.items():
            status_dict = cb.get_status()
            latencies = self._provider_latencies.get(name, [])
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

            state_label = status_dict["state"]
            if state_label == CircuitBreaker.STATE_CLOSED:
                health_status = "healthy"
            elif state_label == CircuitBreaker.STATE_HALF_OPEN:
                health_status = "degraded"
            else:
                health_status = "circuit_open"

            statuses.append(
                ProviderStatus(
                    name=name,
                    status=health_status,
                    circuit_state=state_label,
                    failure_count=status_dict["failure_count"],
                    last_failure_timestamp=status_dict["last_failure_time"],
                    total_calls=status_dict["total_calls"],
                    avg_latency_ms=round(avg_lat, 2),
                )
            )
        return statuses
