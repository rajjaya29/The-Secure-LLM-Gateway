"""Prometheus metrics exporter and internal telemetry tracker."""

import time
import threading
from typing import Dict, List, Any
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)


class MetricsTracker:
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.registry = registry
        self._start_time = time.time()
        self._lock = threading.Lock()

        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.blocked_injections = 0
        self.pii_scrubbed_count = 0
        self.total_tokens_served = 0
        self.estimated_tokens_saved = 0
        self.cached_latencies: List[float] = []
        self.upstream_latencies: List[float] = []

        self.p_requests_total = Counter(
            "llm_gateway_requests_total",
            "Total number of requests handled by the gateway",
            ["status", "model", "cache_status"],
            registry=self.registry,
        )

        self.p_cache_hits = Counter(
            "llm_gateway_cache_hits_total",
            "Total number of semantic cache hits",
            registry=self.registry,
        )

        self.p_cache_misses = Counter(
            "llm_gateway_cache_misses_total",
            "Total number of semantic cache misses",
            registry=self.registry,
        )

        self.p_latency = Histogram(
            "llm_gateway_latency_seconds",
            "Latency distribution in seconds",
            ["type"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        self.p_guardrail_blocks = Counter(
            "llm_gateway_guardrail_blocks_total",
            "Total number of requests blocked by input/output guardrails",
            ["reason"],
            registry=self.registry,
        )

        self.p_pii_scrubbed = Counter(
            "llm_gateway_pii_scrubbed_total",
            "Total number of PII entities scrubbed",
            ["entity_type"],
            registry=self.registry,
        )

        self.p_tokens = Counter(
            "llm_gateway_tokens_total",
            "Total token count consumed/served",
            ["type", "model"],
            registry=self.registry,
        )

        self.p_fallbacks = Counter(
            "llm_gateway_provider_fallbacks_total",
            "Total number of provider fallback transitions",
            ["from_provider", "to_provider"],
            registry=self.registry,
        )

        self.p_active_requests = Gauge(
            "llm_gateway_active_requests",
            "Number of requests currently in-flight",
            registry=self.registry,
        )

    def record_request(
        self,
        status: str,
        model: str,
        cache_hit: bool,
        e2e_latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        with self._lock:
            self.total_requests += 1
            if cache_hit:
                self.cache_hits += 1
                self.p_cache_hits.inc()
                self.estimated_tokens_saved += (prompt_tokens + completion_tokens)
                self.p_tokens.labels(type="saved", model=model).inc(prompt_tokens + completion_tokens)
                self.cached_latencies.append(e2e_latency_ms)
                if len(self.cached_latencies) > 200:
                    self.cached_latencies.pop(0)
            else:
                self.cache_misses += 1
                self.p_cache_misses.inc()
                self.upstream_latencies.append(e2e_latency_ms)
                if len(self.upstream_latencies) > 200:
                    self.upstream_latencies.pop(0)

            total_toks = prompt_tokens + completion_tokens
            self.total_tokens_served += total_toks

            cache_label = "hit" if cache_hit else "miss"
            self.p_requests_total.labels(status=status, model=model, cache_status=cache_label).inc()
            self.p_latency.labels(type="gateway_e2e").observe(e2e_latency_ms / 1000.0)

            if prompt_tokens > 0:
                self.p_tokens.labels(type="prompt", model=model).inc(prompt_tokens)
            if completion_tokens > 0:
                self.p_tokens.labels(type="completion", model=model).inc(completion_tokens)

    def record_guardrail_block(self, reason: str):
        with self._lock:
            self.blocked_injections += 1
            self.p_guardrail_blocks.labels(reason=reason).inc()

    def record_pii_scrubbed(self, entity_type: str, count: int = 1):
        with self._lock:
            self.pii_scrubbed_count += count
            self.p_pii_scrubbed.labels(entity_type=entity_type).inc(count)

    def record_cache_lookup_latency(self, latency_ms: float):
        self.p_latency.labels(type="cache_lookup").observe(latency_ms / 1000.0)

    def record_provider_latency(self, latency_ms: float):
        self.p_latency.labels(type="provider_inference").observe(latency_ms / 1000.0)

    def record_fallback(self, from_provider: str, to_provider: str):
        self.p_fallbacks.labels(from_provider=from_provider, to_provider=to_provider).inc()

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time
            total = self.total_requests
            hit_ratio = round((self.cache_hits / total), 4) if total > 0 else 0.0
            
            avg_cached = (
                sum(self.cached_latencies) / len(self.cached_latencies)
                if self.cached_latencies
                else 0.0
            )
            avg_upstream = (
                sum(self.upstream_latencies) / len(self.upstream_latencies)
                if self.upstream_latencies
                else 0.0
            )

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_ratio": hit_ratio,
                "total_injections_blocked": self.blocked_injections,
                "total_pii_entities_scrubbed": self.pii_scrubbed_count,
                "total_tokens_served": self.total_tokens_served,
                "estimated_tokens_saved": self.estimated_tokens_saved,
                "avg_cached_latency_ms": round(avg_cached, 2),
                "avg_upstream_latency_ms": round(avg_upstream, 2),
            }

    def export_prometheus(self) -> bytes:
        return generate_latest(self.registry)


metrics = MetricsTracker()
