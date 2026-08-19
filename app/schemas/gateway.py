"""Schemas for internal Gateway state, guardrail assessments, and telemetry."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time


class GuardrailThreat(BaseModel):
    category: str
    confidence: float
    matched_pattern: Optional[str] = None
    description: str


class GuardrailResult(BaseModel):
    is_safe: bool = True
    blocked: bool = False
    action_taken: str = "allow"
    injection_score: float = 0.0
    threats: List[GuardrailThreat] = []
    sanitized_text: Optional[str] = None
    pii_entities_detected: List[Dict[str, Any]] = []
    pii_entities_count: int = 0
    processing_time_ms: float = 0.0


class CacheLookupResult(BaseModel):
    hit: bool = False
    similarity: float = 0.0
    cached_response: Optional[Dict[str, Any]] = None
    cache_id: Optional[str] = None
    matched_prompt: Optional[str] = None
    lookup_latency_ms: float = 0.0


class ProviderStatus(BaseModel):
    name: str
    status: str
    circuit_state: str
    failure_count: int = 0
    last_failure_timestamp: Optional[float] = None
    total_calls: int = 0
    avg_latency_ms: float = 0.0


class GatewayStatsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_ratio: float
    total_injections_blocked: int
    total_pii_entities_scrubbed: int
    total_tokens_served: int
    estimated_tokens_saved: int
    avg_cached_latency_ms: float
    avg_upstream_latency_ms: float
    providers: List[ProviderStatus]
    cache_size: int
    cache_max_size: int


class GuardrailTestRequest(BaseModel):
    prompt: str
    check_injection: bool = True
    check_pii: bool = True
    mask_style: str = "tokenized"


class GuardrailTestResponse(BaseModel):
    original_prompt: str
    sanitized_prompt: str
    is_safe: bool
    blocked: bool
    injection_score: float
    threats: List[GuardrailThreat]
    pii_entities: List[Dict[str, Any]]
    processing_time_ms: float
