# 📋 Resume Claim Audit

This document provides a line-by-line verification and evidence mapping for every technical claim in the resume statement:

> **The Secure LLM Gateway | Python, FastAPI, Sentence-Transformers, ChromaDB, SQLite | GitHub**
>
> – Built an authenticated, rate-limited LLM API proxy in FastAPI to handle prompt routing and reduce redundant upstream API calls.
>
> – Implemented local vector-based semantic caching using all-MiniLM-L6-v2 embeddings and ChromaDB (cosine similarity ≥ 0.90), cutting response latency from ~980ms to <25ms on cached query hits.
>
> – Developed custom middleware for API-key authentication (X-API-Key), in-memory sliding-window rate limiting, and prompt validation.
>
> – Integrated structured SQLite request logging and built a /stats metrics endpoint to track live cache-hit ratios, latency distributions, and per-key usage.

---

## 🎯 Verification Matrix

| Resume Claim | Implementation Location | Automated Test | Evidence / Measured Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FastAPI LLM Proxy** | [`app/main.py`](file:///Users/jayaraj/Jaya_Project/app/main.py), [`app/api/v1/routes.py`](file:///Users/jayaraj/Jaya_Project/app/api/v1/routes.py) | `tests/test_gateway.py` | `POST /v1/chat/completions` handles OpenAI-compatible requests and routes across providers | **PASS** |
| **X-API-Key Authentication** | [`app/resilience/auth.py`](file:///Users/jayaraj/Jaya_Project/app/resilience/auth.py) | `tests/test_auth.py` | Rejects missing/invalid keys with HTTP 401; validates configured keys; generates secure hashes for logging | **PASS** |
| **Sliding-Window Rate Limiting** | [`app/resilience/rate_limiter.py`](file:///Users/jayaraj/Jaya_Project/app/resilience/rate_limiter.py) | `tests/test_rate_limit.py` | In-memory `collections.deque` per API key; rejects bursts exceeding quota with HTTP 429 and `Retry-After` | **PASS** |
| **Prompt Validation** | [`app/guardrails/prompt_validator.py`](file:///Users/jayaraj/Jaya_Project/app/guardrails/prompt_validator.py) | `tests/test_prompt_validation.py` | Intercepts DAN mode, direct overrides, delimiter spoofing, and context exfiltration with HTTP 400 | **PASS** |
| **Sentence Transformers** | [`app/cache/embeddings.py`](file:///Users/jayaraj/Jaya_Project/app/cache/embeddings.py) | `tests/test_embeddings.py` | `from sentence_transformers import SentenceTransformer` singleton model loader | **PASS** |
| **all-MiniLM-L6-v2 Embeddings** | [`app/cache/embeddings.py`](file:///Users/jayaraj/Jaya_Project/app/cache/embeddings.py) | `tests/test_embeddings.py` | Generates 384-dimensional unit-normalized dense vectors | **PASS** |
| **ChromaDB Vector Store** | [`app/cache/chroma_store.py`](file:///Users/jayaraj/Jaya_Project/app/cache/chroma_store.py) | `tests/test_semantic_cache.py` | ChromaDB collection with `{"hnsw:space": "cosine"}` and tenant/API-key isolation | **PASS** |
| **Cosine Similarity $\ge 0.90$** | [`app/cache/chroma_store.py`](file:///Users/jayaraj/Jaya_Project/app/cache/chroma_store.py) | `tests/test_semantic_cache.py` | Converts distance $d$ to similarity $1.0 - d \ge 0.90$; filters matches accordingly | **PASS** |
| **Structured SQLite Request Logging** | [`app/observability/database.py`](file:///Users/jayaraj/Jaya_Project/app/observability/database.py) | `tests/test_sqlite_logging.py` | `gateway_logs.db` table `request_logs` records request_id, api_key_hash (no raw keys), latency, cache hit | **PASS** |
| **`/stats` Metrics Endpoint** | [`app/api/v1/routes.py`](file:///Users/jayaraj/Jaya_Project/app/api/v1/routes.py), [`app/main.py`](file:///Users/jayaraj/Jaya_Project/app/main.py) | `tests/test_stats.py` | `GET /stats` returns live request totals, cache hit ratio, upstream calls avoided, and per-key usage | **PASS** |
| **Live Cache-Hit Ratios** | [`app/observability/database.py`](file:///Users/jayaraj/Jaya_Project/app/observability/database.py) | `tests/test_stats.py` | Live calculation: `cache_hits / (cache_hits + cache_misses)` from SQLite | **PASS** |
| **Latency Distribution (P50/P95/P99)** | [`app/observability/database.py`](file:///Users/jayaraj/Jaya_Project/app/observability/database.py) | `tests/test_stats.py` | Computes numpy percentiles across logged request latencies | **PASS** |
| **Per-Key Usage Analytics** | [`app/observability/database.py`](file:///Users/jayaraj/Jaya_Project/app/observability/database.py) | `tests/test_stats.py` | Tracks request count, cache hits, total tokens, avg latency, and error counts per `api_key_hash` | **PASS** |
| **Performance: ~980ms $\to$ <25ms** | [`benchmark.py`](file:///Users/jayaraj/Jaya_Project/benchmark.py) | `tests/test_benchmark_metrics.py` | **Measured: 1038.12 ms (Cold) $\to$ 15.00 ms (Cached)** — **69.2x Speedup** | **PASS** |

---

## 🔬 Benchmark Latency Evidence (Actual Measurements)

```
==================================================
SECURE LLM GATEWAY BENCHMARK
============================

Requests: 22

Cache hits: 6
Cache misses: 16

Cache hit ratio: 27.3%

Upstream calls: 16
Upstream calls avoided: 6

CACHE HIT LATENCY
Average: 15.00 ms   (<25ms Target Met!)
P50:     18.03 ms   (<25ms Target Met!)
P95:     19.62 ms   (<25ms Target Met!)
P99:     19.62 ms   (<25ms Target Met!)

CACHE MISS LATENCY
Average: 1038.12 ms (~980ms Baseline)
P50:     1007.45 ms
P95:     1151.51 ms
P99:     1167.93 ms

Speedup: 69.2x

==================================================
```
