# 🛡️ The Secure LLM Gateway
### *High-Performance Semantic Caching, Guardrails & LLM Reverse-Proxy*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow.svg?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6B6B.svg?style=flat)](https://www.trychroma.com/)
[![SQLite](https://img.shields.io/badge/Audit-SQLite_3-003B57.svg?style=flat&logo=sqlite)](https://www.sqlite.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-16%20passed%20(100%25)-brightgreen.svg?style=flat)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An authenticated, rate-limited LLM API proxy built with **FastAPI**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **ChromaDB**, and **SQLite**. Designed to eliminate redundant upstream model calls through **sub-25ms vector semantic caching ($\ge 0.90$ similarity)**, enforce **sliding-window rate limiting**, validate prompts against adversarial jailbreaks, and provide real-time **per-key observability**.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Performance Benchmarks](#-performance-benchmarks)
- [Quickstart Guide](#-quickstart-guide)
- [API Reference](#-api-reference)
- [Repository Structure](#-repository-structure)
- [Automated Testing](#-automated-testing)
- [Security & Tenant Isolation](#-security--tenant-isolation)
- [Docker Deployment](#-docker-deployment)

---

## 🎯 Executive Overview

| Technical Capability | Implementation Details | Verified Metric / SLA |
| :--- | :--- | :--- |
| **LLM Proxy & Routing** | Asynchronous reverse-proxy handling OpenAI-compatible chat requests | Multi-provider fallback (OpenAI, Anthropic, Ollama, Mock) |
| **API-Key Authentication** | Custom dependency validating `X-API-Key` headers | Rejects unauthorized requests with `HTTP 401` |
| **Sliding-Window Rate Limiting** | Process-local in-memory `collections.deque` per API key | Enforces quotas (60 req/60s) with `HTTP 429` & `Retry-After` |
| **Prompt Validation** | Heuristic & pattern filtering for overrides, jailbreaks, and exfiltrations | Blocks attacks with `HTTP 400 Bad Request` |
| **Semantic Vector Cache** | ChromaDB with `all-MiniLM-L6-v2` dense embeddings ($\ge 0.90$ Cosine Sim) | Cuts latency from **~1038ms to 15.0ms (69.2x speedup)** |
| **Structured Request Logging** | Asynchronous SQLite repository logging hashed client identifiers | `gateway_logs.db` auditing all transactions safely |
| **Live Telemetry & `/stats`** | Dynamic analytical endpoint tracking hit ratios, percentiles & key usage | Live P50, P95, P99 percentiles & per-key analytics |

---

## 🌟 System Architecture

```
                                 CLIENT REQUEST
                         (Headers: X-API-Key: demo-key)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI GATEWAY ENGINE                                │
│                                                                             │
│  [ Step 1: Authentication ] ──► Validate X-API-Key (HTTP 401 if invalid)    │
│  [ Step 2: Rate Limiting ]  ──► Sliding-Window Quota Check (HTTP 429)       │
│  [ Step 3: Validation ]     ──► Prompt Injection & Jailbreak Check (HTTP 400)│
│                                      │                                      │
│  [ Step 4: Semantic Cache ] ─────────┴────────────────────────────────────┐ │
│       ├── Generate query embedding using all-MiniLM-L6-v2                 │ │
│       └── Query ChromaDB collection with Cosine Similarity >= 0.90        │ │
│                                                                           │ │
│       ┌───────────────────────┬────────────────────────┐                  │ │
│       │ Cache HIT (>= 0.90)   │ Cache MISS (< 0.90)    │                  │ │
│       ▼                       ▼                        │                  │ │
│  Return Cached Response  Route to Upstream Provider   │                  │ │
│  (⚡ Latency: < 25ms)     (📡 Latency: ~980ms)          │                  │ │
│                               │                        │                  │ │
│                               ├── Write to ChromaDB    │                  │ │
│                               └── Log to SQLite DB     │                  │ │
└───────────────────────────────────────┬────────────────┴──────────────────┘
                                        │
                                        ▼
                               CLIENT RESPONSE
                      (Headers: X-Cache-Status: HIT/MISS)
```

---

## 🔑 Key Features

### 1. 🔐 API-Key Authentication (`X-API-Key`)
* Validates client access tokens via the `X-API-Key` header (with `Bearer` token fallback).
* Computes deterministic SHA-256 digests (`key_<hash>`) for database storage and tenant isolation.
* **Zero-Secret Leakage**: Raw API keys are strictly forbidden from database logs.

### 2. ⏱️ In-Memory Sliding-Window Rate Limiting
* Implements a rolling timestamp deque (`collections.deque`) per authenticated API key.
* Independently manages burst quotas (default: `60 requests / 60 seconds`).
* Provides HTTP standard response headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`).

### 3. 🛡️ Prompt Validation & Security Filter
* Real-time inspection intercepting:
  * **Instruction Overrides**: `"ignore all previous instructions"`, `"disregard prior directives"`
  * **Persona Jailbreaks**: `"DAN mode"`, `"Developer mode"`, `"unaligned AI"`
  * **Delimiter Spoofing**: `<|im_start|>system`, `<<SYS>>`, `### System:`
  * **Context Exfiltration**: `"reveal system prompt"`, `"repeat words verbatim"`

### 4. ⚡ ChromaDB Semantic Cache (`all-MiniLM-L6-v2`)
* Local vector cache converting queries into 384-dimensional unit-normalized embeddings.
* Configured with cosine distance metric (`{"hnsw:space": "cosine"}`).
* Matches semantically equivalent and paraphrased queries at $\text{Similarity} \ge 0.90$.
* **Tenant Partitioning**: Enforces metadata filters so client keys cannot access other tenants' cached responses.

### 5. 📊 SQLite Audit Logging & `/stats` Analytics
* Asynchronously records each transaction to `gateway_logs.db`.
* **`GET /stats` Endpoint**: Delivers live aggregations including:
  * Overall request counts, cache hits, and cache hit ratios
  * Upstream calls and upstream calls avoided
  * Latency distributions: **P50, P95, and P99**
  * Per-API-key usage breakdowns (requests, cache hits, token volume, error counts).

---

## ⚡ Performance Benchmarks

All benchmark metrics are recorded using high-precision monotonic clock timers (`time.perf_counter()`):

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
Average: 15.00 ms   (< 25ms SLA Target Met)
P50:     18.03 ms   (< 25ms SLA Target Met)
P95:     19.62 ms   (< 25ms SLA Target Met)
P99:     19.62 ms   (< 25ms SLA Target Met)

CACHE MISS LATENCY
Average: 1038.12 ms (~980ms Simulated Baseline)
P50:     1007.45 ms
P95:     1151.51 ms
P99:     1167.93 ms

Speedup: 69.2x

==================================================
```

### 📊 Latency Comparison

| Metric | Cold Request (Cache MISS) | Cached Query (Cache HIT) | Speedup / Improvement |
| :--- | :--- | :--- | :--- |
| **Average Latency** | `1038.12 ms` | **`15.00 ms`** | **69.2x faster** |
| **P50 Latency** | `1007.45 ms` | **`18.03 ms`** | **55.8x faster** |
| **P95 Latency** | `1151.51 ms` | **`19.62 ms`** | **58.6x faster** |
| **SLA Target Compliance** | Baseline | **100% (< 25ms)** | ✅ Target Achieved |

---

## 🛠️ Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/rajjaya29/The-Secure-LLM-Gateway.git
cd The-Secure-LLM-Gateway

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
HOST=0.0.0.0
PORT=8000
REQUIRE_API_KEY=true
VALID_API_KEYS=["sk-test-key-123", "demo-key", "sk-admin-master-key"]
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
EMBEDDING_MODEL=all-MiniLM-L6-v2
SEMANTIC_SIMILARITY_THRESHOLD=0.90
SQLITE_DB_PATH=gateway_logs.db
DEFAULT_PROVIDER=mock
MOCK_LLM_LATENCY_MS=980.0
```

### 3. Launch the Gateway Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Reference

### 1. Chat Completion (`POST /v1/chat/completions`)

#### Request
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-test-key-123" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What is Kubernetes?"}
    ],
    "temperature": 0.7
  }'
```

#### Response Headers
* `X-Cache-Status`: `HIT` or `MISS`
* `X-Cache-Similarity`: Cosine similarity score (e.g., `0.9614`)
* `X-Latency-Ms`: Monotonic latency in milliseconds (e.g., `15.82`)
* `X-RateLimit-Remaining`: Remaining sliding-window requests (e.g., `59`)

#### Response Body
```json
{
  "id": "chatcmpl-cache-a1b2c3d4",
  "object": "chat.completion",
  "created": 1787127600,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Kubernetes is an open-source container orchestration system for automating application deployment, scaling, and management."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 18,
    "total_tokens": 23
  }
}
```

---

### 2. Live Telemetry & Usage Analytics (`GET /stats`)

#### Request
```bash
curl -X GET http://localhost:8000/stats \
  -H "X-API-Key: sk-test-key-123"
```

#### Response Body
```json
{
  "total_requests": 22,
  "cache_hits": 6,
  "cache_misses": 16,
  "cache_hit_ratio": 0.2727,
  "upstream_calls": 16,
  "upstream_calls_avoided": 6,
  "error_count": 0,
  "total_injections_blocked": 0,
  "total_tokens_served": 555,
  "avg_latency_ms": 759.09,
  "avg_cached_latency_ms": 15.0,
  "avg_upstream_latency_ms": 1038.12,
  "latency": {
    "p50_ms": 1002.93,
    "p95_ms": 1144.63,
    "p99_ms": 1166.29
  },
  "per_key_usage": {
    "key_2d550185026d": {
      "requests": 22,
      "cache_hits": 6,
      "cache_hit_ratio": 0.2727,
      "total_tokens": 555,
      "total_latency_ms": 16699.97,
      "avg_latency_ms": 759.09,
      "errors": 0
    }
  }
}
```

---

## 📁 Repository Structure

```
├── app/
│   ├── main.py                     # FastAPI application, CORS, lifespan & middleware
│   ├── config.py                   # Pydantic BaseSettings environment configuration
│   ├── api/v1/routes.py            # /v1/chat/completions, /models, and /stats endpoints
│   ├── schemas/                    # Pydantic schemas (OpenAI compatible & Gateway telemetry)
│   ├── resilience/
│   │   ├── auth.py                 # X-API-Key verification & secure key hashing
│   │   ├── rate_limiter.py         # In-memory sliding-window rate limiter
│   │   └── circuit_breaker.py      # Upstream provider fault tolerance
│   ├── guardrails/
│   │   ├── prompt_validator.py     # Prompt injection and jailbreak filter
│   │   ├── pii_scrubber.py         # PII entity detection and anonymization
│   │   └── output_guardrail.py     # Response verification and leak prevention
│   ├── cache/
│   │   ├── embeddings.py           # Sentence-Transformers (all-MiniLM-L6-v2)
│   │   ├── chroma_store.py         # ChromaDB cosine vector store with tenant isolation
│   │   └── semantic_cache.py       # Async cache manager & lookup logic
│   ├── router/
│   │   ├── providers.py            # OpenAI, Anthropic, Ollama, and Mock providers
│   │   └── llm_router.py           # Priority fallback router
│   ├── observability/
│   │   ├── database.py             # Structured SQLite request logger & analytics
│   │   ├── logging.py              # Structured JSON application logger
│   │   └── metrics.py              # Prometheus metrics collector
│   └── static/                     # Dark-mode dashboard UI (HTML/CSS/JS)
├── tests/                          # 16 automated pytest unit & integration tests
│   ├── test_auth.py                # API key verification & hashing tests
│   ├── test_rate_limit.py          # Sliding-window rate limiter tests
│   ├── test_prompt_validation.py   # Prompt injection & jailbreak tests
│   ├── test_embeddings.py          # Sentence-Transformers embedding tests
│   ├── test_semantic_cache.py      # ChromaDB caching & tenant isolation tests
│   ├── test_sqlite_logging.py      # SQLite database audit logging tests
│   ├── test_stats.py               # /stats metrics & percentile tests
│   ├── test_gateway.py             # Complete proxy end-to-end flow tests
│   └── test_benchmark_metrics.py   # Benchmark speedup & calculation tests
├── benchmark.py                    # Official latency benchmark script
├── benchmark_50_prompts.py         # 50-prompt test workload evaluation
├── RESUME_CLAIM_AUDIT.md           # Claim-by-claim verification & evidence matrix
├── Dockerfile                      # Production container image
├── docker-compose.yml              # Container orchestration configuration
└── requirements.txt                # Python project dependencies
```

---

## 🧪 Automated Testing

Run the automated test suite with pytest:
```bash
pytest -v
```

### Test Coverage (16/16 Passed)
```
tests/test_auth.py::test_api_key_hashing_security PASSED                 [  6%]
tests/test_auth.py::test_auth_missing_api_key_rejected PASSED            [ 12%]
tests/test_auth.py::test_auth_invalid_api_key_rejected PASSED            [ 18%]
tests/test_auth.py::test_auth_valid_api_keys PASSED                      [ 25%]
tests/test_benchmark_metrics.py::test_benchmark_speedup_and_percentiles PASSED [ 31%]
tests/test_embeddings.py::test_embedding_engine_output_dimensions PASSED [ 37%]
tests/test_embeddings.py::test_embedding_engine_unit_normalization PASSED [ 43%]
tests/test_embeddings.py::test_embedding_batch_processing PASSED         [ 50%]
tests/test_gateway.py::test_gateway_full_proxy_lifecycle PASSED          [ 56%]
tests/test_prompt_validation.py::test_prompt_validator_safe_prompts PASSED [ 62%]
tests/test_prompt_validation.py::test_prompt_validator_malicious_prompts PASSED [ 68%]
tests/test_prompt_validation.py::test_prompt_validator_case_and_whitespace_variations PASSED [ 75%]
tests/test_rate_limit.py::test_sliding_window_rate_limiter_burst_and_window PASSED [ 81%]
tests/test_semantic_cache.py::test_semantic_cache_hit_miss_and_isolation PASSED [ 87%]
tests/test_sqlite_logging.py::test_sqlite_request_logging_and_hashing PASSED [ 93%]
tests/test_stats.py::test_stats_metrics_and_percentiles PASSED           [100%]

============================== 16 passed in 15.40s ==============================
```

---

## 🔒 Security & Tenant Isolation

1. **API Key Protection**: Raw client API keys are hashed with SHA-256 (`key_<hash>`) before logging. Database compromises cannot expose live client secrets.
2. **Tenant Isolation in Vector Space**: ChromaDB collections enforce a `where={"api_key_hash": key_scope}` partition. Key A cannot access Key B's cached query responses.
3. **Thread-Safe Rate Limiting**: In-memory rate limiting deques are synchronized using `threading.Lock`.
4. **Adversarial Injection Defense**: Suspicious instruction patterns and delimiter injections are rejected at the edge before consuming upstream LLM tokens.

---

## 🐳 Docker Deployment

The gateway is containerized and ready to deploy without requiring external paid API credentials:

```bash
docker-compose up --build
```

The gateway will be accessible at `http://localhost:8000` with automated health checks enabled.

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
