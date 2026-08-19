"""
Official Benchmark Evaluation Script for The Secure LLM Gateway.

Measures real latency across:
  A. Cold / Cache-Miss Requests (~980ms baseline upstream latency)
  B. Semantic-Cache-Hit Requests (<25ms target via ChromaDB & all-MiniLM-L6-v2)

Calculates:
  - Total requests, cache hits, cache misses, hit ratio
  - Upstream calls & upstream calls avoided
  - Average, P50, P95, P99 latencies
  - Speedup factor
"""

import time
import asyncio
import statistics
import numpy as np
from typing import List, Dict, Any
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.api.v1.routes import vector_store
from app.observability.database import sqlite_logger

API_KEY = "sk-test-key-123"

# Curated benchmark query pool
BENCHMARK_QUERIES = [
    # Cold Baseline Queries
    {"prompt": "What is the capital of France?", "category": "Cold Query"},
    {"prompt": "Explain quantum superposition in simple terms.", "category": "Cold Query"},
    {"prompt": "Write a Python function to check if a string is a palindrome.", "category": "Cold Query"},
    {"prompt": "What were the primary causes of the French Revolution in 1789?", "category": "Cold Query"},
    {"prompt": "How does vector similarity search work in high-dimensional space?", "category": "Cold Query"},
    {"prompt": "What is the difference between TCP and UDP networking protocols?", "category": "Cold Query"},
    {"prompt": "Explain how photosynthesis converts light into chemical energy.", "category": "Cold Query"},
    {"prompt": "What are the core principles of Object-Oriented Programming?", "category": "Cold Query"},
    {"prompt": "Describe the architecture of a transformer neural network.", "category": "Cold Query"},
    {"prompt": "What is the economic principle of supply and demand?", "category": "Cold Query"},

    # Semantically Equivalent / Paraphrased Queries (Cache Hits)
    {"prompt": "What is the capital of France?", "category": "Cache Hit (Exact)"},
    {"prompt": "Can you tell me the capital city of France?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "Could you explain what quantum superposition is simply?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "Write a python function to check if a string is a palindrome.", "category": "Cache Hit (Exact)"},
    {"prompt": "Python code to verify if a word reads the same backwards (palindrome).", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "What caused the French Revolution in 1789?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "How does vector similarity search work in embedding space?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "Differences between TCP and UDP protocols explained.", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "How do plants convert sunlight into energy via photosynthesis?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "What are the fundamental principles of OOP in software?", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "Explain transformer architecture in deep learning models.", "category": "Cache Hit (Paraphrase)"},
    {"prompt": "Principles of supply and demand in market economics.", "category": "Cache Hit (Paraphrase)"},
]


async def run_benchmark():
    print("==================================================")
    print("SECURE LLM GATEWAY BENCHMARK")
    print("==================================================")

    # Initialize in-memory transport
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")

    # Clear state
    vector_store.clear()
    sqlite_logger.clear()

    # Warmup forward pass for SentenceTransformers & ChromaDB index
    warmup_payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "warmup"}]}
    await client.post("/v1/chat/completions", json=warmup_payload, headers={"X-API-Key": API_KEY})
    vector_store.clear()
    sqlite_logger.clear()

    hit_latencies: List[float] = []
    miss_latencies: List[float] = []
    all_latencies: List[float] = []

    total_requests = len(BENCHMARK_QUERIES)

    for q in BENCHMARK_QUERIES:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": q["prompt"]}],
            "temperature": 0.7,
        }
        headers = {"X-API-Key": API_KEY}

        t_start = time.perf_counter()
        res = await client.post("/v1/chat/completions", json=payload, headers=headers)
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        gateway_ms = float(res.headers.get("X-Latency-Ms", elapsed_ms))
        cache_status = res.headers.get("X-Cache-Status", "MISS")

        all_latencies.append(gateway_ms)

        if cache_status == "HIT":
            hit_latencies.append(gateway_ms)
        else:
            miss_latencies.append(gateway_ms)

        await asyncio.sleep(0.05)

    cache_hits = len(hit_latencies)
    cache_misses = len(miss_latencies)
    hit_ratio = (cache_hits / total_requests) * 100 if total_requests > 0 else 0.0

    upstream_calls = cache_misses
    upstream_calls_avoided = cache_hits

    # Calculate statistics
    avg_total = statistics.mean(all_latencies) if all_latencies else 0.0
    p50_total = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
    p95_total = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    p99_total = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

    avg_hit = statistics.mean(hit_latencies) if hit_latencies else 0.0
    p50_hit = float(np.percentile(hit_latencies, 50)) if hit_latencies else 0.0
    p95_hit = float(np.percentile(hit_latencies, 95)) if hit_latencies else 0.0
    p99_hit = float(np.percentile(hit_latencies, 99)) if hit_latencies else 0.0

    avg_miss = statistics.mean(miss_latencies) if miss_latencies else 0.0
    p50_miss = float(np.percentile(miss_latencies, 50)) if miss_latencies else 0.0
    p95_miss = float(np.percentile(miss_latencies, 95)) if miss_latencies else 0.0
    p99_miss = float(np.percentile(miss_latencies, 99)) if miss_latencies else 0.0

    speedup = (avg_miss / avg_hit) if avg_hit > 0 else 0.0

    output = f"""
==================================================
SECURE LLM GATEWAY BENCHMARK
============================

Requests: {total_requests}

Cache hits: {cache_hits}
Cache misses: {cache_misses}

Cache hit ratio: {hit_ratio:.1f}%

Upstream calls: {upstream_calls}
Upstream calls avoided: {upstream_calls_avoided}

CACHE HIT LATENCY
Average: {avg_hit:.2f} ms
P50: {p50_hit:.2f} ms
P95: {p95_hit:.2f} ms
P99: {p99_hit:.2f} ms

CACHE MISS LATENCY
Average: {avg_miss:.2f} ms
P50: {p50_miss:.2f} ms
P95: {p95_miss:.2f} ms
P99: {p99_miss:.2f} ms

Speedup: {speedup:.1f}x

==================================================
"""
    print(output)
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
