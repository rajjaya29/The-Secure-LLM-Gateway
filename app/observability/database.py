"""Structured SQLite Request Logging and Analytics Engine."""

import os
import time
import sqlite3
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
import aiosqlite
from app.config import settings


class SQLiteLogger:
    """
    Asynchronous SQLite Database Logger for full request tracing,
    per-API-key usage quotas, latency distributions, and live cache-hit ratios.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_db()

    def _init_db(self):
        """Synchronously initialize SQLite table schemas and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    api_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT,
                    response TEXT,
                    latency_ms REAL NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    similarity REAL DEFAULT 0.0,
                    provider TEXT,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    pii_scrubbed_count INTEGER DEFAULT 0,
                    injection_blocked INTEGER DEFAULT 0,
                    status_code INTEGER DEFAULT 200
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_api_key ON request_logs(api_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_cache_hit ON request_logs(cache_hit)")
            conn.commit()

    async def log_request(
        self,
        api_key: str,
        model: str,
        prompt: str,
        response: str,
        latency_ms: float,
        cache_hit: bool,
        similarity: float = 0.0,
        provider: str = "mock",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        pii_scrubbed_count: int = 0,
        injection_blocked: bool = False,
        status_code: int = 200,
    ):
        """Asynchronously writes a structured request log entry to SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO request_logs (
                    timestamp, api_key, model, prompt, response,
                    latency_ms, cache_hit, similarity, provider,
                    prompt_tokens, completion_tokens, pii_scrubbed_count,
                    injection_blocked, status_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    api_key,
                    model,
                    prompt[:500] if prompt else "",
                    response[:500] if response else "",
                    round(latency_ms, 3),
                    1 if cache_hit else 0,
                    round(similarity, 4),
                    provider,
                    prompt_tokens,
                    completion_tokens,
                    pii_scrubbed_count,
                    1 if injection_blocked else 0,
                    status_code,
                ),
            )
            await db.commit()

    async def get_analytics(self) -> Dict[str, Any]:
        """Calculates live cache-hit ratios, latency distributions, and per-key usage from SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 1. Total Aggregates
            async with db.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN cache_hit = 0 AND injection_blocked = 0 THEN 1 ELSE 0 END) as cache_misses,
                    SUM(injection_blocked) as blocked_injections,
                    SUM(pii_scrubbed_count) as pii_scrubbed_total,
                    SUM(prompt_tokens + completion_tokens) as total_tokens,
                    SUM(CASE WHEN cache_hit = 1 THEN prompt_tokens + completion_tokens ELSE 0 END) as tokens_saved
                FROM request_logs
            """) as cursor:
                summary = await cursor.fetchone()

            total_reqs = summary["total_requests"] or 0
            cache_hits = summary["cache_hits"] or 0
            cache_misses = summary["cache_misses"] or 0
            blocked_injections = summary["blocked_injections"] or 0
            pii_scrubbed = summary["pii_scrubbed_total"] or 0
            total_tokens = summary["total_tokens"] or 0
            tokens_saved = summary["tokens_saved"] or 0
            hit_ratio = round(cache_hits / (cache_hits + cache_misses), 4) if (cache_hits + cache_misses) > 0 else 0.0

            # 2. Latency Distributions
            async with db.execute("SELECT latency_ms FROM request_logs WHERE cache_hit = 1") as cursor:
                rows_cached = await cursor.fetchall()
                cached_latencies = [r[0] for r in rows_cached]

            async with db.execute("SELECT latency_ms FROM request_logs WHERE cache_hit = 0 AND injection_blocked = 0") as cursor:
                rows_upstream = await cursor.fetchall()
                upstream_latencies = [r[0] for r in rows_upstream]

            avg_cached = np.mean(cached_latencies) if cached_latencies else 0.0
            avg_upstream = np.mean(upstream_latencies) if upstream_latencies else 0.0

            all_latencies = cached_latencies + upstream_latencies
            p50 = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
            p90 = float(np.percentile(all_latencies, 90)) if all_latencies else 0.0
            p95 = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
            p99 = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

            # 3. Per-Key Usage Tracking
            per_key_usage = []
            async with db.execute("""
                SELECT
                    api_key,
                    COUNT(*) as request_count,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(prompt_tokens + completion_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(injection_blocked) as injections_blocked
                FROM request_logs
                GROUP BY api_key
                ORDER BY request_count DESC
            """) as cursor:
                key_rows = await cursor.fetchall()
                for kr in key_rows:
                    per_key_usage.append({
                        "api_key": kr["api_key"],
                        "requests": kr["request_count"],
                        "cache_hits": kr["cache_hits"] or 0,
                        "cache_hit_ratio": round((kr["cache_hits"] or 0) / kr["request_count"], 4),
                        "total_tokens": kr["total_tokens"] or 0,
                        "avg_latency_ms": round(kr["avg_latency_ms"] or 0.0, 2),
                        "injections_blocked": kr["injections_blocked"] or 0,
                    })

            return {
                "total_requests": total_reqs,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio": hit_ratio,
                "total_injections_blocked": blocked_injections,
                "total_pii_entities_scrubbed": pii_scrubbed,
                "total_tokens_served": total_tokens,
                "estimated_tokens_saved": tokens_saved,
                "avg_cached_latency_ms": round(float(avg_cached), 2),
                "avg_upstream_latency_ms": round(float(avg_upstream), 2),
                "latency_distribution": {
                    "p50_ms": round(p50, 2),
                    "p90_ms": round(p90, 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                },
                "per_key_usage": per_key_usage,
            }

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM request_logs")
            conn.commit()


sqlite_logger = SQLiteLogger()
