"""Structured SQLite Request Logging and Analytics Repository."""

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
    per-API-key usage tracking, latency percentiles, and live cache-hit ratios.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_db()

    def _init_db(self):
        """Initializes SQLite tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_length INTEGER DEFAULT 0,
                    response_length INTEGER DEFAULT 0,
                    latency_ms REAL NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    similarity REAL DEFAULT 0.0,
                    provider TEXT,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    pii_scrubbed_count INTEGER DEFAULT 0,
                    injection_blocked INTEGER DEFAULT 0,
                    status_code INTEGER DEFAULT 200,
                    error TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_key_hash ON request_logs(api_key_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_cache_hit ON request_logs(cache_hit)")
            conn.commit()

    async def log_request(
        self,
        request_id: str,
        api_key_hash: str,
        model: str,
        prompt_length: int,
        response_length: int,
        latency_ms: float,
        cache_hit: bool,
        similarity: float = 0.0,
        provider: str = "mock",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        pii_scrubbed_count: int = 0,
        injection_blocked: bool = False,
        status_code: int = 200,
        error: Optional[str] = None,
    ):
        """Asynchronously inserts a structured audit log entry into SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO request_logs (
                    request_id, timestamp, api_key_hash, model, prompt_length,
                    response_length, latency_ms, cache_hit, similarity, provider,
                    prompt_tokens, completion_tokens, pii_scrubbed_count,
                    injection_blocked, status_code, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    time.time(),
                    api_key_hash,
                    model,
                    prompt_length,
                    response_length,
                    round(latency_ms, 3),
                    1 if cache_hit else 0,
                    round(similarity, 4),
                    provider,
                    prompt_tokens,
                    completion_tokens,
                    pii_scrubbed_count,
                    1 if injection_blocked else 0,
                    status_code,
                    error,
                ),
            )
            await db.commit()

    async def get_stats(self) -> Dict[str, Any]:
        """Calculates live analytics from SQLite request logs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 1. Total Aggregates
            async with db.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN cache_hit = 0 AND injection_blocked = 0 AND status_code < 400 THEN 1 ELSE 0 END) as cache_misses,
                    SUM(CASE WHEN cache_hit = 0 AND injection_blocked = 0 AND status_code < 400 THEN 1 ELSE 0 END) as upstream_calls,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as upstream_calls_avoided,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
                    SUM(injection_blocked) as blocked_injections,
                    SUM(pii_scrubbed_count) as pii_scrubbed_total,
                    SUM(prompt_tokens + completion_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms
                FROM request_logs
            """) as cursor:
                summary = await cursor.fetchone()

            total_reqs = summary["total_requests"] or 0
            cache_hits = summary["cache_hits"] or 0
            cache_misses = summary["cache_misses"] or 0
            upstream_calls = summary["upstream_calls"] or 0
            upstream_calls_avoided = summary["upstream_calls_avoided"] or 0
            error_count = summary["error_count"] or 0
            blocked_injections = summary["blocked_injections"] or 0
            pii_scrubbed = summary["pii_scrubbed_total"] or 0
            total_tokens = summary["total_tokens"] or 0
            avg_lat = round(float(summary["avg_latency_ms"] or 0.0), 2)
            
            valid_completed = cache_hits + cache_misses
            hit_ratio = round(cache_hits / valid_completed, 4) if valid_completed > 0 else 0.0

            # 2. Latency Distributions
            async with db.execute("SELECT latency_ms FROM request_logs WHERE cache_hit = 1") as cursor:
                rows_cached = await cursor.fetchall()
                cached_latencies = [r[0] for r in rows_cached]

            async with db.execute("SELECT latency_ms FROM request_logs WHERE cache_hit = 0 AND injection_blocked = 0") as cursor:
                rows_upstream = await cursor.fetchall()
                upstream_latencies = [r[0] for r in rows_upstream]

            all_latencies = cached_latencies + upstream_latencies
            p50 = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
            p95 = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
            p99 = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

            avg_cached = np.mean(cached_latencies) if cached_latencies else 0.0
            avg_upstream = np.mean(upstream_latencies) if upstream_latencies else 0.0

            # 3. Per-Key Usage Tracking (Dictionary by api_key_hash)
            per_key_usage: Dict[str, Any] = {}
            async with db.execute("""
                SELECT
                    api_key_hash,
                    COUNT(*) as request_count,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(prompt_tokens + completion_tokens) as total_tokens,
                    SUM(latency_ms) as total_latency_ms,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                FROM request_logs
                GROUP BY api_key_hash
                ORDER BY request_count DESC
            """) as cursor:
                key_rows = await cursor.fetchall()
                for kr in key_rows:
                    k_id = kr["api_key_hash"]
                    req_cnt = kr["request_count"]
                    k_hits = kr["cache_hits"] or 0
                    per_key_usage[k_id] = {
                        "requests": req_cnt,
                        "cache_hits": k_hits,
                        "cache_hit_ratio": round(k_hits / req_cnt, 4) if req_cnt > 0 else 0.0,
                        "total_tokens": kr["total_tokens"] or 0,
                        "total_latency_ms": round(kr["total_latency_ms"] or 0.0, 2),
                        "avg_latency_ms": round(kr["avg_latency_ms"] or 0.0, 2),
                        "errors": kr["errors"] or 0,
                    }

            return {
                "total_requests": total_reqs,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio": hit_ratio,
                "upstream_calls": upstream_calls,
                "upstream_calls_avoided": upstream_calls_avoided,
                "error_count": error_count,
                "total_injections_blocked": blocked_injections,
                "total_pii_entities_scrubbed": pii_scrubbed,
                "total_tokens_served": total_tokens,
                "avg_latency_ms": avg_lat,
                "avg_cached_latency_ms": round(float(avg_cached), 2),
                "avg_upstream_latency_ms": round(float(avg_upstream), 2),
                "latency": {
                    "p50_ms": round(p50, 2),
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
