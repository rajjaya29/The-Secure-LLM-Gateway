"""Unit tests for /stats metrics calculations from SQLite logs."""

import pytest
from app.observability.database import SQLiteLogger


@pytest.mark.asyncio
async def test_stats_metrics_and_percentiles(tmp_path):
    db_file = str(tmp_path / "test_stats.db")
    logger = SQLiteLogger(db_path=db_file)

    # Insert 4 mock records for key_alpha and 1 for key_beta
    for i in range(3):
        await logger.log_request(
            request_id=f"alpha-hit-{i}",
            api_key_hash="key_alpha",
            model="gpt-4o-mini",
            prompt_length=15,
            response_length=40,
            latency_ms=10.0 + i,
            cache_hit=True,
            similarity=0.95,
            provider="chroma-semantic-cache",
            status_code=200,
        )

    await logger.log_request(
        request_id="alpha-miss-1",
        api_key_hash="key_alpha",
        model="gpt-4o-mini",
        prompt_length=15,
        response_length=40,
        latency_ms=980.0,
        cache_hit=False,
        similarity=0.0,
        provider="mock",
        status_code=200,
    )

    await logger.log_request(
        request_id="beta-miss-1",
        api_key_hash="key_beta",
        model="gpt-4o-mini",
        prompt_length=25,
        response_length=50,
        latency_ms=990.0,
        cache_hit=False,
        similarity=0.0,
        provider="mock",
        status_code=200,
    )

    stats = await logger.get_stats()

    assert stats["total_requests"] == 5
    assert stats["cache_hits"] == 3
    assert stats["cache_misses"] == 2
    assert stats["upstream_calls"] == 2
    assert stats["upstream_calls_avoided"] == 3
    assert stats["cache_hit_ratio"] == 0.60
    assert stats["latency"]["p50_ms"] > 0
    assert stats["latency"]["p95_ms"] > 0
    assert "key_alpha" in stats["per_key_usage"]
    assert stats["per_key_usage"]["key_alpha"]["requests"] == 4
    assert stats["per_key_usage"]["key_alpha"]["cache_hits"] == 3
    assert "key_beta" in stats["per_key_usage"]
