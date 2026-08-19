"""Unit tests for Structured SQLite Request Logging."""

import pytest
import sqlite3
from app.observability.database import SQLiteLogger


@pytest.mark.asyncio
async def test_sqlite_request_logging_and_hashing(tmp_path):
    db_file = str(tmp_path / "test_audit.db")
    logger = SQLiteLogger(db_path=db_file)

    # Log 3 requests
    await logger.log_request(
        request_id="req-001",
        api_key_hash="key_a1b2c3d4e5f6",
        model="gpt-4o-mini",
        prompt_length=20,
        response_length=50,
        latency_ms=12.5,
        cache_hit=True,
        similarity=0.98,
        provider="chroma-semantic-cache",
        prompt_tokens=5,
        completion_tokens=10,
        status_code=200,
    )

    await logger.log_request(
        request_id="req-002",
        api_key_hash="key_a1b2c3d4e5f6",
        model="gpt-4o-mini",
        prompt_length=30,
        response_length=80,
        latency_ms=985.0,
        cache_hit=False,
        similarity=0.0,
        provider="mock",
        prompt_tokens=8,
        completion_tokens=20,
        status_code=200,
    )

    # Verify table contents in SQLite
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT request_id, api_key_hash, cache_hit, latency_ms, provider FROM request_logs")
        rows = cursor.fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "req-001"
    assert rows[0][1] == "key_a1b2c3d4e5f6"  # Only hashed key stored!
    assert rows[0][2] == 1  # Cache hit
    assert rows[0][3] == 12.5
    assert rows[1][2] == 0  # Cache miss
