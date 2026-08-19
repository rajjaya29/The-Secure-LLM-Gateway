"""Token Bucket Rate Limiter for request and token throttling."""

import time
import threading
from typing import Dict, Tuple


class TokenBucket:
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= amount:
                self.tokens -= amount
                return True, 0.0
            else:
                needed = amount - self.tokens
                wait_time = needed / self.refill_rate if self.refill_rate > 0 else 60.0
                return False, wait_time


class TokenBucketRateLimiter:
    def __init__(self, default_rpm: int = 120, default_tpm: int = 100000, enabled: bool = True):
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self.enabled = enabled
        
        self._rpm_buckets: Dict[str, TokenBucket] = {}
        self._tpm_buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_or_create_buckets(self, client_id: str) -> Tuple[TokenBucket, TokenBucket]:
        with self._lock:
            if client_id not in self._rpm_buckets:
                self._rpm_buckets[client_id] = TokenBucket(
                    capacity=float(self.default_rpm),
                    refill_rate_per_sec=self.default_rpm / 60.0,
                )
            if client_id not in self._tpm_buckets:
                self._tpm_buckets[client_id] = TokenBucket(
                    capacity=float(self.default_tpm),
                    refill_rate_per_sec=self.default_tpm / 60.0,
                )
            return self._rpm_buckets[client_id], self._tpm_buckets[client_id]

    def check_limit(self, client_id: str, estimated_tokens: int = 50) -> Tuple[bool, float, str]:
        if not self.enabled:
            return True, 0.0, ""

        rpm_bucket, tpm_bucket = self._get_or_create_buckets(client_id)

        rpm_allowed, rpm_wait = rpm_bucket.consume(1.0)
        if not rpm_allowed:
            return False, round(rpm_wait, 2), "RPM (Requests Per Minute)"

        tpm_allowed, tpm_wait = tpm_bucket.consume(float(estimated_tokens))
        if not tpm_allowed:
            return False, round(tpm_wait, 2), "TPM (Tokens Per Minute)"

        return True, 0.0, ""
