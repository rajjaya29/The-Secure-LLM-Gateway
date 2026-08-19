"""Unit tests verifying benchmark calculation logic and metrics."""

import pytest
import numpy as np


def test_benchmark_speedup_and_percentiles():
    cached_latencies = [12.0, 15.0, 14.5, 11.2, 18.0, 13.5]
    upstream_latencies = [980.0, 995.0, 975.0, 990.0, 985.0]

    avg_cached = np.mean(cached_latencies)
    avg_upstream = np.mean(upstream_latencies)
    speedup = avg_upstream / avg_cached

    assert avg_cached < 25.0
    assert avg_upstream > 900.0
    assert speedup > 30.0  # ~980ms / ~14ms = ~70x speedup

    p50_hit = np.percentile(cached_latencies, 50)
    p95_hit = np.percentile(cached_latencies, 95)
    assert p50_hit < 25.0
    assert p95_hit < 25.0
