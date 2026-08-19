"""Unit tests for Sentence Transformers all-MiniLM-L6-v2 embedding generation."""

import pytest
import numpy as np
from app.cache.embeddings import EmbeddingEngine


def test_embedding_engine_output_dimensions():
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    vec = engine.embed_query("What is the capital of France?")
    
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embedding_engine_unit_normalization():
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    vec = engine.embed_query("Vector search similarity testing")
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 0.01) == 1.0


def test_embedding_batch_processing():
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    texts = ["First query", "Second query", "Third query"]
    batch_vecs = engine.embed_batch(texts)
    
    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert isinstance(v, np.ndarray)
        assert len(v) == 384
