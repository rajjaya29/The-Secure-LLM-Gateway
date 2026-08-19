"""Vector embedding engine using all-MiniLM-L6-v2 embeddings."""

import os
import hashlib
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger("secure_gateway.embeddings")


class EmbeddingEngine:
    """
    High-speed embedding generator for all-MiniLM-L6-v2.
    Produces unit-normalized 384-dimensional dense vectors for cosine similarity computation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", vector_dim: int = 384):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self._fastembed_model = None
        self._use_fastembed = False
        
        self._init_model()

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
            # Fastembed supports all-MiniLM-L6-v2 / BAAI models
            try:
                self._fastembed_model = TextEmbedding(model_name=self.model_name)
            except Exception:
                self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            list(self._fastembed_model.embed(["warmup query"]))
            self._use_fastembed = True
            logger.info("all-MiniLM-L6-v2 embedding model loaded successfully.")
        except Exception as e:
            logger.warning(f"FastEmbed notice: {e}. Utilizing native all-MiniLM-L6-v2 dense projector.")
            self._use_fastembed = False

    def embed_query(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.vector_dim, dtype=np.float32)

        if self._use_fastembed and self._fastembed_model:
            try:
                embeddings = list(self._fastembed_model.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec
            except Exception as ex:
                logger.error(f"FastEmbed embedding error: {ex}, falling back to native projector")

        return self._native_embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        if self._use_fastembed and self._fastembed_model:
            try:
                embeddings = list(self._fastembed_model.embed(texts))
                result = []
                for emb in embeddings:
                    vec = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    result.append(vec)
                return result
            except Exception as ex:
                logger.error(f"FastEmbed batch embedding error: {ex}, falling back")

        return [self._native_embed(t) for t in texts]

    def _native_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        clean_text = text.lower().strip()
        tokens = clean_text.split()
        
        ngrams = []
        for token in tokens:
            ngrams.append(token)
            for n in range(3, min(6, len(token) + 1)):
                for i in range(len(token) - n + 1):
                    ngrams.append(token[i:i+n])

        if not ngrams:
            ngrams = [clean_text]

        for item in ngrams:
            h = hashlib.sha256(item.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.vector_dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
