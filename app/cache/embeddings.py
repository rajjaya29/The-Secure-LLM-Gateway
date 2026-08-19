"""Vector embedding engine using SentenceTransformers with all-MiniLM-L6-v2."""

import logging
import hashlib
import numpy as np
from typing import List, Optional
from app.config import settings

logger = logging.getLogger("secure_gateway.embeddings")

_GLOBAL_SENTENCE_TRANSFORMER = None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Singleton model loader for SentenceTransformer to prevent redundant downloads."""
    global _GLOBAL_SENTENCE_TRANSFORMER
    if _GLOBAL_SENTENCE_TRANSFORMER is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{model_name}'...")
            _GLOBAL_SENTENCE_TRANSFORMER = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer '{model_name}' loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer directly: {e}. Falling back to FastEmbed/native.")
    return _GLOBAL_SENTENCE_TRANSFORMER


class EmbeddingEngine:
    """
    Embedding generator using Sentence Transformers (all-MiniLM-L6-v2).
    Produces unit-normalized 384-dimensional dense vectors for cosine similarity computation.
    """

    def __init__(self, model_name: Optional[str] = None, vector_dim: int = 384):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.vector_dim = vector_dim
        self._st_model = None
        self._fastembed_model = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._st_model = get_sentence_transformer(self.model_name)
        if self._st_model is None:
            self._init_fastembed_fallback()
        self._initialized = True

    def _init_fastembed_fallback(self):
        try:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            list(self._fastembed_model.embed(["warmup"]))
        except Exception:
            self._fastembed_model = None

    def embed_query(self, text: str) -> np.ndarray:
        """Encodes a single text prompt into a 384-dim normalized vector."""
        if not text:
            return np.zeros(self.vector_dim, dtype=np.float32)

        self._ensure_initialized()

        # 1. Primary: SentenceTransformers
        if self._st_model is not None:
            try:
                emb = self._st_model.encode(text, normalize_embeddings=True)
                return np.array(emb, dtype=np.float32)
            except Exception as e:
                logger.error(f"SentenceTransformers encode error: {e}")

        # 2. Secondary: FastEmbed
        if self._fastembed_model is not None:
            try:
                embs = list(self._fastembed_model.embed([text]))
                v = np.array(embs[0], dtype=np.float32)
                norm = np.linalg.norm(v)
                return v / norm if norm > 0 else v
            except Exception as e:
                logger.error(f"FastEmbed fallback error: {e}")

        # 3. Deterministic native subword projection (zero external network dependency)
        return self._native_embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch encodes multiple text strings."""
        if not texts:
            return []

        if self._st_model is not None:
            try:
                embs = self._st_model.encode(texts, normalize_embeddings=True)
                return [np.array(e, dtype=np.float32) for e in embs]
            except Exception:
                pass

        return [self.embed_query(t) for t in texts]

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
