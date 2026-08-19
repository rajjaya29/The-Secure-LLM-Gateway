"""Vector Store implementation for semantic cache storage and retrieval."""

import time
import uuid
import threading
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class CacheItem:
    def __init__(
        self,
        id: str,
        prompt: str,
        embedding: np.ndarray,
        response_payload: Dict[str, Any],
        model: str,
        system_hash: str = "default",
        ttl_seconds: int = 86400,
        token_count: int = 0,
    ):
        self.id = id
        self.prompt = prompt
        self.embedding = embedding
        self.response_payload = response_payload
        self.model = model
        self.system_hash = system_hash
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.last_accessed = self.created_at
        self.hit_count = 0
        self.token_count = token_count

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "model": self.model,
            "system_hash": self.system_hash,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "token_count": self.token_count,
        }


class InMemoryVectorStore:
    def __init__(self, max_entries: int = 10000, default_ttl_seconds: int = 86400):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_seconds
        self._items: Dict[str, CacheItem] = {}
        self._exact_hash_index: Dict[str, str] = {}
        self._lock = threading.RLock()

    def _hash_key(self, prompt: str, model: str, system_hash: str) -> str:
        import hashlib
        return hashlib.sha256(f"{model}:{system_hash}:{prompt.strip()}".encode()).hexdigest()

    def upsert(
        self,
        prompt: str,
        embedding: np.ndarray,
        response_payload: Dict[str, Any],
        model: str,
        system_hash: str = "default",
        ttl_seconds: Optional[int] = None,
        token_count: int = 0,
    ) -> CacheItem:
        with self._lock:
            if len(self._items) >= self.max_entries:
                self._evict()

            item_id = f"cache-{uuid.uuid4().hex[:12]}"
            ttl = ttl_seconds or self.default_ttl
            
            item = CacheItem(
                id=item_id,
                prompt=prompt,
                embedding=embedding,
                response_payload=response_payload,
                model=model,
                system_hash=system_hash,
                ttl_seconds=ttl,
                token_count=token_count,
            )

            self._items[item_id] = item
            exact_key = self._hash_key(prompt, model, system_hash)
            self._exact_hash_index[exact_key] = item_id

            return item

    def get_by_exact_match(self, prompt: str, model: str, system_hash: str = "default") -> Optional[CacheItem]:
        with self._lock:
            exact_key = self._hash_key(prompt, model, system_hash)
            item_id = self._exact_hash_index.get(exact_key)
            if not item_id:
                return None

            item = self._items.get(item_id)
            if not item:
                self._exact_hash_index.pop(exact_key, None)
                return None

            if item.is_expired():
                self._delete(item_id)
                return None

            item.hit_count += 1
            item.last_accessed = time.time()
            return item

    def search(
        self,
        query_vector: np.ndarray,
        model: str,
        system_hash: str = "default",
        threshold: float = 0.92,
        limit: int = 1,
    ) -> List[Tuple[CacheItem, float]]:
        with self._lock:
            candidates: List[CacheItem] = []
            vectors: List[np.ndarray] = []
            
            now = time.time()
            expired_ids = []

            for item_id, item in list(self._items.items()):
                if item.expires_at <= now:
                    expired_ids.append(item_id)
                    continue

                if item.model == model and item.system_hash == system_hash:
                    candidates.append(item)
                    vectors.append(item.embedding)

            for exp_id in expired_ids:
                self._delete(exp_id)

            if not candidates or not vectors:
                return []

            matrix = np.vstack(vectors)
            sim_scores = matrix @ query_vector

            results: List[Tuple[CacheItem, float]] = []
            for idx, score in enumerate(sim_scores):
                score_val = float(score)
                if score_val >= threshold:
                    candidate = candidates[idx]
                    candidate.hit_count += 1
                    candidate.last_accessed = time.time()
                    results.append((candidate, round(score_val, 4)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def _evict(self):
        now = time.time()
        expired = [item_id for item_id, item in self._items.items() if item.expires_at <= now]
        for exp_id in expired:
            self._delete(exp_id)

        if len(self._items) >= self.max_entries:
            sorted_by_lru = sorted(self._items.values(), key=lambda x: x.last_accessed)
            evict_count = max(1, int(len(self._items) * 0.1))
            for item in sorted_by_lru[:evict_count]:
                self._delete(item.id)

    def _delete(self, item_id: str):
        item = self._items.pop(item_id, None)
        if item:
            exact_key = self._hash_key(item.prompt, item.model, item.system_hash)
            self._exact_hash_index.pop(exact_key, None)

    def clear(self):
        with self._lock:
            self._items.clear()
            self._exact_hash_index.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)
