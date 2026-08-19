"""ChromaDB Vector Store with Tenant / API-Key Isolation."""

import json
import time
import uuid
import threading
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import chromadb
from app.config import settings


class ChromaVectorStore:
    """
    Local Vector-Based Semantic Cache backed by ChromaDB with cosine distance metric.
    Features strict API-Key / Tenant Isolation to prevent cross-client cache leakage.
    """

    def __init__(
        self,
        collection_name: str = "semantic_cache",
        persist_directory: Optional[str] = None,
        max_entries: int = 10000,
        isolate_by_api_key: bool = True,
    ):
        self.collection_name = collection_name
        self.max_entries = max_entries
        self.isolate_by_api_key = isolate_by_api_key
        self._exact_hash_index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _hash_key(self, prompt: str, model: str, api_key_hash: str, system_hash: str) -> str:
        import hashlib
        key_scope = api_key_hash if self.isolate_by_api_key else "global"
        return hashlib.sha256(f"{key_scope}:{model}:{system_hash}:{prompt.strip()}".encode()).hexdigest()

    def upsert(
        self,
        prompt: str,
        embedding: np.ndarray,
        response_payload: Dict[str, Any],
        model: str,
        api_key_hash: str = "key_default",
        system_hash: str = "default",
        ttl_seconds: int = 86400,
        token_count: int = 0,
    ) -> str:
        """Indexes prompt vector, payload, and client metadata into ChromaDB."""
        with self._lock:
            doc_id = f"chroma-{uuid.uuid4().hex[:12]}"
            created_at = time.time()
            expires_at = created_at + ttl_seconds
            key_scope = api_key_hash if self.isolate_by_api_key else "global"

            metadata = {
                "model": model,
                "api_key_hash": key_scope,
                "system_hash": system_hash,
                "created_at": created_at,
                "expires_at": expires_at,
                "token_count": token_count,
                "response_json": json.dumps(response_payload),
            }

            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding],
                documents=[prompt],
                metadatas=[metadata],
            )

            # Record in exact match index
            exact_key = self._hash_key(prompt, model, api_key_hash, system_hash)
            self._exact_hash_index[exact_key] = {
                "id": doc_id,
                "prompt": prompt,
                "response_payload": response_payload,
                "model": model,
                "api_key_hash": key_scope,
                "system_hash": system_hash,
                "expires_at": expires_at,
                "token_count": token_count,
            }

            return doc_id

    def get_by_exact_match(
        self,
        prompt: str,
        model: str,
        api_key_hash: str = "key_default",
        system_hash: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Fast-path exact match hash lookup with tenant isolation."""
        with self._lock:
            exact_key = self._hash_key(prompt, model, api_key_hash, system_hash)
            item = self._exact_hash_index.get(exact_key)
            if not item:
                return None

            if time.time() > item["expires_at"]:
                self._exact_hash_index.pop(exact_key, None)
                return None

            return item

    def search(
        self,
        query_vector: np.ndarray,
        model: str,
        api_key_hash: str = "key_default",
        system_hash: str = "default",
        threshold: float = 0.90,
        limit: int = 1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches ChromaDB using cosine distance with client partition filtering.
        Calculates cosine similarity s = 1.0 - distance.
        Returns matches where similarity >= threshold.
        """
        with self._lock:
            if self._collection.count() == 0:
                return []

            vec_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            key_scope = api_key_hash if self.isolate_by_api_key else "global"

            # Filter by model and api_key_hash for tenant isolation
            where_clause = None
            if self._collection.count() > 1:
                where_clause = {
                    "$and": [
                        {"model": model},
                        {"api_key_hash": key_scope},
                    ]
                }

            try:
                query_res = self._collection.query(
                    query_embeddings=[vec_list],
                    n_results=min(limit * 4, max(1, self._collection.count())),
                    where=where_clause,
                )
            except Exception:
                query_res = self._collection.query(
                    query_embeddings=[vec_list],
                    n_results=min(limit * 4, max(1, self._collection.count())),
                )

            results: List[Tuple[Dict[str, Any], float]] = []
            now = time.time()

            if query_res and query_res.get("ids") and query_res["ids"][0]:
                ids = query_res["ids"][0]
                distances = query_res["distances"][0] if "distances" in query_res and query_res["distances"] else []
                documents = query_res["documents"][0] if "documents" in query_res and query_res["documents"] else []
                metadatas = query_res["metadatas"][0] if "metadatas" in query_res and query_res["metadatas"] else []

                for idx, doc_id in enumerate(ids):
                    dist = distances[idx] if idx < len(distances) else 1.0
                    sim = round(float(1.0 - dist), 4)

                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    doc_text = documents[idx] if idx < len(documents) else ""

                    if meta.get("expires_at", 0) <= now:
                        continue
                    if meta.get("model") != model or (self.isolate_by_api_key and meta.get("api_key_hash") != key_scope):
                        continue

                    if sim >= threshold:
                        resp_payload = json.loads(meta.get("response_json", "{}"))
                        item_dict = {
                            "id": doc_id,
                            "prompt": doc_text,
                            "response_payload": resp_payload,
                            "model": model,
                            "api_key_hash": key_scope,
                            "system_hash": system_hash,
                            "token_count": meta.get("token_count", 0),
                        }
                        results.append((item_dict, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def clear(self):
        with self._lock:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._exact_hash_index.clear()

    @property
    def size(self) -> int:
        return self._collection.count()
