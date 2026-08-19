"""ChromaDB Vector Store implementation for semantic cache storage and cosine similarity retrieval."""

import json
import time
import uuid
import threading
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings


class ChromaVectorStore:
    """
    Local Vector-Based Semantic Cache using ChromaDB with cosine distance metric ('hnsw:space': 'cosine').
    Matches queries against previous query embeddings (all-MiniLM-L6-v2) at cosine similarity >= 0.90.
    """

    def __init__(
        self,
        collection_name: str = "semantic_cache",
        persist_directory: Optional[str] = None,
        max_entries: int = 10000,
    ):
        self.collection_name = collection_name
        self.max_entries = max_entries
        self._exact_hash_index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Initialize ChromaDB Client
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Create or get ChromaDB collection with cosine distance
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        ttl_seconds: int = 86400,
        token_count: int = 0,
    ) -> str:
        """Indexes prompt embedding and response payload into ChromaDB."""
        with self._lock:
            doc_id = f"chroma-{uuid.uuid4().hex[:12]}"
            created_at = time.time()
            expires_at = created_at + ttl_seconds
            
            # Prepare metadata for ChromaDB (scalars only)
            metadata = {
                "model": model,
                "system_hash": system_hash,
                "created_at": created_at,
                "expires_at": expires_at,
                "token_count": token_count,
                "response_json": json.dumps(response_payload),
            }

            # Upsert into ChromaDB collection
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding],
                documents=[prompt],
                metadatas=[metadata],
            )

            # Store in exact match hash index for sub-millisecond shortcut
            exact_key = self._hash_key(prompt, model, system_hash)
            self._exact_hash_index[exact_key] = {
                "id": doc_id,
                "prompt": prompt,
                "response_payload": response_payload,
                "model": model,
                "system_hash": system_hash,
                "expires_at": expires_at,
            }

            return doc_id

    def get_by_exact_match(self, prompt: str, model: str, system_hash: str = "default") -> Optional[Dict[str, Any]]:
        """Sub-millisecond exact match hash lookup."""
        with self._lock:
            exact_key = self._hash_key(prompt, model, system_hash)
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
        system_hash: str = "default",
        threshold: float = 0.90,
        limit: int = 1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries ChromaDB using cosine distance.
        Converts cosine distance d to cosine similarity s = 1.0 - d.
        Returns matches where similarity >= threshold.
        """
        with self._lock:
            if self._collection.count() == 0:
                return []

            vec_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            # Query ChromaDB
            try:
                query_res = self._collection.query(
                    query_embeddings=[vec_list],
                    n_results=min(limit * 3, max(1, self._collection.count())),
                    where={"$and": [{"model": model}, {"system_hash": system_hash}]} if self._collection.count() > 1 else None,
                )
            except Exception:
                # Fallback if where filter unsupported in single-doc collections
                query_res = self._collection.query(
                    query_embeddings=[vec_list],
                    n_results=min(limit * 3, max(1, self._collection.count())),
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
                    # Cosine distance to similarity: sim = 1.0 - distance
                    sim = round(float(1.0 - dist), 4)

                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    doc_text = documents[idx] if idx < len(documents) else ""

                    # Check expiration & context isolation
                    if meta.get("expires_at", 0) <= now:
                        continue
                    if meta.get("model") != model or meta.get("system_hash") != system_hash:
                        continue

                    if sim >= threshold:
                        resp_payload = json.loads(meta.get("response_json", "{}"))
                        item_dict = {
                            "id": doc_id,
                            "prompt": doc_text,
                            "response_payload": resp_payload,
                            "model": model,
                            "system_hash": system_hash,
                            "token_count": meta.get("token_count", 0),
                        }
                        results.append((item_dict, sim))

            # Sort by highest similarity
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
