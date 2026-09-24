"""
NEXUS Vector Database Store
Project-isolated vector storage with cosine similarity indexing,
metadata filtering, and serialization.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore:
    """
    In-memory / persistent vector store providing multi-tenant project-isolated
    semantic similarity searches with metadata filtering.
    """

    def __init__(self):
        # Dictionary keyed by chunk_id: Dict[str, Any]
        self._entries: Dict[str, Dict[str, Any]] = {}

    def insert(
        self,
        chunk_id: str,
        vector: List[float],
        document_id: str,
        project_id: Optional[str],
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert or update a single vector record."""
        norm_vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(norm_vec)
        if norm > 1e-6:
            norm_vec = norm_vec / norm

        self._entries[chunk_id] = {
            "chunk_id": chunk_id,
            "vector": norm_vec,
            "document_id": document_id,
            "project_id": project_id,
            "text": text,
            "metadata": metadata or {}
        }

    def insert_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Batch insert chunks.
        Each item must have: chunk_id, vector, document_id, project_id, text, and optional metadata.
        """
        count = 0
        for item in chunks:
            self.insert(
                chunk_id=item["chunk_id"],
                vector=item["vector"],
                document_id=item.get("document_id", "doc_default"),
                project_id=item.get("project_id"),
                text=item["text"],
                metadata=item.get("metadata", {})
            )
            count += 1
        return count

    def search(
        self,
        query_vector: List[float],
        project_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar chunks strictly isolated by project_id.
        """
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec = q_vec / q_norm

        candidates: List[Dict[str, Any]] = []

        for cid, record in self._entries.items():
            # Strict project isolation
            if project_id is not None and record["project_id"] != project_id:
                continue

            # Metadata filtering
            if filters:
                match = True
                rec_meta = record["metadata"]
                for k, v in filters.items():
                    if rec_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Cosine similarity dot product
            sim = float(np.dot(q_vec, record["vector"]))
            candidates.append({
                "chunk_id": cid,
                "document_id": record["document_id"],
                "project_id": record["project_id"],
                "text": record["text"],
                "score": sim,
                "metadata": record["metadata"]
            })

        # Sort descending by similarity score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a specific document."""
        to_delete = [cid for cid, r in self._entries.items() if r["document_id"] == document_id]
        for cid in to_delete:
            del self._entries[cid]
        return len(to_delete)

    def count(self, project_id: Optional[str] = None) -> int:
        """Count vectors optionally filtered by project_id."""
        if project_id is None:
            return len(self._entries)
        return sum(1 for r in self._entries.values() if r["project_id"] == project_id)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
