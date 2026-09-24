"""
NEXUS BM25 Keyword Search Engine
High-efficiency, zero-dependency BM25Okapi implementation for lexical search,
tokenization, IDF weighting, and project-isolated retrieval.
"""

import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter


class BM25Retriever:
    """
    Pure Python BM25Okapi retriever supporting multi-tenant document collections.
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
        "that", "the", "their", "then", "there", "these", "they", "this", "to",
        "was", "will", "with"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: Dict[str, Dict[str, Any]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._doc_freqs: Dict[str, int] = Counter()
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer filtering out common stopwords."""
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the BM25 index.
        Each doc must have: chunk_id, text, document_id, project_id, metadata.
        """
        for doc in documents:
            cid = doc["chunk_id"]
            tokens = self._tokenize(doc["text"])
            self._corpus[cid] = {
                "chunk_id": cid,
                "document_id": doc.get("document_id", "doc_default"),
                "project_id": doc.get("project_id"),
                "text": doc["text"],
                "tokens": tokens,
                "tf": Counter(tokens),
                "metadata": doc.get("metadata", {})
            }
            self._doc_lengths[cid] = len(tokens)

        self._recompute_statistics()

    def _recompute_statistics(self) -> None:
        """Recompute corpus frequency statistics and average document length."""
        self._doc_freqs.clear()
        total_len = sum(self._doc_lengths.values())
        n_docs = len(self._corpus)
        self._avgdl = total_len / n_docs if n_docs > 0 else 1.0

        for record in self._corpus.values():
            unique_tokens = set(record["tokens"])
            for t in unique_tokens:
                self._doc_freqs[t] += 1

    def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search corpus using BM25Okapi formula.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens or not self._corpus:
            return []

        n_docs = len(self._corpus)
        candidates: List[Dict[str, Any]] = []

        for cid, record in self._corpus.items():
            if project_id is not None and record["project_id"] != project_id:
                continue

            if filters:
                match = True
                rec_meta = record["metadata"]
                for k, v in filters.items():
                    if rec_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = 0.0
            doc_len = self._doc_lengths[cid]
            tf_map = record["tf"]

            for token in q_tokens:
                if token not in tf_map:
                    continue

                f = tf_map[token]
                df = self._doc_freqs.get(token, 0)
                # Robertson-Spärck Jones IDF
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

                # BM25 term saturation
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                score += idf * (numerator / denominator)

            if score > 0.0:
                candidates.append({
                    "chunk_id": cid,
                    "document_id": record["document_id"],
                    "project_id": record["project_id"],
                    "text": record["text"],
                    "score": float(score),
                    "metadata": record["metadata"]
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def delete_document(self, document_id: str) -> None:
        """Delete chunks for a document and update index."""
        to_delete = [cid for cid, r in self._corpus.items() if r["document_id"] == document_id]
        for cid in to_delete:
            del self._corpus[cid]
            del self._doc_lengths[cid]
        self._recompute_statistics()
