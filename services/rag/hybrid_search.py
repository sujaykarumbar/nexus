"""
NEXUS Hybrid Search Engine
Combines Dense Semantic Vector Search with Lexical BM25 Keyword Search
via Reciprocal Rank Fusion (RRF) and metadata filtering.
"""

from typing import List, Dict, Any, Optional
from services.rag.vector_store import VectorStore
from services.rag.bm25_retriever import BM25Retriever
from services.rag.embeddings import EmbeddingProvider


class HybridSearchEngine:
    """
    Unified Hybrid Search combining dense semantic embeddings and sparse BM25
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedding_provider: EmbeddingProvider,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel dense vector and BM25 queries, then merge results via RRF.
        """
        # 1. Dense Semantic Search
        q_vec = self.embedding_provider.embed_text(query)
        dense_results = self.vector_store.search(
            query_vector=q_vec,
            project_id=project_id,
            top_k=top_k * 2,
            filters=filters
        )

        # 2. BM25 Keyword Search
        bm25_results = self.bm25_retriever.search(
            query=query,
            project_id=project_id,
            top_k=top_k * 2,
            filters=filters
        )

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        source_ranks: Dict[str, Dict[str, int]] = {}

        # Process dense ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            chunk_lookup[cid] = item
            score = dense_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + score
            if cid not in source_ranks:
                source_ranks[cid] = {}
            source_ranks[cid]["dense_rank"] = rank

        # Process BM25 ranks
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            score = bm25_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + score
            if cid not in source_ranks:
                source_ranks[cid] = {}
            source_ranks[cid]["bm25_rank"] = rank

        # Format candidates
        merged_candidates: List[Dict[str, Any]] = []
        for cid, score in fused_scores.items():
            base = chunk_lookup[cid]
            candidate = dict(base)
            candidate["hybrid_score"] = float(score)
            candidate["retrieval_ranks"] = source_ranks.get(cid, {})
            merged_candidates.append(candidate)

        # Sort descending by fused RRF score
        merged_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return merged_candidates[:top_k]
