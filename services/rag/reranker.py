"""
NEXUS Contextual Relevance Reranker
Re-scores candidate chunks from retrieval layer using contextual term proximity,
query term coverage, and exact sequence matching.
"""

import re
from typing import List, Dict, Any


class Reranker:
    """
    Reranks top candidate chunks using query term density, exact phrase matching,
    and sequence position bonuses.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates down to top_n highest-relevance evidence chunks.
        """
        if not candidates:
            return []

        q_clean = query.lower()
        q_tokens = [w for w in re.findall(r"\b\w+\b", q_clean) if len(w) > 2]

        reranked = []
        for idx, cand in enumerate(candidates):
            text = cand.get("text", "").lower()
            initial_score = cand.get("hybrid_score", cand.get("score", 0.0))

            # 1. Query Term Coverage (0.0 to 1.0)
            if q_tokens:
                matched_tokens = sum(1 for t in q_tokens if t in text)
                coverage_score = matched_tokens / len(q_tokens)
            else:
                coverage_score = 0.5

            # 2. Exact Phrase Match Bonus
            phrase_bonus = 0.0
            if len(q_clean.split()) > 1 and q_clean in text:
                phrase_bonus = 0.35

            # 3. Heading / Section Relevance Bonus
            meta = cand.get("metadata", {})
            sec_bonus = 0.0
            sec_name = meta.get("section", "").lower()
            if any(t in sec_name for t in q_tokens):
                sec_bonus = 0.15

            # Combined reranking score
            rerank_score = (
                0.40 * initial_score +
                0.35 * coverage_score +
                0.15 * phrase_bonus +
                0.10 * sec_bonus
            )

            res = dict(cand)
            res["retrieval_score"] = initial_score
            res["rerank_score"] = float(rerank_score)
            res["rerank_rank"] = 0
            reranked.append(res)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        for r_idx, r in enumerate(reranked, start=1):
            r["rerank_rank"] = r_idx

        return reranked[:top_n]
