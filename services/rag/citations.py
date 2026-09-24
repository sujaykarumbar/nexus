"""
NEXUS Citation & Provenance Manager
Generates verifiable, structured citations and inline markers for retrieved document evidence.
"""

from typing import List, Dict, Any, Optional


class CitationManager:
    """
    Manages document citations, evidence linking, and formatting.
    """

    @staticmethod
    def create_citation(
        chunk: Dict[str, Any],
        citation_index: int,
        claim: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a structured citation dictionary from a retrieved chunk.
        """
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename") or chunk.get("filename", "document")
        page = metadata.get("page") or chunk.get("page_number", 1)
        section = metadata.get("section") or chunk.get("section_title", "General")
        doc_id = chunk.get("document_id", "unknown_doc")
        chunk_id = chunk.get("chunk_id", "unknown_chunk")
        score = chunk.get("rerank_score", chunk.get("score", 0.85))

        snippet = chunk.get("text", "")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        citation_tag = f"[{citation_index}]"
        label = f"Document: {filename}, Page {page}"
        if section and section != "General":
            label += f", Section: {section}"

        return {
            "citation_id": f"cite_{citation_index}",
            "citation_tag": citation_tag,
            "label": label,
            "claim": claim or "",
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "filename": filename,
            "page": page,
            "section": section,
            "confidence": round(float(score), 4),
            "snippet": snippet
        }

    @staticmethod
    def format_citations_markdown(citations: List[Dict[str, Any]]) -> str:
        """Format list of citations into markdown reference block."""
        if not citations:
            return ""

        lines = ["### References & Citations"]
        for cite in citations:
            lines.append(
                f"- **{cite['citation_tag']} {cite['label']}** (Confidence: {cite['confidence'] * 100:.1f}%)\n"
                f"  > *\"{cite['snippet']}\"*"
            )
        return "\n".join(lines)
