"""
NEXUS Research / RAG Agent
Connects the agent swarm with domain document intelligence,
hybrid retrieval (dense vector + BM25), reranking, and citation grounding.
"""

from typing import Dict, Any, List, Optional
from services.rag.rag_engine import RAGEngine
from services.graph.graph_rag import GraphRAGEngine


class RAGAgent:
    """
    Autonomous Document Intelligence & Research Agent that queries domain documents,
    extracts knowledge graph relationships, and grounds analytical reasoning with verifiable citations.
    """

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        graph_rag_engine: Optional[GraphRAGEngine] = None
    ):
        self.engine = rag_engine or RAGEngine()
        self.graph_rag = graph_rag_engine or GraphRAGEngine()

    def retrieve_evidence(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Tool: Search hybrid vector and BM25 store to extract ranked evidence chunks."""
        return self.engine.search_evidence(query=query, project_id=project_id, top_k=top_k)

    def answer_question(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Tool: Generate grounded answer with inline citations and document references."""
        return self.engine.answer_query(query=query, project_id=project_id, top_k=top_k)

    def query_graph_rag(
        self,
        query: str,
        project_id: Optional[str] = None,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Tool: Search Knowledge Graph for multi-hop relationships and structural lineage."""
        return self.graph_rag.query(query=query, project_id=project_id, max_hops=max_hops)

    def ingest_document(
        self,
        file_path: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        project_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tool: Parse, chunk, embed, and index a new document."""
        return self.engine.ingest_document(
            file_path=file_path,
            raw_bytes=raw_bytes,
            filename=filename,
            file_type=file_type,
            project_id=project_id,
            document_id=document_id
        )
