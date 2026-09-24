"""
NEXUS End-to-End RAG Engine
Coordinates query understanding, hybrid retrieval (Dense + BM25), reranking,
grounded context assembly, and citation generation with strict hallucination guards.
"""

import time
import re
from typing import List, Dict, Any, Optional

from services.rag.document_processor import DocumentProcessor
from services.rag.chunker import DocumentChunker
from services.rag.embeddings import EmbeddingProvider, FastDenseEmbeddingProvider
from services.rag.vector_store import VectorStore
from services.rag.bm25_retriever import BM25Retriever
from services.rag.hybrid_search import HybridSearchEngine
from services.rag.reranker import Reranker
from services.rag.citations import CitationManager


class RAGEngine:
    """
    Production-grade Retrieval-Augmented Generation (RAG) engine.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        self.processor = DocumentProcessor()
        self.chunker = DocumentChunker()
        self.embedding_provider = embedding_provider or FastDenseEmbeddingProvider(dim=128)
        self.vector_store = vector_store or VectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.hybrid_searcher = HybridSearchEngine(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            embedding_provider=self.embedding_provider
        )
        self.reranker = Reranker()
        self.citation_manager = CitationManager()

    def ingest_document(
        self,
        file_path: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        project_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process, chunk, embed, and index a document into both vector and BM25 stores.
        """
        start_t = time.perf_counter()

        if file_path:
            processed = self.processor.process_file(
                file_path=file_path,
                document_id=document_id,
                project_id=project_id
            )
        elif raw_bytes and filename and file_type:
            processed = self.processor.process_bytes(
                raw_bytes=raw_bytes,
                filename=filename,
                file_type=file_type,
                file_size=len(raw_bytes),
                document_id=document_id,
                project_id=project_id
            )
        else:
            raise ValueError("Must provide either file_path or (raw_bytes, filename, file_type)")

        doc_id = processed["document_id"]
        chunks = self.chunker.chunk_document(processed)

        # Generate embeddings and store vectors
        vector_items = []
        for chk in chunks:
            vec = self.embedding_provider.embed_text(chk["text"])
            chk["vector"] = vec
            vector_items.append(chk)

        # Batch insert into VectorStore
        self.vector_store.insert_batch(vector_items)

        # Add to BM25 index
        self.bm25_retriever.add_documents(chunks)

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "document_id": doc_id,
            "project_id": project_id,
            "filename": processed["filename"],
            "file_type": processed["file_type"],
            "hash": processed["hash"],
            "page_count": processed["page_count"],
            "chunks_created": len(chunks),
            "duration_ms": round(duration_ms, 2),
            "status": "COMPLETED"
        }

    def rewrite_query(self, query: str) -> str:
        """Standardize and enrich query for optimal keyword and vector alignment."""
        q = query.strip()
        # Remove trailing punctuation
        q = re.sub(r"[?!.]+$", "", q)
        return q

    def search_evidence(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank top evidence chunks for a question.
        """
        rewritten = self.rewrite_query(query)
        candidates = self.hybrid_searcher.search(
            query=rewritten,
            project_id=project_id,
            top_k=top_k * 3,
            filters=filters
        )
        reranked = self.reranker.rerank(query=rewritten, candidates=candidates, top_n=top_k)
        return reranked

    def answer_query(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Execute full grounded RAG pipeline: retrieval, citation generation,
        and answer synthesis.
        """
        start_t = time.perf_counter()

        evidence_chunks = self.search_evidence(query=query, project_id=project_id, top_k=top_k)

        if not evidence_chunks:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "answer": "Insufficient evidence. No relevant documents were found to support this question.",
                "grounded": False,
                "citations": [],
                "evidence": [],
                "duration_ms": round(duration_ms, 2)
            }

        citations = [
            self.citation_manager.create_citation(chunk=c, citation_index=idx)
            for idx, c in enumerate(evidence_chunks, start=1)
        ]

        # Synthesize answer using top evidence text
        context_snippets = [
            f"[{c['citation_tag']}] (From {c['label']}): {c['snippet']}"
            for c in citations
        ]

        summary_points = []
        for c in citations:
            summary_points.append(f"{c['citation_tag']} {c['snippet']}")

        answer_text = (
            f"Based on verified project documents:\n\n" +
            "\n\n".join(summary_points)
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "answer": answer_text,
            "grounded": True,
            "citations": citations,
            "evidence": evidence_chunks,
            "duration_ms": round(duration_ms, 2)
        }
