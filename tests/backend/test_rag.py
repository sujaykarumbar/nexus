"""
Tests for NEXUS RAG Engine & Document Intelligence Components
"""

import pytest
from services.rag.document_processor import DocumentProcessor
from services.rag.chunker import DocumentChunker
from services.rag.embeddings import FastDenseEmbeddingProvider
from services.rag.vector_store import VectorStore
from services.rag.bm25_retriever import BM25Retriever
from services.rag.hybrid_search import HybridSearchEngine
from services.rag.reranker import Reranker
from services.rag.citations import CitationManager
from services.rag.rag_engine import RAGEngine


def test_document_processor_bytes():
    processor = DocumentProcessor()
    text = "# Project Alpha\nThis is a mission-critical analytical document on customer churn.\nSection 2 covers retention strategies."
    doc = processor.process_bytes(
        raw_bytes=text.encode("utf-8"),
        filename="project_alpha.md",
        file_type="md",
        file_size=len(text),
        project_id="proj_1"
    )
    assert doc["filename"] == "project_alpha.md"
    assert doc["page_count"] >= 1
    assert "retention strategies" in doc["full_text"]


def test_document_chunker():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    doc = {
        "document_id": "doc_123",
        "filename": "test.txt",
        "sections": [{
            "title": "Overview",
            "page": 1,
            "text": "Paragraph one has valuable context about machine learning algorithms and feature selection. " * 3
        }]
    }
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    assert chunks[0]["document_id"] == "doc_123"
    assert "text" in chunks[0]
    assert chunks[0]["token_count"] > 0


def test_embedding_and_vector_store():
    embedder = FastDenseEmbeddingProvider(dim=64)
    v1 = embedder.embed_text("customer churn prediction model")
    v2 = embedder.embed_text("churn risk rate analysis")
    v3 = embedder.embed_text("astronomy telescope galaxy image")

    sim_high = embedder.cosine_similarity(v1, v2)
    sim_low = embedder.cosine_similarity(v1, v3)
    assert sim_high > sim_low

    vstore = VectorStore()
    vstore.insert_batch([
        {"chunk_id": "c1", "text": "customer churn rate", "vector": v1, "document_id": "doc_1", "metadata": {"doc": "1"}},
        {"chunk_id": "c2", "text": "astronomy galaxy", "vector": v3, "document_id": "doc_2", "metadata": {"doc": "2"}}
    ])
    results = vstore.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"


def test_bm25_retriever():
    bm25 = BM25Retriever()
    bm25.add_documents([
        {"chunk_id": "c1", "text": "Customer retention rate increased after promotional email campaign"},
        {"chunk_id": "c2", "text": "Quarterly financial balance sheet and revenue disclosures"}
    ])
    results = bm25.search("retention promotional", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_hybrid_search_and_reranker():
    embedder = FastDenseEmbeddingProvider(dim=64)
    vstore = VectorStore()
    bm25 = BM25Retriever()

    docs = [
        {"chunk_id": "c1", "text": "Machine learning model optimization using Optuna hyperband sampler."},
        {"chunk_id": "c2", "text": "Time-series forecasting with ARIMA and Prophet decomposition."}
    ]
    for d in docs:
        d["vector"] = embedder.embed_text(d["text"])
    vstore.insert_batch(docs)
    bm25.add_documents(docs)

    hybrid = HybridSearchEngine(vector_store=vstore, bm25_retriever=bm25, embedding_provider=embedder)
    candidates = hybrid.search(query="Optuna hyperparameter optimization", top_k=2)
    assert len(candidates) >= 1

    reranker = Reranker()
    ranked = reranker.rerank(query="Optuna hyperparameter optimization", candidates=candidates, top_n=1)
    assert len(ranked) == 1
    assert ranked[0]["chunk_id"] == "c1"


def test_citations_manager():
    chunk = {
        "chunk_id": "chk_99",
        "document_id": "doc_1",
        "text": "The verified churn rate for Q3 dropped by 4.2% following onboarding optimization.",
        "score": 0.92,
        "metadata": {
            "filename": "q3_report.pdf",
            "page": 4,
            "section": "Retention Analysis"
        }
    }
    cite = CitationManager.create_citation(chunk=chunk, citation_index=1)
    assert cite["citation_tag"] == "[1]"
    assert cite["filename"] == "q3_report.pdf"
    assert cite["page"] == 4
    assert cite["confidence"] == 0.92

    md = CitationManager.format_citations_markdown([cite])
    assert "[1]" in md
    assert "q3_report.pdf" in md


def test_rag_engine_lifecycle():
    engine = RAGEngine()
    sample_text = (
        "Project NEXUS Architecture Specification.\n"
        "NEXUS uses a deterministic Critic Verification Layer to prevent hallucination.\n"
        "Model training benchmark requires 5-fold cross-validation."
    )
    ingest = engine.ingest_document(
        raw_bytes=sample_text.encode("utf-8"),
        filename="nexus_spec.txt",
        file_type="txt",
        project_id="proj_test"
    )
    assert ingest["status"] == "COMPLETED"
    assert ingest["chunks_created"] >= 1

    # Question answering
    result = engine.answer_query(
        query="How does NEXUS prevent hallucination?",
        project_id="proj_test",
        top_k=2
    )
    assert result["grounded"] is True
    assert len(result["citations"]) >= 1
    assert "Critic Verification Layer" in result["answer"] or "critic" in result["answer"].lower()
