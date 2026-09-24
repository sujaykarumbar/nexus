"""
NEXUS Document Intelligence & RAG API Routes
Endpoints for document ingestion, hybrid search, citation management, and grounded Q&A.
"""

import os
import time
import uuid
import hashlib
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session

from apps.api.core.config import settings
from apps.api.core.database import get_db
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.document import Document, DocumentChunk
from apps.api.routes.auth import get_current_user
from apps.api.schemas.rag import (
    DocumentUploadResponse,
    DocumentDetailResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    CitationSchema
)
from services.rag.rag_engine import RAGEngine

router = APIRouter(prefix="/rag", tags=["Document Intelligence & RAG"])

# Singleton RAGEngine instance for live index state
_rag_engine = RAGEngine()


def get_rag_engine() -> RAGEngine:
    return _rag_engine


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Ingest, chunk, embed, and index a document (PDF, TXT, MD, DOCX, CSV) into hybrid search.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    filename = file.filename or "document.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    # Default to first project if not provided
    if not project_id:
        proj = db.query(Project).filter(Project.owner_id == current_user.id).first()
        if not proj:
            proj = Project(
                id=str(uuid.uuid4()),
                name="Default Workspace",
                owner_id=current_user.id
            )
            db.add(proj)
            db.commit()
            db.refresh(proj)
        project_id = proj.id

    # Verify project ownership
    project = db.query(Project).filter(Project.id == project_id, Project.owner_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found or access denied.")

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    file_hash = hashlib.sha256(content).hexdigest()

    # Index document with RAG engine
    try:
        ingest_result = engine.ingest_document(
            raw_bytes=content,
            filename=filename,
            file_type=ext,
            project_id=project_id,
            document_id=doc_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process document: {str(e)}"
        )

    # Persist document record in SQL database
    db_doc = Document(
        id=doc_id,
        project_id=project_id,
        filename=filename,
        file_type=ext,
        file_size=len(content),
        sha256_hash=file_hash,
        page_count=ingest_result.get("page_count", 1),
        chunk_count=ingest_result.get("chunks_created", 0),
        status="indexed",
        metadata_json=ingest_result
    )
    db.add(db_doc)

    # Persist chunks in SQL database
    doc_chunks = engine.chunker.chunk_document({
        "document_id": doc_id,
        "filename": filename,
        "text": content.decode("utf-8", errors="ignore"),
        "page_count": ingest_result.get("page_count", 1),
        "pages": [{"page_number": 1, "text": content.decode("utf-8", errors="ignore")}]
    })

    for chk in doc_chunks:
        db_chunk = DocumentChunk(
            id=chk.get("chunk_id", str(uuid.uuid4())),
            document_id=doc_id,
            chunk_index=chk.get("chunk_index", 0),
            page_number=chk.get("page_number", 1),
            section_title=chk.get("section_title", "General"),
            text_content=chk.get("text", ""),
            token_count=chk.get("token_count", 0),
            metadata_json=chk.get("metadata", {})
        )
        db.add(db_chunk)

    db.commit()

    return DocumentUploadResponse(
        document_id=doc_id,
        project_id=project_id,
        filename=filename,
        file_type=ext,
        file_size=len(content),
        sha256_hash=file_hash,
        page_count=ingest_result.get("page_count", 1),
        chunks_created=ingest_result.get("chunks_created", 0),
        duration_ms=ingest_result.get("duration_ms", 0.0),
        status="indexed"
    )


@router.get("/documents", response_model=List[DocumentDetailResponse])
def list_documents(
    project_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all ingested documents for the project or user.
    """
    query = db.query(Document).join(Project).filter(Project.owner_id == current_user.id)
    if project_id:
        query = query.filter(Document.project_id == project_id)

    docs = query.order_by(Document.created_at.desc()).all()
    return [
        DocumentDetailResponse(
            document_id=d.id,
            project_id=d.project_id,
            filename=d.filename,
            file_type=d.file_type,
            file_size=d.file_size or 0,
            sha256_hash=d.sha256_hash,
            page_count=d.page_count or 1,
            chunk_count=d.chunk_count or 0,
            status=d.status,
            created_at=d.created_at.isoformat() if d.created_at else None
        )
        for d in docs
    ]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Delete a document and its indexed vectors from the knowledge store.
    """
    doc = db.query(Document).join(Project).filter(
        Document.id == document_id,
        Project.owner_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    db.delete(doc)
    db.commit()

    return {"status": "deleted", "document_id": document_id}


@router.post("/search", response_model=HybridSearchResponse)
def hybrid_search(
    req: HybridSearchRequest,
    current_user: User = Depends(get_current_user),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Perform hybrid dense + BM25 keyword search with cross-entropy reranking.
    """
    start_t = time.perf_counter()
    results = engine.search_evidence(
        query=req.query,
        project_id=req.project_id,
        top_k=req.top_k,
        filters=req.filters
    )
    dur = (time.perf_counter() - start_t) * 1000.0

    return HybridSearchResponse(
        query=req.query,
        total_found=len(results),
        results=results,
        duration_ms=round(dur, 2)
    )


@router.post("/query", response_model=RAGQueryResponse)
def rag_grounded_query(
    req: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Execute grounded RAG query returning synthesized answer with verified inline citations.
    """
    result = engine.answer_query(
        query=req.query,
        project_id=req.project_id,
        top_k=req.top_k
    )

    citations = [
        CitationSchema(
            citation_id=c["citation_id"],
            citation_tag=c["citation_tag"],
            label=c["label"],
            claim=c.get("claim", ""),
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            filename=c["filename"],
            page=c["page"],
            section=c.get("section", "General"),
            confidence=c["confidence"],
            snippet=c["snippet"]
        )
        for c in result.get("citations", [])
    ]

    return RAGQueryResponse(
        query=req.query,
        answer=result.get("answer", ""),
        grounded=result.get("grounded", False),
        citations=citations,
        evidence=result.get("evidence", []),
        duration_ms=result.get("duration_ms", 0.0)
    )
