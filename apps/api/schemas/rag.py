"""
NEXUS Document Intelligence & RAG Schemas
Pydantic schemas for document ingestion, hybrid search, citations, and grounded Q&A.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    project_id: Optional[str] = None
    filename: str
    file_type: str
    file_size: int
    sha256_hash: Optional[str] = None
    page_count: int
    chunks_created: int
    duration_ms: float
    status: str


class DocumentDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    project_id: str
    filename: str
    file_type: str
    file_size: int
    sha256_hash: Optional[str] = None
    page_count: int
    chunk_count: int
    status: str
    created_at: Optional[str] = None


class DocumentChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    section_title: Optional[str] = None
    text_content: str
    token_count: int


class HybridSearchRequest(BaseModel):
    query: str
    project_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class HybridSearchResponse(BaseModel):
    query: str
    total_found: int
    results: List[Dict[str, Any]]
    duration_ms: float


class CitationSchema(BaseModel):
    citation_id: str
    citation_tag: str
    label: str
    claim: Optional[str] = ""
    document_id: str
    chunk_id: str
    filename: str
    page: int
    section: Optional[str] = "General"
    confidence: float
    snippet: str


class RAGQueryRequest(BaseModel):
    query: str
    project_id: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    citations: List[CitationSchema]
    evidence: List[Dict[str, Any]]
    duration_ms: float
