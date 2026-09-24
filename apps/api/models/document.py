"""
NEXUS Document & Document Chunk Database Models
Persistent storage for Document Intelligence and RAG.
"""

from sqlalchemy import Column, String, Integer, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, md, csv
    file_size = Column(Integer, default=0)
    sha256_hash = Column(String(64), nullable=True, index=True)
    page_count = Column(Integer, default=1)
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default="indexed", nullable=False)  # indexed, processing, error
    metadata_json = Column(JSON, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, default=1)
    section_title = Column(String(255), nullable=True)
    text_content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    metadata_json = Column(JSON, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")
