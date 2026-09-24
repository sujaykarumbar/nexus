from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(50), default="active", nullable=False)

    # Relationships
    owner = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("AnalysisJob", back_populates="project", cascade="all, delete-orphan")
    models = relationship("MLModel", back_populates="project", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")

