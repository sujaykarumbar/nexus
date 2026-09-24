from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # csv, excel, json, parquet
    file_size_bytes = Column(Integer, default=0)
    
    # Statistical metadata (deterministic metrics)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    data_quality_score = Column(Float, nullable=True)
    detected_problem_type = Column(String(50), nullable=True)  # classification, regression, time_series, clustering
    target_column = Column(String(255), nullable=True)
    
    # JSON metadata storage
    schema_metadata = Column(JSON, nullable=True)  # column names, dtypes, null counts
    eda_summary = Column(JSON, nullable=True)       # correlations, distributions, outliers
    
    # Relationships
    project = relationship("Project", back_populates="datasets")
    jobs = relationship("AnalysisJob", back_populates="dataset", cascade="all, delete-orphan")
    models = relationship("MLModel", back_populates="dataset", cascade="all, delete-orphan")
    quality_reports = relationship("DataQualityReport", back_populates="dataset", cascade="all, delete-orphan")
    eda_reports = relationship("EDAReport", back_populates="dataset", cascade="all, delete-orphan")
