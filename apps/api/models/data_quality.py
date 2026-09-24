from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class DataQualityReport(Base, TimestampMixin):
    __tablename__ = "data_quality_reports"

    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    overall_score = Column(Float, nullable=False)
    grade = Column(String(10), nullable=False)
    score_breakdown = Column(JSON, nullable=False)  # completeness, uniqueness, validity, etc.
    issues_summary = Column(JSON, nullable=False)    # missing values, duplicate counts, outliers
    recommendations = Column(JSON, nullable=False)   # structured cleaning recommendations

    # Relationships
    dataset = relationship("Dataset", back_populates="quality_reports")
