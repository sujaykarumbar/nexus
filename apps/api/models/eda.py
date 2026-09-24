from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class EDAReport(Base, TimestampMixin):
    __tablename__ = "eda_reports"

    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    summary = Column(JSON, nullable=False)
    correlations = Column(JSON, nullable=False)
    distributions = Column(JSON, nullable=False)
    insights = Column(JSON, nullable=False)
    visualization_specs = Column(JSON, nullable=False)

    # Relationships
    dataset = relationship("Dataset", back_populates="eda_reports")
