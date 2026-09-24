from sqlalchemy import Column, String, Float, Text, ForeignKey, JSON
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class AnomalyEvent(Base, TimestampMixin):
    __tablename__ = "anomaly_events"

    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    score = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    affected_features = Column(JSON, nullable=False)
    explanation = Column(Text, nullable=False)
    raw_record = Column(JSON, nullable=True)
