from sqlalchemy import Column, String, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class PredictionLog(Base, TimestampMixin):
    """Audit log for real-time and batch model predictions."""
    __tablename__ = "prediction_logs"

    model_id = Column(String(36), ForeignKey("ml_models.id"), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    input_data = Column(JSON, nullable=True)  # sanitized feature values
    prediction = Column(String(255), nullable=False)
    probability = Column(Float, nullable=True)
    contributing_features = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)
    batch_id = Column(String(36), nullable=True, index=True)

    model = relationship("MLModel")
