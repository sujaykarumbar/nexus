"""
NEXUS SQLAlchemy model for persisted streaming anomaly alerts.
"""

from sqlalchemy import Column, String, Float, Integer, Text, ForeignKey, Index
from sqlalchemy.orm import relationship

from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class StreamingAlert(Base, TimestampMixin):
    __tablename__ = "streaming_alerts"

    session_id   = Column(String(36), nullable=False, index=True)
    dataset_id   = Column(String(36), ForeignKey("datasets.id"), nullable=True, index=True)
    anomaly_id   = Column(String(36), nullable=False, unique=True)
    timestamp    = Column(String(50), nullable=False)
    column       = Column(String(255), nullable=False)
    value        = Column(Float, nullable=False)
    z_score      = Column(Float, nullable=False)
    iqr_score    = Column(Float, nullable=True)
    severity     = Column(String(20), nullable=False)   # low | medium | high | critical
    anomaly_type = Column(String(30), nullable=False)   # z_score | iqr | combined
    window_mean  = Column(Float, nullable=True)
    window_std   = Column(Float, nullable=True)
    window_size  = Column(Integer, nullable=True)

    # Relationships
    dataset = relationship("Dataset", foreign_keys=[dataset_id])

    __table_args__ = (
        Index("ix_streaming_alerts_session_severity", "session_id", "severity"),
    )
