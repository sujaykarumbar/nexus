from sqlalchemy import Column, String, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class AgentLog(Base, TimestampMixin):
    __tablename__ = "agent_logs"

    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)  # Data Agent, EDA Agent, ML Agent, Critic Agent
    action = Column(String(255), nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=True)
    verification_status = Column(String(50), nullable=True)  # SUPPORTED, PARTIALLY_SUPPORTED, REJECTED

    # Relationships
    job = relationship("AnalysisJob", back_populates="agent_logs")
