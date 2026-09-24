from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class AnalysisJob(Base, TimestampMixin):
    __tablename__ = "analysis_jobs"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True, index=True)
    creator_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    
    job_type = Column(String(50), nullable=False)  # profiling, eda, automl, forecast, anomaly, multi_agent
    status = Column(String(50), default="queued", nullable=False, index=True)  # queued, running, completed, failed, cancelled
    progress_percentage = Column(Float, default=0.0)
    current_stage = Column(String(100), nullable=True)
    
    # Execution artifacts
    logs = Column(JSON, default=list)
    results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="jobs")
    dataset = relationship("Dataset", back_populates="jobs")
    creator = relationship("User", back_populates="jobs")
    agent_logs = relationship("AgentLog", back_populates="job", cascade="all, delete-orphan")
