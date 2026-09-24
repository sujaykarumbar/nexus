from sqlalchemy import Column, String, Float, Text, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class MLModel(Base, TimestampMixin):
    __tablename__ = "ml_models"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    algorithm = Column(String(100), nullable=False)
    problem_type = Column(String(50), nullable=False)  # binary_classification, multiclass_classification, regression
    target_column = Column(String(255), nullable=False)
    version = Column(String(50), default="v1.0", nullable=False)
    lifecycle_stage = Column(String(50), default="REGISTERED", nullable=False)  # TRAINED, VALIDATED, REGISTERED, STAGING, PRODUCTION, ARCHIVED
    
    # Quantitative performance metrics (deterministic)
    primary_metric_name = Column(String(50), nullable=False)  # f1, accuracy, roc_auc, r2, rmse
    primary_metric_value = Column(Float, nullable=False)
    all_metrics = Column(JSON, nullable=False)  # full evaluation dict
    cv_scores = Column(JSON, nullable=True)  # cross validation summary
    delta_improvement_pct = Column(Float, nullable=True)  # improvement over baseline %
    
    # Visual & diagnostic artifacts
    confusion_matrix = Column(JSON, nullable=True)
    roc_curve = Column(JSON, nullable=True)
    actual_vs_pred = Column(JSON, nullable=True)
    
    # Model configuration & governance
    hyperparameters = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)  # global feature attribution
    feature_names = Column(JSON, nullable=True)  # active feature names
    model_card = Column(JSON, nullable=True)  # full generated model card
    artifact_path = Column(String(500), nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    status = Column(String(50), default="ready", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="models")
    dataset = relationship("Dataset", back_populates="models")
