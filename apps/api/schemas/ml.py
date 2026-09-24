from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class TargetSuggestionItem(BaseModel):
    column: str
    confidence: float
    suggested_type: str
    unique_count: int
    null_count: int
    reasons: List[str]
    sample_values: List[str]


class TargetSuggestionsResponse(BaseModel):
    dataset_id: str
    total_columns: int
    suggestions: List[TargetSuggestionItem]


class LeakageWarningItem(BaseModel):
    column: str
    risk_type: str
    severity: str
    description: str
    recommendation: str
    correlation: Optional[float] = None


class LeakageAuditResponse(BaseModel):
    has_leakage_risk: bool
    has_critical_risk: bool
    target_column: str
    warning_count: int
    warnings: List[LeakageWarningItem]
    recommended_drop_columns: List[str]


class MLTrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    candidate_algorithms: Optional[List[str]] = None
    excluded_columns: Optional[List[str]] = None
    cv_splits: int = Field(5, ge=2, le=10)
    optimize_hyperparameters: bool = False
    optuna_trials: int = Field(10, ge=3, le=50)


class ModelLeaderboardItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    algorithm: str
    display_name: str
    primary_metric_name: str
    primary_metric_value: float
    metrics: Dict[str, Any]
    cv_summary: Dict[str, Any]
    delta_improvement_pct: float
    training_duration_seconds: float
    inference_latency_ms: float
    is_best: bool


class MLTrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class MLModelSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    id: str
    project_id: str
    dataset_id: str
    name: str
    algorithm: str
    problem_type: str
    target_column: str
    version: str
    lifecycle_stage: str
    primary_metric_name: str
    primary_metric_value: float
    delta_improvement_pct: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    created_at: datetime


class MLModelDetailResponse(MLModelSummaryResponse):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    all_metrics: Dict[str, Any]
    cv_scores: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[Dict[str, Any]] = None
    roc_curve: Optional[List[Dict[str, float]]] = None
    actual_vs_pred: Optional[List[Dict[str, float]]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    feature_names: Optional[List[str]] = None
    model_card: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    status: str


class SinglePredictRequest(BaseModel):
    features: Dict[str, Any]


class ContributingFeatureItem(BaseModel):
    feature: str
    feature_value: Any
    impact: float
    direction: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_version: str
    prediction: Any
    probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    contributing_features: List[ContributingFeatureItem] = []
    latency_ms: float


class PromoteModelRequest(BaseModel):
    lifecycle_stage: str  # STAGING, PRODUCTION, ARCHIVED
