"""
NEXUS Forecasting API Schemas
Pydantic validation schemas for time-series detection, model training,
leaderboard comparisons, and multi-step future forecasting with prediction intervals.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class TimeSeriesDetectRequest(BaseModel):
    dataset_id: str
    time_column: Optional[str] = None
    target_column: Optional[str] = None


class CandidateTimeColumn(BaseModel):
    column_name: str
    confidence_score: float
    is_parsed: bool
    reasons: List[str]


class TimeSeriesDetectResponse(BaseModel):
    dataset_id: str
    candidates: List[CandidateTimeColumn]
    profile: Optional[Dict[str, Any]] = None


class ForecastTrainRequest(BaseModel):
    project_id: str
    dataset_id: str
    time_column: str
    target_column: str
    horizon: int = Field(default=7, ge=1, le=90)
    candidate_models: Optional[List[str]] = Field(default=None)
    covariates: Optional[List[str]] = Field(default=None)


class ForecastIntervalPoint(BaseModel):
    step: int
    timestamp: str
    prediction: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    uncertainty_sigma: float
    estimation_method: str
    disclaimer: str


class ForecastModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    project_id: str
    dataset_id: str
    time_column: str
    target_column: str
    horizon: int
    best_model_name: str
    lifecycle_stage: str
    metrics: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    temporal_profile: Optional[Dict[str, Any]] = None
    future_forecast: List[Dict[str, Any]]
    recent_history: Optional[List[Dict[str, Any]]] = None
    validation_comparison: Optional[List[Dict[str, Any]]] = None
    feature_importances: Optional[Dict[str, float]] = None
    training_duration_seconds: Optional[float] = None
    created_at: Optional[str] = None


class DynamicForecastRequest(BaseModel):
    horizon: int = Field(default=7, ge=1, le=90)


class DynamicForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    best_model_name: str
    horizon: int
    forecast: List[Dict[str, Any]]
