"""
NEXUS Anomaly Detection API Schemas
Pydantic validation schemas for anomaly scans, severity distributions,
change-point detection, and diagnostic evidence explanations.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnomalyDetectRequest(BaseModel):
    project_id: str
    dataset_id: str
    time_column: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    include_deep_learning: bool = True


class AnomalyEventItem(BaseModel):
    id: str
    index: int
    timestamp: str
    metric_name: str
    actual: float
    expected: float
    deviation: float
    anomaly_score: float
    severity: str # LOW, MEDIUM, HIGH, CRITICAL
    detection_methods: List[str]
    explanation: str
    recommendation: str


class ChangePointItem(BaseModel):
    index: int
    timestamp: str
    disparity_score: float
    previous_mean: float
    new_mean: float
    magnitude: float
    percentage_change: float
    regime_type: str
    confidence: float


class AnomalySummary(BaseModel):
    total_observations: int
    total_anomalies: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    anomaly_percentage: float
    change_points_count: int


class AnomalyReportResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    analyzed_metric: str
    dataset_summary: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    change_points: List[Dict[str, Any]]
    timeline: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None
