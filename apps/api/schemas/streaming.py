"""
NEXUS Streaming API Schemas (Pydantic v2)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────────────────

class StreamStartRequest(BaseModel):
    dataset_id: str
    target_columns: Optional[List[str]] = Field(
        default=None,
        description="Numeric columns to monitor. If None, all numeric columns are used."
    )
    rows_per_second: float = Field(default=5.0, ge=0.1, le=500.0)
    window_size: int = Field(default=50, ge=10, le=500)
    z_threshold: float = Field(default=2.8, ge=1.5, le=5.0)
    spike_probability: float = Field(default=0.04, ge=0.0, le=0.5)
    spike_factor: float = Field(default=4.0, ge=2.0, le=10.0)


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class StreamSessionResponse(BaseModel):
    session_id: str
    dataset_id: str
    status: str                         # queued | running | stopped | error
    rows_per_second: float
    window_size: int
    z_threshold: float
    columns_monitored: List[str]
    records_processed: int
    anomalies_detected: int
    anomaly_rate: float
    spikes_injected: int
    message: Optional[str] = None


class StreamingAlertResponse(BaseModel):
    anomaly_id: str
    session_id: str
    timestamp: str
    column: str
    value: float
    z_score: float
    iqr_score: Optional[float] = None
    severity: str
    anomaly_type: str
    window_mean: Optional[float] = None
    window_std: Optional[float] = None
    window_size: Optional[int] = None
    created_at: Optional[str] = None


class StreamAlertsResponse(BaseModel):
    session_id: str
    total: int
    alerts: List[StreamingAlertResponse]
    severity_counts: Dict[str, int]
