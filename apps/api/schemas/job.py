from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class JobCreate(BaseModel):
    project_id: str
    dataset_id: Optional[str] = None
    job_type: str  # profiling, eda, automl, forecast, anomaly, multi_agent


class JobResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: Optional[str] = None
    creator_id: str
    job_type: str
    status: str
    progress_percentage: float
    current_stage: Optional[str] = None
    logs: List[Dict[str, Any]]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
