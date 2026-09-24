from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    project_id: str


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    file_path: str
    file_type: str
    file_size_bytes: int
    row_count: int
    column_count: int
    data_quality_score: Optional[float] = None
    detected_problem_type: Optional[str] = None
    target_column: Optional[str] = None
    schema_metadata: Optional[Dict[str, Any]] = None
    eda_summary: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    rows: List[Dict[str, Any]]
    total_rows: int
    total_columns: int
