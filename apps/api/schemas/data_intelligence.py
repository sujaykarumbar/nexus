from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class SchemaColumnInfo(BaseModel):
    name: str
    inferred_type: str
    raw_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    is_nullable: bool
    sample_values: List[Any]


class DatasetProfileResponse(BaseModel):
    row_count: int
    column_count: int
    memory_usage_bytes: int
    memory_usage_mb: float
    schema: Dict[str, Any]
    columns: Dict[str, Any]


class DataQualityResponse(BaseModel):
    score: float
    grade: str
    score_breakdown: Dict[str, Any]
    issues: Dict[str, Any]
    recommendations: List[Dict[str, Any]]


class EDAResponse(BaseModel):
    summary: Dict[str, Any]
    correlations: Dict[str, Any]
    distributions: Dict[str, Any]


class InsightItem(BaseModel):
    category: str
    title: str
    description: str
    importance: str
    evidence: Dict[str, Any]
    verification_status: str


class VisualizationSpecItem(BaseModel):
    id: str
    type: str
    title: str
    description: str
    data: Any
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    x_key: Optional[str] = None
    y_key: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    color: Optional[str] = None
    columns: Optional[List[str]] = None


class PaginatedPreviewResponse(BaseModel):
    page: int
    page_size: int
    total_rows: int
    total_pages: int
    total_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    inferred_types: Dict[str, str]
    rows: List[Dict[str, Any]]
