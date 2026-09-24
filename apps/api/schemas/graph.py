"""
NEXUS Knowledge Graph API Schemas
Pydantic validation schemas for Graph endpoints, GraphRAG queries, and lineage visualization.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GraphNodeOut(BaseModel):
    id: str
    name: str
    node_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None


class GraphEdgeOut(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphResponse(BaseModel):
    project_id: Optional[str] = None
    nodes: List[GraphNodeOut]
    edges: List[GraphEdgeOut]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class GraphBuildRequest(BaseModel):
    project_id: str
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    correlation_threshold: float = 0.25
    include_models: bool = True
    include_anomalies: bool = True
    include_forecasts: bool = True


class GraphBuildResponse(BaseModel):
    project_id: str
    dataset_id: Optional[str] = None
    nodes_created: int
    edges_created: int
    total_nodes: int
    total_edges: int
    duration_ms: float
    status: str = "COMPLETED"


class GraphRAGQueryRequest(BaseModel):
    query: str
    project_id: Optional[str] = None
    max_hops: int = 2
    top_k_entities: int = 4


class GraphRAGQueryResponse(BaseModel):
    answer: str
    grounded: bool
    seed_entities: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    subgraph: Dict[str, Any] = Field(default_factory=dict)
    paths: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float


class NeighborhoodResponse(BaseModel):
    root_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_count: int
    edge_count: int
