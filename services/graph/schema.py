"""
NEXUS Knowledge Graph Schema & Domain Models
Defines node types, edge relations, graph nodes, edges, and query payloads.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    DATASET = "dataset"
    COLUMN = "column"
    PROBLEM = "problem"
    MODEL = "model"
    METRIC = "metric"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CONCEPT = "concept"


class RelationType(str, Enum):
    HAS_COLUMN = "HAS_COLUMN"
    CORRELATES_WITH = "CORRELATES_WITH"
    TARGET_OF = "TARGET_OF"
    TRAINED_ON = "TRAINED_ON"
    ACHIEVED_METRIC = "ACHIEVED_METRIC"
    HAS_ANOMALY = "HAS_ANOMALY"
    HAS_FORECAST = "HAS_FORECAST"
    MENTIONS = "MENTIONS"
    DERIVED_FROM = "DERIVED_FROM"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"


class GraphNode(BaseModel):
    id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value if isinstance(self.node_type, NodeType) else str(self.node_type),
            "properties": self.properties,
            "project_id": self.project_id
        }


class GraphEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value if isinstance(self.relation, RelationType) else str(self.relation),
            "weight": self.weight,
            "properties": self.properties
        }


class GraphPath(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    score: float = 1.0
    description: str = ""
