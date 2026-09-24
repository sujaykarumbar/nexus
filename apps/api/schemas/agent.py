"""
NEXUS Multi-Agent Swarm & Critic Schemas
Pydantic schemas for DAG execution, live step traces, and critic mathematical audits.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class AgentDefinition(BaseModel):
    id: str
    name: str
    role: str
    status: str = "idle"
    tools: List[str]
    color: str


class SwarmRunRequest(BaseModel):
    project_id: str
    dataset_id: str
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    goal: Optional[str] = "Comprehensive autonomous data analysis and decision intelligence."
    run_automl: bool = True
    run_forecasting: bool = True
    run_anomalies: bool = True
    run_rag: bool = True
    document_query: Optional[str] = None


class SwarmStepTrace(BaseModel):
    step_id: str
    agent_id: str
    agent_name: str
    status: str  # queued, running, completed, verified, rejected, failed
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    summary: str
    tools_used: List[str] = []
    reasoning: Optional[str] = None
    data_artifacts: Dict[str, Any] = {}


class CriticAuditItem(BaseModel):
    claim: str
    metric_name: str
    reported_value: Any
    ground_truth_value: Any
    verified: bool
    status: str  # SUPPORTED, PARTIALLY_SUPPORTED, REJECTED
    delta: Optional[float] = None
    confidence: float
    reason: str


class CriticVerificationResult(BaseModel):
    status: str  # SUPPORTED, PARTIALLY_SUPPORTED, REJECTED
    audit_count: int
    supported_count: int
    rejected_count: int
    overall_confidence: float
    gatekeeper_verdict: str
    audits: List[CriticAuditItem]


class CriticVerifyRequest(BaseModel):
    claims: List[Dict[str, Any]]
    computed_metrics: Dict[str, Any]


class RecommendationItem(BaseModel):
    id: str
    title: str
    category: str  # strategic, risk_mitigation, operational, model_deployment
    action: str
    expected_impact: str
    risk_level: str  # LOW, MEDIUM, HIGH
    confidence: float
    supporting_metrics: List[str]


class SwarmRunResponse(BaseModel):
    job_id: str
    project_id: str
    dataset_id: str
    status: str  # running, completed, failed, rejected_by_critic
    progress_percentage: float
    current_step: Optional[str] = None
    trace: List[SwarmStepTrace]
    critic_verdict: Optional[CriticVerificationResult] = None
    recommendations: List[RecommendationItem] = []
    report_markdown: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
