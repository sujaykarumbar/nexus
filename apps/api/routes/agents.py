"""
NEXUS Multi-Agent Swarm API Routes
Endpoints for swarm execution, live DAG status, and Critic mathematical verification.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.models.agent_log import AgentLog
from apps.api.routes.auth import get_current_user
from apps.api.schemas.agent import (
    AgentDefinition,
    SwarmRunRequest,
    SwarmRunResponse,
    SwarmStepTrace,
    CriticVerifyRequest,
    CriticVerificationResult,
    CriticAuditItem,
    RecommendationItem
)
from services.agents.critic_agent import CriticAgent
from services.agents.swarm_orchestrator import SwarmOrchestrator
from services.ingestion.factory import IngestionFactory

router = APIRouter(prefix="/agents", tags=["Agent Swarm"])

# In-memory execution store for instant retrieval & live polling
_swarm_runs: Dict[str, Dict[str, Any]] = {}

AGENT_REGISTRY = [
    {
        "id": "data_agent",
        "name": "Data Agent",
        "role": "Schema inference, data quality scoring, missingness analysis & hygiene.",
        "status": "idle",
        "tools": ["profile_dataset", "detect_missing_values", "detect_outliers", "clean_data"],
        "color": "#3B82F6"
    },
    {
        "id": "eda_agent",
        "name": "Data Scientist Agent",
        "role": "Automated statistical distributions, correlation maps, and hypothesis tests.",
        "status": "idle",
        "tools": ["compute_correlations", "distribution_test", "cardinality_check", "skew_analysis"],
        "color": "#10B981"
    },
    {
        "id": "ml_agent",
        "name": "ML Engineer Agent",
        "role": "AutoML training, multi-metric benchmarking, Optuna hyperparameter optimization.",
        "status": "idle",
        "tools": ["train_automl", "cross_validate", "tune_hyperparameters", "calculate_shap"],
        "color": "#8B5CF6"
    },
    {
        "id": "forecast_agent",
        "name": "Forecasting Agent",
        "role": "Time-series trend decomposition, stationarity checks & prediction intervals.",
        "status": "idle",
        "tools": ["decompose_series", "fit_arima", "fit_prophet", "evaluate_mape"],
        "color": "#F59E0B"
    },
    {
        "id": "anomaly_agent",
        "name": "Anomaly Agent",
        "role": "Unsupervised outlier detection & multi-dimensional anomaly ranking.",
        "status": "idle",
        "tools": ["isolation_forest", "dbscan_outliers", "rank_severity", "explain_anomaly"],
        "color": "#EF4444"
    },
    {
        "id": "rag_agent",
        "name": "Research & RAG Agent",
        "role": "Domain document retrieval, hybrid dense/sparse search & verified citations.",
        "status": "idle",
        "tools": ["retrieve_evidence", "answer_question", "ingest_document"],
        "color": "#6366F1"
    },
    {
        "id": "critic_agent",
        "name": "Critic Agent (Gatekeeper)",
        "role": "Strict mathematical evidence checking, hallucination rejection & claim verification.",
        "status": "active",
        "tools": ["verify_claim", "audit_metrics", "reject_hallucination", "validate_confidence"],
        "color": "#EC4899"
    },
    {
        "id": "recommendation_agent",
        "name": "Recommendation & Report Agent",
        "role": "Actionable decision intelligence with quantified risks and executive dossiers.",
        "status": "idle",
        "tools": ["formulate_actions", "quantify_risk", "estimate_roi", "generate_dossier"],
        "color": "#06B6D4"
    }
]


@router.get("", response_model=List[AgentDefinition])
def get_agent_swarm_status(current_user: User = Depends(get_current_user)):
    """Retrieve multi-agent swarm architecture status and available tools."""
    return AGENT_REGISTRY


@router.post("/swarm/run", response_model=SwarmRunResponse)
def run_agent_swarm(
    req: SwarmRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger full autonomous multi-agent swarm DAG execution on a dataset.
    """
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == req.dataset_id,
        Project.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or access denied."
        )

    if not os.path.exists(dataset.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset file missing on disk: {dataset.file_path}"
        )

    # Load dataset into pandas dataframe
    try:
        loader = IngestionFactory.get_adapter(dataset.file_type)
        df = loader.load(dataset.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse dataset file: {str(e)}"
        )

    orchestrator = SwarmOrchestrator()
    result = orchestrator.run_swarm(
        df=df,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        project_name=dataset.project.name if dataset.project else "Project",
        dataset_name=dataset.name,
        target_column=req.target_column,
        time_column=req.time_column,
        goal=req.goal,
        run_automl=req.run_automl,
        run_forecasting=req.run_forecasting,
        run_anomalies=req.run_anomalies,
        run_rag=req.run_rag,
        document_query=req.document_query
    )

    # Cache execution run
    job_id = result["job_id"]
    _swarm_runs[job_id] = result

    # Persist job and agent logs in database
    db_job = AnalysisJob(
        id=job_id,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        creator_id=current_user.id,
        job_type="multi_agent",
        status=result["status"],
        progress_percentage=100.0,
        current_stage="Completed",
        results={
            "status": result["status"],
            "critic_verdict": result["critic_verdict"],
            "recommendations_count": len(result["recommendations"])
        }
    )
    db.add(db_job)

    # Record agent logs
    for step in result["trace"]:
        db_log = AgentLog(
            job_id=job_id,
            agent_name=step["agent_name"],
            action=step["step_id"],
            input_data={"tools_used": step["tools_used"]},
            output_data=step["data_artifacts"],
            reasoning=step.get("reasoning"),
            verification_status="SUPPORTED" if step["status"] == "verified" else step["status"].upper()
        )
        db.add(db_log)

    db.commit()

    return result


@router.get("/swarm/runs/{job_id}", response_model=SwarmRunResponse)
def get_swarm_run_trace(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve live execution trace and Critic verification logs for a swarm run.
    """
    if job_id in _swarm_runs:
        return _swarm_runs[job_id]

    # Look up in DB
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Swarm run job '{job_id}' not found."
        )

    # Reconstruct from DB
    trace = [
        SwarmStepTrace(
            step_id=l.action,
            agent_id=l.agent_name.lower().replace(" ", "_"),
            agent_name=l.agent_name,
            status=l.verification_status.lower() if l.verification_status else "completed",
            started_at=l.created_at.isoformat() if l.created_at else "",
            summary=f"Action: {l.action}",
            tools_used=l.input_data.get("tools_used", []) if l.input_data else [],
            reasoning=l.reasoning,
            data_artifacts=l.output_data or {}
        )
        for l in job.agent_logs
    ]

    return SwarmRunResponse(
        job_id=job.id,
        project_id=job.project_id,
        dataset_id=job.dataset_id or "",
        status=job.status,
        progress_percentage=job.progress_percentage or 100.0,
        current_step=job.current_stage,
        trace=trace,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.updated_at.isoformat() if job.updated_at else None
    )


@router.post("/critic/verify", response_model=CriticVerificationResult)
def verify_claim_standalone(
    req: CriticVerifyRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Directly audit a batch of numerical claims against computed ground-truth metrics.
    """
    critic = CriticAgent()
    return critic.verify_artifacts(
        claims=req.claims,
        computed_metrics=req.computed_metrics
    )
