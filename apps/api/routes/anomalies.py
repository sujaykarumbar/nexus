"""
NEXUS Anomaly Detection API Routes
Endpoints for statistical, ML, and deep autoencoder anomaly scanning,
severity breakdowns, change-point identification, and timeline diagnostics.
"""

import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db, SessionLocal
from apps.api.core.exceptions import EntityNotFoundException, ValidationException
from apps.api.routes.auth import get_current_user
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.models.forecast import AnomalyReport

from apps.api.schemas.anomaly import (
    AnomalyDetectRequest,
    AnomalyReportResponse,
    AnomalySummary,
    ChangePointItem
)
from services.ingestion.factory import IngestionFactory
from services.anomaly_detection.engine import AnomalyDetectionEngine

router = APIRouter(prefix="/anomalies", tags=["Anomaly Detection Engine"])


def _run_anomaly_detection_task(
    project_id: str,
    dataset_id: str,
    time_column: Optional[str],
    target_column: Optional[str],
    feature_columns: Optional[List[str]],
    include_deep_learning: bool,
    job_id: str,
    db: Optional[Session] = None
):
    """Background task executing multi-detector anomaly scan."""
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            if job:
                job.status = "failed"
                job.error_message = f"Dataset {dataset_id} not found."
                db.commit()
            return

        adapter = IngestionFactory.get_adapter(dataset.file_type)
        df = adapter.load(dataset.file_path)

        engine = AnomalyDetectionEngine()
        result = engine.detect(
            df=df,
            time_col=time_column,
            target_col=target_column,
            feature_cols=feature_columns,
            include_deep_learning=include_deep_learning
        )

        report = AnomalyReport(
            id=str(uuid.uuid4()),
            project_id=project_id,
            dataset_id=dataset_id,
            analyzed_metric=result["analyzed_metric"],
            summary=result["dataset_summary"],
            anomalies=result["anomalies"],
            change_points=result["change_points"],
            timeline=result["timeline"]
        )
        db.add(report)

        if job:
            job.status = "completed"
            job.progress_percentage = 100.0
            job.current_stage = "COMPLETED"
            job.results = result["dataset_summary"]
        db.commit()

    except Exception as e:
        db.rollback()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
    finally:
        if should_close:
            db.close()


@router.post("/detect", status_code=status.HTTP_202_ACCEPTED)
def start_anomaly_detection(
    payload: AnomalyDetectRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Triggers consensus anomaly detection scan across dataset.
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", payload.dataset_id)

    project = db.query(Project).filter(Project.id == payload.project_id).first()
    if not project or (project.owner_id != current_user.id and not current_user.is_superuser):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this project.")

    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        creator_id=current_user.id,
        job_type="ANOMALY_DETECTION",
        status="queued",
        current_stage="QUEUED",
        progress_percentage=0.0
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(
        _run_anomaly_detection_task,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        time_column=payload.time_column,
        target_column=payload.target_column,
        feature_columns=payload.feature_columns,
        include_deep_learning=payload.include_deep_learning,
        job_id=job_id
    )

    return {
        "message": "Anomaly detection scan dispatched.",
        "job_id": job_id,
        "status": "queued"
    }


@router.get("/jobs/{job_id}")
def get_anomaly_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Checks status of an anomaly detection scan job.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise EntityNotFoundException("AnalysisJob", job_id)

    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress_percentage,
        "stage": job.current_stage,
        "error_message": job.error_message,
        "result_summary": job.results,
        "created_at": job.created_at.isoformat() if job.created_at else None
    }


@router.get("/{dataset_id}", response_model=AnomalyReportResponse)
def get_latest_anomaly_report(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the most recent anomaly report for a dataset.
    """
    report = db.query(AnomalyReport).filter(AnomalyReport.dataset_id == dataset_id).order_by(AnomalyReport.created_at.desc()).first()
    if not report:
        raise EntityNotFoundException("AnomalyReport", dataset_id)

    project = db.query(Project).filter(Project.id == report.project_id).first()
    if not project or (project.owner_id != current_user.id and not current_user.is_superuser):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this report.")

    return AnomalyReportResponse(
        id=report.id,
        project_id=report.project_id,
        dataset_id=report.dataset_id,
        analyzed_metric=report.analyzed_metric,
        dataset_summary=report.summary,
        anomalies=report.anomalies,
        change_points=report.change_points or [],
        timeline=report.timeline,
        created_at=report.created_at.isoformat() if report.created_at else None
    )


@router.get("/{dataset_id}/summary")
def get_anomaly_summary(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns counts and severity breakdown for dataset anomalies.
    """
    report = db.query(AnomalyReport).filter(AnomalyReport.dataset_id == dataset_id).order_by(AnomalyReport.created_at.desc()).first()
    if not report:
        raise EntityNotFoundException("AnomalyReport", dataset_id)

    return report.summary


@router.get("/{dataset_id}/timeline")
def get_anomaly_timeline(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns timeline chart points with anomaly markers.
    """
    report = db.query(AnomalyReport).filter(AnomalyReport.dataset_id == dataset_id).order_by(AnomalyReport.created_at.desc()).first()
    if not report:
        raise EntityNotFoundException("AnomalyReport", dataset_id)

    return {"timeline": report.timeline or [], "analyzed_metric": report.analyzed_metric}


@router.get("/{dataset_id}/change-points", response_model=List[Dict[str, Any]])
def get_anomaly_change_points(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns detected regime shift change points.
    """
    report = db.query(AnomalyReport).filter(AnomalyReport.dataset_id == dataset_id).order_by(AnomalyReport.created_at.desc()).first()
    if not report:
        raise EntityNotFoundException("AnomalyReport", dataset_id)

    return report.change_points or []
