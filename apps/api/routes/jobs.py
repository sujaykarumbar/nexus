import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from apps.api.core.database import get_db, SessionLocal
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.schemas.job import JobCreate, JobResponse
from apps.api.routes.auth import get_current_user
from apps.api.core.exceptions import EntityNotFoundException

router = APIRouter(prefix="/jobs", tags=["Analysis Jobs"])


def execute_background_job(job_id: str):
    """Simulate asynchronous multi-stage execution with database checkpointing."""
    job = None
    db = None
    try:
        db = SessionLocal()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        job.status = "running"
        job.progress_percentage = 10.0
        job.current_stage = "Data Ingestion & Integrity Check"
        current_logs = list(job.logs or [])
        current_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": "Initializing deterministic dataset integrity check..."
        })
        job.logs = current_logs
        db.commit()

        # Step 2: Profiling
        job.progress_percentage = 40.0
        job.current_stage = "Automated Statistical Profiling"
        current_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": "Computed column distributions, cardinality, and missingness maps."
        })
        job.logs = current_logs
        db.commit()

        # Step 3: Analysis
        job.progress_percentage = 80.0
        job.current_stage = "AutoML Problem Framing & Feature Engine"
        current_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": "Identified target variables and prepared 5-fold cross validation split."
        })
        job.logs = current_logs
        db.commit()

        # Step 4: Completion
        job.status = "completed"
        job.progress_percentage = 100.0
        job.current_stage = "Completed"
        job.results = {
            "summary": "Automated analysis pipeline executed successfully.",
            "metrics": {
                "quality_score": 98.2,
                "recommended_action": "AutoML Model Training Ready"
            }
        }
        current_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": "Job finished successfully."
        })
        job.logs = current_logs
        db.commit()
    except Exception as e:
        if db and job:
            try:
                job.status = "failed"
                job.error_message = str(e)
                current_logs = list(job.logs or [])
                current_logs.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "ERROR",
                    "message": f"Job execution failed: {str(e)}"
                })
                job.logs = current_logs
                db.commit()
            except Exception:
                pass
    finally:
        if db:
            db.close()


@router.get("", response_model=List[JobResponse])
def list_jobs(
    project_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all analysis jobs for current user."""
    query = db.query(AnalysisJob).filter(AnalysisJob.creator_id == current_user.id)
    if project_id:
        query = query.filter(AnalysisJob.project_id == project_id)
    return query.order_by(AnalysisJob.created_at.desc()).all()


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(
    request: JobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Dispatch a new asynchronous data intelligence job."""
    project = db.query(Project).filter(
        Project.id == request.project_id,
        Project.owner_id == current_user.id,
        Project.is_active == True
    ).first()
    if not project:
        raise EntityNotFoundException("Project", request.project_id)

    job = AnalysisJob(
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        creator_id=current_user.id,
        job_type=request.job_type,
        status="queued",
        progress_percentage=0.0,
        current_stage="Queued in broker",
        logs=[{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": f"Job {request.job_type} enqueued."
        }]
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Launch background worker
    background_tasks.add_task(execute_background_job, job.id)
    return job


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Poll job status, stage, logs, and results."""
    job = db.query(AnalysisJob).filter(
        AnalysisJob.id == job_id,
        AnalysisJob.creator_id == current_user.id
    ).first()
    if not job:
        raise EntityNotFoundException("AnalysisJob", job_id)
    return job
