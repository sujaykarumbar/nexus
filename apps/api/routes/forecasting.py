"""
NEXUS Forecasting API Routes
Endpoints for time-series detection, model training, evaluation leaderboards,
uncertainty prediction intervals, and dynamic future forecasting.
"""

import os
import uuid
import joblib
from datetime import datetime, timezone
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
from apps.api.models.forecast import ForecastModel

from apps.api.schemas.forecasting import (
    TimeSeriesDetectRequest,
    TimeSeriesDetectResponse,
    CandidateTimeColumn,
    ForecastTrainRequest,
    ForecastModelResponse,
    DynamicForecastRequest,
    DynamicForecastResponse
)
from services.ingestion.factory import IngestionFactory
from services.forecasting.time_series_detector import TimeSeriesDetector
from services.forecasting.forecaster import ForecastingEngine

router = APIRouter(prefix="/forecast", tags=["Forecasting Engine"])


@router.post("/detect", response_model=TimeSeriesDetectResponse)
def detect_time_series_properties(
    payload: TimeSeriesDetectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Scans dataset to identify temporal columns, sampling frequency, trend, and seasonality.
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", payload.dataset_id)

    # Check project ownership
    project = db.query(Project).filter(Project.id == dataset.project_id).first()
    if not project or (project.owner_id != current_user.id and not current_user.is_superuser):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this dataset.")

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)

    detector = TimeSeriesDetector()
    candidates = detector.detect_candidate_time_columns(df)

    profile = None
    target_time_col = payload.time_column or (candidates[0]["column_name"] if candidates else None)

    if target_time_col and target_time_col in df.columns:
        try:
            profile = detector.analyze_temporal_profile(
                df=df,
                time_col=target_time_col,
                target_col=payload.target_column
            )
        except Exception:
            pass

    return TimeSeriesDetectResponse(
        dataset_id=dataset.id,
        candidates=[CandidateTimeColumn(**c) for c in candidates],
        profile=profile
    )


def _run_forecast_training_task(
    project_id: str,
    dataset_id: str,
    time_column: str,
    target_column: str,
    horizon: int,
    candidate_models: Optional[List[str]],
    covariates: Optional[List[str]],
    job_id: str,
    db: Optional[Session] = None
):
    """Background task executing the forecasting tournament."""
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

        def progress_cb(info: Dict[str, Any]):
            if job:
                job.progress_percentage = float(info.get("progress", 0))
                job.current_stage = f"{info.get('status', 'TRAINING')} ({info.get('current_model', '')})"
                db.commit()

        engine = ForecastingEngine()
        result = engine.train_and_evaluate(
            df=df,
            time_col=time_column,
            target_col=target_column,
            horizon=horizon,
            candidate_models=candidate_models,
            covariates=covariates,
            progress_callback=progress_cb
        )

        # Save to ForecastModel table
        forecast_record = ForecastModel(
            id=result["model_id"],
            project_id=project_id,
            dataset_id=dataset_id,
            time_column=time_column,
            target_column=target_column,
            horizon=horizon,
            best_model_name=result["best_model"],
            lifecycle_stage="TRAINED",
            metrics=result["best_metrics"],
            leaderboard=result["leaderboard"],
            temporal_profile=result.get("temporal_profile"),
            future_forecast=result["future_forecast"],
            recent_history=result.get("recent_history"),
            validation_comparison=result.get("validation_comparison"),
            feature_importances=result.get("feature_importances"),
            artifact_path=result.get("artifact_path"),
            training_duration_seconds=result.get("training_duration_seconds")
        )
        db.add(forecast_record)

        if job:
            job.status = "completed"
            job.progress_percentage = 100.0
            job.current_stage = "COMPLETED"
            job.results = {
                "model_id": result["model_id"],
                "best_model": result["best_model"],
                "mae": result["best_metrics"].get("mae"),
                "rmse": result["best_metrics"].get("rmse")
            }
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


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
def start_forecast_training(
    payload: ForecastTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Dispatches asynchronous forecasting tournament across baseline, statistical, ML, and DL models.
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", payload.dataset_id)

    project = db.query(Project).filter(Project.id == payload.project_id).first()
    if not project or (project.owner_id != current_user.id and not current_user.is_superuser):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this project.")

    # Create AnalysisJob
    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        creator_id=current_user.id,
        job_type="FORECAST_TRAINING",
        status="queued",
        current_stage="QUEUED",
        progress_percentage=0.0
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(
        _run_forecast_training_task,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        time_column=payload.time_column,
        target_column=payload.target_column,
        horizon=payload.horizon,
        candidate_models=payload.candidate_models,
        covariates=payload.covariates,
        job_id=job_id
    )

    return {
        "message": "Forecasting training job dispatched successfully.",
        "job_id": job_id,
        "status": "queued"
    }


@router.get("/jobs/{job_id}")
def get_forecast_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Checks status of a running or completed forecasting job.
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


@router.get("/models", response_model=List[ForecastModelResponse])
def list_forecast_models(
    project_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Lists trained forecast models accessible to current user.
    """
    query = db.query(ForecastModel).join(Project, ForecastModel.project_id == Project.id)
    if not current_user.is_superuser:
        query = query.filter(Project.owner_id == current_user.id)

    if project_id:
        query = query.filter(ForecastModel.project_id == project_id)

    records = query.order_by(ForecastModel.created_at.desc()).all()
    results = []
    for r in records:
        results.append(ForecastModelResponse(
            id=r.id,
            project_id=r.project_id,
            dataset_id=r.dataset_id,
            time_column=r.time_column,
            target_column=r.target_column,
            horizon=r.horizon,
            best_model_name=r.best_model_name,
            lifecycle_stage=r.lifecycle_stage,
            metrics=r.metrics or {},
            leaderboard=r.leaderboard or [],
            temporal_profile=r.temporal_profile,
            future_forecast=r.future_forecast or [],
            recent_history=r.recent_history,
            validation_comparison=r.validation_comparison,
            feature_importances=r.feature_importances,
            training_duration_seconds=r.training_duration_seconds,
            created_at=r.created_at.isoformat() if r.created_at else None
        ))
    return results


@router.get("/models/{model_id}", response_model=ForecastModelResponse)
def get_forecast_model_detail(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves full metrics, leaderboard, prediction intervals, and diagnostic curves.
    """
    r = db.query(ForecastModel).filter(ForecastModel.id == model_id).first()
    if not r:
        raise EntityNotFoundException("ForecastModel", model_id)

    project = db.query(Project).filter(Project.id == r.project_id).first()
    if not project or (project.owner_id != current_user.id and not current_user.is_superuser):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this model.")

    return ForecastModelResponse(
        id=r.id,
        project_id=r.project_id,
        dataset_id=r.dataset_id,
        time_column=r.time_column,
        target_column=r.target_column,
        horizon=r.horizon,
        best_model_name=r.best_model_name,
        lifecycle_stage=r.lifecycle_stage,
        metrics=r.metrics or {},
        leaderboard=r.leaderboard or [],
        temporal_profile=r.temporal_profile,
        future_forecast=r.future_forecast or [],
        recent_history=r.recent_history,
        validation_comparison=r.validation_comparison,
        feature_importances=r.feature_importances,
        training_duration_seconds=r.training_duration_seconds,
        created_at=r.created_at.isoformat() if r.created_at else None
    )


@router.post("/models/{model_id}/forecast", response_model=DynamicForecastResponse)
def generate_dynamic_forecast(
    model_id: str,
    payload: DynamicForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generates new multi-step forecast for a requested horizon using saved model artifacts.
    """
    r = db.query(ForecastModel).filter(ForecastModel.id == model_id).first()
    if not r:
        raise EntityNotFoundException("ForecastModel", model_id)

    if not r.artifact_path or not os.path.exists(r.artifact_path):
        # Fallback to precalculated forecast if artifact unavailable
        return DynamicForecastResponse(
            model_id=r.id,
            best_model_name=r.best_model_name,
            horizon=len(r.future_forecast[:payload.horizon]),
            forecast=r.future_forecast[:payload.horizon]
        )

    saved_obj = joblib.load(r.artifact_path)
    forecaster = saved_obj["forecaster"]

    if hasattr(forecaster, "predict_recursive"):
        pts = forecaster.predict_recursive(horizon=payload.horizon)
    elif hasattr(forecaster, "predict"):
        pts = forecaster.predict(horizon=payload.horizon)
    else:
        pts = [f.get("prediction", 0.0) for f in r.future_forecast[:payload.horizon]]

    from services.forecasting.uncertainty import UncertaintyEstimator
    intervals = UncertaintyEstimator.estimate_intervals(point_forecasts=pts)

    # Calculate forward timestamps
    from datetime import timedelta
    last_ts_str = r.future_forecast[0]["timestamp"] if r.future_forecast else "2024-01-01"
    try:
        last_dt = datetime.fromisoformat(last_ts_str)
    except Exception:
        last_dt = datetime.now(timezone.utc)

    for idx, inter in enumerate(intervals):
        t = last_dt + timedelta(days=idx + 1)
        inter["timestamp"] = t.strftime("%Y-%m-%d")

    return DynamicForecastResponse(
        model_id=r.id,
        best_model_name=r.best_model_name,
        horizon=payload.horizon,
        forecast=intervals
    )


@router.post("/models/{model_id}/promote")
def promote_forecast_model(
    model_id: str,
    payload: Dict[str, str],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Promotes model lifecycle stage (e.g. STAGING, PRODUCTION, ARCHIVED).
    """
    r = db.query(ForecastModel).filter(ForecastModel.id == model_id).first()
    if not r:
        raise EntityNotFoundException("ForecastModel", model_id)

    stage = payload.get("lifecycle_stage", "PRODUCTION").upper()
    r.lifecycle_stage = stage
    db.commit()

    return {"model_id": r.id, "lifecycle_stage": r.lifecycle_stage}
