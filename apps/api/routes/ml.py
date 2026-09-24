import os
import io
import time
import uuid
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db, SessionLocal
from apps.api.core.exceptions import EntityNotFoundException, ValidationException
from apps.api.routes.auth import get_current_user
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.models.ml_model import MLModel
from apps.api.models.prediction_log import PredictionLog

from apps.api.schemas.ml import (
    TargetSuggestionsResponse,
    TargetSuggestionItem,
    LeakageAuditResponse,
    LeakageWarningItem,
    MLTrainRequest,
    MLTrainResponse,
    MLModelSummaryResponse,
    MLModelDetailResponse,
    SinglePredictRequest,
    PredictResponse,
    ContributingFeatureItem,
    PromoteModelRequest
)
from services.ingestion.factory import IngestionFactory
from services.ml.target_detector import TargetDetector
from services.ml.leakage_detector import LeakageDetector
from services.ml.trainer import AutoMLTrainer

router = APIRouter(prefix="/ml", tags=["AutoML & Model Registry"])


def run_automl_training_job(
    dataset_id: str,
    target_column: str,
    job_id: str,
    candidate_algorithms: Optional[List[str]] = None,
    excluded_columns: Optional[List[str]] = None,
    cv_splits: int = 5,
    optimize_hyperparameters: bool = False,
    optuna_trials: int = 10,
    db: Optional[Session] = None
):
    """Background execution task for autonomous ML training pipeline."""
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
                job.error_message = f"Dataset '{dataset_id}' not found."
                db.commit()
            return

        adapter = IngestionFactory.get_adapter(dataset.file_type)
        df = adapter.load(dataset.file_path)

        def progress_tracker(pct: float, stage: str):
            if job:
                job.progress_percentage = pct
                job.current_stage = stage
                job.logs = list(job.logs or []) + [{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "message": f"[{pct:.0f}%] {stage}"
                }]
                db.commit()

        # Run complete real ML pipeline
        result = AutoMLTrainer.run_automl_pipeline(
            df=df,
            target_column=target_column,
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            project_id=dataset.project_id,
            candidate_algorithms=candidate_algorithms,
            excluded_columns=excluded_columns,
            cv_splits=cv_splits,
            optimize_hyperparameters=optimize_hyperparameters,
            optuna_trials=optuna_trials,
            progress_callback=progress_tracker
        )

        models_to_persist = result.get("all_models", [result["best_model"]])
        persisted_models = []
        best_persisted_id = None

        for m_item in models_to_persist:
            ml_model = MLModel(
                id=m_item["id"],
                project_id=dataset.project_id,
                dataset_id=dataset.id,
                name=f"{dataset.name} — {m_item['display_name']}",
                algorithm=m_item["algorithm"],
                problem_type=result["problem_type"],
                target_column=target_column,
                version=m_item.get("version", "v1.0"),
                lifecycle_stage=m_item.get("lifecycle_stage", "CANDIDATE"),
                primary_metric_name=m_item["primary_metric_name"],
                primary_metric_value=m_item["primary_metric_value"],
                all_metrics=m_item["metrics"],
                cv_scores=m_item.get("cv_summary"),
                delta_improvement_pct=m_item.get("delta_improvement_pct"),
                confusion_matrix=m_item.get("confusion_matrix"),
                roc_curve=m_item.get("roc_curve"),
                actual_vs_pred=m_item.get("actual_vs_pred"),
                hyperparameters=m_item.get("hyperparameters"),
                feature_importance=m_item.get("feature_importance"),
                feature_names=m_item.get("feature_names"),
                model_card=m_item.get("model_card"),
                artifact_path=m_item.get("artifact_path"),
                training_duration_seconds=result["total_duration_seconds"],
                status="ready",
                is_active=True
            )
            db.add(ml_model)
            persisted_models.append(ml_model)
            if m_item.get("is_best", False) or best_persisted_id is None:
                best_persisted_id = ml_model.id

        db.commit()

        if job:
            job.status = "completed"
            job.progress_percentage = 100.0
            job.current_stage = "Completed"
            job.results = {
                "model_id": best_persisted_id,
                "algorithm": result["best_model"]["algorithm"],
                "problem_type": result["problem_type"],
                "primary_metric": f"{result['best_model']['primary_metric_name']}: {result['best_model']['primary_metric_value']}",
                "improvement_pct": result["best_model"]["delta_improvement_pct"],
                "leaderboard_size": len(result["leaderboard"]),
                "leaderboard": result["leaderboard"]
            }
            db.commit()

        db.commit()
    except Exception as e:
        import traceback
        print(f"DEBUG TRAINING EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "message": f"Training failed: {str(e)}"
            }]
            db.commit()
    finally:
        if should_close:
            db.close()


@router.get("/datasets/{dataset_id}/targets", response_model=TargetSuggestionsResponse)
def get_target_suggestions(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze dataset columns and suggest candidate target variables with confidence."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)

    candidates = TargetDetector.detect_candidate_targets(df)
    items = [
        TargetSuggestionItem(
            column=c["column"],
            confidence=c["confidence"],
            suggested_type=c["suggested_type"],
            unique_count=c["unique_count"],
            null_count=c["null_count"],
            reasons=c["reasons"],
            sample_values=c["sample_values"]
        )
        for c in candidates
    ]
    return TargetSuggestionsResponse(
        dataset_id=dataset_id,
        total_columns=len(df.columns),
        suggestions=items
    )


@router.get("/datasets/{dataset_id}/leakage", response_model=LeakageAuditResponse)
@router.get("/datasets/{dataset_id}/audit", response_model=LeakageAuditResponse)
@router.post("/datasets/{dataset_id}/audit", response_model=LeakageAuditResponse)
def audit_target_leakage(
    dataset_id: str,
    target_column: str = Query(..., description="Target column to audit for leakage"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Audit dataset for potential data leakage, perfect correlations, and identifiers."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)

    try:
        leakage = LeakageDetector.audit_leakage(df, target_column)
    except ValueError as ve:
        raise ValidationException(str(ve))

    warning_items = [
        LeakageWarningItem(
            column=w["column"],
            risk_type=w["risk_type"],
            severity=w["severity"],
            description=w["description"],
            recommendation=w["recommendation"],
            correlation=w.get("correlation")
        )
        for w in leakage["warnings"]
    ]

    return LeakageAuditResponse(
        has_leakage_risk=leakage["has_leakage_risk"],
        has_critical_risk=leakage["has_critical_risk"],
        target_column=target_column,
        warning_count=leakage["warning_count"],
        warnings=warning_items,
        recommended_drop_columns=leakage["recommended_drop_columns"]
    )


@router.post("/train", response_model=MLTrainResponse, status_code=status.HTTP_202_ACCEPTED)
def train_automl(
    request: MLTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Dispatch an autonomous machine learning training pipeline asynchronously."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == request.dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", request.dataset_id)

    job = AnalysisJob(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        creator_id=current_user.id,
        job_type="automl_training",
        status="queued",
        progress_percentage=0.0,
        current_stage="Queued for AutoML training"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch to background task or run synchronously in tests
    if os.environ.get("ENVIRONMENT") == "testing":
        run_automl_training_job(
            dataset_id=dataset.id,
            target_column=request.target_column,
            job_id=job.id,
            candidate_algorithms=request.candidate_algorithms,
            excluded_columns=request.excluded_columns,
            cv_splits=request.cv_splits,
            optimize_hyperparameters=request.optimize_hyperparameters,
            optuna_trials=request.optuna_trials,
            db=db
        )
    else:
        background_tasks.add_task(
            run_automl_training_job,
            dataset_id=dataset.id,
            target_column=request.target_column,
            job_id=job.id,
            candidate_algorithms=request.candidate_algorithms,
            excluded_columns=request.excluded_columns,
            cv_splits=request.cv_splits,
            optimize_hyperparameters=request.optimize_hyperparameters,
            optuna_trials=request.optuna_trials
        )

    return MLTrainResponse(
        job_id=job.id,
        status="queued",
        message=f"AutoML training pipeline dispatched for dataset '{dataset.name}' target '{request.target_column}'."
    )


@router.get("/models", response_model=List[MLModelSummaryResponse])
def list_models(
    project_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List registered machine learning models for user's projects."""
    query = db.query(MLModel).join(Project).filter(
        Project.owner_id == current_user.id,
        MLModel.is_active == True
    )
    if project_id:
        query = query.filter(MLModel.project_id == project_id)
    if dataset_id:
        query = query.filter(MLModel.dataset_id == dataset_id)

    return query.order_by(MLModel.created_at.desc()).all()


@router.get("/models/{model_id}", response_model=MLModelDetailResponse)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve full model specifications, metrics, curves, and Model Card."""
    model = db.query(MLModel).join(Project).filter(
        MLModel.id == model_id,
        Project.owner_id == current_user.id,
        MLModel.is_active == True
    ).first()
    if not model:
        raise EntityNotFoundException("MLModel", model_id)
    return model


@router.post("/models/{model_id}/promote", response_model=MLModelSummaryResponse)
def promote_model(
    model_id: str,
    request: PromoteModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Promote model lifecycle stage (e.g. STAGING, PRODUCTION, ARCHIVED)."""
    model = db.query(MLModel).join(Project).filter(
        MLModel.id == model_id,
        Project.owner_id == current_user.id,
        MLModel.is_active == True
    ).first()
    if not model:
        raise EntityNotFoundException("MLModel", model_id)

    valid_stages = ["TRAINED", "VALIDATED", "REGISTERED", "STAGING", "PRODUCTION", "ARCHIVED"]
    stage = request.lifecycle_stage.upper().strip()
    if stage not in valid_stages:
        raise ValidationException(f"Invalid lifecycle stage '{stage}'. Must be one of {valid_stages}.")

    model.lifecycle_stage = stage
    db.commit()
    db.refresh(model)
    return model


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict_model(
    model_id: str,
    request: SinglePredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Perform real-time prediction using the serialized pipeline artifact."""
    model = db.query(MLModel).join(Project).filter(
        MLModel.id == model_id,
        Project.owner_id == current_user.id,
        MLModel.is_active == True
    ).first()
    if not model:
        raise EntityNotFoundException("MLModel", model_id)

    if not model.artifact_path or not os.path.exists(model.artifact_path):
        raise ValidationException("Model artifact not found on server storage.")

    start_t = time.time()
    try:
        pred_res = AutoMLTrainer.predict_instance(model.artifact_path, request.features)
    except Exception as e:
        raise ValidationException(f"Inference error: {str(e)}")

    latency_ms = round((time.time() - start_t) * 1000, 2)

    # Log prediction audit record
    log = PredictionLog(
        model_id=model.id,
        model_version=model.version,
        input_data=request.features,
        prediction=str(pred_res["prediction"]),
        probability=pred_res.get("probability"),
        contributing_features=pred_res.get("contributing_features"),
        latency_ms=latency_ms
    )
    db.add(log)
    db.commit()

    feature_items = [
        ContributingFeatureItem(
            feature=f["feature"],
            feature_value=f["feature_value"],
            impact=f["impact"],
            direction=f["direction"]
        )
        for f in pred_res.get("contributing_features", [])
    ]

    return PredictResponse(
        model_id=model.id,
        model_version=model.version,
        prediction=pred_res["prediction"],
        probability=pred_res.get("probability"),
        probabilities=pred_res.get("probabilities"),
        contributing_features=feature_items,
        latency_ms=latency_ms
    )


@router.post("/models/{model_id}/predict/batch")
def predict_batch_model(
    model_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Batch prediction endpoint for uploaded CSV datasets."""
    model = db.query(MLModel).join(Project).filter(
        MLModel.id == model_id,
        Project.owner_id == current_user.id,
        MLModel.is_active == True
    ).first()
    if not model:
        raise EntityNotFoundException("MLModel", model_id)

    if not model.artifact_path or not os.path.exists(model.artifact_path):
        raise ValidationException("Model artifact not found on server storage.")

    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValidationException(f"Failed to parse CSV: {str(e)}")

    artifact = AutoMLTrainer.load_pipeline_artifact(model.artifact_path)
    preprocessor = artifact["preprocessor"]
    model_inst = artifact["model"]

    try:
        X_trans = preprocessor.transform(df)
        preds = model_inst.predict(X_trans)
        df["predicted_" + model.target_column] = preds
    except Exception as e:
        raise ValidationException(f"Batch inference failed: {str(e)}")

    batch_id = str(uuid.uuid4())
    # Save first 10 predictions to audit log
    for idx in range(min(10, len(df))):
        log = PredictionLog(
            model_id=model.id,
            model_version=model.version,
            prediction=str(preds[idx]),
            batch_id=batch_id,
            latency_ms=1.0
        )
        db.add(log)
    db.commit()

    return {
        "batch_id": batch_id,
        "model_id": model.id,
        "rows_processed": len(df),
        "predictions_sample": [
            int(p) if isinstance(p, (int, pd.Int64Dtype)) else (float(p) if isinstance(p, float) else str(p))
            for p in preds[:20]
        ]
    }
