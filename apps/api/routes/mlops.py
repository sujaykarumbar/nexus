"""
NEXUS MLOps API Routes — Phase 9
Model version registry, drift detection, and pipeline scheduling endpoints.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db
from apps.api.core.exceptions import EntityNotFoundException
from apps.api.routes.auth import get_current_user
from apps.api.models.user import User
from apps.api.models.dataset import Dataset
from apps.api.models.ml_model import MLModel
from apps.api.models.forecast import ForecastModel

from services.mlops.registry import get_registry
from services.mlops.drift_detector import DriftDetector
from services.mlops.scheduler import get_scheduler
from services.ingestion.factory import IngestionFactory

router = APIRouter(prefix="/mlops", tags=["MLOps & Model Registry"])


# ─────────────────────────────────────────────────────────────────────────────
# Model Version Registry
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/registry", response_model=List[Dict[str, Any]])
def list_model_versions(
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """Lists all registered model versions, optionally filtered by model_id."""
    registry = get_registry()
    versions = registry.list_versions(model_id=model_id)
    return [v.to_dict() for v in versions]


@router.get("/registry/{version_id}", response_model=Dict[str, Any])
def get_model_version(
    version_id: str,
    current_user: User = Depends(get_current_user),
):
    """Returns full detail for a specific model version."""
    registry = get_registry()
    v = registry.get_version(version_id)
    if not v:
        raise EntityNotFoundException("ModelVersion", version_id)
    return v.to_dict()


@router.post("/registry/{model_id}/versions", response_model=Dict[str, Any])
def register_model_version(
    model_id: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Registers a new version for an existing ML or Forecast model.
    Automatically computes dataset hash for reproducibility.
    """
    # Try to find model in either table
    ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
    forecast_model = db.query(ForecastModel).filter(ForecastModel.id == model_id).first()

    if not ml_model and not forecast_model:
        raise EntityNotFoundException("Model", model_id)

    if ml_model:
        model_type = "automl"
        algorithm = ml_model.best_model_name or "unknown"
        metrics = ml_model.metrics or {}
        dataset_id = ml_model.dataset_id
        artifact_path = ml_model.artifact_path
    else:
        model_type = "forecast"
        algorithm = forecast_model.best_model_name or "unknown"
        metrics = forecast_model.metrics or {}
        dataset_id = forecast_model.dataset_id
        artifact_path = forecast_model.artifact_path

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    file_path = dataset.file_path if dataset else dataset_id

    registry = get_registry()
    version = registry.register(
        model_id=model_id,
        model_type=model_type,
        algorithm=algorithm,
        metrics=metrics,
        dataset_id=dataset_id,
        dataset_file_path=file_path,
        artifact_path=artifact_path,
        parent_version_id=payload.get("parent_version_id"),
        tags=payload.get("tags", []),
        lifecycle_stage=payload.get("lifecycle_stage", "experimental"),
        notes=payload.get("notes"),
    )
    return version.to_dict()


@router.post("/registry/{version_id}/compare", response_model=Dict[str, Any])
def compare_versions(
    version_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Side-by-side metric comparison between two model versions."""
    compare_to = payload.get("compare_to")
    if not compare_to:
        raise HTTPException(status_code=400, detail="`compare_to` version_id is required.")
    registry = get_registry()
    return registry.compare(version_id, compare_to)


@router.post("/registry/{version_id}/promote", response_model=Dict[str, Any])
def promote_version(
    version_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Promotes a version to staging, production, or archived."""
    stage = payload.get("lifecycle_stage", "staging")
    registry = get_registry()
    v = registry.promote(version_id, stage)
    if not v:
        raise EntityNotFoundException("ModelVersion", version_id)
    return v.to_dict()


@router.get("/registry/{version_id}/lineage", response_model=List[Dict[str, Any]])
def get_version_lineage(
    version_id: str,
    current_user: User = Depends(get_current_user),
):
    """Returns the full training lineage chain for a version (oldest first)."""
    registry = get_registry()
    chain = registry.get_lineage(version_id)
    if not chain:
        raise EntityNotFoundException("ModelVersion", version_id)
    return [v.to_dict() for v in chain]


# ─────────────────────────────────────────────────────────────────────────────
# Data Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/drift", response_model=Dict[str, Any])
def compute_drift(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Computes PSI + KS-test based drift between a reference and current dataset.
    Required: reference_dataset_id, current_dataset_id.
    Optional: feature_columns (list of column names to check).
    """
    ref_id = payload.get("reference_dataset_id")
    cur_id = payload.get("current_dataset_id")
    feature_columns = payload.get("feature_columns")

    if not ref_id or not cur_id:
        raise HTTPException(status_code=400, detail="Both reference_dataset_id and current_dataset_id are required.")

    ref_dataset = db.query(Dataset).filter(Dataset.id == ref_id).first()
    cur_dataset = db.query(Dataset).filter(Dataset.id == cur_id).first()

    if not ref_dataset:
        raise EntityNotFoundException("Dataset", ref_id)
    if not cur_dataset:
        raise EntityNotFoundException("Dataset", cur_id)

    ref_adapter = IngestionFactory.get_adapter(ref_dataset.file_type)
    cur_adapter = IngestionFactory.get_adapter(cur_dataset.file_type)

    ref_df = ref_adapter.load(ref_dataset.file_path)
    cur_df = cur_adapter.load(cur_dataset.file_path)

    report = DriftDetector.detect(
        reference_df=ref_df,
        current_df=cur_df,
        reference_dataset_id=ref_id,
        current_dataset_id=cur_id,
        feature_columns=feature_columns,
    )
    return report.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Scheduler
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/pipelines", response_model=List[Dict[str, Any]])
def list_pipelines(current_user: User = Depends(get_current_user)):
    """Lists all defined re-training pipelines."""
    scheduler = get_scheduler()
    return [p.to_dict() for p in scheduler.list_all()]


@router.post("/pipelines", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_pipeline(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Creates a scheduled re-training pipeline.
    Required: name, project_id, dataset_id, model_type, target_column.
    Optional: cron_expression (default '0 2 * * *'), description.
    """
    required = ["name", "project_id", "dataset_id", "model_type", "target_column"]
    for field in required:
        if not payload.get(field):
            raise HTTPException(status_code=400, detail=f"Field `{field}` is required.")

    dataset = db.query(Dataset).filter(Dataset.id == payload["dataset_id"]).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", payload["dataset_id"])

    scheduler = get_scheduler()
    pipeline = scheduler.create(
        name=payload["name"],
        project_id=payload["project_id"],
        dataset_id=payload["dataset_id"],
        model_type=payload["model_type"],
        target_column=payload["target_column"],
        cron_expression=payload.get("cron_expression", "0 2 * * *"),
        description=payload.get("description"),
    )
    return pipeline.to_dict()


@router.post("/pipelines/{pipeline_id}/trigger", response_model=Dict[str, Any])
def trigger_pipeline(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
):
    """Manually triggers a pipeline run immediately."""
    scheduler = get_scheduler()
    pipeline = scheduler.trigger_now(pipeline_id)
    if not pipeline:
        raise EntityNotFoundException("Pipeline", pipeline_id)
    return pipeline.to_dict()


@router.delete("/pipelines/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
):
    """Deletes and unschedules a pipeline."""
    scheduler = get_scheduler()
    if not scheduler.delete(pipeline_id):
        raise EntityNotFoundException("Pipeline", pipeline_id)
