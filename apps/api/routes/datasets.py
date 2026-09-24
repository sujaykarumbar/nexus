import os
import shutil
import uuid
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, status
from sqlalchemy.orm import Session
from apps.api.core.config import settings
from apps.api.core.database import get_db
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.models.data_quality import DataQualityReport
from apps.api.models.eda import EDAReport
from apps.api.schemas.dataset import DatasetResponse
from apps.api.schemas.data_intelligence import (
    DatasetProfileResponse,
    DataQualityResponse,
    EDAResponse,
    InsightItem,
    VisualizationSpecItem,
    PaginatedPreviewResponse
)
from apps.api.routes.auth import get_current_user
from apps.api.core.exceptions import EntityNotFoundException, ValidationException
from services.ingestion.factory import IngestionFactory
from services.ingestion.secure_upload import save_secure_upload
from services.profiling.schema_detector import SchemaDetector
from services.profiling.profiler import DataProfiler
from services.quality.quality_engine import DataQualityEngine
from services.eda.eda_engine import EDAEngine
from apps.api.services.dataset_pipeline import run_dataset_analysis_pipeline

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    project_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List datasets for a project or all projects owned by the user."""
    query = db.query(Dataset).join(Project).filter(
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    )
    if project_id:
        query = query.filter(Dataset.project_id == project_id)
        
    return query.order_by(Dataset.created_at.desc()).all()


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: str = Form(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Securely upload and register a CSV, Excel, JSON, or Parquet dataset."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.is_active == True
    ).first()
    if not project:
        raise EntityNotFoundException("Project", project_id)

    raw_filename = file.filename or "dataset.csv"
    
    # Secure storage
    try:
        stored_path, clean_name, file_size, checksum = save_secure_upload(file.file, raw_filename)
    except Exception as e:
        raise ValidationException(str(e))

    ext = clean_name.split(".")[-1].lower()
    adapter = IngestionFactory.get_adapter(ext)
    
    # Validation & Initial Metadata
    if not adapter.validate(stored_path):
        if os.path.exists(stored_path):
            os.remove(stored_path)
        raise ValidationException(f"Failed to validate file structure for '{raw_filename}'. File may be corrupt or malformed.")

    metadata = adapter.get_metadata(stored_path)
    
    dataset = Dataset(
        project_id=project_id,
        name=name or clean_name,
        description=description,
        file_path=stored_path,
        file_type=ext,
        file_size_bytes=file_size,
        row_count=metadata.get("row_count", 0),
        column_count=metadata.get("column_count", 0),
        data_quality_score=95.0
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Immediately run analysis pipeline synchronously or add to background
    run_dataset_analysis_pipeline(dataset.id, db=db)
    db.refresh(dataset)
    return dataset


@router.post("/sample", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def load_sample_dataset(
    project_id: str = Form(...),
    sample_type: str = Form(...),  # churn, sales, sensor
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Load one of the bundled production sample datasets into a project and analyze it."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id,
        Project.is_active == True
    ).first()
    if not project:
        raise EntityNotFoundException("Project", project_id)

    sample_map = {
        "churn": ("customer_churn.csv", "Customer Churn Intelligence Dataset", "classification", "churn"),
        "customer_churn": ("customer_churn.csv", "Customer Churn Intelligence Dataset", "classification", "churn"),
        "sales": ("sales_forecasting.csv", "Retail Store Sales & Demand Forecasting", "time_series", "sales_amount"),
        "sensor": ("sensor_anomalies.csv", "Industrial IoT Vibration & Thermal Sensors", "anomaly", "status_flag"),
        "sales_timeseries": ("sales_timeseries.csv", "Retail Omnichannel Sales & Demand Forecasting", "time_series", "sales"),
        "iot_sensor": ("iot_sensor_data.csv", "Industrial IoT Machine Telemetry & Vibration", "anomaly", "temperature"),
        "traffic": ("website_traffic.csv", "E-Commerce Website Traffic & Conversions", "time_series", "visitors"),
        "energy": ("energy_consumption.csv", "Smart Grid Energy Consumption & Demand", "time_series", "energy_usage")
    }

    if sample_type not in sample_map:
        raise ValidationException("Invalid sample_type. Choose 'churn', 'sales', 'sensor', 'sales_timeseries', 'iot_sensor', 'traffic', or 'energy'.")

    sample_filename, sample_title, prob_type, target_col = sample_map[sample_type]
    source_path = os.path.join(settings.SAMPLE_DIR, sample_filename)
    if not os.path.exists(source_path):
        raise ValidationException(f"Sample dataset file '{sample_filename}' not found on server.")

    target_uuid = str(uuid.uuid4())
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{target_uuid}_{sample_filename}")
    shutil.copyfile(source_path, dest_path)

    df = pd.read_csv(dest_path)
    
    dataset = Dataset(
        project_id=project_id,
        name=sample_title,
        description=f"Automated benchmark dataset for {prob_type} intelligence.",
        file_path=dest_path,
        file_type="csv",
        file_size_bytes=os.path.getsize(dest_path),
        row_count=len(df),
        column_count=len(df.columns),
        data_quality_score=98.5,
        detected_problem_type=prob_type,
        target_column=target_col
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Run complete analysis
    run_dataset_analysis_pipeline(dataset.id, db=db)
    db.refresh(dataset)
    return dataset


@router.post("/{dataset_id}/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_dataset(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger the automated data intelligence pipeline in the background."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    job = AnalysisJob(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        creator_id=current_user.id,
        job_type="data_intelligence",
        status="queued",
        progress_percentage=0.0,
        current_stage="Queued for analysis"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_dataset_analysis_pipeline, dataset.id, job.id)
    return {"job_id": job.id, "status": "queued", "message": "Data intelligence pipeline dispatched."}


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get dataset metadata."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)
    return dataset


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get full column-level statistical profile and schema inferences."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)
    profile = DataProfiler.profile_dataset(df)
    return profile


@router.get("/{dataset_id}/quality", response_model=DataQualityResponse)
def get_dataset_quality(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get data quality report, score breakdown, issues, and cleaning recommendations."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    report = db.query(DataQualityReport).filter(DataQualityReport.dataset_id == dataset_id).first()
    if report:
        return {
            "score": report.overall_score,
            "grade": report.grade,
            "score_breakdown": report.score_breakdown,
            "issues": report.issues_summary,
            "recommendations": report.recommendations
        }

    # If report not cached in DB, compute deterministically on the fly
    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)
    quality_info = DataQualityEngine.evaluate_quality(df)
    return quality_info


@router.get("/{dataset_id}/eda", response_model=EDAResponse)
def get_dataset_eda(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get EDA report containing correlation matrices, distributions, and summary metrics."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    report = db.query(EDAReport).filter(EDAReport.dataset_id == dataset_id).first()
    if report:
        return {
            "summary": report.summary,
            "correlations": report.correlations,
            "distributions": report.distributions
        }

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)
    return EDAEngine.run_full_eda(df)


@router.get("/{dataset_id}/insights", response_model=List[InsightItem])
def get_dataset_insights(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get verified automated data insights."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    report = db.query(EDAReport).filter(EDAReport.dataset_id == dataset_id).first()
    if report and report.insights:
        return report.insights

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)
    eda_info = EDAEngine.run_full_eda(df)
    quality_info = DataQualityEngine.evaluate_quality(df)
    insights = InsightGenerator.generate_insights(df, eda_info, quality_info)
    return insights


@router.get("/{dataset_id}/visualizations", response_model=List[VisualizationSpecItem])
def get_dataset_visualizations(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get declarative chart specifications for frontend rendering."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    report = db.query(EDAReport).filter(EDAReport.dataset_id == dataset_id).first()
    if report and report.visualization_specs:
        return report.visualization_specs

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)
    eda_info = EDAEngine.run_full_eda(df)
    return VisualizationEngine.generate_chart_specs(df, eda_info)


@router.get("/{dataset_id}/preview", response_model=PaginatedPreviewResponse)
def preview_dataset_paginated(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Server-side paginated preview of raw dataset records."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    page_df = df.iloc[start_idx:end_idx]
    clean_df = page_df.where(pd.notnull(page_df), None)

    schema_info = SchemaDetector.detect_schema(df)
    inferred_types = {col: info["inferred_type"] for col, info in schema_info["columns"].items()}

    return PaginatedPreviewResponse(
        page=page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        total_columns=len(df.columns),
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        inferred_types=inferred_types,
        rows=clean_df.to_dict(orient="records")
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Soft-delete dataset."""
    dataset = db.query(Dataset).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id,
        Dataset.is_active == True
    ).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", dataset_id)

    dataset.is_active = False
    db.commit()
    return None
