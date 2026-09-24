from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from apps.api.core.database import SessionLocal
from apps.api.models.dataset import Dataset
from apps.api.models.job import AnalysisJob
from apps.api.models.data_quality import DataQualityReport
from apps.api.models.eda import EDAReport

from services.ingestion.factory import IngestionFactory
from services.profiling.schema_detector import SchemaDetector
from services.profiling.profiler import DataProfiler
from services.quality.quality_engine import DataQualityEngine
from services.eda.eda_engine import EDAEngine
from services.insights.insight_generator import InsightGenerator
from services.insights.verifier import ClaimVerifier
from services.visualization.vis_engine import VisualizationEngine


def run_dataset_analysis_pipeline(
    dataset_id: str, 
    job_id: Optional[str] = None, 
    db: Optional[Session] = None
):
    """
    Executes the complete deterministic Data Intelligence Pipeline for a dataset:
    Validation -> Ingestion -> Schema Detection -> Profiling -> Data Quality -> EDA -> Insights -> Visualizations.
    """
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True
    job = None
    try:
        if job_id:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "running"
                job.progress_percentage = 10.0
                job.current_stage = "File Ingestion & Validation"
                job.logs = list(job.logs or []) + [{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "message": "Validating file format and parsing raw data records..."
                }]
                db.commit()

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        # 1. Load DataFrame via Adapter
        adapter = IngestionFactory.get_adapter(dataset.file_type)
        df = adapter.load(dataset.file_path)

        if job:
            job.progress_percentage = 30.0
            job.current_stage = "Heuristic Schema Detection"
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": f"Successfully parsed {len(df)} rows across {len(df.columns)} columns."
            }]
            db.commit()

        # 2. Schema Detection & Profiling
        schema_info = SchemaDetector.detect_schema(df)
        profiling_info = DataProfiler.profile_dataset(df)

        if job:
            job.progress_percentage = 55.0
            job.current_stage = "Data Quality Scoring & Outlier Auditing"
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": "Computed statistical distributions, quantiles, and category frequencies."
            }]
            db.commit()

        # 3. Data Quality Engine
        quality_info = DataQualityEngine.evaluate_quality(df)

        # Persist or update DataQualityReport
        db.query(DataQualityReport).filter(DataQualityReport.dataset_id == dataset_id).delete()
        quality_report = DataQualityReport(
            dataset_id=dataset_id,
            overall_score=quality_info["score"],
            grade=quality_info["grade"],
            score_breakdown=quality_info["score_breakdown"],
            issues_summary=quality_info["issues"],
            recommendations=quality_info["recommendations"]
        )
        db.add(quality_report)

        if job:
            job.progress_percentage = 75.0
            job.current_stage = "Exploratory Data Analysis & Correlation Matrix"
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": f"Data Quality Score: {quality_info['score']}/100. Issues detected: {len(quality_info['issues']['missing_values'])} missing cols, {len(quality_info['issues']['outliers'])} outlier features."
            }]
            db.commit()

        # 4. Automated EDA Engine
        eda_info = EDAEngine.run_full_eda(df)

        # 5. Insight Generation & Verification
        raw_insights = InsightGenerator.generate_insights(df, eda_info, quality_info)
        ground_truth = {
            "correlations": eda_info["correlations"],
            "quality": quality_info
        }
        verified_insights = ClaimVerifier.audit_all_insights(raw_insights, ground_truth)

        # 6. Visualization Specs
        vis_specs = VisualizationEngine.generate_chart_specs(df, eda_info)

        # Persist or update EDAReport
        db.query(EDAReport).filter(EDAReport.dataset_id == dataset_id).delete()
        eda_report = EDAReport(
            dataset_id=dataset_id,
            summary=eda_info["summary"],
            correlations=eda_info["correlations"],
            distributions=eda_info["distributions"],
            insights=verified_insights,
            visualization_specs=vis_specs
        )
        db.add(eda_report)

        # Update Dataset table metadata
        dataset.row_count = len(df)
        dataset.column_count = len(df.columns)
        dataset.data_quality_score = quality_info["score"]
        dataset.schema_metadata = schema_info
        dataset.eda_summary = eda_info["summary"]

        # Infer target column and problem type candidates
        for col, cinfo in schema_info["columns"].items():
            clower = col.lower()
            if any(k in clower for k in ["churn", "target", "label", "is_fraud", "class"]):
                dataset.target_column = col
                if cinfo["inferred_type"] in ["BOOLEAN", "CATEGORICAL"] or cinfo["unique_count"] <= 10:
                    dataset.detected_problem_type = "classification"
                else:
                    dataset.detected_problem_type = "regression"
                break
            elif any(k in clower for k in ["sales", "revenue", "price", "amount", "demand"]):
                dataset.target_column = col
                dataset.detected_problem_type = "regression"

        if job:
            job.status = "completed"
            job.progress_percentage = 100.0
            job.current_stage = "Completed"
            job.results = {
                "quality_score": quality_info["score"],
                "grade": quality_info["grade"],
                "insights_count": len(verified_insights),
                "charts_count": len(vis_specs)
            }
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": "Data Intelligence analysis pipeline finished successfully."
            }]
            db.commit()

        db.commit()
    except Exception as e:
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.logs = list(job.logs or []) + [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "message": f"Pipeline failed: {str(e)}"
            }]
            db.commit()
        raise e
    finally:
        if should_close:
            db.close()
