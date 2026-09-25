"""
NEXUS MLOps Engine Tests — Phase 9
Validates Model Registry, Versioning, Drift Detection, and Pipeline Scheduling.
Both unit logic and FastAPI endpoints are tested.
"""

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from services.mlops.registry import ModelRegistry, ModelVersionRecord, get_registry
from services.mlops.drift_detector import DriftDetector, _compute_psi
from services.mlops.scheduler import PipelineScheduler, get_scheduler
from apps.api.models.ml_model import MLModel
from apps.api.models.project import Project


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def test_model_registry_lifecycle():
    registry = ModelRegistry()

    # 1. Register v1
    v1 = registry.register(
        model_id="model-101",
        model_type="automl",
        algorithm="RandomForestClassifier",
        metrics={"accuracy": 0.88, "f1": 0.85},
        dataset_id="ds-001",
        dataset_file_path=None,
        tags=["baseline"],
        lifecycle_stage="experimental",
        notes="Initial experiment",
    )

    assert v1.version_tag == "v1"
    assert v1.model_id == "model-101"
    assert v1.lifecycle_stage == "experimental"

    # 2. Register v2 linked to v1
    v2 = registry.register(
        model_id="model-101",
        model_type="automl",
        algorithm="XGBoostClassifier",
        metrics={"accuracy": 0.94, "f1": 0.92},
        dataset_id="ds-001",
        parent_version_id=v1.version_id,
        tags=["tuned", "candidate"],
        lifecycle_stage="staging",
    )

    assert v2.version_tag == "v2"
    assert v2.parent_version_id == v1.version_id

    # 3. Retrieve versions
    versions = registry.list_versions(model_id="model-101")
    assert len(versions) == 2
    assert versions[0].version_id == v1.version_id
    assert versions[1].version_id == v2.version_id

    # 4. Compare versions
    cmp_res = registry.compare(v1.version_id, v2.version_id)
    assert len(cmp_res["metric_comparison"]) >= 2
    acc_comp = next(c for c in cmp_res["metric_comparison"] if c["metric"] == "accuracy")
    assert acc_comp["delta"] > 0
    assert cmp_res["overall_winner"] == "version_b"

    # 5. Promote version
    promoted = registry.promote(v2.version_id, "production")
    assert promoted is not None
    assert promoted.lifecycle_stage == "production"

    # 6. Lineage chain
    lineage = registry.get_lineage(v2.version_id)
    assert len(lineage) == 2
    assert lineage[0].version_id == v1.version_id
    assert lineage[1].version_id == v2.version_id


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: Drift Detector
# ─────────────────────────────────────────────────────────────────────────────

def test_drift_detector_stable():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "num1": np.random.normal(50, 10, 500),
        "num2": np.random.uniform(0, 100, 500),
        "cat1": np.random.choice(["A", "B", "C"], 500),
    })
    # Same distribution
    cur_df = pd.DataFrame({
        "num1": np.random.normal(50, 10, 500),
        "num2": np.random.uniform(0, 100, 500),
        "cat1": np.random.choice(["A", "B", "C"], 500),
    })

    report = DriftDetector.detect(
        reference_df=ref_df,
        current_df=cur_df,
        reference_dataset_id="ref",
        current_dataset_id="cur",
    )

    assert report.alert is False
    assert report.overall_drift_level in ["none", "minor"]
    assert len(report.drifted_features) == 0


def test_drift_detector_severe_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "metric_a": np.random.normal(10, 2, 500),
        "metric_b": np.random.normal(100, 15, 500),
    })
    # Drastic shift in metric_a
    cur_df = pd.DataFrame({
        "metric_a": np.random.normal(30, 2, 500),
        "metric_b": np.random.normal(100, 15, 500),
    })

    report = DriftDetector.detect(
        reference_df=ref_df,
        current_df=cur_df,
        reference_dataset_id="ref",
        current_dataset_id="cur",
    )

    assert "metric_a" in report.drifted_features
    assert report.alert is True
    assert report.overall_psi > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: Pipeline Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_scheduler_crud():
    scheduler = PipelineScheduler()

    pipe = scheduler.create(
        name="Nightly Churn Retrain",
        project_id="proj-1",
        dataset_id="ds-1",
        model_type="automl",
        target_column="churn",
        cron_expression="0 2 * * *",
        description="Daily retrain on updated data",
    )

    assert pipe.name == "Nightly Churn Retrain"
    assert pipe.run_count == 0

    # Trigger now
    triggered = scheduler.trigger_now(pipe.pipeline_id)
    assert triggered is not None
    assert triggered.run_count == 1
    assert triggered.last_run_status in ["manually_triggered", "COMPLETED"]

    # List
    all_pipes = scheduler.list_all()
    assert len(all_pipes) == 1

    # Delete
    assert scheduler.delete(pipe.pipeline_id) is True
    assert len(scheduler.list_all()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# API Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_mlops_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Create a project
    proj_res = client.post("/api/v1/projects", json={"name": "MLOps Test Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest sample dataset
    sample_res = client.post(
        "/api/v1/datasets/sample",
        data={"project_id": project_id, "sample_type": "churn"},
        headers=auth_headers
    )
    assert sample_res.status_code == 201
    dataset_id = sample_res.json()["id"]

    # 3. Create a mock MLModel record in DB to register versions against
    model_record = MLModel(
        project_id=project_id,
        dataset_id=dataset_id,
        name="Churn Classifier",
        algorithm="XGBoostClassifier",
        problem_type="binary_classification",
        target_column="churn",
        primary_metric_name="accuracy",
        primary_metric_value=0.91,
        all_metrics={"accuracy": 0.91, "f1": 0.87, "roc_auc": 0.93},
        status="ready",
    )
    db_session.add(model_record)
    db_session.commit()
    db_session.refresh(model_record)
    model_id = model_record.id

    # 4. Register a version via API
    v_res = client.post(
        f"/api/v1/mlops/registry/{model_id}/versions",
        json={"tags": ["candidate", "production-ready"], "lifecycle_stage": "staging", "notes": "Tested in staging"},
        headers=auth_headers
    )
    assert v_res.status_code == 200
    v_data = v_res.json()
    version_id = v_data["version_id"]
    assert v_data["version_tag"] == "v1"
    assert v_data["algorithm"] == "XGBoostClassifier"

    # 5. List versions
    list_res = client.get(f"/api/v1/mlops/registry?model_id={model_id}", headers=auth_headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 6. Get version detail
    get_res = client.get(f"/api/v1/mlops/registry/{version_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["version_id"] == version_id

    # 7. Promote version
    promote_res = client.post(
        f"/api/v1/mlops/registry/{version_id}/promote",
        json={"lifecycle_stage": "production"},
        headers=auth_headers
    )
    assert promote_res.status_code == 200
    assert promote_res.json()["lifecycle_stage"] == "production"

    # 8. Test Drift API endpoint
    drift_res = client.post(
        "/api/v1/mlops/drift",
        json={
            "reference_dataset_id": dataset_id,
            "current_dataset_id": dataset_id,
        },
        headers=auth_headers
    )
    assert drift_res.status_code == 200
    d_data = drift_res.json()
    assert "overall_psi" in d_data
    assert "alert" in d_data
    assert d_data["alert"] is False

    # 9. Pipeline Scheduler API endpoints
    pipe_res = client.post(
        "/api/v1/mlops/pipelines",
        json={
            "name": "Weekly Churn Retraining",
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_type": "automl",
            "target_column": "churn",
            "cron_expression": "0 0 * * 0",
        },
        headers=auth_headers
    )
    assert pipe_res.status_code == 201
    pipe_data = pipe_res.json()
    pipe_id = pipe_data["pipeline_id"]

    # Trigger pipeline
    trig_res = client.post(f"/api/v1/mlops/pipelines/{pipe_id}/trigger", headers=auth_headers)
    assert trig_res.status_code == 200
    assert trig_res.json()["run_count"] == 1

    # Delete pipeline
    del_res = client.delete(f"/api/v1/mlops/pipelines/{pipe_id}", headers=auth_headers)
    assert del_res.status_code == 204
