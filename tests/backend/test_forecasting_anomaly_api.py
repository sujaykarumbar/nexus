"""
End-to-end API Integration Tests for Forecasting and Anomaly Detection.
Tests /api/v1/forecast/... and /api/v1/anomalies/... endpoints.
"""

from fastapi.testclient import TestClient
from apps.api.routes.forecasting import _run_forecast_training_task
from apps.api.routes.anomalies import _run_anomaly_detection_task
from apps.api.models.forecast import ForecastModel, AnomalyReport


def test_forecasting_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Create project
    proj_res = client.post("/api/v1/projects", json={"name": "Forecasting API Test Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest sales_timeseries sample
    sample_res = client.post(
        "/api/v1/datasets/sample",
        data={"project_id": project_id, "sample_type": "sales_timeseries"},
        headers=auth_headers
    )
    assert sample_res.status_code == 201
    dataset_id = sample_res.json()["id"]

    # 3. Detect temporal properties
    detect_res = client.post(
        "/api/v1/forecast/detect",
        json={"dataset_id": dataset_id, "target_column": "sales"},
        headers=auth_headers
    )
    assert detect_res.status_code == 200
    det = detect_res.json()
    assert len(det["candidates"]) > 0
    assert det["candidates"][0]["column_name"] == "date"
    assert det["profile"] is not None
    assert det["profile"]["is_chronological"] is True

    # 4. Trigger training job directly
    train_res = client.post(
        "/api/v1/forecast/train",
        json={
            "project_id": project_id,
            "dataset_id": dataset_id,
            "time_column": "date",
            "target_column": "sales",
            "horizon": 7,
            "candidate_models": ["naive", "holt_winters", "random_forest"]
        },
        headers=auth_headers
    )
    assert train_res.status_code == 202
    job_id = train_res.json()["job_id"]

    # Execute synchronous task helper for determinism in test
    _run_forecast_training_task(
        project_id=project_id,
        dataset_id=dataset_id,
        time_column="date",
        target_column="sales",
        horizon=7,
        candidate_models=["naive", "holt_winters", "random_forest"],
        covariates=["marketing_spend"],
        job_id=job_id,
        db=db_session
    )

    # 5. Verify job status completed
    job_res = client.get(f"/api/v1/forecast/jobs/{job_id}", headers=auth_headers)
    assert job_res.status_code == 200
    assert job_res.json()["status"] == "completed"

    # 6. List and inspect forecast models
    models_res = client.get(f"/api/v1/forecast/models?project_id={project_id}", headers=auth_headers)
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert len(models_list) > 0
    model_id = models_list[0]["id"]

    detail_res = client.get(f"/api/v1/forecast/models/{model_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    model_detail = detail_res.json()
    assert model_detail["id"] == model_id
    assert len(model_detail["leaderboard"]) >= 3
    assert len(model_detail["future_forecast"]) == 7

    # 7. Generate dynamic forecast with new horizon
    dyn_res = client.post(
        f"/api/v1/forecast/models/{model_id}/forecast",
        json={"horizon": 14},
        headers=auth_headers
    )
    assert dyn_res.status_code == 200
    dyn_forecast = dyn_res.json()
    assert dyn_forecast["horizon"] == 14
    assert len(dyn_forecast["forecast"]) == 14

    # 8. Promote model stage
    promote_res = client.post(
        f"/api/v1/forecast/models/{model_id}/promote",
        json={"lifecycle_stage": "PRODUCTION"},
        headers=auth_headers
    )
    assert promote_res.status_code == 200
    assert promote_res.json()["lifecycle_stage"] == "PRODUCTION"


def test_anomaly_detection_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Create project & ingest iot_sensor sample
    proj_res = client.post("/api/v1/projects", json={"name": "Anomaly API Test Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    sample_res = client.post(
        "/api/v1/datasets/sample",
        data={"project_id": project_id, "sample_type": "iot_sensor"},
        headers=auth_headers
    )
    assert sample_res.status_code == 201
    dataset_id = sample_res.json()["id"]

    # 2. Trigger anomaly detection scan
    detect_res = client.post(
        "/api/v1/anomalies/detect",
        json={
            "project_id": project_id,
            "dataset_id": dataset_id,
            "time_column": "timestamp",
            "target_column": "temperature",
            "include_deep_learning": True
        },
        headers=auth_headers
    )
    assert detect_res.status_code == 202
    job_id = detect_res.json()["job_id"]

    # Run synchronous task
    _run_anomaly_detection_task(
        project_id=project_id,
        dataset_id=dataset_id,
        time_column="timestamp",
        target_column="temperature",
        feature_columns=["temperature", "pressure", "vibration", "voltage", "current"],
        include_deep_learning=True,
        job_id=job_id,
        db=db_session
    )

    # 3. Retrieve latest anomaly report
    report_res = client.get(f"/api/v1/anomalies/{dataset_id}", headers=auth_headers)
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["dataset_id"] == dataset_id
    assert report["dataset_summary"]["total_anomalies"] > 0
    assert report["dataset_summary"]["critical_count"] >= 1
    assert len(report["anomalies"]) > 0

    # 4. Summary endpoint
    sum_res = client.get(f"/api/v1/anomalies/{dataset_id}/summary", headers=auth_headers)
    assert sum_res.status_code == 200
    assert "critical_count" in sum_res.json()

    # 5. Timeline endpoint
    timeline_res = client.get(f"/api/v1/anomalies/{dataset_id}/timeline", headers=auth_headers)
    assert timeline_res.status_code == 200
    assert len(timeline_res.json()["timeline"]) > 0

    # 6. Change-points endpoint
    cp_res = client.get(f"/api/v1/anomalies/{dataset_id}/change-points", headers=auth_headers)
    assert cp_res.status_code == 200
    assert isinstance(cp_res.json(), list)
