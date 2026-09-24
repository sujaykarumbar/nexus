import time
from fastapi.testclient import TestClient
from apps.api.routes.ml import run_automl_training_job
from apps.api.models.ml_model import MLModel


def test_ml_target_suggestions_and_leakage_audit(client, auth_headers):
    # 1. Create project
    proj_res = client.post("/api/v1/projects", json={"name": "ML Target Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest churn sample dataset
    sample_res = client.post("/api/v1/datasets/sample", data={"project_id": project_id, "sample_type": "churn"}, headers=auth_headers)
    assert sample_res.status_code == 201
    dataset_id = sample_res.json()["id"]

    # 3. Request target suggestions
    target_res = client.get(f"/api/v1/ml/datasets/{dataset_id}/targets", headers=auth_headers)
    assert target_res.status_code == 200
    suggestions = target_res.json()["suggestions"]
    assert len(suggestions) > 0
    top_col = suggestions[0]["column"]
    assert top_col == "churn"

    # 4. Request leakage audit for 'churn'
    leak_res = client.get(f"/api/v1/ml/datasets/{dataset_id}/leakage?target_column=churn", headers=auth_headers)
    assert leak_res.status_code == 200
    leakage = leak_res.json()
    assert leakage["target_column"] == "churn"
    assert "warnings" in leakage
    assert "recommended_drop_columns" in leakage
    # customer_id should be flagged as ID
    assert "customer_id" in leakage["recommended_drop_columns"]


def test_ml_train_job_and_model_lifecycle(client, auth_headers, db_session):
    # 1. Create project & ingest dataset
    proj_res = client.post("/api/v1/projects", json={"name": "ML Lifecycle Project"}, headers=auth_headers)
    project_id = proj_res.json()["id"]

    sample_res = client.post("/api/v1/datasets/sample", data={"project_id": project_id, "sample_type": "churn"}, headers=auth_headers)
    dataset_id = sample_res.json()["id"]

    # 2. Trigger training job (synchronous execution via background tasks or direct call)
    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "churn",
        "candidate_algorithms": ["LogisticRegression", "RandomForest"],
        "excluded_columns": ["customer_id"],
        "cv_splits": 3,
        "optimize_hyperparameters": False,
        "optuna_trials": 3
    }
    train_res = client.post("/api/v1/ml/train", json=train_payload, headers=auth_headers)
    assert train_res.status_code == 202
    job_info = train_res.json()
    job_id = job_info["job_id"]
    assert job_info["status"] in ["queued", "running", "completed"]

    # In Starlette TestClient, background task finishes after the post call returns
    # Check if models were registered in the database
    models_res = client.get(f"/api/v1/ml/models?project_id={project_id}", headers=auth_headers)
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert len(models_list) >= 2  # At least LogisticRegression and RandomForest

    best_model_summary = models_list[0]
    model_id = best_model_summary["id"]

    # 3. Retrieve model details
    detail_res = client.get(f"/api/v1/ml/models/{model_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == model_id
    assert "all_metrics" in detail
    assert "accuracy" in detail["all_metrics"]
    assert detail["confusion_matrix"] is not None
    assert detail["model_card"] is not None

    # 4. Perform single instance prediction with feature contribution explainability
    predict_payload = {
        "features": {
            "age": 42,
            "tenure_months": 12,
            "monthly_charges": 79.85,
            "total_charges": 958.20,
            "contract_type": "Month-to-Month",
            "tech_support": "No",
            "payment_method": "Electronic Check",
            "num_tickets": 3
        }
    }
    pred_res = client.post(f"/api/v1/ml/models/{model_id}/predict", json=predict_payload, headers=auth_headers)
    assert pred_res.status_code == 200
    pred = pred_res.json()
    assert "prediction" in pred
    assert pred["prediction"] in [0, 1, 0.0, 1.0]
    assert pred["latency_ms"] >= 0
    assert len(pred["contributing_features"]) > 0

    # 5. Model stage promotion
    promote_res = client.post(f"/api/v1/ml/models/{model_id}/promote", json={"lifecycle_stage": "PRODUCTION"}, headers=auth_headers)
    assert promote_res.status_code == 200
    promoted = promote_res.json()
    assert promoted["lifecycle_stage"] == "PRODUCTION"
