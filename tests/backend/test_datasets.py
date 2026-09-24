import io


def test_load_sample_dataset(client, auth_headers):
    # 1. Create a project
    proj_res = client.post("/api/v1/projects", json={"name": "Sample Test Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest customer churn sample dataset
    sample_payload = {
        "project_id": project_id,
        "sample_type": "churn"
    }
    sample_res = client.post("/api/v1/datasets/sample", data=sample_payload, headers=auth_headers)
    assert sample_res.status_code == 201
    dataset_data = sample_res.json()
    assert dataset_data["row_count"] > 0
    assert dataset_data["column_count"] > 0
    assert dataset_data["detected_problem_type"] == "classification"
    assert dataset_data["target_column"] == "churn"
    assert dataset_data["data_quality_score"] is not None

    dataset_id = dataset_data["id"]

    # 3. Preview dataset
    prev_res = client.get(f"/api/v1/datasets/{dataset_id}/preview", headers=auth_headers)
    assert prev_res.status_code == 200
    preview_data = prev_res.json()
    assert "columns" in preview_data
    assert len(preview_data["rows"]) > 0


def test_upload_csv_dataset(client, auth_headers):
    # Create project
    proj_res = client.post("/api/v1/projects", json={"name": "Upload Test Project"}, headers=auth_headers)
    project_id = proj_res.json()["id"]

    # Create dummy CSV file in memory
    csv_content = b"user_id,feature_a,feature_b,target\n1,10.5,20.2,1\n2,12.1,19.8,0\n3,9.4,22.0,1"
    files = {
        "file": ("test_data.csv", io.BytesIO(csv_content), "text/csv")
    }
    data = {
        "project_id": project_id,
        "name": "Custom Uploaded Data"
    }

    upload_res = client.post("/api/v1/datasets/upload", data=data, files=files, headers=auth_headers)
    assert upload_res.status_code == 201
    dataset = upload_res.json()
    assert dataset["row_count"] == 3
    assert dataset["column_count"] == 4
    assert dataset["data_quality_score"] is not None
