def test_full_dataset_intelligence_api_flow(client, auth_headers):
    # 1. Create a project
    proj_res = client.post("/api/v1/projects", json={"name": "Intelligence Test Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest customer churn sample dataset
    sample_payload = {
        "project_id": project_id,
        "sample_type": "churn"
    }
    sample_res = client.post("/api/v1/datasets/sample", data=sample_payload, headers=auth_headers)
    assert sample_res.status_code == 201
    dataset = sample_res.json()
    dataset_id = dataset["id"]

    # 3. Test GET /profile
    prof_res = client.get(f"/api/v1/datasets/{dataset_id}/profile", headers=auth_headers)
    assert prof_res.status_code == 200
    profile = prof_res.json()
    assert "schema" in profile
    assert "columns" in profile

    # 4. Test GET /quality
    qual_res = client.get(f"/api/v1/datasets/{dataset_id}/quality", headers=auth_headers)
    assert qual_res.status_code == 200
    quality = qual_res.json()
    assert "score" in quality
    assert "score_breakdown" in quality
    assert "recommendations" in quality

    # 5. Test GET /eda
    eda_res = client.get(f"/api/v1/datasets/{dataset_id}/eda", headers=auth_headers)
    assert eda_res.status_code == 200
    eda = eda_res.json()
    assert "correlations" in eda
    assert "distributions" in eda

    # 6. Test GET /insights
    ins_res = client.get(f"/api/v1/datasets/{dataset_id}/insights", headers=auth_headers)
    assert ins_res.status_code == 200
    insights = ins_res.json()
    assert len(insights) >= 1

    # 7. Test GET /visualizations
    vis_res = client.get(f"/api/v1/datasets/{dataset_id}/visualizations", headers=auth_headers)
    assert vis_res.status_code == 200
    vis_specs = vis_res.json()
    assert len(vis_specs) >= 1

    # 8. Test GET /preview with pagination
    prev_res = client.get(f"/api/v1/datasets/{dataset_id}/preview?page=1&page_size=10", headers=auth_headers)
    assert prev_res.status_code == 200
    preview = prev_res.json()
    assert preview["page"] == 1
    assert preview["page_size"] == 10
    assert len(preview["rows"]) <= 10
    assert "inferred_types" in preview

    # 9. Test POST /analyze background dispatch
    analyze_res = client.post(f"/api/v1/datasets/{dataset_id}/analyze", headers=auth_headers)
    assert analyze_res.status_code == 202
    assert "job_id" in analyze_res.json()
