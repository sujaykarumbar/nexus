def test_create_and_poll_job(client, auth_headers):
    # 1. Create project
    proj_res = client.post("/api/v1/projects", json={"name": "Job Test Project"}, headers=auth_headers)
    project_id = proj_res.json()["id"]

    # 2. Dispatch async job
    job_payload = {
        "project_id": project_id,
        "job_type": "profiling"
    }
    create_res = client.post("/api/v1/jobs", json=job_payload, headers=auth_headers)
    assert create_res.status_code == 202
    job_data = create_res.json()
    assert job_data["job_type"] == "profiling"
    job_id = job_data["id"]

    # 3. List jobs
    list_res = client.get("/api/v1/jobs", headers=auth_headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 4. Poll specific job
    get_res = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == job_id
