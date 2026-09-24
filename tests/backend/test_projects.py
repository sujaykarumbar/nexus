def test_create_and_list_projects(client, auth_headers):
    # Create Project
    create_payload = {
        "name": "Telecom Churn Analysis 2026",
        "description": "Multi-agent exploratory churn modeling project."
    }
    create_res = client.post("/api/v1/projects", json=create_payload, headers=auth_headers)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["name"] == "Telecom Churn Analysis 2026"
    project_id = created_data["id"]

    # List Projects
    list_res = client.get("/api/v1/projects", headers=auth_headers)
    assert list_res.status_code == 200
    projects = list_res.json()
    assert len(projects) >= 1
    assert any(p["id"] == project_id for p in projects)

    # Get Single Project
    get_res = client.get(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Telecom Churn Analysis 2026"

    # Soft Delete Project
    del_res = client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert del_res.status_code == 204

    # Verify Project no longer listed
    list_res_after = client.get("/api/v1/projects", headers=auth_headers)
    assert not any(p["id"] == project_id for p in list_res_after.json())
