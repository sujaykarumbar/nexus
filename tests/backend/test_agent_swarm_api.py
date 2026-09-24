"""
API Integration Tests for Multi-Agent Swarm & Critic Verification Endpoints
Tests /api/v1/agents/...
"""

from fastapi.testclient import TestClient


def test_agent_swarm_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Get agent swarm status
    agents_res = client.get("/api/v1/agents", headers=auth_headers)
    assert agents_res.status_code == 200
    agents = agents_res.json()
    assert len(agents) >= 7
    agent_ids = [a["id"] for a in agents]
    assert "critic_agent" in agent_ids
    assert "data_agent" in agent_ids
    assert "ml_agent" in agent_ids
    assert "rag_agent" in agent_ids

    # 2. Standalone Critic verification endpoint
    critic_req = {
        "claims": [
            {
                "claim": "The dataset contains 1200 records",
                "metric_name": "row_count",
                "reported_value": 1200
            },
            {
                "claim": "Model F1-score is 0.94",
                "metric_name": "f1_score",
                "reported_value": 0.94
            }
        ],
        "computed_metrics": {
            "row_count": 1200,
            "f1_score": 0.938
        }
    }
    critic_res = client.post("/api/v1/agents/critic/verify", json=critic_req, headers=auth_headers)
    assert critic_res.status_code == 200
    c_data = critic_res.json()
    assert c_data["status"] == "SUPPORTED"
    assert c_data["supported_count"] == 2
    assert c_data["gatekeeper_verdict"] == "PASS"

    # 3. Create project & sample dataset for Swarm execution
    proj_res = client.post("/api/v1/projects", json={"name": "Swarm Execution Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    sample_res = client.post(
        "/api/v1/datasets/sample",
        data={"project_id": project_id, "sample_type": "churn"},
        headers=auth_headers
    )
    assert sample_res.status_code == 201
    dataset_id = sample_res.json()["id"]

    # 4. Run Swarm execution
    swarm_req = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "target_column": "churn",
        "run_automl": True,
        "run_forecasting": False,
        "run_anomalies": True,
        "run_rag": False
    }
    run_res = client.post("/api/v1/agents/swarm/run", json=swarm_req, headers=auth_headers)
    assert run_res.status_code == 200
    r_data = run_res.json()
    assert r_data["status"] in ["completed", "rejected_by_critic"]
    assert len(r_data["trace"]) >= 4
    job_id = r_data["job_id"]

    # 5. Fetch trace
    trace_res = client.get(f"/api/v1/agents/swarm/runs/{job_id}", headers=auth_headers)
    assert trace_res.status_code == 200
    t_data = trace_res.json()
    assert t_data["job_id"] == job_id
    assert len(t_data["trace"]) >= 4
