"""
API Lifecycle Tests for Knowledge Graph & GraphRAG Routes
"""

import pytest
from starlette.testclient import TestClient


def test_knowledge_graph_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Create a project
    proj_res = client.post("/api/v1/projects", json={"name": "Graph API Project"}, headers=auth_headers)
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

    # 3. Build knowledge graph
    build_req = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "target_column": "churn",
        "correlation_threshold": 0.3,
        "include_models": True,
        "include_anomalies": True,
        "include_forecasts": True
    }
    build_res = client.post("/api/v1/graph/build", json=build_req, headers=auth_headers)
    assert build_res.status_code == 200
    b_data = build_res.json()
    assert b_data["status"] == "COMPLETED"
    assert b_data["total_nodes"] > 5
    assert b_data["total_edges"] > 5

    # 4. Fetch project knowledge graph
    graph_res = client.get(f"/api/v1/graph/projects/{project_id}", headers=auth_headers)
    assert graph_res.status_code == 200
    g_data = graph_res.json()
    assert len(g_data["nodes"]) == b_data["total_nodes"]
    assert len(g_data["edges"]) == b_data["total_edges"]
    assert g_data["metrics"]["node_count"] == b_data["total_nodes"]

    # 5. Query GraphRAG
    query_req = {
        "query": "What features correlate with monthly_charges and what is the target churn?",
        "project_id": project_id,
        "max_hops": 2
    }
    q_res = client.post("/api/v1/graph/query", json=query_req, headers=auth_headers)
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["grounded"] is True
    assert len(q_data["seed_entities"]) >= 1
    assert "monthly_charges" in str(q_data["seed_entities"]).lower() or "churn" in str(q_data["seed_entities"]).lower()

    # 6. Neighborhood exploration
    sample_node_id = g_data["nodes"][0]["id"]
    neigh_res = client.get(f"/api/v1/graph/node/{sample_node_id}/neighborhood", headers=auth_headers)
    assert neigh_res.status_code == 200
    n_data = neigh_res.json()
    assert n_data["node_count"] >= 1
