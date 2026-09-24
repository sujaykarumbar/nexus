"""
API Integration Tests for Document Intelligence & RAG Endpoints
Tests /api/v1/rag/... endpoints (upload, list, search, query).
"""

import io
from fastapi.testclient import TestClient


def test_rag_api_lifecycle(client: TestClient, auth_headers: dict, db_session):
    # 1. Create a project
    proj_res = client.post("/api/v1/projects", json={"name": "RAG Intelligence Project"}, headers=auth_headers)
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Upload a text document
    doc_content = (
        "# Enterprise Cloud Infrastructure Report\n"
        "The cluster deployment achieved 99.98% uptime in Q3.\n"
        "Security audits confirmed zero unauthorized breaches.\n"
        "Recommended action: Maintain automated failover standby nodes."
    )
    files = {
        "file": ("cloud_infra_report.md", io.BytesIO(doc_content.encode("utf-8")), "text/markdown")
    }
    data = {"project_id": project_id}

    upload_res = client.post("/api/v1/rag/documents/upload", files=files, data=data, headers=auth_headers)
    assert upload_res.status_code == 201
    up_data = upload_res.json()
    assert up_data["filename"] == "cloud_infra_report.md"
    assert up_data["chunks_created"] >= 1
    doc_id = up_data["document_id"]

    # 3. List documents
    list_res = client.get(f"/api/v1/rag/documents?project_id={project_id}", headers=auth_headers)
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) >= 1
    assert any(d["document_id"] == doc_id for d in docs)

    # 4. Hybrid Search
    search_res = client.post(
        "/api/v1/rag/search",
        json={"query": "cluster deployment uptime", "project_id": project_id, "top_k": 3},
        headers=auth_headers
    )
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert s_data["total_found"] >= 1
    assert "uptime" in s_data["results"][0]["text"].lower()

    # 5. Grounded RAG Query with Citations
    query_res = client.post(
        "/api/v1/rag/query",
        json={"query": "What was the cluster deployment uptime?", "project_id": project_id, "top_k": 2},
        headers=auth_headers
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["grounded"] is True
    assert len(q_data["citations"]) >= 1
    assert q_data["citations"][0]["citation_tag"] == "[1]"
    assert "cloud_infra_report.md" in q_data["citations"][0]["filename"]

    # 6. Delete document
    del_res = client.delete(f"/api/v1/rag/documents/{doc_id}", headers=auth_headers)
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"
