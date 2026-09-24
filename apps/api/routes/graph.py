"""
NEXUS Knowledge Graph & GraphRAG API Routes
Provides endpoints for graph construction, topological queries, node exploration,
and multi-hop GraphRAG question answering.
"""

import os
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db
from apps.api.models.user import User
from apps.api.models.project import Project
from apps.api.models.dataset import Dataset
from apps.api.models.ml_model import MLModel
from apps.api.models.anomaly import AnomalyEvent
from apps.api.models.forecast import ForecastModel
from apps.api.routes.auth import get_current_user
from apps.api.schemas.graph import (
    KnowledgeGraphResponse,
    GraphNodeOut,
    GraphEdgeOut,
    GraphBuildRequest,
    GraphBuildResponse,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    NeighborhoodResponse
)
from services.graph.graph_store import get_graph_store
from services.graph.extractor import KnowledgeGraphExtractor
from services.graph.graph_rag import GraphRAGEngine
from services.ingestion.factory import IngestionFactory

router = APIRouter(prefix="/graph", tags=["Knowledge Graph & GraphRAG"])


@router.get("/projects/{project_id}", response_model=KnowledgeGraphResponse)
def get_project_knowledge_graph(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve full project knowledge graph (nodes, edges, topological metrics).
    """
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied."
        )

    store = get_graph_store()
    # Attempt loading from disk cache if empty in memory
    nodes = store.get_nodes(project_id=project_id)
    if not nodes:
        store.load_from_disk(project_id)
        nodes = store.get_nodes(project_id=project_id)

    edges = store.get_edges(project_id=project_id)
    metrics = store.get_graph_metrics(project_id=project_id)

    return KnowledgeGraphResponse(
        project_id=project_id,
        nodes=[GraphNodeOut(**n.to_dict()) for n in nodes],
        edges=[GraphEdgeOut(**e.to_dict()) for e in edges],
        metrics=metrics
    )


@router.post("/build", response_model=GraphBuildResponse)
def build_project_knowledge_graph(
    req: GraphBuildRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Autonomously extract, construct, and index Knowledge Graph from project datasets,
    features, correlations, ML models, and detected anomalies.
    """
    start_t = time.perf_counter()

    project = db.query(Project).filter(
        Project.id == req.project_id,
        Project.owner_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied."
        )

    dataset_query = db.query(Dataset).filter(Dataset.project_id == req.project_id)
    if req.dataset_id:
        dataset_query = dataset_query.filter(Dataset.id == req.dataset_id)
    dataset = dataset_query.first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No dataset found for this project."
        )

    if not os.path.exists(dataset.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset file missing on disk: {dataset.file_path}"
        )

    try:
        loader = IngestionFactory.get_adapter(dataset.file_type)
        df = loader.load(dataset.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse dataset file: {str(e)}"
        )

    store = get_graph_store()
    extractor = KnowledgeGraphExtractor(store=store)

    target_col = req.target_column or dataset.target_column

    # 1. Extract Tabular Schema & Correlations
    ds_res = extractor.extract_from_dataset(
        df=df,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        project_id=project.id,
        target_column=target_col,
        correlation_threshold=req.correlation_threshold
    )

    total_nodes_created = ds_res["nodes_created"]
    total_edges_created = ds_res["edges_created"]

    # 2. Extract ML Models
    if req.include_models:
        db_models = db.query(MLModel).filter(
            MLModel.project_id == project.id,
            MLModel.dataset_id == dataset.id
        ).all()

        if db_models:
            models_data = [
                {
                    "id": m.id,
                    "name": m.name,
                    "algorithm": m.algorithm,
                    "hyperparameters": m.hyperparameters or {},
                    "metrics": m.all_metrics or {m.primary_metric_name: m.primary_metric_value},
                    "status": m.status
                }
                for m in db_models
            ]
            m_res = extractor.extract_from_ml_models(
                dataset_id=dataset.id,
                models_data=models_data,
                project_id=project.id,
                target_column=target_col
            )
            total_nodes_created += m_res["nodes_created"]
            total_edges_created += m_res["edges_created"]

    # 3. Extract Anomalies
    if req.include_anomalies:
        anom_events = db.query(AnomalyEvent).filter(AnomalyEvent.dataset_id == dataset.id).all()
        if anom_events:
            sev_counts: Dict[str, int] = {}
            for ev in anom_events:
                sev_counts[ev.severity] = sev_counts.get(ev.severity, 0) + 1

            anom_summary = {
                "total_anomalies": len(anom_events),
                "contamination_rate": round(len(anom_events) / max(1, len(df)), 4),
                "severity_breakdown": sev_counts
            }
            a_res = extractor.extract_from_anomalies(
                dataset_id=dataset.id,
                anomalies_summary=anom_summary,
                project_id=project.id
            )
            total_nodes_created += a_res["nodes_created"]
            total_edges_created += a_res["edges_created"]

    # 4. Extract Forecasts
    if req.include_forecasts:
        fc_series = db.query(ForecastModel).filter(ForecastModel.dataset_id == dataset.id).all()
        if fc_series:
            fc_summary = {
                "forecast_series_count": len(fc_series),
                "targets": [f.target_column for f in fc_series],
                "horizon": fc_series[0].horizon if fc_series else 30
            }
            fc_res = extractor.extract_from_forecast(
                dataset_id=dataset.id,
                forecast_summary=fc_summary,
                project_id=project.id
            )
            total_nodes_created += fc_res["nodes_created"]
            total_edges_created += fc_res["edges_created"]

    # Persist graph snapshot to disk
    store.save_to_disk(project.id)

    duration_ms = (time.perf_counter() - start_t) * 1000.0
    all_nodes = store.get_nodes(project_id=project.id)
    all_edges = store.get_edges(project_id=project.id)

    return GraphBuildResponse(
        project_id=project.id,
        dataset_id=dataset.id,
        nodes_created=total_nodes_created,
        edges_created=total_edges_created,
        total_nodes=len(all_nodes),
        total_edges=len(all_edges),
        duration_ms=round(duration_ms, 2),
        status="COMPLETED"
    )


@router.post("/query", response_model=GraphRAGQueryResponse)
def query_knowledge_graph(
    req: GraphRAGQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Execute GraphRAG query with entity linking, multi-hop relational path traversal,
    and lineage-verified answer synthesis.
    """
    if req.project_id:
        project = db.query(Project).filter(
            Project.id == req.project_id,
            Project.owner_id == current_user.id
        ).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or access denied."
            )

    store = get_graph_store()
    rag_engine = GraphRAGEngine(store=store)

    result = rag_engine.query(
        query=req.query,
        project_id=req.project_id,
        max_hops=req.max_hops,
        top_k_entities=req.top_k_entities
    )

    return GraphRAGQueryResponse(**result)


@router.get("/node/{node_id}/neighborhood", response_model=NeighborhoodResponse)
def get_node_neighborhood(
    node_id: str,
    max_hops: int = 2,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve k-hop ego subgraph for exploring neighborhood around a specific entity.
    """
    store = get_graph_store()
    subgraph = store.get_neighborhood_subgraph(node_id=node_id, max_hops=max_hops)

    return NeighborhoodResponse(
        root_id=node_id,
        nodes=subgraph.get("nodes", []),
        edges=subgraph.get("edges", []),
        node_count=subgraph.get("node_count", 0),
        edge_count=subgraph.get("edge_count", 0)
    )
