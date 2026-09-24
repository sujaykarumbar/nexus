"""
Unit & Integration Tests for NEXUS Knowledge Graph Store & Extractor
"""

import pytest
import pandas as pd
import numpy as np

from services.graph.schema import GraphNode, GraphEdge, NodeType, RelationType
from services.graph.graph_store import NetworkXGraphStore
from services.graph.extractor import KnowledgeGraphExtractor


@pytest.fixture
def mock_graph_store():
    store = NetworkXGraphStore()
    store.clear()
    return store


@pytest.fixture
def sample_churn_df():
    np.random.seed(42)
    n = 100
    day_mins = np.random.uniform(50, 350, n)
    # Strongly collinear charge = day_mins * 0.17
    day_charge = day_mins * 0.17 + np.random.normal(0, 0.01, n)
    cust_service_calls = np.random.poisson(1.5, n)
    churn = (cust_service_calls > 3).astype(int)

    return pd.DataFrame({
        "account_id": [f"acc_{i}" for i in range(n)],
        "total_day_minutes": day_mins,
        "total_day_charge": day_charge,
        "customer_service_calls": cust_service_calls,
        "churn": churn
    })


def test_graph_store_node_edge_operations(mock_graph_store):
    store = mock_graph_store

    # 1. Add nodes
    n1 = GraphNode(id="n1", name="Churn Dataset", node_type=NodeType.DATASET, project_id="p1")
    n2 = GraphNode(id="n2", name="churn", node_type=NodeType.COLUMN, project_id="p1")
    n3 = GraphNode(id="n3", name="Random Forest", node_type=NodeType.MODEL, project_id="p1")

    store.add_node(n1)
    store.add_node(n2)
    store.add_node(n3)

    assert len(store.get_nodes(project_id="p1")) == 3
    assert store.get_node("n1").name == "Churn Dataset"

    # 2. Add edges
    e1 = GraphEdge(id="e1", source_id="n1", target_id="n2", relation=RelationType.HAS_COLUMN)
    e2 = GraphEdge(id="e2", source_id="n3", target_id="n1", relation=RelationType.TRAINED_ON)
    store.add_edge(e1)
    store.add_edge(e2)

    assert len(store.get_edges(project_id="p1")) == 2

    # 3. Neighborhood & Subgraph
    neighbors = store.get_neighbors("n1")
    assert len(neighbors) == 2

    subgraph = store.get_neighborhood_subgraph("n1", max_hops=1)
    assert subgraph["node_count"] == 3
    assert subgraph["edge_count"] == 2

    # 4. Search nodes
    search_res = store.search_nodes("churn", project_id="p1")
    assert len(search_res) >= 2

    # 5. Graph metrics
    metrics = store.get_graph_metrics(project_id="p1")
    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 2
    assert "dataset" in metrics["node_type_distribution"]


def test_knowledge_graph_extractor_dataset(mock_graph_store, sample_churn_df):
    extractor = KnowledgeGraphExtractor(store=mock_graph_store)

    res = extractor.extract_from_dataset(
        df=sample_churn_df,
        dataset_id="ds_123",
        dataset_name="Customer Churn Test",
        project_id="proj_1",
        target_column="churn",
        correlation_threshold=0.5
    )

    assert res["nodes_created"] >= 6  # 1 dataset + 5 columns + 1 problem
    assert res["edges_created"] >= 6

    # Verify strong correlation edge between total_day_minutes and total_day_charge
    col_mins_id = "col_ds_123_total_day_minutes"
    col_charge_id = "col_ds_123_total_day_charge"

    edges = mock_graph_store.get_edges(relation=RelationType.CORRELATES_WITH)
    corr_edge = [
        e for e in edges
        if (e.source_id == col_mins_id and e.target_id == col_charge_id) or
           (e.source_id == col_charge_id and e.target_id == col_mins_id)
    ]
    assert len(corr_edge) == 1
    assert corr_edge[0].weight >= 0.95


def test_knowledge_graph_extractor_models(mock_graph_store):
    extractor = KnowledgeGraphExtractor(store=mock_graph_store)

    # First add dataset node
    ds_node = GraphNode(id="ds_456", name="Sales Dataset", node_type=NodeType.DATASET, project_id="p2")
    mock_graph_store.add_node(ds_node)

    models_data = [
        {
            "id": "m1",
            "name": "XGBoost Classifier",
            "algorithm": "xgboost",
            "hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "metrics": {"f1": 0.92, "roc_auc": 0.96, "accuracy": 0.94},
            "status": "ready"
        }
    ]

    res = extractor.extract_from_ml_models(
        dataset_id="456",
        models_data=models_data,
        project_id="p2"
    )

    assert res["models_ingested"] == 1
    assert res["nodes_created"] == 4  # 1 model + 3 metrics
    assert res["edges_created"] == 4  # 1 TRAINED_ON + 3 ACHIEVED_METRIC

    # Check metric nodes
    metric_nodes = mock_graph_store.get_nodes(project_id="p2", node_type=NodeType.METRIC)
    assert len(metric_nodes) == 3
