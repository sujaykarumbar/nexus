"""
Unit & Integration Tests for NEXUS GraphRAG Engine
"""

import pytest
from services.graph.schema import GraphNode, GraphEdge, NodeType, RelationType
from services.graph.graph_store import NetworkXGraphStore
from services.graph.graph_rag import GraphRAGEngine


@pytest.fixture
def populated_graph_store():
    store = NetworkXGraphStore()
    store.clear()

    # Create connected knowledge graph
    ds = GraphNode(id="ds_1", name="Telco Churn Dataset", node_type=NodeType.DATASET, project_id="p1")
    col_churn = GraphNode(id="col_churn", name="churn", node_type=NodeType.COLUMN, project_id="p1")
    col_mins = GraphNode(id="col_mins", name="total_day_minutes", node_type=NodeType.COLUMN, project_id="p1")
    col_charge = GraphNode(id="col_charge", name="total_day_charge", node_type=NodeType.COLUMN, project_id="p1")
    model = GraphNode(id="model_rf", name="Random Forest Classifier", node_type=NodeType.MODEL, project_id="p1")
    metric = GraphNode(id="met_f1", name="F1: 0.94", node_type=NodeType.METRIC, project_id="p1", properties={"value": 0.94})

    for n in [ds, col_churn, col_mins, col_charge, model, metric]:
        store.add_node(n)

    store.add_edge(GraphEdge(id="e1", source_id="ds_1", target_id="col_churn", relation=RelationType.HAS_COLUMN))
    store.add_edge(GraphEdge(id="e2", source_id="ds_1", target_id="col_mins", relation=RelationType.HAS_COLUMN))
    store.add_edge(GraphEdge(id="e3", source_id="col_mins", target_id="col_charge", relation=RelationType.CORRELATES_WITH, properties={"correlation_r": 0.99}))
    store.add_edge(GraphEdge(id="e4", source_id="model_rf", target_id="ds_1", relation=RelationType.TRAINED_ON))
    store.add_edge(GraphEdge(id="e5", source_id="model_rf", target_id="met_f1", relation=RelationType.ACHIEVED_METRIC))

    return store


def test_graph_rag_query_resolution(populated_graph_store):
    engine = GraphRAGEngine(store=populated_graph_store)

    # Query for model and metric
    res = engine.query(query="What is the Random Forest performance on Churn?", project_id="p1")

    assert res["grounded"] is True
    assert len(res["seed_entities"]) >= 1
    assert len(res["facts"]) >= 1
    assert "Random Forest" in res["answer"]
    assert len(res["subgraph"]["nodes"]) >= 2


def test_graph_rag_unmatched_query(populated_graph_store):
    engine = GraphRAGEngine(store=populated_graph_store)

    res = engine.query(query="Quantum cryptography satellite orbit velocity", project_id="p1")
    assert res["grounded"] is False
    assert len(res["facts"]) == 0
