"""
NEXUS Knowledge Graph Extractor
Autonomously extracts entities, attributes, and relationships across:
1. Tabular datasets, schemas, and column statistics.
2. Correlation matrices and feature collinearity.
3. Machine learning problems, target features, and model benchmarks.
4. Time-series forecasts and anomaly clusters.
5. Ingested documents, text chunks, and business concepts.
"""

import re
import uuid
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .schema import GraphNode, GraphEdge, NodeType, RelationType
from .graph_store import BaseGraphStore, get_graph_store
from services.profiling.schema_detector import SchemaDetector
from services.eda.eda_engine import EDAEngine

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """
    Extracts structured knowledge graphs from multi-modal analytical artifacts.
    """

    def __init__(self, store: Optional[BaseGraphStore] = None):
        self.store = store or get_graph_store()

    def extract_from_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_name: str,
        project_id: Optional[str] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        correlation_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Extract complete dataset graph: Dataset Node -> Column Nodes -> Correlation Edges -> Target Problem.
        """
        nodes_created = 0
        edges_created = 0

        # 1. Dataset Node
        dataset_node_id = f"ds_{dataset_id}"
        dataset_node = GraphNode(
            id=dataset_node_id,
            name=dataset_name,
            node_type=NodeType.DATASET,
            project_id=project_id,
            properties={
                "dataset_id": dataset_id,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns)
            }
        )
        self.store.add_node(dataset_node)
        nodes_created += 1

        # 2. Column Nodes & HAS_COLUMN Edges
        schema_info = SchemaDetector.detect_schema(df)
        col_nodes: Dict[str, str] = {}

        for col_name, col_meta in schema_info.get("columns", {}).items():
            col_node_id = f"col_{dataset_id}_{col_name}"
            col_node = GraphNode(
                id=col_node_id,
                name=col_name,
                node_type=NodeType.COLUMN,
                project_id=project_id,
                properties={
                    "column_name": col_name,
                    "dataset_id": dataset_id,
                    "inferred_type": col_meta.get("inferred_type", "UNKNOWN"),
                    "data_type": col_meta.get("data_type", "object"),
                    "null_count": col_meta.get("null_count", 0),
                    "null_percentage": col_meta.get("null_percentage", 0.0),
                    "unique_count": col_meta.get("unique_count", 0)
                }
            )
            self.store.add_node(col_node)
            col_nodes[col_name] = col_node_id
            nodes_created += 1

            # Edge: Dataset -> Column
            has_col_edge = GraphEdge(
                id=f"edge_has_col_{dataset_id}_{col_name}",
                source_id=dataset_node_id,
                target_id=col_node_id,
                relation=RelationType.HAS_COLUMN,
                weight=1.0,
                properties={"column_name": col_name}
            )
            self.store.add_edge(has_col_edge)
            edges_created += 1

        # 3. Correlation Matrix & CORRELATES_WITH Edges
        numeric_cols = [
            c for c, m in schema_info.get("columns", {}).items()
            if m.get("inferred_type") == "NUMERICAL" or pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) >= 2:
            corr_result = EDAEngine.calculate_correlations(df)
            corr_matrix = corr_result.get("matrix", {})

            seen_pairs = set()
            for c1 in numeric_cols:
                for c2 in numeric_cols:
                    if c1 == c2:
                        continue
                    pair_key = tuple(sorted([c1, c2]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    r_val = corr_matrix.get(c1, {}).get(c2)
                    if r_val is not None and not np.isnan(r_val):
                        r_abs = abs(float(r_val))
                        if r_abs >= correlation_threshold:
                            edge_corr = GraphEdge(
                                id=f"edge_corr_{dataset_id}_{pair_key[0]}_{pair_key[1]}",
                                source_id=col_nodes[pair_key[0]],
                                target_id=col_nodes[pair_key[1]],
                                relation=RelationType.CORRELATES_WITH,
                                weight=round(r_abs, 3),
                                properties={
                                    "correlation_r": round(float(r_val), 3),
                                    "strength": "strong" if r_abs >= 0.7 else ("moderate" if r_abs >= 0.4 else "weak")
                                }
                            )
                            self.store.add_edge(edge_corr)
                            edges_created += 1

        # 4. Target Column & Problem Classification Node
        if target_column and target_column in col_nodes:
            target_node_id = col_nodes[target_column]
            prob_node_id = f"prob_{dataset_id}_{target_column}"
            detected_prob = problem_type or ("binary_classification" if df[target_column].nunique() == 2 else "multiclass_classification")

            prob_node = GraphNode(
                id=prob_node_id,
                name=f"{detected_prob.replace('_', ' ').title()}: {target_column}",
                node_type=NodeType.PROBLEM,
                project_id=project_id,
                properties={
                    "problem_type": detected_prob,
                    "target_column": target_column,
                    "dataset_id": dataset_id
                }
            )
            self.store.add_node(prob_node)
            nodes_created += 1

            # Target feature edge: Column -> Target Problem
            target_edge = GraphEdge(
                id=f"edge_target_{dataset_id}_{target_column}",
                source_id=target_node_id,
                target_id=prob_node_id,
                relation=RelationType.TARGET_OF,
                weight=1.0,
                properties={"target_column": target_column}
            )
            self.store.add_edge(target_edge)
            edges_created += 1

            # Problem -> Dataset edge
            part_edge = GraphEdge(
                id=f"edge_prob_ds_{dataset_id}_{target_column}",
                source_id=prob_node_id,
                target_id=dataset_node_id,
                relation=RelationType.BELONGS_TO,
                weight=1.0
            )
            self.store.add_edge(part_edge)
            edges_created += 1

        return {
            "dataset_id": dataset_id,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "root_node_id": dataset_node_id
        }

    def extract_from_ml_models(
        self,
        dataset_id: str,
        models_data: List[Dict[str, Any]],
        project_id: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract ML models and their benchmark metrics into the knowledge graph.
        """
        nodes_created = 0
        edges_created = 0
        dataset_node_id = f"ds_{dataset_id}"

        for m_dict in models_data:
            model_id = m_dict.get("id") or m_dict.get("model_id") or f"model_{uuid.uuid4().hex[:8]}"
            model_name = m_dict.get("name") or m_dict.get("algorithm") or "Machine Learning Model"
            algo = m_dict.get("algorithm", model_name)

            model_node_id = f"model_{model_id}"
            model_node = GraphNode(
                id=model_node_id,
                name=f"{model_name}",
                node_type=NodeType.MODEL,
                project_id=project_id,
                properties={
                    "model_id": model_id,
                    "algorithm": algo,
                    "hyperparameters": m_dict.get("hyperparameters", {}),
                    "is_champion": m_dict.get("is_champion", False),
                    "status": m_dict.get("status", "completed")
                }
            )
            self.store.add_node(model_node)
            nodes_created += 1

            # Edge: Model -> Dataset (TRAINED_ON)
            trained_edge = GraphEdge(
                id=f"edge_trained_{model_id}_{dataset_id}",
                source_id=model_node_id,
                target_id=dataset_node_id,
                relation=RelationType.TRAINED_ON,
                weight=1.0
            )
            self.store.add_edge(trained_edge)
            edges_created += 1

            # Edge: Model -> Target Problem (if target_column provided)
            if target_column:
                prob_node_id = f"prob_{dataset_id}_{target_column}"
                if self.store.get_node(prob_node_id):
                    target_edge = GraphEdge(
                        id=f"edge_model_prob_{model_id}",
                        source_id=model_node_id,
                        target_id=prob_node_id,
                        relation=RelationType.TARGET_OF,
                        weight=1.0
                    )
                    self.store.add_edge(target_edge)
                    edges_created += 1

            # Extract Metrics
            metrics = m_dict.get("metrics") or m_dict.get("evaluation_metrics") or {}
            for metric_name, val in metrics.items():
                if isinstance(val, (int, float)) and not np.isnan(val):
                    metric_node_id = f"metric_{model_id}_{metric_name}"
                    metric_node = GraphNode(
                        id=metric_node_id,
                        name=f"{metric_name.upper()}: {round(float(val), 4)}",
                        node_type=NodeType.METRIC,
                        project_id=project_id,
                        properties={
                            "metric_name": metric_name,
                            "value": round(float(val), 4),
                            "model_id": model_id
                        }
                    )
                    self.store.add_node(metric_node)
                    nodes_created += 1

                    # Edge: Model -> Metric (ACHIEVED_METRIC)
                    achieved_edge = GraphEdge(
                        id=f"edge_achieved_{model_id}_{metric_name}",
                        source_id=model_node_id,
                        target_id=metric_node_id,
                        relation=RelationType.ACHIEVED_METRIC,
                        weight=min(1.0, max(0.0, float(val))) if "loss" not in metric_name.lower() and "error" not in metric_name.lower() else 0.5,
                        properties={"value": float(val)}
                    )
                    self.store.add_edge(achieved_edge)
                    edges_created += 1

        return {
            "models_ingested": len(models_data),
            "nodes_created": nodes_created,
            "edges_created": edges_created
        }

    def extract_from_anomalies(
        self,
        dataset_id: str,
        anomalies_summary: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract detected anomaly clusters and high-severity outliers into the graph.
        """
        nodes_created = 0
        edges_created = 0
        dataset_node_id = f"ds_{dataset_id}"

        anomaly_count = anomalies_summary.get("total_anomalies", 0)
        contamination = anomalies_summary.get("contamination_rate", 0.05)

        anomaly_node_id = f"anom_{dataset_id}"
        anomaly_node = GraphNode(
            id=anomaly_node_id,
            name=f"Anomalies: {anomaly_count} records ({round(contamination * 100, 1)}%)",
            node_type=NodeType.ANOMALY,
            project_id=project_id,
            properties=anomalies_summary
        )
        self.store.add_node(anomaly_node)
        nodes_created += 1

        has_anom_edge = GraphEdge(
            id=f"edge_has_anom_{dataset_id}",
            source_id=dataset_node_id,
            target_id=anomaly_node_id,
            relation=RelationType.HAS_ANOMALY,
            weight=1.0,
            properties={"count": anomaly_count}
        )
        self.store.add_edge(has_anom_edge)
        edges_created += 1

        return {"nodes_created": nodes_created, "edges_created": edges_created}

    def extract_from_forecast(
        self,
        dataset_id: str,
        forecast_summary: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract time-series forecast projections into the graph.
        """
        nodes_created = 0
        edges_created = 0
        dataset_node_id = f"ds_{dataset_id}"

        horizon = forecast_summary.get("horizon", 30)
        best_model = forecast_summary.get("best_model", "AutoARIMA")

        forecast_node_id = f"fc_{dataset_id}"
        forecast_node = GraphNode(
            id=forecast_node_id,
            name=f"Forecast: {horizon} periods via {best_model}",
            node_type=NodeType.FORECAST,
            project_id=project_id,
            properties=forecast_summary
        )
        self.store.add_node(forecast_node)
        nodes_created += 1

        edge_fc = GraphEdge(
            id=f"edge_has_fc_{dataset_id}",
            source_id=dataset_node_id,
            target_id=forecast_node_id,
            relation=RelationType.HAS_FORECAST,
            weight=1.0
        )
        self.store.add_edge(edge_fc)
        edges_created += 1

        return {"nodes_created": nodes_created, "edges_created": edges_created}

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        title: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract domain concepts and relate document chunks to relevant tabular features or models.
        """
        nodes_created = 0
        edges_created = 0

        doc_node_id = f"doc_{document_id}"
        doc_node = GraphNode(
            id=doc_node_id,
            name=title,
            node_type=NodeType.DOCUMENT,
            project_id=project_id,
            properties={"document_id": document_id, "text_length": len(text)}
        )
        self.store.add_node(doc_node)
        nodes_created += 1

        # Search existing column and model nodes in this project to see if the text mentions them
        existing_nodes = self.store.get_nodes(project_id=project_id)
        text_lower = text.lower()

        for node in existing_nodes:
            if node.node_type in (NodeType.COLUMN, NodeType.MODEL, NodeType.DATASET):
                node_name_lower = node.name.lower()
                # Check if exact token or phrase appears in document
                if len(node_name_lower) >= 3 and re.search(r'\b' + re.escape(node_name_lower) + r'\b', text_lower):
                    mention_edge = GraphEdge(
                        id=f"edge_mention_{document_id}_{node.id}",
                        source_id=doc_node_id,
                        target_id=node.id,
                        relation=RelationType.MENTIONS,
                        weight=0.8,
                        properties={"term": node.name}
                    )
                    self.store.add_edge(mention_edge)
                    edges_created += 1

        return {
            "document_id": document_id,
            "nodes_created": nodes_created,
            "edges_created": edges_created
        }
