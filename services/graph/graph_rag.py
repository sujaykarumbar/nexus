"""
NEXUS GraphRAG Engine
Combines Knowledge Graph multi-hop relational traversal with hybrid document retrieval.
Transforms complex entity-relationship graphs into grounded, lineage-verified analytical answers.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from .schema import GraphNode, GraphEdge, NodeType, RelationType, GraphPath
from .graph_store import BaseGraphStore, get_graph_store

logger = logging.getLogger(__name__)


class GraphRAGEngine:
    """
    Graph-Augmented Retrieval & Question Answering Engine.
    Executes entity linking, multi-hop sub-graph traversal, relational fact extraction,
    and synthesized lineage answers.
    """

    def __init__(self, store: Optional[BaseGraphStore] = None):
        self.store = store or get_graph_store()

    def link_entities(self, query: str, project_id: Optional[str] = None, max_entities: int = 4) -> List[GraphNode]:
        """
        Identify seed nodes in the knowledge graph that correspond to terms in the query.
        """
        matched_nodes = self.store.search_nodes(query=query, project_id=project_id, limit=max_entities)
        return matched_nodes

    def extract_relational_facts(self, edges: List[Dict[str, Any]], nodes_map: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Format graph edges into clear, structured natural language factual assertions.
        """
        facts: List[str] = []
        for e in edges:
            src = nodes_map.get(e["source_id"], {"name": e["source_id"]})
            tgt = nodes_map.get(e["target_id"], {"name": e["target_id"]})
            rel = e.get("relation", "RELATES_TO")
            props = e.get("properties", {})

            if rel == "HAS_COLUMN":
                facts.append(f"Dataset '{src['name']}' includes feature '{tgt['name']}'.")
            elif rel == "CORRELATES_WITH":
                r_val = props.get("correlation_r", e.get("weight", "N/A"))
                facts.append(f"Feature '{src['name']}' correlates with '{tgt['name']}' (Pearson r = {r_val}).")
            elif rel == "TARGET_OF":
                facts.append(f"'{src['name']}' is targeted by '{tgt['name']}'.")
            elif rel == "TRAINED_ON":
                facts.append(f"Model '{src['name']}' was trained on dataset '{tgt['name']}'.")
            elif rel == "ACHIEVED_METRIC":
                facts.append(f"Model '{src['name']}' achieved performance metric '{tgt['name']}'.")
            elif rel == "HAS_ANOMALY":
                count = props.get("count", "multiple")
                facts.append(f"Dataset '{src['name']}' has detected anomalies: '{tgt['name']}'.")
            elif rel == "HAS_FORECAST":
                facts.append(f"Dataset '{src['name']}' has projected time-series forecasts: '{tgt['name']}'.")
            elif rel == "MENTIONS":
                facts.append(f"Document '{src['name']}' references domain concept '{tgt['name']}'.")
            else:
                facts.append(f"'{src['name']}' {rel} '{tgt['name']}'.")

        return facts

    def query(
        self,
        query: str,
        project_id: Optional[str] = None,
        max_hops: int = 2,
        top_k_entities: int = 4
    ) -> Dict[str, Any]:
        """
        Execute full GraphRAG pipeline:
        1. Link entities from query.
        2. Extract connected relational subgraph across k-hops.
        3. Discover multi-hop paths between candidate entities.
        4. Synthesize verifiable response with lineage citations.
        """
        start_t = time.perf_counter()

        seed_nodes = self.link_entities(query=query, project_id=project_id, max_entities=top_k_entities)

        if not seed_nodes:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "answer": "No relevant entities or analytical relationships were identified in the knowledge graph for this query.",
                "grounded": False,
                "seed_entities": [],
                "facts": [],
                "subgraph": {"nodes": [], "edges": []},
                "paths": [],
                "duration_ms": round(duration_ms, 2)
            }

        # Accumulate k-hop subgraphs around seed nodes
        collected_node_ids: Set[str] = set()
        for seed in seed_nodes:
            sub = self.store.get_neighborhood_subgraph(seed.id, max_hops=max_hops)
            for n in sub.get("nodes", []):
                collected_node_ids.add(n["id"])

        all_nodes = [self.store.get_node(nid).to_dict() for nid in collected_node_ids if self.store.get_node(nid)]
        nodes_map = {n["id"]: n for n in all_nodes}

        all_edges = [
            e.to_dict() for e in self.store.get_edges(project_id=project_id)
            if e.source_id in collected_node_ids and e.target_id in collected_node_ids
        ]

        # Multi-hop paths between seeds if >= 2 seeds
        paths: List[Dict[str, Any]] = []
        if len(seed_nodes) >= 2:
            for i in range(len(seed_nodes)):
                for j in range(i + 1, len(seed_nodes)):
                    found = self.store.find_paths(seed_nodes[i].id, seed_nodes[j].id, max_depth=3)
                    paths.extend([p.model_dump() for p in found])

        # Extract human-readable relational facts
        facts = self.extract_relational_facts(all_edges, nodes_map)

        # Synthesize Answer
        seed_names = [f"'{s.name}' ({s.node_type.value})" for s in seed_nodes]
        fact_lines = "\n".join([f"• {f}" for f in facts[:8]]) if facts else "• Direct entity matched without further relations."

        answer = (
            f"**Knowledge Graph Relational Intelligence**:\n\n"
            f"Identified core entities: {', '.join(seed_names)}.\n\n"
            f"**Verified Graph Relations & Lineage**:\n"
            f"{fact_lines}\n\n"
            f"*Discovered {len(all_nodes)} connected entities and {len(all_edges)} structural relationships.*"
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "answer": answer,
            "grounded": True,
            "seed_entities": [s.to_dict() for s in seed_nodes],
            "facts": facts,
            "subgraph": {
                "nodes": all_nodes,
                "edges": all_edges,
                "node_count": len(all_nodes),
                "edge_count": len(all_edges)
            },
            "paths": paths,
            "duration_ms": round(duration_ms, 2)
        }
