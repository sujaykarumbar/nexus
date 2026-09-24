"""
NEXUS Knowledge Graph Store
Dual-engine Graph Storage:
- NetworkXGraphStore: High-performance, in-memory MultiDiGraph with JSON disk persistence.
- Neo4jGraphStore: Optional enterprise connector for Neo4j Bolt graph instances with Cypher queries.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple
import networkx as nx

from .schema import GraphNode, GraphEdge, NodeType, RelationType, GraphPath

logger = logging.getLogger(__name__)


class BaseGraphStore(ABC):
    """Abstract interface for NEXUS Graph Storage engines."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        pass

    @abstractmethod
    def get_nodes(self, project_id: Optional[str] = None, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        pass

    @abstractmethod
    def get_edges(self, project_id: Optional[str] = None, relation: Optional[RelationType] = None) -> List[GraphEdge]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[GraphNode]:
        pass

    @abstractmethod
    def get_neighborhood_subgraph(self, node_id: str, max_hops: int = 2) -> Dict[str, Any]:
        pass

    @abstractmethod
    def find_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[GraphPath]:
        pass

    @abstractmethod
    def search_nodes(self, query: str, project_id: Optional[str] = None, limit: int = 10) -> List[GraphNode]:
        pass

    @abstractmethod
    def get_graph_metrics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def clear(self, project_id: Optional[str] = None) -> None:
        pass


class NetworkXGraphStore(BaseGraphStore):
    """
    In-memory and JSON-persisted MultiDiGraph store using NetworkX.
    Zero external dependencies, ideal for instant local development and unit tests.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "data", "graphs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._nodes_map: Dict[str, GraphNode] = {}
        self._edges_map: Dict[str, GraphEdge] = {}

    def add_node(self, node: GraphNode) -> None:
        self._nodes_map[node.id] = node
        self.graph.add_node(
            node.id,
            name=node.name,
            node_type=node.node_type.value if isinstance(node.node_type, NodeType) else str(node.node_type),
            properties=node.properties,
            project_id=node.project_id
        )

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges_map[edge.id] = edge
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            id=edge.id,
            relation=edge.relation.value if isinstance(edge.relation, RelationType) else str(edge.relation),
            weight=edge.weight,
            properties=edge.properties
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes_map.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges_map.get(edge_id)

    def get_nodes(self, project_id: Optional[str] = None, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        result = list(self._nodes_map.values())
        if project_id:
            result = [n for n in result if n.project_id == project_id or n.project_id is None]
        if node_type:
            target_type = node_type.value if isinstance(node_type, NodeType) else str(node_type)
            result = [n for n in result if (n.node_type.value if isinstance(n.node_type, NodeType) else str(n.node_type)) == target_type]
        return result

    def get_edges(self, project_id: Optional[str] = None, relation: Optional[RelationType] = None) -> List[GraphEdge]:
        result = list(self._edges_map.values())
        if project_id:
            valid_node_ids = {n.id for n in self.get_nodes(project_id=project_id)}
            result = [e for e in result if e.source_id in valid_node_ids or e.target_id in valid_node_ids]
        if relation:
            target_rel = relation.value if isinstance(relation, RelationType) else str(relation)
            result = [e for e in result if (e.relation.value if isinstance(e.relation, RelationType) else str(e.relation)) == target_rel]
        return result

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[GraphNode]:
        if node_id not in self.graph:
            return []

        neighbor_ids: Set[str] = set()
        if direction in ("out", "both"):
            neighbor_ids.update(self.graph.successors(node_id))
        if direction in ("in", "both"):
            neighbor_ids.update(self.graph.predecessors(node_id))

        return [self._nodes_map[nid] for nid in neighbor_ids if nid in self._nodes_map]

    def get_neighborhood_subgraph(self, node_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """Extract k-hop ego subgraph centered around node_id."""
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}

        # Ego graph traversal
        visited: Set[str] = {node_id}
        current_layer: Set[str] = {node_id}

        for _ in range(max_hops):
            next_layer: Set[str] = set()
            for nid in current_layer:
                if nid in self.graph:
                    succ = set(self.graph.successors(nid))
                    pred = set(self.graph.predecessors(nid))
                    next_layer.update((succ | pred) - visited)
            visited.update(next_layer)
            current_layer = next_layer
            if not current_layer:
                break

        sub_nodes = [self._nodes_map[nid].to_dict() for nid in visited if nid in self._nodes_map]
        sub_edges = [
            e.to_dict() for e in self._edges_map.values()
            if e.source_id in visited and e.target_id in visited
        ]

        return {
            "root_id": node_id,
            "nodes": sub_nodes,
            "edges": sub_edges,
            "node_count": len(sub_nodes),
            "edge_count": len(sub_edges)
        }

    def find_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[GraphPath]:
        """Find all simple paths between two nodes up to max_depth."""
        if source_id not in self.graph or target_id not in self.graph:
            return []

        simple_g = nx.DiGraph(self.graph)
        try:
            raw_paths = list(nx.all_simple_paths(simple_g, source=source_id, target=target_id, cutoff=max_depth))
        except Exception:
            return []

        paths: List[GraphPath] = []
        for p in raw_paths[:5]:  # Top 5 paths
            path_nodes = [self._nodes_map[nid] for nid in p if nid in self._nodes_map]
            path_edges: List[GraphEdge] = []
            for u, v in zip(p[:-1], p[1:]):
                matching = [
                    e for e in self._edges_map.values()
                    if (e.source_id == u and e.target_id == v) or (e.source_id == v and e.target_id == u)
                ]
                if matching:
                    path_edges.append(matching[0])

            desc = " -> ".join([n.name for n in path_nodes])
            paths.append(GraphPath(
                nodes=path_nodes,
                edges=path_edges,
                score=round(1.0 / (len(p)), 3),
                description=desc
            ))

        return paths

    def search_nodes(self, query: str, project_id: Optional[str] = None, limit: int = 10) -> List[GraphNode]:
        """Fuzzy keyword matching over node names and properties."""
        q_tokens = query.lower().split()
        candidates = self.get_nodes(project_id=project_id)
        scored: List[Tuple[float, GraphNode]] = []

        for node in candidates:
            score = 0.0
            name_lower = node.name.lower()
            id_lower = node.id.lower()
            props_str = json.dumps(node.properties).lower()

            for token in q_tokens:
                if token == name_lower:
                    score += 5.0
                elif token in name_lower:
                    score += 2.5
                elif token in id_lower:
                    score += 1.5
                elif token in props_str:
                    score += 1.0

            if score > 0.0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def get_graph_metrics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute topological graph statistics, density, and influential nodes."""
        nodes = self.get_nodes(project_id=project_id)
        edges = self.get_edges(project_id=project_id)

        node_ids = {n.id for n in nodes}
        subg = self.graph.subgraph(node_ids) if project_id else self.graph

        num_nodes = len(nodes)
        num_edges = len(edges)

        density = round(nx.density(subg), 4) if num_nodes > 1 else 0.0

        # Degree Centrality / PageRank
        try:
            pagerank = nx.pagerank(subg, alpha=0.85, max_iter=100)
            top_ranked = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            key_entities = [
                {
                    "node_id": nid,
                    "name": self._nodes_map[nid].name if nid in self._nodes_map else nid,
                    "type": self._nodes_map[nid].node_type.value if nid in self._nodes_map else "unknown",
                    "score": round(score, 4)
                }
                for nid, score in top_ranked
            ]
        except Exception:
            key_entities = []

        # Count by node type
        type_counts: Dict[str, int] = {}
        for n in nodes:
            t = n.node_type.value if isinstance(n.node_type, NodeType) else str(n.node_type)
            type_counts[t] = type_counts.get(t, 0) + 1

        # Count by edge relation
        relation_counts: Dict[str, int] = {}
        for e in edges:
            r = e.relation.value if isinstance(e.relation, RelationType) else str(e.relation)
            relation_counts[r] = relation_counts.get(r, 0) + 1

        return {
            "node_count": num_nodes,
            "edge_count": num_edges,
            "density": density,
            "node_type_distribution": type_counts,
            "relation_distribution": relation_counts,
            "key_entities": key_entities,
            "is_connected": nx.is_weakly_connected(subg) if num_nodes > 0 else False
        }

    def save_to_disk(self, project_id: str) -> str:
        """Persist project graph to JSON on disk."""
        filepath = os.path.join(self.storage_dir, f"{project_id}.json")
        nodes = [n.to_dict() for n in self.get_nodes(project_id=project_id)]
        edges = [e.to_dict() for e in self.get_edges(project_id=project_id)]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"project_id": project_id, "nodes": nodes, "edges": edges}, f, indent=2)
        return filepath

    def load_from_disk(self, project_id: str) -> bool:
        """Load graph from JSON file if available."""
        filepath = os.path.join(self.storage_dir, f"{project_id}.json")
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for n_dict in data.get("nodes", []):
                self.add_node(GraphNode(**n_dict))
            for e_dict in data.get("edges", []):
                self.add_edge(GraphEdge(**e_dict))
            return True
        except Exception as e:
            logger.warning(f"Failed to load graph from {filepath}: {e}")
            return False

    def clear(self, project_id: Optional[str] = None) -> None:
        if project_id:
            node_ids_to_del = [n.id for n in self.get_nodes(project_id=project_id)]
            for nid in node_ids_to_del:
                if nid in self.graph:
                    self.graph.remove_node(nid)
                self._nodes_map.pop(nid, None)
            edge_ids_to_del = [
                eid for eid, e in self._edges_map.items()
                if e.source_id in node_ids_to_del or e.target_id in node_ids_to_del
            ]
            for eid in edge_ids_to_del:
                self._edges_map.pop(eid, None)
        else:
            self.graph.clear()
            self._nodes_map.clear()
            self._edges_map.clear()


class Neo4jGraphStore(BaseGraphStore):
    """
    Production Neo4j Bolt driver adapter.
    Executes Cypher queries against a live Neo4j 5 instance.
    Falls back gracefully to NetworkX if Neo4j driver is not present.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        fallback_store: Optional[BaseGraphStore] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "nexus_password")
        self.fallback = fallback_store or NetworkXGraphStore()
        self._driver = None
        self._connected = False

        try:
            import neo4j
            self._driver = neo4j.GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            with self._driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info(f"Connected successfully to Neo4j at {self.uri}")
        except Exception as e:
            logger.info(f"Neo4j not active or unavailable ({e}); falling back to in-memory NetworkX store.")
            self._connected = False

    def add_node(self, node: GraphNode) -> None:
        self.fallback.add_node(node)
        if self._connected and self._driver:
            try:
                cypher = """
                MERGE (n:Entity {id: $id})
                SET n.name = $name,
                    n.node_type = $node_type,
                    n.project_id = $project_id,
                    n += $properties
                """
                with self._driver.session() as session:
                    session.run(
                        cypher,
                        id=node.id,
                        name=node.name,
                        node_type=node.node_type.value if isinstance(node.node_type, NodeType) else str(node.node_type),
                        project_id=node.project_id,
                        properties=node.properties
                    )
            except Exception as e:
                logger.warning(f"Neo4j node sync failed: {e}")

    def add_edge(self, edge: GraphEdge) -> None:
        self.fallback.add_edge(edge)
        if self._connected and self._driver:
            try:
                rel_type = edge.relation.value if isinstance(edge.relation, RelationType) else str(edge.relation)
                cypher = f"""
                MATCH (s:Entity {{id: $source_id}}), (t:Entity {{id: $target_id}})
                MERGE (s)-[r:{rel_type} {{id: $id}}]->(t)
                SET r.weight = $weight,
                    r += $properties
                """
                with self._driver.session() as session:
                    session.run(
                        cypher,
                        id=edge.id,
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        weight=edge.weight,
                        properties=edge.properties
                    )
            except Exception as e:
                logger.warning(f"Neo4j edge sync failed: {e}")

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.fallback.get_node(node_id)

    def get_nodes(self, project_id: Optional[str] = None, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        return self.fallback.get_nodes(project_id=project_id, node_type=node_type)

    def get_edges(self, project_id: Optional[str] = None, relation: Optional[RelationType] = None) -> List[GraphEdge]:
        return self.fallback.get_edges(project_id=project_id, relation=relation)

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[GraphNode]:
        return self.fallback.get_neighbors(node_id=node_id, direction=direction)

    def get_neighborhood_subgraph(self, node_id: str, max_hops: int = 2) -> Dict[str, Any]:
        return self.fallback.get_neighborhood_subgraph(node_id=node_id, max_hops=max_hops)

    def find_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[GraphPath]:
        return self.fallback.find_paths(source_id=source_id, target_id=target_id, max_depth=max_depth)

    def search_nodes(self, query: str, project_id: Optional[str] = None, limit: int = 10) -> List[GraphNode]:
        return self.fallback.search_nodes(query=query, project_id=project_id, limit=limit)

    def get_graph_metrics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        metrics = self.fallback.get_graph_metrics(project_id=project_id)
        metrics["backend"] = "neo4j" if self._connected else "networkx_memory"
        return metrics

    def clear(self, project_id: Optional[str] = None) -> None:
        self.fallback.clear(project_id=project_id)
        if self._connected and self._driver:
            try:
                with self._driver.session() as session:
                    if project_id:
                        session.run("MATCH (n:Entity {project_id: $project_id}) DETACH DELETE n", project_id=project_id)
                    else:
                        session.run("MATCH (n) DETACH DELETE n")
            except Exception as e:
                logger.warning(f"Neo4j clear failed: {e}")


# Global Singleton Store instance
_GLOBAL_STORE: Optional[BaseGraphStore] = None


def get_graph_store() -> BaseGraphStore:
    """Return configured Graph Store singleton."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        neo4j_uri = os.getenv("NEO4J_URI")
        if neo4j_uri:
            _GLOBAL_STORE = Neo4jGraphStore(uri=neo4j_uri)
        else:
            _GLOBAL_STORE = NetworkXGraphStore()
    return _GLOBAL_STORE
