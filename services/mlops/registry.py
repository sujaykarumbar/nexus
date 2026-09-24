"""
NEXUS Model Registry
Tracks model versions, training lineage, dataset hashes, metric snapshots, and artifact paths.
Provides CRUD operations, version comparison, and parent → child lineage traversal.
"""

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ModelVersionRecord:
    """Immutable snapshot of a trained model version."""
    version_id: str
    model_id: str                   # Parent ML/Forecast model ID
    model_type: str                 # 'automl' | 'forecast' | 'anomaly'
    version_tag: str                # e.g. 'v1', 'v2', 'v2.1-retrain'
    algorithm: str                  # e.g. 'XGBoostClassifier', 'HoltWinters'
    metrics: Dict[str, float]       # e.g. {'accuracy': 0.91, 'f1': 0.88}
    dataset_id: str
    dataset_hash: str               # SHA-256 of dataset bytes (reproducibility)
    artifact_path: Optional[str]
    parent_version_id: Optional[str]   # For lineage chaining
    tags: List[str]
    lifecycle_stage: str            # 'experimental' | 'staging' | 'production' | 'archived'
    created_at: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def metric_delta(self, other: "ModelVersionRecord") -> Dict[str, float]:
        """Compute signed metric change vs. another version."""
        delta: Dict[str, float] = {}
        for k, v in self.metrics.items():
            if k in other.metrics:
                delta[k] = round(v - other.metrics[k], 6)
        return delta


class ModelRegistry:
    """
    In-process model version registry.

    Stores versions in an ordered list keyed by version_id. In production this
    would be backed by a dedicated DB table or MLflow tracking server.
    """

    def __init__(self) -> None:
        self._versions: Dict[str, ModelVersionRecord] = {}
        # model_id -> ordered list of version_ids
        self._model_index: Dict[str, List[str]] = {}

    @staticmethod
    def compute_dataset_hash(file_path: str) -> str:
        """Compute SHA-256 fingerprint of a dataset file for reproducibility tracking."""
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        except (OSError, IOError):
            h.update(file_path.encode())
        return h.hexdigest()[:16]

    def register(
        self,
        model_id: str,
        model_type: str,
        algorithm: str,
        metrics: Dict[str, float],
        dataset_id: str,
        dataset_file_path: str,
        artifact_path: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        lifecycle_stage: str = "experimental",
        notes: Optional[str] = None,
    ) -> ModelVersionRecord:
        """Register a new model version and return the record."""
        # Determine version tag
        existing = self._model_index.get(model_id, [])
        version_tag = f"v{len(existing) + 1}"

        version = ModelVersionRecord(
            version_id=str(uuid.uuid4()),
            model_id=model_id,
            model_type=model_type,
            version_tag=version_tag,
            algorithm=algorithm,
            metrics=metrics,
            dataset_id=dataset_id,
            dataset_hash=self.compute_dataset_hash(dataset_file_path),
            artifact_path=artifact_path,
            parent_version_id=parent_version_id,
            tags=tags or [],
            lifecycle_stage=lifecycle_stage,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self._versions[version.version_id] = version
        self._model_index.setdefault(model_id, []).append(version.version_id)
        return version

    def get_version(self, version_id: str) -> Optional[ModelVersionRecord]:
        return self._versions.get(version_id)

    def list_versions(self, model_id: Optional[str] = None) -> List[ModelVersionRecord]:
        if model_id:
            ids = self._model_index.get(model_id, [])
            return [self._versions[i] for i in ids if i in self._versions]
        return list(self._versions.values())

    def compare(
        self, version_id_a: str, version_id_b: str
    ) -> Dict[str, Any]:
        """Side-by-side metric comparison between two versions."""
        a = self._versions.get(version_id_a)
        b = self._versions.get(version_id_b)
        if not a or not b:
            return {"error": "One or both version IDs not found"}

        all_metrics = set(a.metrics) | set(b.metrics)
        comparison = []
        for m in sorted(all_metrics):
            va = a.metrics.get(m)
            vb = b.metrics.get(m)
            delta = None
            winner = None
            if va is not None and vb is not None:
                delta = round(vb - va, 6)
                # Higher is better for accuracy/f1/r2/auc; lower for mae/mse/rmse
                lower_better = any(k in m.lower() for k in ("mae", "mse", "rmse", "loss", "error"))
                if lower_better:
                    winner = "b" if vb < va else ("a" if va < vb else "tie")
                else:
                    winner = "b" if vb > va else ("a" if va > vb else "tie")
            comparison.append({
                "metric": m,
                "version_a": va,
                "version_b": vb,
                "delta": delta,
                "winner": winner,
            })

        return {
            "version_a": a.to_dict(),
            "version_b": b.to_dict(),
            "metric_comparison": comparison,
            "overall_winner": _compute_winner(comparison),
        }

    def promote(self, version_id: str, stage: str) -> Optional[ModelVersionRecord]:
        v = self._versions.get(version_id)
        if v:
            v.lifecycle_stage = stage.lower()
        return v

    def get_lineage(self, version_id: str) -> List[ModelVersionRecord]:
        """Traverse parent chain and return ordered lineage (oldest first)."""
        chain: List[ModelVersionRecord] = []
        current_id: Optional[str] = version_id
        visited = set()
        while current_id and current_id not in visited:
            v = self._versions.get(current_id)
            if not v:
                break
            chain.append(v)
            visited.add(current_id)
            current_id = v.parent_version_id
        return list(reversed(chain))


def _compute_winner(comparison: List[Dict[str, Any]]) -> str:
    """Simple majority vote on per-metric winners."""
    score = {"a": 0, "b": 0}
    for c in comparison:
        w = c.get("winner")
        if w in score:
            score[w] += 1
    if score["b"] > score["a"]:
        return "version_b"
    elif score["a"] > score["b"]:
        return "version_a"
    return "tie"


# Global singleton registry
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
