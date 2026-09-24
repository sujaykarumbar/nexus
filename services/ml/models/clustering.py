from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ClusteringModelFactory:
    """Unsupervised Clustering Engine with Metric Evaluation."""

    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        return ["kmeans", "dbscan", "agglomerative"]

    @classmethod
    def fit_and_evaluate(
        cls,
        algorithm: str,
        X: np.ndarray,
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train clustering model and calculate silhouette and Davies-Bouldin scores."""
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
            labels = model.fit_predict(X)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X)
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")

        unique_labels = set(labels)
        n_clusters_found = len(unique_labels - {-1})

        metrics = {
            "n_clusters": int(n_clusters_found),
            "n_samples": len(X),
            "noise_samples": int((labels == -1).sum()) if -1 in labels else 0
        }

        # Calculate scores if more than 1 cluster found
        if 1 < n_clusters_found < len(X):
            # Mask out noise for DBSCAN if applicable
            valid_idx = labels != -1
            if valid_idx.sum() > n_clusters_found:
                sil = float(silhouette_score(X[valid_idx], labels[valid_idx]))
                db = float(davies_bouldin_score(X[valid_idx], labels[valid_idx]))
                metrics["silhouette_score"] = round(sil, 4)
                metrics["davies_bouldin_index"] = round(db, 4)

        return {
            "algorithm": algorithm,
            "metrics": metrics,
            "labels": labels.tolist(),
            "model_instance": model
        }
