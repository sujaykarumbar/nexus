from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class ModelExplainer:
    """Model Explainability and Feature Attribution Engine."""

    @classmethod
    def compute_global_feature_importance(
        cls,
        model,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute normalized global feature importance scores using tree feature importances,
        linear regression / logistic regression coefficients, or permutation importance.
        """
        importances = None

        # 1. Tree-based feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        # 2. Linear model coefficients
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)

        # 3. Fallback: Permutation Importance if validation data is provided
        elif X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                res = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
                importances = res.importances_mean
            except Exception:
                pass

        if importances is None or len(importances) == 0:
            # Equal attribution fallback
            n_feats = len(feature_names)
            importances = np.ones(n_feats) / max(1, n_feats)

        # Normalize to sum to 1.0 (or 100%)
        sum_imp = float(np.sum(importances))
        if sum_imp > 0:
            normalized_imp = (importances / sum_imp).tolist()
        else:
            normalized_imp = [1.0 / len(feature_names)] * len(feature_names)

        results = []
        for i, name in enumerate(feature_names):
            score = float(normalized_imp[i]) if i < len(normalized_imp) else 0.0
            results.append({
                "feature": name,
                "importance_score": round(score, 4),
                "percentage": round(score * 100, 2)
            })

        # Rank features descending by importance
        results.sort(key=lambda x: x["importance_score"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results

    @classmethod
    def explain_instance_prediction(
        cls,
        model,
        instance_features: np.ndarray,
        feature_names: List[str],
        base_value: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Compute top contributing features for a single prediction instance.
        """
        contributions = []
        if hasattr(model, "feature_importances_"):
            weights = model.feature_importances_
        elif hasattr(model, "coef_"):
            weights = np.abs(model.coef_).flatten()
        else:
            weights = np.ones(len(feature_names))

        # Approximate directional contribution
        flat_instance = instance_features.flatten()
        for i, name in enumerate(feature_names):
            if i < len(flat_instance):
                val = float(flat_instance[i])
                weight = float(weights[i]) if i < len(weights) else 0.01
                impact = round(val * weight, 4)
                contributions.append({
                    "feature": name,
                    "feature_value": round(val, 2),
                    "impact": impact,
                    "direction": "positive" if impact >= 0 else "negative"
                })

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:8]

    explain_prediction = explain_instance_prediction


ExplainabilityEngine = ModelExplainer
