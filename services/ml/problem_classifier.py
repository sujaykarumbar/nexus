from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ProblemClassifier:
    """Classifies Machine Learning Problem Type and Target Distribution."""

    @classmethod
    def classify_problem(cls, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Classify problem type (binary_classification, multiclass_classification, regression)
        and assess class imbalance / distribution.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset columns.")

        series = df[target_column].dropna()
        n_unique = series.nunique()
        total_count = len(series)

        if total_count == 0:
            return {
                "problem_type": "unknown",
                "is_classification": False,
                "is_regression": False,
                "classes": [],
                "class_distribution": {},
                "is_imbalanced": False,
                "reasoning": "Target column contains only null values."
            }

        # Check numeric continuous vs categorical
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # 1. Binary Classification
        if n_unique == 2:
            unique_classes = [str(x) for x in series.unique()]
            val_counts = series.value_counts(normalize=True).to_dict()
            class_dist = {str(k): round(float(v * 100), 2) for k, v in val_counts.items()}
            
            # Check for class imbalance (minority class <= 25%)
            min_class_pct = min(class_dist.values())
            is_imbalanced = min_class_pct <= 25.0

            return {
                "problem_type": "binary_classification",
                "is_classification": True,
                "is_regression": False,
                "classes": unique_classes,
                "class_count": 2,
                "num_classes": 2,
                "class_distribution": class_dist,
                "is_imbalanced": is_imbalanced,
                "imbalance_ratio": round(max(class_dist.values()) / max(0.1, min_class_pct), 2),
                "recommended_primary_metric": "f1" if is_imbalanced else "roc_auc",
                "primary_metric": "f1" if is_imbalanced else "roc_auc",
                "reasoning": f"Target has exactly 2 distinct classes. {'Severe class imbalance detected (<= 25% minority).' if is_imbalanced else 'Balanced binary targets.'}"
            }

        # 2. Multiclass Classification
        if not is_numeric or n_unique <= 20:
            unique_classes = [str(x) for x in series.unique()]
            val_counts = series.value_counts(normalize=True).head(10).to_dict()
            class_dist = {str(k): round(float(v * 100), 2) for k, v in val_counts.items()}
            min_class_pct = min(class_dist.values())
            is_imbalanced = min_class_pct < (100.0 / (n_unique * 2))

            return {
                "problem_type": "multiclass_classification",
                "is_classification": True,
                "is_regression": False,
                "classes": unique_classes,
                "class_count": n_unique,
                "num_classes": n_unique,
                "class_distribution": class_dist,
                "is_imbalanced": is_imbalanced,
                "recommended_primary_metric": "f1_macro",
                "primary_metric": "f1_macro",
                "reasoning": f"Discrete categorical target with {n_unique} distinct classes."
            }

        # 3. Regression
        stats_dict = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "skewness": round(float(series.skew()), 4)
        }
        return {
            "problem_type": "regression",
            "is_classification": False,
            "is_regression": True,
            "classes": [],
            "class_count": 0,
            "num_classes": 0,
            "class_distribution": {},
            "is_imbalanced": False,
            "stats": stats_dict,
            "distribution_stats": stats_dict,
            "recommended_primary_metric": "rmse",
            "primary_metric": "rmse",
            "reasoning": f"Continuous numeric target across {n_unique} unique numeric values."
        }

    classify = classify_problem
