from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score


class BaselineModelRunner:
    """Trains Dummy Baselines and Computes Actual Benchmark Improvements."""

    @classmethod
    def train_and_evaluate_baseline(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Train a deterministic Dummy baseline and evaluate on holdout test set.
        """
        if is_classification:
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
            
            acc = float(accuracy_score(y_test, y_pred))
            # Average micro/weighted for multi-class, binary otherwise
            unique_classes = np.unique(y_train)
            avg = "binary" if len(unique_classes) <= 2 else "weighted"
            f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))

            return {
                "model_name": "Dummy Baseline (Most Frequent)",
                "algorithm": "DummyClassifier",
                "is_classification": True,
                "metrics": {
                    "accuracy": round(acc, 4),
                    "f1": round(f1, 4)
                },
                "primary_metric_name": "f1",
                "primary_metric_value": round(f1, 4),
                "model_instance": dummy
            }
        else:
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            return {
                "model_name": "Dummy Baseline (Mean)",
                "algorithm": "DummyRegressor",
                "is_classification": False,
                "metrics": {
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4)
                },
                "primary_metric_name": "rmse",
                "primary_metric_value": round(rmse, 4),
                "model_instance": dummy
            }

    @classmethod
    def run_baseline(cls, problem_type: str, X: Any, y: Any) -> Dict[str, Any]:
        """Convenience method to evaluate baseline on a given dataset split."""
        from sklearn.model_selection import train_test_split
        is_classification = "classification" in problem_type.lower()
        X_arr = np.array(X) if hasattr(X, "values") else X
        y_arr = np.array(y) if hasattr(y, "values") else y
        
        # If dataset has categorical columns in X, dummy doesn't even need X features, only shape
        X_dummy = np.zeros((len(y_arr), 1))
        X_tr, X_te, y_tr, y_te = train_test_split(X_dummy, y_arr, test_size=0.2, random_state=42)
        return cls.train_and_evaluate_baseline(X_tr, y_tr, X_te, y_te, is_classification)

    @classmethod
    def calculate_improvement(
        cls,
        baseline_score: float,
        model_score: float,
        metric_name: str,
        higher_is_better: bool = True
    ) -> float:
        """
        Calculate relative percentage improvement over baseline:
        E.g. F1 from 0.60 to 0.90 -> +50.0%
        RMSE from 100 to 60 -> +40.0% (reduction in error)
        """
        if baseline_score == 0.0:
            return 100.0 if model_score > 0 else 0.0

        if higher_is_better:
            delta = ((model_score - baseline_score) / abs(baseline_score)) * 100.0
        else:
            # Lower is better (RMSE, MAE)
            delta = ((baseline_score - model_score) / abs(baseline_score)) * 100.0

        return round(float(delta), 2)
