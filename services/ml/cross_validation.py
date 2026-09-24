from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error, r2_score


class CrossValidationRunner:
    """Robust Cross-Validation Engine for Classification and Regression."""

    @classmethod
    def evaluate_cv(
        cls,
        model,
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Execute n-fold Stratified or Standard Cross Validation and record full fold distribution.
        """
        # Ensure n_splits does not exceed minimum class count
        if is_classification:
            _, counts = np.unique(y, return_counts=True)
            min_count = min(counts)
            splits = max(2, min(n_splits, min_count))
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
            
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted" if len(np.unique(y)) > 2 else "f1"
            }
            primary_key = "test_f1"
        else:
            splits = max(2, min(n_splits, len(y)))
            cv = KFold(n_splits=splits, shuffle=True, random_state=random_state)
            scoring = {
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            }
            primary_key = "test_rmse"

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        fold_scores = {}
        summary = {}

        for score_name, scores in cv_results.items():
            if score_name.startswith("test_"):
                metric_name = score_name.replace("test_", "")
                # If negative RMSE, flip sign back to positive
                clean_scores = [-s if "neg_" in score_name else s for s in scores]
                fold_scores[metric_name] = [round(float(s), 4) for s in clean_scores]
                mean_val = float(np.mean(clean_scores))
                std_val = float(np.std(clean_scores))
                summary[metric_name] = {
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "formatted": f"{round(mean_val, 3)} ± {round(std_val, 3)}"
                }

        return {
            "n_splits": splits,
            "fold_scores": fold_scores,
            "summary": summary
        }
