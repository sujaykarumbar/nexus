from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC


class ClassificationModelFactory:
    """Library of Production Classification Algorithms with default configurations."""

    @classmethod
    def normalize_algorithm(cls, name: str) -> str:
        s = name.strip().lower().replace(" ", "_").replace("-", "")
        aliases = {
            "logisticregression": "logistic_regression",
            "randomforest": "random_forest",
            "randomforestclassifier": "random_forest",
            "decisiontree": "decision_tree",
            "decisiontreeclassifier": "decision_tree",
            "gradientboosting": "gradient_boosting",
            "gradientboostingclassifier": "gradient_boosting",
            "histgradientboosting": "hist_gradient_boosting",
            "histgradientboostingclassifier": "hist_gradient_boosting",
            "svc": "svm",
            "supportvectormachine": "svm",
        }
        return aliases.get(s.replace("_", ""), s)

    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        return [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
            "decision_tree",
            "svm"
        ]

    @classmethod
    def create_model(
        cls, 
        algorithm: str, 
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """Create a classifier instance with configured or default hyperparameters."""
        algo = cls.normalize_algorithm(algorithm)
        params = dict(hyperparameters or {})

        if algo == "logistic_regression":
            default_params = {
                "max_iter": 1000,
                "random_state": random_state,
                "class_weight": "balanced"
            }
            default_params.update(params)
            return LogisticRegression(**default_params)

        elif algo == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": random_state,
                "n_jobs": -1
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)

        elif algo == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": random_state
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)

        elif algo == "hist_gradient_boosting":
            default_params = {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": random_state
            }
            default_params.update(params)
            return HistGradientBoostingClassifier(**default_params)

        elif algo == "decision_tree":
            default_params = {
                "max_depth": 6,
                "min_samples_split": 2,
                "random_state": random_state
            }
            default_params.update(params)
            return DecisionTreeClassifier(**default_params)

        elif algo == "svm":
            default_params = {
                "probability": True,
                "random_state": random_state
            }
            default_params.update(params)
            return SVC(**default_params)

        else:
            raise ValueError(f"Unsupported classification algorithm '{algorithm}'. Choose from {cls.get_supported_algorithms()}.")

    @classmethod
    def get_default_candidates(cls, n_rows: int) -> List[str]:
        """Intelligently select models based on sample size constraints."""
        if n_rows < 10000:
            return ["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting"]
        else:
            # For larger datasets, prefer fast histogram gradient boosting and logistic regression
            return ["logistic_regression", "hist_gradient_boosting", "random_forest"]
