from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)


class RegressionModelFactory:
    """Library of Production Regression Algorithms."""

    @classmethod
    def normalize_algorithm(cls, name: str) -> str:
        s = name.strip().lower().replace(" ", "_").replace("-", "")
        aliases = {
            "linearregression": "linear_regression",
            "ridge": "ridge",
            "lasso": "lasso",
            "randomforest": "random_forest",
            "randomforestregressor": "random_forest",
            "decisiontree": "decision_tree",
            "decisiontreeregressor": "decision_tree",
            "gradientboosting": "gradient_boosting",
            "gradientboostingregressor": "gradient_boosting",
            "histgradientboosting": "hist_gradient_boosting",
            "histgradientboostingregressor": "hist_gradient_boosting",
        }
        return aliases.get(s.replace("_", ""), s)

    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        return [
            "linear_regression",
            "ridge",
            "lasso",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
            "decision_tree"
        ]

    @classmethod
    def create_model(
        cls,
        algorithm: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """Create a regressor instance with configured or default hyperparameters."""
        algo = cls.normalize_algorithm(algorithm)
        params = dict(hyperparameters or {})

        if algo == "linear_regression":
            return LinearRegression(**params)

        elif algo == "ridge":
            default_params = {"alpha": 1.0, "random_state": random_state}
            default_params.update(params)
            return Ridge(**default_params)

        elif algo == "lasso":
            default_params = {"alpha": 1.0, "random_state": random_state}
            default_params.update(params)
            return Lasso(**default_params)

        elif algo == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": random_state,
                "n_jobs": -1
            }
            default_params.update(params)
            return RandomForestRegressor(**default_params)

        elif algo == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": random_state
            }
            default_params.update(params)
            return GradientBoostingRegressor(**default_params)

        elif algo == "hist_gradient_boosting":
            default_params = {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": random_state
            }
            default_params.update(params)
            return HistGradientBoostingRegressor(**default_params)

        elif algo == "decision_tree":
            default_params = {
                "max_depth": 6,
                "min_samples_split": 2,
                "random_state": random_state
            }
            default_params.update(params)
            return DecisionTreeRegressor(**default_params)

        else:
            raise ValueError(f"Unsupported regression algorithm '{algorithm}'. Choose from {cls.get_supported_algorithms()}.")

    @classmethod
    def get_default_candidates(cls, n_rows: int) -> List[str]:
        if n_rows < 10000:
            return ["linear_regression", "ridge", "random_forest", "gradient_boosting"]
        else:
            return ["ridge", "hist_gradient_boosting", "random_forest"]
