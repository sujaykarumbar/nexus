import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from .models.classification import ClassificationModelFactory
from .models.regression import RegressionModelFactory

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Bayesian Hyperparameter Optimization Engine powered by Optuna."""

    @classmethod
    def optimize_model(
        cls,
        algorithm: str,
        is_classification: bool,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 15,
        timeout_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Run Optuna study across algorithm search space using Cross-Validation.
        Returns best parameters, best validation score, and trial histories.
        """
        if is_classification:
            splits = max(2, min(3, len(y_train) // 5))
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
            scoring = "f1_weighted" if len(np.unique(y_train)) > 2 else "f1"
            direction = "maximize"
        else:
            splits = max(2, min(3, len(y_train) // 5))
            cv = KFold(n_splits=splits, shuffle=True, random_state=42)
            scoring = "neg_root_mean_squared_error"
            direction = "maximize"

        trials_log = []

        def objective(trial: optuna.Trial) -> float:
            start_t = time.time()
            params = {}

            if algorithm in ["random_forest", "decision_tree"]:
                params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
                if algorithm == "random_forest":
                    params["n_estimators"] = trial.suggest_int("n_estimators", 25, 120, step=25)

            elif algorithm in ["gradient_boosting", "hist_gradient_boosting"]:
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                params["max_depth"] = trial.suggest_int("max_depth", 2, 7)
                if algorithm == "gradient_boosting":
                    params["n_estimators"] = trial.suggest_int("n_estimators", 30, 100, step=20)
                else:
                    params["max_iter"] = trial.suggest_int("max_iter", 30, 100, step=20)

            elif algorithm in ["logistic_regression", "ridge", "lasso"]:
                if algorithm == "logistic_regression":
                    params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
                else:
                    params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)

            elif algorithm == "svm":
                params["C"] = trial.suggest_float("C", 0.1, 10.0, log=True)

            if is_classification:
                model = ClassificationModelFactory.create_model(algorithm, params)
            else:
                model = RegressionModelFactory.create_model(algorithm, params)

            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            mean_score = float(np.mean(scores))

            duration = round(time.time() - start_t, 3)
            trials_log.append({
                "trial_number": trial.number,
                "parameters": params,
                "score": round(mean_score, 4),
                "duration_seconds": duration
            })

            return mean_score

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

        best_score = float(study.best_value)
        # If regression negative RMSE, convert back to positive
        if not is_classification:
            best_score = -best_score

        return {
            "algorithm": algorithm,
            "best_params": study.best_params,
            "best_cv_score": round(best_score, 4),
            "total_trials": len(study.trials),
            "trials_summary": trials_log
        }
