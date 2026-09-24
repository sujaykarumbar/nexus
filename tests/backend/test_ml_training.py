import os
import pandas as pd
import numpy as np
import pytest
from services.ml.trainer import AutoMLTrainer
from services.ml.preprocessor import MLPreprocessor
from services.ml.baseline import BaselineModelRunner
from services.ml.evaluator import ModelEvaluator


@pytest.fixture
def sample_classification_df():
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "age": np.random.randint(20, 65, size=n),
        "income": np.random.uniform(20000, 120000, size=n),
        "contract": np.random.choice(["month-to-month", "one-year", "two-year"], size=n),
        "support_calls": np.random.randint(0, 10, size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
    })


@pytest.fixture
def sample_regression_df():
    np.random.seed(42)
    n = 120
    X1 = np.random.uniform(10, 50, size=n)
    X2 = np.random.randint(1, 5, size=n)
    y = 3.5 * X1 + 10.2 * X2 + np.random.normal(0, 2.0, size=n)
    return pd.DataFrame({
        "sqft": X1,
        "bedrooms": X2,
        "neighborhood": np.random.choice(["Downtown", "Suburbs", "Uptown"], size=n),
        "price": y
    })


def test_baseline_runner_classification(sample_classification_df):
    runner = BaselineModelRunner()
    X = sample_classification_df.drop(columns=["churn"])
    y = sample_classification_df["churn"]
    
    baseline = runner.run_baseline("binary_classification", X, y)
    assert baseline["algorithm"] == "DummyClassifier"
    assert "metrics" in baseline
    assert "accuracy" in baseline["metrics"]
    assert "f1" in baseline["metrics"]


def test_baseline_runner_regression(sample_regression_df):
    runner = BaselineModelRunner()
    X = sample_regression_df.drop(columns=["price"])
    y = sample_regression_df["price"]
    
    baseline = runner.run_baseline("regression", X, y)
    assert baseline["algorithm"] == "DummyRegressor"
    assert "metrics" in baseline
    assert "rmse" in baseline["metrics"]
    assert "r2" in baseline["metrics"]


def test_automl_trainer_classification(sample_classification_df):
    trainer = AutoMLTrainer()
    # Fast test with 2 candidate algorithms, 3 folds, 3 optuna trials
    result = trainer.train(
        df=sample_classification_df,
        target_column="churn",
        candidate_algorithms=["LogisticRegression", "RandomForest"],
        cv_splits=3,
        optimize_hyperparameters=True,
        optuna_trials=3
    )

    assert result["status"] == "COMPLETED"
    assert result["problem_type"] == "binary_classification"
    assert len(result["leaderboard"]) == 2
    assert result["best_model"] is not None
    
    # Verify leaderboard has real metrics and positive or computed delta vs baseline
    best = result["best_model"]
    assert "primary_metric_value" in best
    assert "delta_improvement_pct" in best
    assert "cv_summary" in best
    assert "feature_importance" in best
    assert len(best["feature_importance"]) > 0

    # Verify model artifact exists on disk
    artifact_path = best["artifact_path"]
    assert os.path.exists(artifact_path)
    assert artifact_path.endswith(".joblib")


def test_automl_trainer_regression(sample_regression_df):
    trainer = AutoMLTrainer()
    result = trainer.train(
        df=sample_regression_df,
        target_column="price",
        candidate_algorithms=["LinearRegression", "RandomForest"],
        cv_splits=3,
        optimize_hyperparameters=False
    )

    assert result["status"] == "COMPLETED"
    assert result["problem_type"] == "regression"
    assert len(result["leaderboard"]) == 2
    best = result["best_model"]
    assert best["primary_metric_name"] == "rmse"
    assert best["primary_metric_value"] > 0
    # Artifact created
    assert os.path.exists(best["artifact_path"])
