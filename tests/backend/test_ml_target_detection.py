import pandas as pd
import numpy as np
from services.ml.target_detector import TargetDetector
from services.ml.problem_classifier import ProblemClassifier


def test_target_detector_identifies_candidates():
    data = {
        "customer_id": [f"CUST_{i}" for i in range(100)],
        "age": np.random.randint(18, 70, size=100),
        "monthly_spend": np.random.uniform(10.0, 500.0, size=100),
        "signup_date": pd.date_range("2024-01-01", periods=100),
        "churn": np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)

    detector = TargetDetector()
    candidates = detector.detect_candidates(df)

    assert len(candidates) > 0
    candidate_names = [c["column_name"] for c in candidates]
    # "churn" matches target name patterns and has binary values, so should be top candidate
    assert "churn" in candidate_names
    top_candidate = candidates[0]
    assert top_candidate["column_name"] == "churn"
    assert top_candidate["confidence_score"] > 0.6
    assert top_candidate["recommended_problem_type"] == "binary_classification"
    # customer_id should not be top candidate
    assert candidate_names.index("churn") < candidate_names.index("customer_id") if "customer_id" in candidate_names else True


def test_problem_classifier_binary():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "is_fraud": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    })

    classifier = ProblemClassifier()
    result = classifier.classify(df, "is_fraud")

    assert result["problem_type"] == "binary_classification"
    assert result["num_classes"] == 2
    assert result["is_imbalanced"] is True
    assert result["primary_metric"] in ["f1", "f1_weighted"]
    assert "0" in result["class_distribution"]
    assert "1" in result["class_distribution"]


def test_problem_classifier_multiclass():
    df = pd.DataFrame({
        "feature_a": range(30),
        "tier": ["Bronze"] * 10 + ["Silver"] * 10 + ["Gold"] * 10,
    })

    classifier = ProblemClassifier()
    result = classifier.classify(df, "tier")

    assert result["problem_type"] == "multiclass_classification"
    assert result["num_classes"] == 3
    assert result["primary_metric"] in ["f1_macro", "f1_weighted"]


def test_problem_classifier_regression():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_a": np.random.randn(100),
        "revenue": np.random.uniform(50.0, 10000.0, size=100),
    })

    classifier = ProblemClassifier()
    result = classifier.classify(df, "revenue")

    assert result["problem_type"] == "regression"
    assert result["primary_metric"] == "rmse"
    assert "distribution_stats" in result
    assert result["distribution_stats"]["mean"] > 0
