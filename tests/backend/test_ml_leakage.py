import pandas as pd
import numpy as np
import pytest
from services.ml.leakage_detector import LeakageDetector


def test_leakage_detector_catches_target_duplicate():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "churn": [0, 1, 0, 1, 0],
        "churn_label": [0, 1, 0, 1, 0],  # Duplicate name & value
    })

    audit = LeakageDetector.audit_leakage(df, "churn")
    assert audit["has_leakage_risk"] is True
    assert "churn_label" in audit["recommended_drop_columns"]
    warning_types = [w["risk_type"] for w in audit["warnings"]]
    assert any(t in warning_types for t in ["TARGET_DERIVED_FEATURE", "TARGET_NAME_DERIVATIVE", "PERFECT_TARGET_CORRELATION"])


def test_leakage_detector_catches_id_column():
    df = pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(100)],
        "tenure": np.random.randint(1, 10, size=100),
        "target": np.random.choice([0, 1], size=100),
    })

    audit = LeakageDetector.audit_leakage(df, "target")
    assert "customer_id" in audit["recommended_drop_columns"]
    col_warnings = [w for w in audit["warnings"] if w["column"] == "customer_id"]
    assert len(col_warnings) > 0
    assert col_warnings[0]["risk_type"] == "HIGH_CARDINALITY_IDENTIFIER"


def test_leakage_detector_catches_near_perfect_correlation():
    np.random.seed(42)
    clean_target = np.random.choice([0, 1], size=200)
    # create feature that correlates 0.99 with target
    corrupt_noise = np.random.choice([0, 1], size=200, p=[0.99, 0.01])
    leaked_feature = (clean_target + corrupt_noise) % 2

    # Or directly continuous highly correlated
    cont_target = np.random.uniform(10, 100, size=200)
    leaked_cont = cont_target + np.random.normal(0, 0.01, size=200)

    df = pd.DataFrame({
        "safe_feature": np.random.randn(200),
        "leaked_var": leaked_cont,
        "target_reg": cont_target
    })

    audit = LeakageDetector.audit_leakage(df, "target_reg")
    assert audit["has_leakage_risk"] is True
    assert "leaked_var" in audit["recommended_drop_columns"]
    leaked_warn = [w for w in audit["warnings"] if w["column"] == "leaked_var"][0]
    assert leaked_warn["risk_type"] == "PERFECT_TARGET_CORRELATION"
    assert leaked_warn["severity"] == "CRITICAL"


def test_leakage_detector_clean_dataset():
    df = pd.DataFrame({
        "feature_1": np.random.randn(50),
        "feature_2": np.random.choice(["A", "B", "C"], size=50),
        "target": np.random.choice([0, 1], size=50)
    })

    audit = LeakageDetector.audit_leakage(df, "target")
    assert audit["has_critical_risk"] is False
    assert len(audit["recommended_drop_columns"]) == 0
