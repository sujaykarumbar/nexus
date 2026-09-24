import pandas as pd
import numpy as np
from services.quality.quality_engine import DataQualityEngine
from services.quality.scoring import DataQualityScorer


def test_data_quality_scoring_calculation():
    # Perfect dataset
    score_perf = DataQualityScorer.calculate_score(
        total_cells=1000,
        total_nulls=0,
        total_rows=100,
        duplicate_rows=0,
        invalid_cells=0,
        constant_columns_count=0,
        total_columns=10,
        outlier_cells=0
    )
    assert score_perf["overall_score"] == 100.0
    assert score_perf["grade"] == "A"

    # Dataset with 10% nulls and 5% duplicates
    score_deg = DataQualityScorer.calculate_score(
        total_cells=1000,
        total_nulls=100,
        total_rows=100,
        duplicate_rows=5,
        invalid_cells=0,
        constant_columns_count=0,
        total_columns=10,
        outlier_cells=0
    )
    assert score_deg["overall_score"] < 100.0
    assert score_deg["components"]["completeness"] == 90.0


def test_data_quality_engine_issue_detection():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 4],  # duplicate row
        "age": [25, -5, 45, 150, 30],  # 2 invalid age values
        "income": [50000, 55000, 52000, 51000, 10000000],  # 1 extreme outlier
        "constant_col": ["FIXED", "FIXED", "FIXED", "FIXED", "FIXED"],  # constant column
        "nullable_feat": [10.5, np.nan, 12.0, np.nan, 14.5]  # 40% missing
    })

    quality_report = DataQualityEngine.evaluate_quality(df)
    issues = quality_report["issues"]

    assert len(issues["missing_values"]) == 1
    assert issues["missing_values"][0]["percentage"] == 40.0
    assert issues["duplicates"]["count"] == 1
    assert "constant_col" in issues["constant_columns"]
    assert len(issues["outliers"]) >= 1
    assert len(issues["invalid_values"]) >= 1
    assert len(quality_report["recommendations"]) >= 3
