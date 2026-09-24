"""
Tests for NEXUS Time-Series Intelligence & Feature Engineering.
Validates temporal detection, frequency inference, anti-leakage lags, and rolling features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.forecasting.time_series_detector import TimeSeriesDetector
from services.forecasting.feature_engineer import TimeSeriesFeatureEngineer


@pytest.fixture
def sample_temporal_df():
    """Generates a synthetic 40-day time series with trend and weekly seasonality."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    t = np.arange(40)
    sales = 100.0 + 2.5 * t + 15.0 * np.sin(2 * np.pi * t / 7)
    df = pd.DataFrame({
        "timestamp": [d.strftime("%Y-%m-%d") for d in dates],
        "sales": sales,
        "visitors": np.round(sales * 3.5),
        "promo": [1 if i % 7 == 0 else 0 for i in range(40)]
    })
    return df


def test_detect_candidate_time_columns(sample_temporal_df):
    detector = TimeSeriesDetector()
    candidates = detector.detect_candidate_time_columns(sample_temporal_df)

    assert len(candidates) > 0
    top_candidate = candidates[0]
    assert top_candidate["column_name"] == "timestamp"
    assert top_candidate["confidence_score"] >= 0.7
    assert top_candidate["is_parsed"] is True


def test_temporal_profile_analysis(sample_temporal_df):
    detector = TimeSeriesDetector()
    profile = detector.analyze_temporal_profile(
        df=sample_temporal_df,
        time_col="timestamp",
        target_col="sales"
    )

    assert profile["total_observations"] == 40
    assert profile["valid_timestamps"] == 40
    assert profile["is_chronological"] is True
    assert profile["duplicate_timestamps"] == 0
    assert profile["inferred_frequency"] in ["1D", "D"]
    assert profile["trend"] == "UPWARD"
    assert profile["seasonality_period"] is not None
    assert profile["autocorrelation_lag1"] is not None


def test_feature_engineering_no_temporal_leakage(sample_temporal_df):
    engineer = TimeSeriesFeatureEngineer(lags=[1, 2, 7], rolling_windows=[7])
    df_eng = engineer.transform(sample_temporal_df, time_col="timestamp", target_col="sales")

    feature_names = engineer.get_feature_names(df_eng, time_col="timestamp", target_col="sales")

    # Verify expected feature columns exist
    assert "lag_1" in feature_names
    assert "lag_2" in feature_names
    assert "lag_7" in feature_names
    assert "rolling_mean_7" in feature_names
    assert "rolling_std_7" in feature_names
    assert "is_weekend" in feature_names
    assert "dayofweek" in feature_names

    # Anti-Leakage Verification:
    # Row 10's lag_1 must strictly equal Row 9's sales
    row_10_lag1 = df_eng.loc[10, "lag_1"]
    row_9_sales = sample_temporal_df.loc[9, "sales"]
    assert np.isclose(row_10_lag1, row_9_sales)

    # Row 10's rolling_mean_7 must equal average of rows 3 to 9 (strictly past 7 values, excluding row 10)
    past_7_sales = sample_temporal_df.loc[3:9, "sales"].values
    expected_rolling_mean = np.mean(past_7_sales)
    assert np.isclose(df_eng.loc[10, "rolling_mean_7"], expected_rolling_mean, atol=1e-3)
