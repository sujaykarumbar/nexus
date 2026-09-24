"""
Tests for NEXUS Anomaly Detection Engine.
Validates Statistical (Z-score, MAD, IQR), ML (Isolation Forest, LOF),
Residuals, Change-Points, Autoencoders, and Severity Classification.
"""

import pytest
import numpy as np
import pandas as pd

from services.anomaly_detection.statistical import StatisticalAnomalyDetector
from services.anomaly_detection.ml_detector import MLAnomalyDetector
from services.anomaly_detection.residual_detector import ResidualAnomalyDetector
from services.anomaly_detection.change_point import ChangePointDetector
from services.anomaly_detection.autoencoder import AutoencoderAnomalyDetector
from services.anomaly_detection.engine import AnomalyDetectionEngine


@pytest.fixture
def synthetic_anomaly_df():
    """Generates synthetic sensor data with known injected point and regime anomalies."""
    np.random.seed(42)
    n = 60
    base_temp = 70.0 + np.random.normal(0, 1.0, n)
    vibration = 0.04 + np.random.normal(0, 0.005, n)

    # Injected point anomaly: normal ~70, spike to 105.0 at row 20
    base_temp[20] = 105.0

    # Injected change point at row 40: baseline shifts from 70 to 82
    base_temp[40:] += 12.0

    df = pd.DataFrame({
        "timestamp": [f"2024-05-01 {i:02d}:00:00" for i in range(n)],
        "temperature": np.round(base_temp, 2),
        "vibration": np.round(vibration, 4)
    })
    return df


def test_known_injected_anomaly_detection():
    """Prompt requirement: Normal values [100, 102, 99, 101, 103], Injected 500 -> Must detect!"""
    series = pd.Series([100.0, 102.0, 99.0, 101.0, 103.0, 500.0, 101.0, 100.0, 102.0])
    
    # 1. Z-Score
    z_res = StatisticalAnomalyDetector.z_score_detect(series, threshold=2.0)
    assert bool(z_res.loc[5, "is_anomaly"]) is True

    # 2. Modified Z-Score (MAD)
    mad_res = StatisticalAnomalyDetector.modified_z_score_detect(series, threshold=3.5)
    assert bool(mad_res.loc[5, "is_anomaly"]) is True

    # 3. IQR
    iqr_res = StatisticalAnomalyDetector.iqr_detect(series, factor=1.5)
    assert bool(iqr_res.loc[5, "is_anomaly"]) is True


def test_isolation_forest_detect(synthetic_anomaly_df):
    detector = MLAnomalyDetector(contamination=0.05)
    iso_res = detector.isolation_forest_detect(synthetic_anomaly_df, ["temperature", "vibration"])

    # Row 20 should be flagged
    assert bool(iso_res.loc[20, "is_anomaly"]) is True
    assert iso_res.loc[20, "anomaly_score"] > 0.5


def test_residual_anomaly_detector(synthetic_anomaly_df):
    detector = ResidualAnomalyDetector(threshold_std=2.8)
    res = detector.detect_residuals(synthetic_anomaly_df["temperature"])

    assert bool(res.loc[20, "is_anomaly"]) is True
    assert abs(res.loc[20, "deviation"]) > 10.0


def test_change_point_detector(synthetic_anomaly_df):
    detector = ChangePointDetector(min_segment_length=7, significance_threshold=2.5)
    change_points = detector.detect_change_points(
        series=synthetic_anomaly_df["temperature"],
        timestamps=synthetic_anomaly_df["timestamp"]
    )

    assert len(change_points) > 0
    # Step change injected at row 40
    detected_indices = [cp["index"] for cp in change_points]
    assert any(35 <= idx <= 45 for idx in detected_indices)
    assert any(cp["regime_type"] in ["UPWARD_SHIFT", "DOWNWARD_SHIFT"] for cp in change_points)


def test_autoencoder_anomaly_detector(synthetic_anomaly_df):
    ae = AutoencoderAnomalyDetector(latent_dim=2, epochs=25, lr=0.01)
    res = ae.fit_detect(synthetic_anomaly_df, ["temperature", "vibration"], percentile_threshold=95.0)

    assert "reconstruction_error" in res.columns
    # Extreme spike should have highest reconstruction error
    assert res.loc[20, "reconstruction_error"] > res["reconstruction_error"].median()


def test_unified_anomaly_engine_consensus_and_severity(synthetic_anomaly_df):
    engine = AnomalyDetectionEngine()
    results = engine.detect(
        df=synthetic_anomaly_df,
        time_col="timestamp",
        target_col="temperature",
        include_deep_learning=True
    )

    summary = results["dataset_summary"]
    assert summary["total_anomalies"] > 0
    assert summary["critical_count"] >= 1

    anomalies = results["anomalies"]
    row_20_anom = next((a for a in anomalies if a["index"] == 20), None)
    assert row_20_anom is not None
    # 105.0°C exceeds the 95.0°C critical temperature threshold
    assert row_20_anom["severity"] == "CRITICAL"
    assert "Critical temperature threshold breached" in row_20_anom["explanation"]
