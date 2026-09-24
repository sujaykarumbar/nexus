"""
Tests for NEXUS Forecasting Models & Tournament Engine.
Validates Baselines, Holt-Winters, ARIMA, ML recursive forecasting,
evaluation metrics, prediction intervals, and the ForecastingEngine tournament.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.forecasting.baselines import BaselineForecaster
from services.forecasting.statistical import StatisticalForecaster
from services.forecasting.ml_forecast import MLForecaster
from services.forecasting.evaluator import ForecastEvaluator
from services.forecasting.uncertainty import UncertaintyEstimator
from services.forecasting.forecaster import ForecastingEngine


@pytest.fixture
def sales_series_df():
    """Generates 45 days of sales data with trend and weekly seasonality."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(45)]
    t = np.arange(45)
    sales = 200.0 + 3.0 * t + 25.0 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2.0, 45)
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "sales": np.round(sales, 2),
        "marketing": np.round(50.0 + t * 0.8, 2)
    })


def test_baseline_forecaster(sales_series_df):
    series = sales_series_df["sales"]
    
    # 1. Naive
    naive = BaselineForecaster(method="naive").fit(series)
    preds = naive.predict(horizon=5)
    assert len(preds) == 5
    assert np.all(preds == series.iloc[-1])

    # 2. Moving Average
    ma = BaselineForecaster(method="moving_average", window_size=7).fit(series)
    ma_preds = ma.predict(horizon=5)
    assert len(ma_preds) == 5
    assert np.isclose(ma_preds[0], np.mean(series.iloc[-7:]), atol=1e-2)


def test_statistical_forecaster(sales_series_df):
    series = sales_series_df["sales"]

    # Holt-Winters
    hw = StatisticalForecaster(model_type="holt_winters", seasonal_periods=7).fit(series)
    hw_preds = hw.predict(horizon=7)
    assert len(hw_preds) == 7
    assert not np.any(np.isnan(hw_preds))

    # ARIMA
    arima = StatisticalForecaster(model_type="arima", order=(1, 1, 1)).fit(series)
    arima_preds = arima.predict(horizon=7)
    assert len(arima_preds) == 7
    assert not np.any(np.isnan(arima_preds))


def test_ml_forecaster_recursive(sales_series_df):
    ml_f = MLForecaster(model_name="random_forest")
    ml_f.fit(sales_series_df, time_col="date", target_col="sales")

    preds = ml_f.predict_recursive(horizon=7)
    assert len(preds) == 7
    assert not np.any(np.isnan(preds))

    importances = ml_f.get_feature_importances()
    assert len(importances) > 0


def test_forecast_evaluator():
    y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    y_pred = np.array([98.0, 112.0, 119.0, 133.0, 138.0])

    metrics = ForecastEvaluator.evaluate(y_true, y_pred)
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] > 0.90
    assert metrics["mape"] < 5.0
    assert len(metrics["residuals"]) == 5


def test_uncertainty_estimator():
    point_forecasts = np.array([150.0, 155.0, 160.0, 165.0, 170.0])
    residuals = [2.1, -1.8, 0.5, 3.2, -2.4]

    intervals = UncertaintyEstimator.estimate_intervals(
        point_forecasts=point_forecasts,
        residuals=residuals
    )

    assert len(intervals) == 5
    for inter in intervals:
        assert inter["lower_95"] < inter["lower_80"] < inter["prediction"] < inter["upper_80"] < inter["upper_95"]


def test_forecasting_engine_tournament(sales_series_df):
    engine = ForecastingEngine()
    result = engine.train_and_evaluate(
        df=sales_series_df,
        time_col="date",
        target_col="sales",
        horizon=5,
        candidate_models=["naive", "holt_winters", "random_forest"]
    )

    assert "model_id" in result
    assert result["best_model"] in ["naive", "holt_winters", "random_forest"]
    assert len(result["leaderboard"]) >= 3
    assert len(result["future_forecast"]) == 5
    assert "lower_95" in result["future_forecast"][0]
    assert "upper_95" in result["future_forecast"][0]
