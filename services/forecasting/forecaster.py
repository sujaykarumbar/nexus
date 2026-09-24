"""
NEXUS Forecasting Engine Orchestrator
Coordinates temporal validation, multi-model benchmark leaderboard,
best model selection, multi-step future forecasting, and uncertainty estimation.
"""

from typing import Dict, Any, List, Optional, Callable
import os
import uuid
import time
import joblib
import numpy as np
import pandas as pd

from .time_series_detector import TimeSeriesDetector
from .feature_engineer import TimeSeriesFeatureEngineer
from .baselines import BaselineForecaster
from .statistical import StatisticalForecaster
from .ml_forecast import MLForecaster
from .deep_learning import LSTMForecaster, GRUForecaster, DeepLearningTrainer
from .evaluator import ForecastEvaluator
from .uncertainty import UncertaintyEstimator


class ForecastingEngine:
    """
    End-to-end forecasting pipeline executing multi-model benchmarks,
    chronological holdout evaluation, best model promotion, and uncertainty estimation.
    """

    def __init__(self, artifact_dir: str = "data/models"):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.detector = TimeSeriesDetector()
        self.evaluator = ForecastEvaluator()
        self.uncertainty_estimator = UncertaintyEstimator()

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        horizon: int = 7,
        candidate_models: Optional[List[str]] = None,
        covariates: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive forecasting tournament across baseline, statistical, ML, and DL models.
        """
        start_time = time.time()
        candidates = [m.lower() for m in (candidate_models or ["naive", "holt_winters", "arima", "random_forest", "gradient_boosting", "lstm", "gru"])]

        # 1. Clean and chronologically order data
        df_clean = df.copy()
        df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)

        n_rows = len(df_clean)
        if n_rows < horizon + 4:
            raise ValueError(f"Dataset has {n_rows} rows; at least {horizon + 4} required for horizon {horizon}.")

        # 2. Temporal intelligence profile
        profile = self.detector.analyze_temporal_profile(df_clean, time_col=time_col, target_col=target_col)
        seasonal_period = profile.get("seasonality_period") or 7

        # 3. Chronological Train / Holdout Validation Split (Holdout = horizon)
        train_df = df_clean.iloc[:-horizon].copy()
        val_df = df_clean.iloc[-horizon:].copy()

        y_train = pd.to_numeric(train_df[target_col], errors="coerce").values
        y_val = pd.to_numeric(val_df[target_col], errors="coerce").values
        time_train = train_df[time_col]
        time_val = val_df[time_col]

        leaderboard: List[Dict[str, Any]] = []
        trained_artifacts: Dict[str, Any] = {}
        total_candidates = len(candidates)

        # Iterate over candidate models
        for idx, model_name in enumerate(candidates):
            if progress_callback:
                progress_callback({
                    "status": "TRAINING",
                    "current_model": model_name,
                    "progress": int((idx / total_candidates) * 90)
                })

            model_start = time.time()
            val_preds = None
            future_forecaster = None
            training_loss_curves = None

            try:
                # --- BASELINES ---
                if model_name == "naive":
                    forecaster = BaselineForecaster(method="naive").fit(train_df[target_col])
                    val_preds = forecaster.predict(horizon=horizon)
                    future_forecaster = BaselineForecaster(method="naive").fit(df_clean[target_col])

                elif model_name == "seasonal_naive":
                    forecaster = BaselineForecaster(method="seasonal_naive", seasonal_period=seasonal_period).fit(train_df[target_col])
                    val_preds = forecaster.predict(horizon=horizon)
                    future_forecaster = BaselineForecaster(method="seasonal_naive", seasonal_period=seasonal_period).fit(df_clean[target_col])

                elif model_name == "moving_average":
                    forecaster = BaselineForecaster(method="moving_average", window_size=seasonal_period).fit(train_df[target_col])
                    val_preds = forecaster.predict(horizon=horizon)
                    future_forecaster = BaselineForecaster(method="moving_average", window_size=seasonal_period).fit(df_clean[target_col])

                # --- STATISTICAL ---
                elif model_name in ["holt_winters", "exponential_smoothing"]:
                    forecaster = StatisticalForecaster(model_type="holt_winters", seasonal_periods=seasonal_period).fit(train_df[target_col])
                    val_preds = forecaster.predict(horizon=horizon)
                    future_forecaster = StatisticalForecaster(model_type="holt_winters", seasonal_periods=seasonal_period).fit(df_clean[target_col])

                elif model_name in ["arima", "sarima"]:
                    mtype = "sarima" if model_name == "sarima" else "arima"
                    forecaster = StatisticalForecaster(model_type=mtype, seasonal_periods=seasonal_period).fit(train_df[target_col])
                    val_preds = forecaster.predict(horizon=horizon)
                    future_forecaster = StatisticalForecaster(model_type=mtype, seasonal_periods=seasonal_period).fit(df_clean[target_col])

                # --- MACHINE LEARNING ---
                elif model_name in ["random_forest", "gradient_boosting", "hist_gradient_boosting"]:
                    ml_f = MLForecaster(model_name=model_name)
                    ml_f.fit(train_df, time_col=time_col, target_col=target_col, covariates=covariates)
                    val_preds = ml_f.predict_recursive(horizon=horizon)
                    # Future model fitted on full dataset
                    future_f = MLForecaster(model_name=model_name)
                    future_f.fit(df_clean, time_col=time_col, target_col=target_col, covariates=covariates)
                    future_forecaster = future_f

                # --- DEEP LEARNING (PyTorch LSTM / GRU) ---
                elif model_name in ["lstm", "gru"]:
                    seq_len = min(14, max(3, (len(train_df) - horizon) // 2))
                    dl_trainer = DeepLearningTrainer(
                        architecture=model_name,
                        seq_length=seq_len,
                        horizon=horizon,
                        hidden_dim=32,
                        num_layers=2,
                        max_epochs=35,
                        patience=5
                    )
                    dl_info = dl_trainer.fit(train_df[target_col])
                    val_preds = dl_trainer.predict()
                    training_loss_curves = {
                        "train_loss": dl_info["train_loss_history"],
                        "val_loss": dl_info["val_loss_history"]
                    }

                    # Full future trainer
                    future_dl = DeepLearningTrainer(
                        architecture=model_name,
                        seq_length=seq_len,
                        horizon=horizon,
                        hidden_dim=32,
                        num_layers=2,
                        max_epochs=35,
                        patience=5
                    )
                    future_dl.fit(df_clean[target_col])
                    future_forecaster = future_dl

            except Exception as e:
                # Log model failure gracefully in tournament
                leaderboard.append({
                    "model_name": model_name,
                    "status": "FAILED",
                    "error": str(e),
                    "mae": 999999.0,
                    "rmse": 999999.0
                })
                continue

            model_duration = time.time() - model_start

            # Calculate genuine evaluation metrics on unseen holdout horizon
            if val_preds is not None:
                metrics = self.evaluator.evaluate(y_true=y_val, y_pred=val_preds, y_train=y_train)
                leaderboard.append({
                    "model_name": model_name,
                    "status": "SUCCESS",
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "smape": metrics["smape"],
                    "r2": metrics["r2"],
                    "mase": metrics["mase"],
                    "training_duration_seconds": round(model_duration, 2),
                    "loss_curves": training_loss_curves
                })
                trained_artifacts[model_name] = {
                    "forecaster": future_forecaster,
                    "val_preds": [round(float(p), 2) for p in val_preds],
                    "residuals": metrics["residuals"]
                }

        # Filter successful models and sort leaderboard by MAE ascending
        successful = [m for m in leaderboard if m["status"] == "SUCCESS"]
        if not successful:
            raise RuntimeError("All forecasting candidate models failed to converge.")

        successful.sort(key=lambda x: x["mae"])
        best_model_name = successful[0]["model_name"]
        best_metrics = successful[0]
        best_artifact = trained_artifacts[best_model_name]

        # 4. Generate Future Forecast from Best Model
        best_forecaster = best_artifact["forecaster"]
        if hasattr(best_forecaster, "predict_recursive"):
            future_point_forecasts = best_forecaster.predict_recursive(horizon=horizon)
        else:
            future_point_forecasts = best_forecaster.predict(horizon=horizon) if hasattr(best_forecaster, "predict") else np.full(horizon, float(y_train[-1]))

        # Calculate Prediction Intervals
        intervals = self.uncertainty_estimator.estimate_intervals(
            point_forecasts=future_point_forecasts,
            residuals=best_artifact["residuals"]
        )

        # Generate future timestamps
        time_series = df_clean[time_col]
        if len(time_series) > 1:
            step_delta = time_series.iloc[-1] - time_series.iloc[-2]
        else:
            step_delta = pd.Timedelta(days=1)

        future_timestamps = []
        curr_t = time_series.iloc[-1]
        for h_i in range(horizon):
            curr_t += step_delta
            future_timestamps.append(curr_t.strftime("%Y-%m-%d %H:%M:%S") if curr_t.hour != 0 else curr_t.strftime("%Y-%m-%d"))

        for idx, inter in enumerate(intervals):
            inter["timestamp"] = future_timestamps[idx]

        # 5. Persist Model Artifact
        model_id = str(uuid.uuid4())
        model_filepath = os.path.join(self.artifact_dir, f"forecast_{model_id}.joblib")
        joblib.dump({
            "model_id": model_id,
            "best_model_name": best_model_name,
            "forecaster": best_forecaster,
            "time_col": time_col,
            "target_col": target_col,
            "horizon": horizon,
            "metrics": best_metrics,
            "temporal_profile": profile
        }, model_filepath)

        total_duration = time.time() - start_time

        # Feature importances if best model is ML
        feature_importances = {}
        if hasattr(best_forecaster, "get_feature_importances"):
            feature_importances = best_forecaster.get_feature_importances()

        # Historical series points for chart overlay
        recent_history = [
            {"timestamp": str(t), "actual": round(float(v), 2)}
            for t, v in zip(df_clean[time_col].tail(30), df_clean[target_col].tail(30))
        ]

        # Holdout validation comparison (actual vs predicted)
        validation_comparison = [
            {"timestamp": str(t), "actual": round(float(act), 2), "predicted": round(float(pred), 2)}
            for t, act, pred in zip(time_val, y_val, best_artifact["val_preds"])
        ]

        if progress_callback:
            progress_callback({"status": "COMPLETED", "progress": 100})

        return {
            "model_id": model_id,
            "artifact_path": model_filepath,
            "time_column": time_col,
            "target_column": target_col,
            "horizon": horizon,
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "leaderboard": leaderboard,
            "temporal_profile": profile,
            "future_forecast": intervals,
            "recent_history": recent_history,
            "validation_comparison": validation_comparison,
            "feature_importances": feature_importances,
            "training_duration_seconds": round(total_duration, 2)
        }
