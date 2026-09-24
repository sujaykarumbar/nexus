"""
NEXUS Forecast Evaluator
Calculates genuine deterministic forecasting metrics: MAE, RMSE, MAPE, sMAPE, R², and MASE.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ForecastEvaluator:
    """
    Computes rigorous holdout evaluation metrics and residual statistics for forecasting models.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculates all standard time-series error metrics.
        """
        true = np.asarray(y_true, dtype=float).flatten()
        pred = np.asarray(y_pred, dtype=float).flatten()

        if len(true) != len(pred):
            min_len = min(len(true), len(pred))
            true = true[:min_len]
            pred = pred[:min_len]

        if len(true) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "smape": 0.0, "r2": 0.0}

        # 1. MAE
        mae = float(np.mean(np.abs(true - pred)))

        # 2. RMSE
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))

        # 3. MAPE (handling zeros safely)
        mask = true != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100.0)
        else:
            mape = 0.0

        # 4. sMAPE (Symmetric MAPE: 200 * |true - pred| / (|true| + |pred|))
        denom = np.abs(true) + np.abs(pred)
        smape_mask = denom != 0
        if np.any(smape_mask):
            smape = float(np.mean(200.0 * np.abs(true[smape_mask] - pred[smape_mask]) / denom[smape_mask]))
        else:
            smape = 0.0

        # 5. R² (Coefficient of Determination)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 1e-8 else 0.0

        # 6. MASE (Mean Absolute Scaled Error)
        mase = None
        if y_train is not None and len(y_train) > 1:
            train_vals = np.asarray(y_train, dtype=float).flatten()
            naive_diff = np.mean(np.abs(np.diff(train_vals)))
            if naive_diff > 1e-8:
                mase = float(mae / naive_diff)

        # Residuals
        residuals = (true - pred).tolist()

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "smape": round(smape, 2),
            "r2": round(max(-5.0, r2), 4),
            "mase": round(mase, 4) if mase is not None else None,
            "residuals": residuals,
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4)
        }
