"""
NEXUS Forecast Uncertainty Estimator
Calculates calibrated prediction intervals (80% and 95% confidence bounds)
using historical residual variance scaling across forecast horizons.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class UncertaintyEstimator:
    """
    Computes statistical prediction intervals for multi-step forecasts.
    Expands bounds dynamically with horizon uncertainty sqrt(h).
    """

    Z_SCORE_80 = 1.282 # 80% confidence (two-tailed)
    Z_SCORE_95 = 1.960 # 95% confidence (two-tailed)

    @classmethod
    def estimate_intervals(
        cls,
        point_forecasts: np.ndarray,
        residuals: Optional[List[float]] = None,
        residual_std: Optional[float] = None,
        method: str = "residual_variance_scaling"
    ) -> List[Dict[str, Any]]:
        """
        Generates 80% and 95% prediction intervals for each step h in point_forecasts.
        """
        pts = np.asarray(point_forecasts, dtype=float).flatten()
        horizon = len(pts)

        if residual_std is not None and residual_std > 0:
            sigma = float(residual_std)
        elif residuals and len(residuals) > 0:
            sigma = float(np.std(residuals))
            if sigma < 1e-6:
                sigma = float(np.mean(np.abs(pts)) * 0.05 + 1.0)
        else:
            # Heuristic default 5% of mean prediction
            sigma = float(np.mean(np.abs(pts)) * 0.05 + 1.0) if len(pts) > 0 else 1.0

        intervals = []
        for h_idx in range(horizon):
            h_step = h_idx + 1
            # Uncertainty expands with the square root of forecast lead time
            lead_factor = np.sqrt(h_step)
            step_sigma = sigma * lead_factor

            val = float(pts[h_idx])
            lower_95 = val - (cls.Z_SCORE_95 * step_sigma)
            upper_95 = val + (cls.Z_SCORE_95 * step_sigma)
            lower_80 = val - (cls.Z_SCORE_80 * step_sigma)
            upper_80 = val + (cls.Z_SCORE_80 * step_sigma)

            intervals.append({
                "step": h_step,
                "prediction": round(val, 2),
                "lower_80": round(lower_80, 2),
                "upper_80": round(upper_80, 2),
                "lower_95": round(lower_95, 2),
                "upper_95": round(upper_95, 2),
                "uncertainty_sigma": round(step_sigma, 3),
                "estimation_method": method,
                "disclaimer": "Statistical prediction intervals represent model uncertainty under Gaussian residual assumptions, not guaranteed bounds."
            })

        return intervals
