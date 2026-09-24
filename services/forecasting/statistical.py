"""
NEXUS Statistical Forecasting Engine
Implements Exponential Smoothing, Holt-Winters, ARIMA, and SARIMA using statsmodels.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class StatisticalForecaster:
    """
    Statistical time-series models with automated fallback and multi-step prediction.
    """

    def __init__(
        self,
        model_type: str = "holt_winters", # 'exponential_smoothing', 'holt_winters', 'arima', 'sarima'
        seasonal_periods: Optional[int] = 7,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        self.model_type = model_type.lower()
        self.seasonal_periods = seasonal_periods
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.history: np.ndarray = np.array([])

    def fit(self, y: pd.Series) -> "StatisticalForecaster":
        """
        Fits the selected statistical model on univariate historical series.
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels package is required for StatisticalForecaster.")

        clean_y = pd.to_numeric(y, errors="coerce").dropna().values
        self.history = clean_y
        n = len(clean_y)

        if n < 5:
            raise ValueError(f"Insufficient history for statistical modeling (length {n}, minimum 5 required).")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model_type in ["exponential_smoothing", "simple_exp"]:
                # Simple exponential smoothing without trend or seasonality
                model = ExponentialSmoothing(clean_y, initialization_method="estimated")
                self.fitted_model = model.fit()

            elif self.model_type in ["holt_winters", "hw"]:
                # Holt-Winters with additive trend and optional additive seasonality if enough points
                sp = self.seasonal_periods or 7
                if n >= 2 * sp and sp > 1:
                    model = ExponentialSmoothing(
                        clean_y,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=sp,
                        initialization_method="estimated"
                    )
                elif n >= 4:
                    model = ExponentialSmoothing(
                        clean_y,
                        trend="add",
                        seasonal=None,
                        initialization_method="estimated"
                    )
                else:
                    model = ExponentialSmoothing(clean_y, initialization_method="estimated")
                self.fitted_model = model.fit()

            elif self.model_type == "arima":
                # Standard ARIMA(p,d,q)
                p, d, q = self.order
                # Ensure d is at most 1 for short series
                d = min(d, 1)
                model = ARIMA(clean_y, order=(p, d, q))
                self.fitted_model = model.fit()

            elif self.model_type == "sarima":
                # Seasonal ARIMA
                sp = self.seasonal_periods or 7
                s_order = self.seasonal_order or (1, 0, 0, sp) if (n >= 2 * sp and sp > 1) else (0, 0, 0, 0)
                model = SARIMAX(
                    clean_y,
                    order=self.order,
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = model.fit(disp=False)

            else:
                raise ValueError(f"Unknown statistical model type: {self.model_type}")

        return self

    def predict(self, horizon: int = 7) -> np.ndarray:
        """
        Produces point forecasts for horizon h.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet.")

        try:
            if self.model_type in ["exponential_smoothing", "simple_exp", "holt_winters", "hw"]:
                forecast = self.fitted_model.forecast(horizon)
                return np.asarray(forecast, dtype=float)
            elif self.model_type in ["arima", "sarima"]:
                forecast = self.fitted_model.forecast(steps=horizon)
                return np.asarray(forecast, dtype=float)
            else:
                # Fallback to last value
                return np.full(horizon, float(self.history[-1]))
        except Exception:
            # Resilient fallback to last observed value if optimization diverges
            return np.full(horizon, float(self.history[-1]))
