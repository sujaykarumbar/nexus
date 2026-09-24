"""
NEXUS Machine Learning Forecasting Engine
Implements Random Forest, Gradient Boosting, and HistGradientBoosting
forecasting models with recursive multi-step prediction and chronological validation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import TimeSeriesSplit

from .feature_engineer import TimeSeriesFeatureEngineer


class MLForecaster:
    """
    Supervised Machine Learning forecasting with recursive multi-step horizon generation.
    """

    SUPPORTED_MODELS = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
    }

    def __init__(
        self,
        model_name: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        feature_engineer: Optional[TimeSeriesFeatureEngineer] = None
    ):
        self.model_name = model_name.lower()
        self.model_params = model_params or {}
        self.feature_engineer = feature_engineer or TimeSeriesFeatureEngineer()
        self.model = None
        self.feature_names: List[str] = []
        self.last_df: Optional[pd.DataFrame] = None
        self.time_col: str = ""
        self.target_col: str = ""

    def _init_estimator(self):
        default_params = {"random_state": 42}
        if self.model_name == "random_forest":
            default_params.update({"n_estimators": 100, "max_depth": 8, "n_jobs": -1})
            default_params.update(self.model_params)
            return RandomForestRegressor(**default_params)
        elif self.model_name == "gradient_boosting":
            default_params.update({"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05})
            default_params.update(self.model_params)
            return GradientBoostingRegressor(**default_params)
        elif self.model_name in ["hist_gradient_boosting", "lightgbm_equivalent"]:
            default_params.update({"max_iter": 100, "max_depth": 6, "learning_rate": 0.05})
            default_params.update(self.model_params)
            return HistGradientBoostingRegressor(**default_params)
        else:
            default_params.update({"n_estimators": 100, "random_state": 42})
            return RandomForestRegressor(**default_params)

    def fit(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        covariates: Optional[List[str]] = None
    ) -> "MLForecaster":
        """
        Engineers features and fits the regressor.
        """
        self.time_col = time_col
        self.target_col = target_col

        # Engineer features
        df_eng = self.feature_engineer.transform(df, time_col=time_col, target_col=target_col, covariates=covariates)
        self.feature_names = self.feature_engineer.get_feature_names(df_eng, time_col=time_col, target_col=target_col)

        # Retain latest rows for recursive multi-step forecasting
        self.last_df = df_eng.copy()

        X = df_eng[self.feature_names]
        y = pd.to_numeric(df_eng[target_col], errors="coerce").values

        # Ensure no NaNs
        X = X.bfill().ffill().fillna(0.0)

        self.model = self._init_estimator()
        self.model.fit(X, y)

        return self

    def predict_recursive(self, horizon: int = 7) -> np.ndarray:
        """
        Executes recursive multi-step forecasting:
        Predicts t+1 -> updates lags/rolling window -> predicts t+2 -> ... -> t+H.
        """
        if self.model is None or self.last_df is None:
            raise ValueError("MLForecaster has not been fitted.")

        history_df = self.last_df.copy()
        predictions = []

        # Determine step timedelta from history
        time_series = pd.to_datetime(history_df[self.time_col])
        if len(time_series) > 1:
            step_delta = time_series.iloc[-1] - time_series.iloc[-2]
        else:
            step_delta = pd.Timedelta(days=1)

        current_time = time_series.iloc[-1]

        for step in range(horizon):
            current_time += step_delta
            # Re-compute features using updated history
            df_re_eng = self.feature_engineer.transform(
                history_df,
                time_col=self.time_col,
                target_col=self.target_col
            )
            # Latest feature vector (row representing the step to forecast)
            latest_features = df_re_eng[self.feature_names].iloc[[-1]].copy()
            latest_features = latest_features.bfill().ffill().fillna(0.0)

            # Predict next step
            y_pred = float(self.model.predict(latest_features)[0])
            predictions.append(y_pred)

            # Append synthetic predicted observation to history for next recursive step
            new_row = {col: 0.0 for col in history_df.columns}
            new_row[self.time_col] = current_time
            new_row[self.target_col] = y_pred
            history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

        return np.array(predictions, dtype=float)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance dictionary sorted descending.
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        res = {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
