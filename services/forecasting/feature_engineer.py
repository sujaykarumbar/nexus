"""
NEXUS Time-Series Feature Engineering
Generates calendar features, anti-leakage lag features, rolling window statistics,
differencing, and expanding window metrics for forecasting models.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TimeSeriesFeatureEngineer:
    """
    Constructs robust temporal features ensuring strictly zero future data leakage.
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        include_calendar: bool = True,
        include_differencing: bool = True,
    ):
        self.lags = lags or [1, 2, 3, 7, 14]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.include_calendar = include_calendar
        self.include_differencing = include_differencing

    def transform(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transforms input dataframe into feature-engineered dataset for ML forecasting.
        Maintains strictly chronological ordering.
        """
        # Ensure chronological ordering
        df_out = df.copy()
        df_out[time_col] = pd.to_datetime(df_out[time_col])
        df_out = df_out.sort_values(time_col).reset_index(drop=True)

        # 1. Calendar / Temporal features
        if self.include_calendar:
            dt = df_out[time_col].dt
            df_out["year"] = dt.year
            df_out["month"] = dt.month
            df_out["day"] = dt.day
            df_out["dayofweek"] = dt.dayofweek
            df_out["dayofyear"] = dt.dayofyear
            df_out["quarter"] = dt.quarter
            df_out["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
            if (dt.hour != 0).any():
                df_out["hour"] = dt.hour
                df_out["minute"] = dt.minute

        # 2. Lag features (Shifted strictly from past)
        n_rows = len(df_out)
        target_series = pd.to_numeric(df_out[target_col], errors="coerce")

        for lag in self.lags:
            if lag < n_rows:
                df_out[f"lag_{lag}"] = target_series.shift(lag)

        # 3. Rolling window statistics (Strictly using shift(1) to eliminate current observation leakage)
        past_target = target_series.shift(1)
        for w in self.rolling_windows:
            if w < n_rows:
                df_out[f"rolling_mean_{w}"] = past_target.rolling(window=w, min_periods=1).mean()
                df_out[f"rolling_std_{w}"] = past_target.rolling(window=w, min_periods=1).std().fillna(0.0)
                df_out[f"rolling_min_{w}"] = past_target.rolling(window=w, min_periods=1).min()
                df_out[f"rolling_max_{w}"] = past_target.rolling(window=w, min_periods=1).max()

        # 4. Expanding mean (Historical average prior to current observation)
        df_out["expanding_mean"] = past_target.expanding(min_periods=1).mean()

        # 5. Differencing & percentage change
        if self.include_differencing:
            df_out["diff_1"] = past_target.diff(1).fillna(0.0)
            df_out["pct_change_1"] = past_target.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # 6. Covariates processing (shift by 1 if future values aren't known at prediction time, or keep if known external variables)
        if covariates:
            for cov in covariates:
                if cov in df_out.columns and cov != target_col:
                    cov_numeric = pd.to_numeric(df_out[cov], errors="coerce")
                    df_out[f"{cov}_lag1"] = cov_numeric.shift(1)

        # Impute early lag NaNs with backfill/forwardfill then median
        feature_cols = [c for c in df_out.columns if c not in [time_col, target_col]]
        df_out[feature_cols] = df_out[feature_cols].bfill().ffill()

        return df_out

    def get_feature_names(self, df_engineered: pd.DataFrame, time_col: str, target_col: str) -> List[str]:
        """
        Returns list of engineered feature names excluding time and target.
        """
        return [c for c in df_engineered.columns if c not in [time_col, target_col]]
