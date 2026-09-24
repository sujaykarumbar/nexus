"""
NEXUS Time-Series Intelligence & Detection Engine
Automatically detects temporal columns, regularity, sampling frequency,
missing intervals, duplicate timestamps, trend, and seasonality.
"""

from typing import Dict, Any, List, Optional
import re
import numpy as np
import pandas as pd


class TimeSeriesDetector:
    """
    Analyzes datasets to identify temporal features, frequencies,
    trends, seasonality, and chronological consistency.
    """

    TIME_COLUMN_PATTERNS = [
        r"date", r"time", r"timestamp", r"created_at", r"updated_at",
        r"event_time", r"transaction_date", r"datetime", r"day", r"month", r"year"
    ]

    def detect_candidate_time_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scans dataframe columns and returns scored candidate time columns.
        """
        candidates = []

        for col in df.columns:
            score = 0.0
            reasons = []
            col_lower = str(col).lower()

            # Check column name heuristic
            for pattern in self.TIME_COLUMN_PATTERNS:
                if re.search(pattern, col_lower):
                    score += 0.4
                    reasons.append(f"Name matches pattern '{pattern}'")
                    break

            # Check data type
            is_parsed = False
            parsed_series = None
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                score += 0.6
                reasons.append("Native datetime64 dtype")
                is_parsed = True
                parsed_series = df[col]
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                # Sample non-null values to test datetime conversion
                sample = df[col].dropna().head(30)
                if len(sample) > 0:
                    try:
                        converted = pd.to_datetime(sample, errors="coerce")
                        valid_ratio = converted.notna().mean()
                        if valid_ratio >= 0.8:
                            score += 0.5 * valid_ratio
                            reasons.append(f"{int(valid_ratio * 100)}% sample parseable as datetime")
                            is_parsed = True
                    except Exception:
                        pass
            elif pd.api.types.is_integer_dtype(df[col]):
                # Potential Unix timestamp in seconds or milliseconds
                sample = df[col].dropna().head(20)
                if len(sample) > 0:
                    min_val = sample.min()
                    max_val = sample.max()
                    # 2000-01-01 (946684800) to 2035-01-01 (2051222400)
                    if 900000000 <= min_val <= 2500000000 and 900000000 <= max_val <= 2500000000:
                        score += 0.35
                        reasons.append("Values fall within plausible Unix timestamp range (seconds)")
                        is_parsed = True
                    # Milliseconds range
                    elif 900000000000 <= min_val <= 2500000000000:
                        score += 0.35
                        reasons.append("Values fall within plausible Unix timestamp range (milliseconds)")
                        is_parsed = True

            if score >= 0.35:
                candidates.append({
                    "column_name": col,
                    "confidence_score": round(min(1.0, score), 2),
                    "is_parsed": is_parsed,
                    "reasons": reasons
                })

        candidates.sort(key=lambda c: c["confidence_score"], reverse=True)
        return candidates

    def analyze_temporal_profile(self, df: pd.DataFrame, time_col: str, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Deeply inspects the chosen time column and optional target series.
        """
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in dataframe.")

        # Convert to datetime series
        try:
            if pd.api.types.is_integer_dtype(df[time_col]) and df[time_col].dropna().median() > 90000000000:
                dt_series = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
            elif pd.api.types.is_integer_dtype(df[time_col]) and df[time_col].dropna().median() > 90000000:
                dt_series = pd.to_datetime(df[time_col], unit="s", errors="coerce")
            else:
                dt_series = pd.to_datetime(df[time_col], errors="coerce")
        except Exception as e:
            raise ValueError(f"Unable to parse '{time_col}' as datetime: {str(e)}")

        valid_mask = dt_series.notna()
        clean_dt = dt_series[valid_mask]
        total_rows = len(df)
        valid_rows = len(clean_dt)

        if valid_rows < 3:
            raise ValueError("Insufficient valid timestamp observations (minimum 3 required).")

        # Chronological verification
        is_monotonic_increasing = clean_dt.is_monotonic_increasing
        duplicates_count = int(clean_dt.duplicated().sum())
        min_ts = clean_dt.min().isoformat()
        max_ts = clean_dt.max().isoformat()

        # Frequency analysis
        sorted_dt = clean_dt.sort_values().reset_index(drop=True)
        deltas = sorted_dt.diff().dropna()
        
        inferred_freq = pd.infer_freq(sorted_dt)
        median_delta_seconds = float(deltas.dt.total_seconds().median()) if len(deltas) > 0 else 0.0
        
        # Categorize human readable frequency
        frequency_label = "IRREGULAR"
        if inferred_freq:
            frequency_label = inferred_freq
        elif median_delta_seconds > 0:
            if 3500 <= median_delta_seconds <= 3700:
                frequency_label = "1H"
            elif 85000 <= median_delta_seconds <= 87000:
                frequency_label = "1D"
            elif 590000 <= median_delta_seconds <= 610000:
                frequency_label = "1W"
            elif 2400000 <= median_delta_seconds <= 2700000:
                frequency_label = "1M"
            elif 50 <= median_delta_seconds <= 70:
                frequency_label = "1min"

        # Check regularity
        if len(deltas) > 1:
            std_delta = float(deltas.dt.total_seconds().std())
            is_regular = bool(std_delta < 0.1 * max(median_delta_seconds, 1.0))
        else:
            is_regular = True

        # Missing intervals detection
        missing_intervals = 0
        if is_regular and median_delta_seconds > 0:
            expected_intervals = int((sorted_dt.iloc[-1] - sorted_dt.iloc[0]).total_seconds() / median_delta_seconds)
            missing_intervals = max(0, expected_intervals - len(sorted_dt) + 1)

        result: Dict[str, Any] = {
            "time_column": time_col,
            "total_observations": total_rows,
            "valid_timestamps": valid_rows,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "is_chronological": bool(is_monotonic_increasing),
            "duplicate_timestamps": duplicates_count,
            "inferred_frequency": frequency_label,
            "median_interval_seconds": median_delta_seconds,
            "is_regular_interval": is_regular,
            "estimated_missing_intervals": missing_intervals,
            "trend": "UNKNOWN",
            "seasonality_period": None,
            "autocorrelation_lag1": None
        }

        # Analyze target series if provided
        if target_col and target_col in df.columns:
            sorted_df = df.copy()
            sorted_df["_dt_temp_"] = dt_series
            sorted_df = sorted_df.dropna(subset=["_dt_temp_", target_col]).sort_values("_dt_temp_")
            target_vals = pd.to_numeric(sorted_df[target_col], errors="coerce").dropna()

            if len(target_vals) >= 5:
                # 1. Autocorrelation
                try:
                    autocorr_1 = float(target_vals.autocorr(lag=1))
                    result["autocorrelation_lag1"] = round(autocorr_1, 3) if not np.isnan(autocorr_1) else 0.0
                except Exception:
                    pass

                # 2. Trend direction via linear regression slope
                x = np.arange(len(target_vals))
                y = target_vals.values
                if np.std(x) > 0 and np.std(y) > 0:
                    slope, _ = np.polyfit(x, y, 1)
                    rel_slope = slope / (np.mean(np.abs(y)) + 1e-6)
                    if rel_slope > 0.002:
                        result["trend"] = "UPWARD"
                    elif rel_slope < -0.002:
                        result["trend"] = "DOWNWARD"
                    else:
                        result["trend"] = "STATIONARY"

                # 3. Seasonality detection via peak autocorrelation
                max_lag = min(30, len(target_vals) // 2)
                best_period = None
                best_corr = 0.35 # Minimum threshold
                for lag in range(2, max_lag + 1):
                    try:
                        c = target_vals.autocorr(lag=lag)
                        if not np.isnan(c) and c > best_corr:
                            best_corr = c
                            best_period = lag
                    except Exception:
                        pass
                result["seasonality_period"] = best_period

        return result
