from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from .schema_detector import SchemaDetector


class DataProfiler:
    """Statistical Data Profiling Engine."""

    @classmethod
    def profile_column(cls, series: pd.Series, col_name: str, inferred_type: str) -> Dict[str, Any]:
        """Profile statistics for a single column based on its inferred type."""
        clean_series = series.dropna()
        n_total = len(series)
        n_non_null = len(clean_series)
        n_null = n_total - n_non_null
        n_unique = int(clean_series.nunique())

        base_profile = {
            "name": col_name,
            "inferred_type": inferred_type,
            "total_count": n_total,
            "non_null_count": n_non_null,
            "null_count": n_null,
            "null_percentage": round((n_null / n_total * 100), 2) if n_total > 0 else 0.0,
            "unique_count": n_unique,
            "unique_percentage": round((n_unique / n_total * 100), 2) if n_total > 0 else 0.0,
        }

        if n_non_null == 0:
            return base_profile

        # Numerical Profiling
        if inferred_type == "NUMERICAL" or pd.api.types.is_numeric_dtype(clean_series):
            try:
                num_series = pd.to_numeric(clean_series, errors="coerce").dropna()
                if len(num_series) > 0:
                    mean_val = float(num_series.mean())
                    std_val = float(num_series.std()) if len(num_series) > 1 else 0.0
                    min_val = float(num_series.min())
                    max_val = float(num_series.max())
                    median_val = float(num_series.median())
                    q25 = float(num_series.quantile(0.25))
                    q75 = float(num_series.quantile(0.75))
                    iqr = q75 - q25
                    
                    skew_val = float(stats.skew(num_series)) if len(num_series) > 2 else 0.0
                    kurt_val = float(stats.kurtosis(num_series)) if len(num_series) > 3 else 0.0

                    # Histogram bins for visualization
                    counts, bin_edges = np.histogram(num_series, bins=min(15, max(5, n_unique)))
                    histogram_data = [
                        {
                            "bin_start": round(float(bin_edges[i]), 2),
                            "bin_end": round(float(bin_edges[i+1]), 2),
                            "count": int(counts[i])
                        }
                        for i in range(len(counts))
                    ]

                    base_profile.update({
                        "min": round(min_val, 4),
                        "max": round(max_val, 4),
                        "mean": round(mean_val, 4),
                        "median": round(median_val, 4),
                        "std": round(std_val, 4),
                        "variance": round(std_val ** 2, 4),
                        "q25": round(q25, 4),
                        "q75": round(q75, 4),
                        "iqr": round(iqr, 4),
                        "skewness": round(skew_val, 4),
                        "kurtosis": round(kurt_val, 4),
                        "histogram": histogram_data
                    })
            except Exception:
                pass

        # Categorical Profiling
        if inferred_type in ["CATEGORICAL", "BOOLEAN", "TEXT"]:
            try:
                top_counts = clean_series.value_counts().head(10)
                freq_table = [
                    {
                        "category": str(cat),
                        "count": int(count),
                        "percentage": round((int(count) / n_total * 100), 2)
                    }
                    for cat, count in top_counts.items()
                ]
                base_profile.update({
                    "top_categories": freq_table,
                    "cardinality": n_unique
                })
            except Exception:
                pass

        # Text Length Metrics
        if inferred_type == "TEXT":
            try:
                str_lens = clean_series.astype(str).str.len()
                base_profile.update({
                    "avg_length": round(float(str_lens.mean()), 2),
                    "min_length": int(str_lens.min()),
                    "max_length": int(str_lens.max()),
                    "empty_string_count": int((clean_series.astype(str).str.strip() == "").sum())
                })
            except Exception:
                pass

        # Datetime Metrics
        if inferred_type == "DATETIME":
            try:
                dt_series = pd.to_datetime(clean_series, errors="coerce").dropna()
                if len(dt_series) > 0:
                    min_date = dt_series.min()
                    max_date = dt_series.max()
                    date_range_days = (max_date - min_date).days
                    base_profile.update({
                        "min_date": str(min_date),
                        "max_date": str(max_date),
                        "date_range_days": date_range_days
                    })
            except Exception:
                pass

        return base_profile

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile entire dataset and return complete structured profile."""
        schema_info = SchemaDetector.detect_schema(df)
        columns_profile = {}

        for col, col_info in schema_info["columns"].items():
            col_profile = cls.profile_column(df[col], col, col_info["inferred_type"])
            columns_profile[col] = col_profile

        memory_usage_bytes = int(df.memory_usage(deep=True).sum())

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_bytes": memory_usage_bytes,
            "memory_usage_mb": round(memory_usage_bytes / (1024 * 1024), 2),
            "schema": schema_info,
            "columns": columns_profile
        }
