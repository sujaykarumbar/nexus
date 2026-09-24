from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from services.profiling.schema_detector import SchemaDetector


class EDAEngine:
    """Automated Exploratory Data Analysis (EDA) Engine."""

    @classmethod
    def calculate_correlations(cls, df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
        """Compute correlation matrix and highlight strong feature relationships."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {
                "columns": list(num_df.columns),
                "matrix": {},
                "strong_pairs": []
            }

        corr_matrix = num_df.corr(method=method)
        
        # Clean matrix for JSON
        matrix_dict = {}
        for col in corr_matrix.columns:
            matrix_dict[col] = {
                row: round(float(val), 3) if not pd.isna(val) else 0.0
                for row, val in corr_matrix[col].items()
            }

        # Identify strong pairs (|r| >= 0.5, excluding diagonal)
        strong_pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a, col_b = cols[i], cols[j]
                val = corr_matrix.iloc[i, j]
                if not pd.isna(val) and abs(val) >= 0.5:
                    strong_pairs.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(float(val), 3),
                        "relationship": "Strong Positive" if val > 0 else "Strong Negative"
                    })

        # Sort by absolute correlation magnitude
        strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "columns": cols,
            "matrix": matrix_dict,
            "strong_pairs": strong_pairs
        }

    @classmethod
    def analyze_distributions(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate binned distributions for numerical columns and top categories."""
        schema_info = SchemaDetector.detect_schema(df)
        distributions = {}

        for col, col_info in schema_info["columns"].items():
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if col_info["inferred_type"] == "NUMERICAL" or pd.api.types.is_numeric_dtype(df[col]):
                num_series = pd.to_numeric(series, errors="coerce").dropna()
                if len(num_series) > 0:
                    counts, bin_edges = np.histogram(num_series, bins=10)
                    distributions[col] = {
                        "type": "numerical",
                        "bins": [
                            {
                                "range": f"{round(float(bin_edges[i]), 1)} - {round(float(bin_edges[i+1]), 1)}",
                                "count": int(counts[i])
                            }
                            for i in range(len(counts))
                        ]
                    }
            elif col_info["inferred_type"] in ["CATEGORICAL", "BOOLEAN"]:
                top_cats = series.value_counts().head(8)
                distributions[col] = {
                    "type": "categorical",
                    "categories": [
                        {"label": str(k), "count": int(v)}
                        for k, v in top_cats.items()
                    ]
                }

        return distributions

    @classmethod
    def run_full_eda(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute full automated exploratory data analysis."""
        schema_info = SchemaDetector.detect_schema(df)
        correlations = cls.calculate_correlations(df)
        distributions = cls.analyze_distributions(df)

        return {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numerical_features_count": schema_info["column_types_summary"]["NUMERICAL"],
                "categorical_features_count": schema_info["column_types_summary"]["CATEGORICAL"],
                "datetime_features_count": schema_info["column_types_summary"]["DATETIME"]
            },
            "correlations": correlations,
            "distributions": distributions
        }
