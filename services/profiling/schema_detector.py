import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SchemaDetector:
    """Heuristic Schema Detector and Data Type Classifier."""

    ID_NAME_PATTERNS = [
        r'(?i)^id$', r'(?i)_id$', r'(?i)^id_', r'(?i)customer_id', r'(?i)user_id',
        r'(?i)order_id', r'(?i)transaction_id', r'(?i)uuid', r'(?i)guid', r'(?i)email',
        r'(?i)phone', r'(?i)ssn', r'(?i)passport', r'(?i)account_number'
    ]

    DATE_PATTERNS = [
        r'^\d{4}-\d{2}-\d{2}', r'^\d{2}/\d{2}/\d{4}', r'^\d{4}/\d{2}/\d{2}',
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    ]

    BOOLEAN_VALUES = {'0', '1', 'true', 'false', 'yes', 'no', 't', 'f', 'y', 'n'}

    @classmethod
    def infer_column_type(cls, series: pd.Series, col_name: str) -> str:
        """Infer semantic type for a single column using heuristics."""
        clean_series = series.dropna()
        n_total = len(clean_series)
        
        if n_total == 0:
            return "UNKNOWN"

        n_unique = clean_series.nunique()
        unique_ratio = n_unique / n_total if n_total > 0 else 0

        # 1. Identifier Check (Name heuristics + High uniqueness)
        is_id_name = any(re.search(pat, col_name) for pat in cls.ID_NAME_PATTERNS)
        if is_id_name and (unique_ratio > 0.85 or n_unique > 500):
            return "IDENTIFIER"

        # 2. Boolean Check
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if n_unique <= 2:
            unique_str_set = {str(x).strip().lower() for x in clean_series.unique()}
            if unique_str_set.issubset(cls.BOOLEAN_VALUES):
                return "BOOLEAN"

        # 3. Datetime Check
        if pd.api.types.is_datetime64_any_dtype(series):
            return "DATETIME"
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            sample_vals = clean_series.head(50).astype(str)
            date_matches = sum(any(re.match(pat, str(val)) for pat in cls.DATE_PATTERNS) for val in sample_vals)
            if date_matches / len(sample_vals) > 0.8:
                try:
                    pd.to_datetime(sample_vals, errors="raise")
                    return "DATETIME"
                except Exception:
                    pass

        # 4. Numerical Check
        if pd.api.types.is_numeric_dtype(series):
            # Check if low cardinality numeric code is actually categorical
            if n_unique <= 10 and unique_ratio < 0.05 and not is_id_name:
                return "CATEGORICAL"
            return "NUMERICAL"

        # 5. Text Check — long strings should be TEXT regardless of cardinality
        str_lengths = clean_series.astype(str).str.len()
        avg_len = str_lengths.mean()
        if avg_len > 35:
            return "TEXT"

        # 6. Categorical vs Identifier Check for Objects / Strings
        if n_unique <= 50 or unique_ratio < 0.15:
            return "CATEGORICAL"
        elif unique_ratio > 0.9:
            return "IDENTIFIER"

        return "CATEGORICAL"

    @classmethod
    def detect_schema(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and return complete schema profile for a dataset."""
        schema = {}
        total_rows = len(df)

        for col in df.columns:
            series = df[col]
            inferred_type = cls.infer_column_type(series, str(col))
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique())
            
            schema[col] = {
                "name": str(col),
                "inferred_type": inferred_type,
                "raw_dtype": str(series.dtype),
                "null_count": null_count,
                "null_percentage": round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
                "unique_count": unique_count,
                "unique_percentage": round((unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
                "is_nullable": null_count > 0,
                "sample_values": [None if pd.isna(x) else (int(x) if isinstance(x, (np.integer, int)) else (float(x) if isinstance(x, (np.floating, float)) else str(x))) for x in series.dropna().head(5).tolist()]
            }

        return {
            "columns": schema,
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "column_types_summary": {
                t: sum(1 for col_info in schema.values() if col_info["inferred_type"] == t)
                for t in ["NUMERICAL", "CATEGORICAL", "DATETIME", "BOOLEAN", "TEXT", "IDENTIFIER", "UNKNOWN"]
            }
        }
