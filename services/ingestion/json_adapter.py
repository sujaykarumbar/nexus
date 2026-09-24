import json
import os
from typing import Dict, Any, Optional
import pandas as pd
from .base import BaseIngestionAdapter


class JSONAdapter(BaseIngestionAdapter):
    """Ingestion adapter for standard and line-delimited JSON datasets."""

    def validate(self, file_path: str) -> bool:
        """Validate JSON format."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, (list, dict))
        except Exception:
            # Check lines JSON
            try:
                df = pd.read_json(file_path, lines=True, nrows=5)
                return len(df.columns) > 0
            except Exception:
                return False

    def load(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load JSON into normalized pandas DataFrame."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # Check if dictionary contains a root data array
                for key, val in data.items():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        df = pd.json_normalize(val)
                        break
                else:
                    df = pd.json_normalize([data])
            else:
                df = pd.read_json(file_path)
        except Exception:
            df = pd.read_json(file_path, lines=True)

        if max_rows:
            df = df.head(max_rows)
        return df

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract JSON metadata and schema."""
        df = self.load(file_path, max_rows=50)
        full_df = self.load(file_path)
        return {
            "format": "json",
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "row_count": len(full_df),
            "file_size_bytes": os.path.getsize(file_path)
        }
