import os
from typing import Dict, Any, Optional
import pandas as pd
from .base import BaseIngestionAdapter


class ParquetAdapter(BaseIngestionAdapter):
    """Ingestion adapter for columnar Apache Parquet datasets."""

    def validate(self, file_path: str) -> bool:
        """Validate Parquet file header."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            df = pd.read_parquet(file_path)
            return len(df.columns) > 0
        except Exception:
            return False

    def load(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load Parquet dataset."""
        df = pd.read_parquet(file_path)
        if max_rows:
            df = df.head(max_rows)
        return df

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract Parquet schema and metadata."""
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema
            row_count = parquet_file.metadata.num_rows
            columns = schema.names
        except Exception:
            df = pd.read_parquet(file_path)
            row_count = len(df)
            columns = list(df.columns)

        return {
            "format": "parquet",
            "columns": columns,
            "column_count": len(columns),
            "row_count": row_count,
            "file_size_bytes": os.path.getsize(file_path)
        }
