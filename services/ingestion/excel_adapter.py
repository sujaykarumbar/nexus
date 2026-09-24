import os
from typing import Dict, Any, Optional
import pandas as pd
from .base import BaseIngestionAdapter


class ExcelAdapter(BaseIngestionAdapter):
    """Ingestion adapter for Excel (.xlsx, .xls) workbooks."""

    def validate(self, file_path: str) -> bool:
        """Validate Excel workbook."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            excel_file = pd.ExcelFile(file_path)
            return len(excel_file.sheet_names) > 0
        except Exception:
            return False

    def load(self, file_path: str, max_rows: Optional[int] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load Excel sheet into pandas DataFrame."""
        return pd.read_excel(file_path, sheet_name=sheet_name or 0, nrows=max_rows)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract sheet names and preliminary column metadata."""
        excel_file = pd.ExcelFile(file_path)
        first_sheet = excel_file.sheet_names[0]
        sample_df = pd.read_excel(file_path, sheet_name=first_sheet, nrows=10)
        
        # Approximate row count by loading sheet
        full_df = pd.read_excel(file_path, sheet_name=first_sheet)

        return {
            "format": "excel",
            "sheet_names": excel_file.sheet_names,
            "active_sheet": first_sheet,
            "columns": list(sample_df.columns),
            "column_count": len(sample_df.columns),
            "row_count": len(full_df),
            "file_size_bytes": os.path.getsize(file_path)
        }
