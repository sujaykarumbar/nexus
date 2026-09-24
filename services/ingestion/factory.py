from typing import Optional
from .base import BaseIngestionAdapter
from .csv_adapter import CSVAdapter
from .excel_adapter import ExcelAdapter
from .json_adapter import JSONAdapter
from .parquet_adapter import ParquetAdapter


class IngestionFactory:
    """Factory for selecting the appropriate ingestion adapter."""

    _ADAPTERS = {
        "csv": CSVAdapter,
        "xlsx": ExcelAdapter,
        "xls": ExcelAdapter,
        "json": JSONAdapter,
        "parquet": ParquetAdapter
    }

    @classmethod
    def get_adapter(cls, file_type_or_path: str) -> BaseIngestionAdapter:
        ext = file_type_or_path.split(".")[-1].lower() if "." in file_type_or_path else file_type_or_path.lower()
        adapter_cls = cls._ADAPTERS.get(ext)
        if not adapter_cls:
            raise ValueError(f"Unsupported dataset format '{ext}'. Supported formats: {list(cls._ADAPTERS.keys())}")
        return adapter_cls()

    @classmethod
    def get_loader(cls, file_type_or_path: str) -> BaseIngestionAdapter:
        """Alias for get_adapter."""
        return cls.get_adapter(file_type_or_path)
