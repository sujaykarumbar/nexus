from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator
import os
import pandas as pd


class BaseIngestionAdapter(ABC):
    """Abstract base class for all dataset format ingestion adapters."""

    @abstractmethod
    def validate(self, file_path: str) -> bool:
        """Validate file format, headers, and structure integrity."""
        pass

    @abstractmethod
    def load(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset into a pandas DataFrame."""
        pass

    @abstractmethod
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract quick file metadata without loading full dataset into memory."""
        pass

    def stream_chunks(self, file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream dataset in fixed row chunks for memory-efficient processing."""
        df = self.load(file_path)
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
