import csv
import os
from typing import Dict, Any, Optional, Iterator
import pandas as pd
from .base import BaseIngestionAdapter


class CSVAdapter(BaseIngestionAdapter):
    """Robust CSV ingestion adapter with delimiter and encoding sniffing."""

    SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    def _detect_encoding(self, file_path: str) -> str:
        """Detect best readable encoding for the file."""
        for enc in self.SUPPORTED_ENCODINGS:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    f.read(4096)
                return enc
            except (UnicodeDecodeError, Exception):
                continue
        return "utf-8"

    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Sniff CSV delimiter from sample text."""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                sample = f.read(8192)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                return dialect.delimiter
        except Exception:
            return ","

    def validate(self, file_path: str) -> bool:
        """Validate if file is a non-empty, parseable CSV."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            enc = self._detect_encoding(file_path)
            sep = self._detect_delimiter(file_path, enc)
            df = pd.read_csv(file_path, encoding=enc, sep=sep, nrows=5)
            return len(df.columns) > 0
        except Exception:
            return False

    def load(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load CSV into pandas DataFrame."""
        enc = self._detect_encoding(file_path)
        sep = self._detect_delimiter(file_path, enc)
        return pd.read_csv(
            file_path,
            encoding=enc,
            sep=sep,
            nrows=max_rows,
            on_bad_lines="skip",
            low_memory=False
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract quick schema and row count without loading full file into RAM."""
        enc = self._detect_encoding(file_path)
        sep = self._detect_delimiter(file_path, enc)
        sample_df = pd.read_csv(file_path, encoding=enc, sep=sep, nrows=10)
        
        # Count total rows
        row_count = 0
        with open(file_path, "r", encoding=enc) as f:
            for _ in f:
                row_count += 1
        row_count = max(0, row_count - 1)  # Subtract header

        return {
            "format": "csv",
            "encoding": enc,
            "delimiter": sep,
            "columns": list(sample_df.columns),
            "column_count": len(sample_df.columns),
            "row_count": row_count,
            "file_size_bytes": os.path.getsize(file_path)
        }

    def stream_chunks(self, file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Native chunked streaming for large CSVs."""
        enc = self._detect_encoding(file_path)
        sep = self._detect_delimiter(file_path, enc)
        for chunk in pd.read_csv(file_path, encoding=enc, sep=sep, chunksize=chunk_size, on_bad_lines="skip"):
            yield chunk
