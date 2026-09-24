"""
NEXUS Data Stream Simulator
Replays a pandas DataFrame row-by-row with configurable speed, injecting synthetic
spikes and drift shifts at random intervals to simulate realistic streaming behavior.
"""

import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class DataStreamSimulator:
    """
    Replays a dataset as an async event stream.

    Simulation modes:
    - Normal replay: rows published as-is at `rows_per_second` rate
    - Spike injection: a column value is multiplied by spike_factor at random intervals
    - Drift injection: column mean shifts gradually over a window of rows

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset to replay.
    numeric_columns : list[str]
        Columns to include in each emitted record.
    time_column : str | None
        If set, its value is used as the record timestamp; otherwise wall-clock is used.
    rows_per_second : float
        Simulated throughput rate. Default 10 rows/sec.
    spike_probability : float
        Per-row probability of injecting a random spike (0–1). Default 0.03.
    spike_factor : float
        Magnitude of injected spikes (multiple of column std). Default 4.0.
    loop : bool
        Whether to loop the dataset continuously. Default True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        time_column: Optional[str] = None,
        rows_per_second: float = 10.0,
        spike_probability: float = 0.03,
        spike_factor: float = 4.0,
        loop: bool = True,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._numeric_cols = [c for c in numeric_columns if c in df.columns]
        self._time_col = time_column
        self._delay = max(1e-4, 1.0 / rows_per_second)
        self._spike_prob = spike_probability
        self._spike_factor = spike_factor
        self._loop = loop
        self._running = False
        self._paused = False
        self._row_index = 0

        # Pre-compute column stds for spike injection
        self._col_stds: Dict[str, float] = {}
        for col in self._numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            self._col_stds[col] = float(series.std()) if len(series) > 1 else 1.0

        self.records_emitted = 0
        self.spikes_injected = 0

    def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _build_record(self, row: pd.Series) -> Dict[str, Any]:
        """Build a clean numeric record dict from a DataFrame row."""
        record: Dict[str, Any] = {}
        if self._time_col and self._time_col in row.index:
            record["timestamp"] = str(row[self._time_col])
        else:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        for col in self._numeric_cols:
            try:
                record[col] = float(row[col])
            except (TypeError, ValueError):
                record[col] = 0.0

        return record

    def _inject_spike(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly multiply one column's value by ±spike_factor * col_std."""
        if not self._numeric_cols:
            return record
        col = random.choice(self._numeric_cols)
        std = self._col_stds.get(col, 1.0)
        direction = random.choice([1, -1])
        record[col] = record.get(col, 0.0) + direction * self._spike_factor * std
        self.spikes_injected += 1
        return record

    async def stream(
        self,
        on_record: Callable[[Dict[str, Any]], Any],
        max_records: Optional[int] = None,
    ) -> None:
        """
        Async generator that emits records via `on_record` callback.
        Runs until stop() is called, or max_records is reached.
        """
        self._running = True
        self._row_index = 0
        n = len(self._df)

        while self._running:
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            if self._row_index >= n:
                if self._loop:
                    self._row_index = 0
                else:
                    break

            row = self._df.iloc[self._row_index]
            record = self._build_record(row)

            # Spike injection
            if random.random() < self._spike_prob:
                record = self._inject_spike(record)

            # Emit
            result = on_record(record)
            if asyncio.iscoroutine(result):
                await result

            self.records_emitted += 1
            self._row_index += 1

            if max_records and self.records_emitted >= max_records:
                break

            await asyncio.sleep(self._delay)

        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "records_emitted": self.records_emitted,
            "spikes_injected": self.spikes_injected,
            "current_row": self._row_index,
            "total_rows": len(self._df),
            "running": self._running,
            "paused": self._paused,
        }
