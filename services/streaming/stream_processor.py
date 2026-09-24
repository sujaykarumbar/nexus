"""
NEXUS Stream Processor
Consumes records from an EventBus topic, maintains a sliding window per numeric column,
computes real-time Z-score and IQR drift, and emits StreamingAnomaly events.
"""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from services.streaming.event_bus import EventBus


@dataclass
class StreamingAnomaly:
    """A detected anomaly event in the streaming pipeline."""
    anomaly_id: str
    session_id: str
    timestamp: str
    column: str
    value: float
    z_score: float
    iqr_score: float
    severity: str               # 'low' | 'medium' | 'high' | 'critical'
    anomaly_type: str           # 'z_score' | 'iqr' | 'combined'
    window_mean: float
    window_std: float
    window_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_severity(z: float) -> str:
    az = abs(z)
    if az >= 4.0:
        return "critical"
    elif az >= 3.0:
        return "high"
    elif az >= 2.5:
        return "medium"
    return "low"


class StreamProcessor:
    """
    Sliding-window streaming anomaly detector.

    Maintains per-column rolling statistics over a configurable window.
    On each new record, checks Z-score (> z_threshold) AND IQR fences.
    Detected anomalies are published to the `alerts` topic and optionally
    passed to an on_anomaly callback.
    """

    ALERTS_TOPIC = "nexus.streaming.alerts"

    def __init__(
        self,
        session_id: str,
        bus: EventBus,
        numeric_columns: List[str],
        window_size: int = 50,
        z_threshold: float = 2.8,
        iqr_multiplier: float = 2.5,
        on_anomaly: Optional[Callable[[StreamingAnomaly], None]] = None,
    ) -> None:
        self.session_id = session_id
        self.bus = bus
        self.numeric_columns = numeric_columns
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.on_anomaly = on_anomaly

        # Per-column sliding window buffers
        self._windows: Dict[str, Deque[float]] = {
            col: deque(maxlen=window_size) for col in numeric_columns
        }

        self.records_processed = 0
        self.anomalies_detected = 0
        self._running = False

    async def process_record(self, record: Dict[str, Any]) -> List[StreamingAnomaly]:
        """
        Process a single incoming data record. Returns list of detected anomalies
        (usually 0 or 1 per column per record).
        """
        self.records_processed += 1
        timestamp = record.get(
            "timestamp",
            datetime.now(timezone.utc).isoformat()
        )
        detected: List[StreamingAnomaly] = []

        for col in self.numeric_columns:
            raw_val = record.get(col)
            if raw_val is None:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue

            window = self._windows[col]

            # Need minimum window before we can score
            if len(window) >= max(10, self.window_size // 5):
                arr = np.array(window)
                mean = float(np.mean(arr))
                std = float(np.std(arr)) if np.std(arr) > 1e-8 else 1.0

                # Z-score test
                z = (val - mean) / std
                z_triggered = abs(z) >= self.z_threshold

                # IQR test
                q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
                iqr = q3 - q1 if (q3 - q1) > 1e-8 else 1.0
                lower_fence = q1 - self.iqr_multiplier * iqr
                upper_fence = q3 + self.iqr_multiplier * iqr
                iqr_triggered = val < lower_fence or val > upper_fence

                if z_triggered or iqr_triggered:
                    if z_triggered and iqr_triggered:
                        atype = "combined"
                    elif z_triggered:
                        atype = "z_score"
                    else:
                        atype = "iqr"

                    iqr_score = max(
                        0.0,
                        (val - upper_fence) / iqr if val > upper_fence else (lower_fence - val) / iqr
                    )

                    anomaly = StreamingAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        session_id=self.session_id,
                        timestamp=timestamp,
                        column=col,
                        value=round(val, 4),
                        z_score=round(z, 3),
                        iqr_score=round(iqr_score, 3),
                        severity=_compute_severity(z),
                        anomaly_type=atype,
                        window_mean=round(mean, 4),
                        window_std=round(std, 4),
                        window_size=len(window),
                    )
                    detected.append(anomaly)
                    self.anomalies_detected += 1

                    # Publish to event bus
                    await self.bus.publish(
                        f"{self.ALERTS_TOPIC}.{self.session_id}",
                        anomaly.to_dict()
                    )

                    if self.on_anomaly:
                        self.on_anomaly(anomaly)

            # Always append value to window after scoring
            window.append(val)

        return detected

    def get_stats(self) -> Dict[str, Any]:
        """Current runtime statistics for this processor."""
        return {
            "session_id": self.session_id,
            "records_processed": self.records_processed,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": round(
                self.anomalies_detected / max(1, self.records_processed), 4
            ),
            "columns_monitored": self.numeric_columns,
            "window_size": self.window_size,
            "window_fills": {
                col: len(self._windows[col]) for col in self.numeric_columns
            },
        }
