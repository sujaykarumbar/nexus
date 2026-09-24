"""
NEXUS Change-Point Detection Engine
Detects structural regime shifts in time-series metrics (mean, variance, trend changes).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ChangePointDetector:
    """
    Identifies change points, step shifts, and variance regimes in temporal data.
    """

    def __init__(self, min_segment_length: int = 7, significance_threshold: float = 2.5):
        self.min_segment_length = min_segment_length
        self.significance_threshold = significance_threshold

    def detect_change_points(
        self,
        series: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        Scans time series using dual sliding window disparity test to find significant regime breaks.
        """
        vals = pd.to_numeric(series, errors="coerce").dropna().values
        n = len(vals)
        w = self.min_segment_length

        if n < 2 * w:
            return []

        ts_list = list(timestamps) if timestamps is not None else [f"t_{i}" for i in range(n)]

        change_points = []
        scores = np.zeros(n)

        # Sliding disparity across window w before and window w after point i
        for i in range(w, n - w):
            before = vals[i - w:i]
            after = vals[i:i + w]

            mean_before = np.mean(before)
            mean_after = np.mean(after)
            var_before = np.var(before, ddof=1) if len(before) > 1 else 1e-4
            var_after = np.var(after, ddof=1) if len(after) > 1 else 1e-4

            pooled_se = np.sqrt((var_before / w) + (var_after / w) + 1e-8)
            t_stat = np.abs(mean_after - mean_before) / pooled_se
            scores[i] = t_stat

        # Identify local peaks in disparity scores above threshold
        for i in range(w, n - w):
            if scores[i] >= self.significance_threshold:
                # Check if local maximum
                if scores[i] == np.max(scores[max(0, i - w // 2):min(n, i + w // 2 + 1)]):
                    before = vals[max(0, i - w):i]
                    after = vals[i:min(n, i + w)]
                    mean_before = float(np.mean(before))
                    mean_after = float(np.mean(after))
                    magnitude = float(mean_after - mean_before)
                    rel_change_pct = float((magnitude / (abs(mean_before) + 1e-6)) * 100)

                    change_points.append({
                        "index": int(i),
                        "timestamp": str(ts_list[i]),
                        "disparity_score": round(float(scores[i]), 3),
                        "previous_mean": round(mean_before, 2),
                        "new_mean": round(mean_after, 2),
                        "magnitude": round(magnitude, 2),
                        "percentage_change": round(rel_change_pct, 1),
                        "regime_type": "UPWARD_SHIFT" if magnitude > 0 else "DOWNWARD_SHIFT",
                        "confidence": round(min(1.0, float(scores[i]) / (self.significance_threshold * 2.0)), 2)
                    })

        return change_points
