"""
NEXUS Data Drift Detector — Phase 9
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov tests to measure
statistical drift between a reference dataset (training) and a current dataset (serving).
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureDrift:
    """Drift metrics for a single feature."""
    feature: str
    psi: float              # Population Stability Index (0=no drift, >0.2=significant)
    ks_statistic: float     # KS test statistic
    ks_p_value: float       # KS test p-value
    ref_mean: float
    cur_mean: float
    ref_std: float
    cur_std: float
    mean_shift_pct: float   # % change in mean
    drift_level: str        # 'none' | 'minor' | 'moderate' | 'major'
    alert: bool             # True if actionable drift detected


@dataclass
class DriftReport:
    """Complete dataset drift report."""
    reference_dataset_id: str
    current_dataset_id: str
    feature_drifts: List[FeatureDrift]
    overall_psi: float
    overall_drift_level: str
    drifted_features: List[str]
    stable_features: List[str]
    alert: bool
    recommendation: str
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_drifts"] = [asdict(f) for f in self.feature_drifts]
        return d


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between two numeric distributions.
    PSI < 0.1: no significant change
    PSI 0.1-0.2: minor change
    PSI > 0.2: major shift — model retraining recommended
    """
    # Use quantile-based bins from reference
    eps = 1e-8
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(expected, bins=breakpoints)
    cur_counts, _ = np.histogram(actual, bins=breakpoints)

    ref_pct = ref_counts / (len(expected) + eps)
    cur_pct = cur_counts / (len(actual) + eps)

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(abs(psi), 4)


def _drift_level(psi: float, ks_p: float) -> str:
    if psi < 0.1 and ks_p > 0.05:
        return "none"
    elif psi < 0.1 or ks_p > 0.05:
        return "minor"
    elif psi < 0.25:
        return "moderate"
    return "major"


class DriftDetector:
    """
    Detects statistical data drift between reference and current datasets.
    Uses PSI (production-standard) and KS-test (non-parametric) per feature.
    """

    PSI_ALERT_THRESHOLD = 0.2
    KS_P_ALERT_THRESHOLD = 0.05

    @classmethod
    def detect(
        cls,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_dataset_id: str = "reference",
        current_dataset_id: str = "current",
        feature_columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Run drift detection across all shared numeric columns.
        Returns a DriftReport with per-feature and overall metrics.
        """
        import time
        t0 = time.time()

        # Identify numeric columns shared by both DataFrames
        ref_numeric = set(reference_df.select_dtypes(include="number").columns)
        cur_numeric = set(current_df.select_dtypes(include="number").columns)
        shared = ref_numeric & cur_numeric
        if feature_columns:
            shared = shared & set(feature_columns)
        columns = sorted(shared)

        feature_drifts: List[FeatureDrift] = []
        drifted: List[str] = []
        stable: List[str] = []

        for col in columns:
            ref_arr = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
            cur_arr = pd.to_numeric(current_df[col], errors="coerce").dropna().values

            if len(ref_arr) < 10 or len(cur_arr) < 10:
                continue

            psi = _compute_psi(ref_arr, cur_arr)
            ks_stat, ks_p = stats.ks_2samp(ref_arr, cur_arr)

            ref_mean = float(np.mean(ref_arr))
            cur_mean = float(np.mean(cur_arr))
            ref_std = float(np.std(ref_arr))
            cur_std = float(np.std(cur_arr))
            mean_shift_pct = round(
                abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100, 2
            )

            level = _drift_level(psi, ks_p)
            alert = bool(psi >= cls.PSI_ALERT_THRESHOLD or ks_p <= cls.KS_P_ALERT_THRESHOLD)

            fd = FeatureDrift(
                feature=col,
                psi=float(psi),
                ks_statistic=float(round(float(ks_stat), 4)),
                ks_p_value=float(round(float(ks_p), 4)),
                ref_mean=float(round(ref_mean, 4)),
                cur_mean=float(round(cur_mean, 4)),
                ref_std=float(round(ref_std, 4)),
                cur_std=float(round(cur_std, 4)),
                mean_shift_pct=float(mean_shift_pct),
                drift_level=str(level),
                alert=bool(alert),
            )
            feature_drifts.append(fd)
            if alert:
                drifted.append(col)
            else:
                stable.append(col)

        overall_psi = float(round(
            float(np.mean([f.psi for f in feature_drifts])) if feature_drifts else 0.0, 4
        ))

        if overall_psi < 0.1 and not drifted:
            overall_level = "none"
        elif overall_psi < 0.2:
            overall_level = "minor"
        elif overall_psi < 0.3:
            overall_level = "moderate"
        else:
            overall_level = "major"

        global_alert = bool(len(drifted) > 0)

        if not global_alert:
            recommendation = "No significant drift detected. Current model remains valid."
        elif overall_level == "minor":
            recommendation = f"{len(drifted)} feature(s) show minor drift. Monitor closely."
        elif overall_level == "moderate":
            recommendation = (
                f"{len(drifted)} feature(s) show moderate drift. "
                "Consider retraining with recent data within 1-2 weeks."
            )
        else:
            recommendation = (
                f"CRITICAL: {len(drifted)} feature(s) show major drift. "
                "Immediate retraining recommended — model reliability is compromised."
            )

        duration_ms = (time.time() - t0) * 1000

        return DriftReport(
            reference_dataset_id=reference_dataset_id,
            current_dataset_id=current_dataset_id,
            feature_drifts=feature_drifts,
            overall_psi=overall_psi,
            overall_drift_level=overall_level,
            drifted_features=drifted,
            stable_features=stable,
            alert=global_alert,
            recommendation=recommendation,
            duration_ms=round(duration_ms, 2),
        )
