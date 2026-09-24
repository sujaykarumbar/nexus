"""
NEXUS Unified Anomaly Detection Engine
Coordinates statistical, machine learning, residual, autoencoder, and change-point detectors.
Applies deterministic severity scoring and evidence-based explanations.
"""

from typing import Dict, Any, List, Optional
import uuid
import numpy as np
import pandas as pd

from .statistical import StatisticalAnomalyDetector
from .ml_detector import MLAnomalyDetector
from .residual_detector import ResidualAnomalyDetector
from .change_point import ChangePointDetector
from .autoencoder import AutoencoderAnomalyDetector


class AnomalyDetectionEngine:
    """
    Unified anomaly detection suite with multi-detector consensus,
    deterministic severity classification (LOW, MEDIUM, HIGH, CRITICAL),
    and explainable diagnostics.
    """

    # Domain sensor safety ranges (min, max, critical_max)
    SENSOR_BOUNDS = {
        "temperature": (50.0, 80.0, 95.0), # Normal 50-80°C, Critical >95°C
        "voltage": (210.0, 245.0, 190.0), # Critical drop <190V
        "vibration": (0.0, 0.08, 0.12), # Normal <0.08, Critical >0.12
        "pressure": (25.0, 35.0, 42.0),
        "current": (8.0, 16.0, 20.0)
    }

    def __init__(self):
        self.statistical = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        self.residual_detector = ResidualAnomalyDetector()
        self.change_point_detector = ChangePointDetector()
        self.autoencoder = AutoencoderAnomalyDetector()

    @staticmethod
    def classify_severity(
        score: float,
        deviation_ratio: float = 0.0,
        is_sensor_breach: bool = False
    ) -> str:
        """
        Determines anomaly severity via deterministic rules.
        """
        if is_sensor_breach or score >= 0.85 or abs(deviation_ratio) >= 4.0:
            return "CRITICAL"
        elif score >= 0.65 or abs(deviation_ratio) >= 3.0:
            return "HIGH"
        elif score >= 0.45 or abs(deviation_ratio) >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"

    def detect(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        include_deep_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full anomaly detection across dataset.
        """
        df_clean = df.copy()

        # Parse time column if provided
        if time_col and time_col in df_clean.columns:
            df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors="coerce")
            df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
            timestamps = df_clean[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_col = None
            timestamps = pd.Series([f"Row {i}" for i in range(len(df_clean))])

        # Identify numeric columns
        numeric_cols = [c for c in df_clean.columns if c != time_col and pd.api.types.is_numeric_dtype(df_clean[c])]
        if not numeric_cols:
            raise ValueError("No numeric columns available for anomaly detection.")

        primary_col = target_col if (target_col and target_col in numeric_cols) else numeric_cols[0]
        eval_features = feature_cols or numeric_cols

        anomalies_list: List[Dict[str, Any]] = []

        # 1. Univariate Residual & Statistical Detection on Primary Series
        actual_series = df_clean[primary_col]
        res_df = self.residual_detector.detect_residuals(actual_series)
        z_df = self.statistical.modified_z_score_detect(actual_series)
        iqr_df = self.statistical.iqr_detect(actual_series)

        # 2. Multivariate Isolation Forest Detection
        iso_df = None
        if len(eval_features) >= 1 and len(df_clean) >= 10:
            try:
                iso_df = self.ml_detector.isolation_forest_detect(df_clean, eval_features)
            except Exception:
                pass

        # 3. Deep Learning Autoencoder Detection
        ae_df = None
        if include_deep_learning and len(eval_features) >= 1 and len(df_clean) >= 15:
            try:
                ae_df = self.autoencoder.fit_detect(df_clean, eval_features)
            except Exception:
                pass

        # 4. Change-Point Detection on Primary Series
        change_points = []
        try:
            change_points = self.change_point_detector.detect_change_points(
                series=actual_series,
                timestamps=timestamps
            )
        except Exception:
            pass

        # 5. Sensor Threshold Validation & Aggregation
        n_rows = len(df_clean)
        for i in range(n_rows):
            is_anom = False
            scores = []
            reasons = []

            # Check residual deviation
            if res_df["is_anomaly"].iloc[i]:
                is_anom = True
                scores.append(res_df["anomaly_score"].iloc[i])
                reasons.append(f"Forecast deviation {res_df['deviation'].iloc[i]:+.2f} exceeds dynamic envelope")

            # Check modified z-score
            if z_df["is_anomaly"].iloc[i]:
                is_anom = True
                scores.append(z_df["anomaly_score"].iloc[i])
                reasons.append(f"Modified Z-score {z_df['metric_score'].iloc[i]:.2f} exceeds threshold")

            # Check IQR
            if iqr_df["is_anomaly"].iloc[i]:
                is_anom = True
                scores.append(iqr_df["anomaly_score"].iloc[i])
                reasons.append("Value outside interquartile range bounds")

            # Check Isolation Forest
            if iso_df is not None and iso_df["is_anomaly"].iloc[i]:
                is_anom = True
                scores.append(iso_df["anomaly_score"].iloc[i])
                reasons.append(f"Multivariate Isolation Forest score {iso_df['anomaly_score'].iloc[i]:.2f}")

            # Check Autoencoder
            if ae_df is not None and ae_df["is_anomaly"].iloc[i]:
                is_anom = True
                scores.append(ae_df["anomaly_score"].iloc[i])
                reasons.append(f"Autoencoder reconstruction error ({ae_df['reconstruction_error'].iloc[i]:.4f}) exceeds threshold")

            # Sensor critical limits check
            val = float(actual_series.iloc[i])
            is_sensor_breach = False
            col_key = primary_col.lower()
            for sensor_name, (norm_min, norm_max, crit_val) in self.SENSOR_BOUNDS.items():
                if sensor_name in col_key:
                    if "voltage" in sensor_name and val < crit_val:
                        is_anom = True
                        is_sensor_breach = True
                        scores.append(1.0)
                        reasons.append(f"Critical voltage sag ({val:.1f}V < {crit_val}V)")
                    elif val > crit_val:
                        is_anom = True
                        is_sensor_breach = True
                        scores.append(1.0)
                        reasons.append(f"Critical {sensor_name} threshold breached ({val:.1f} > {crit_val})")
                    elif val < norm_min or val > norm_max:
                        is_anom = True
                        scores.append(0.6)
                        reasons.append(f"{sensor_name} operating outside normal range ({norm_min} - {norm_max})")

            if is_anom:
                composite_score = float(np.max(scores)) if scores else 0.5
                dev_ratio = float(res_df["residual_z"].iloc[i]) if not np.isnan(res_df["residual_z"].iloc[i]) else 0.0
                severity = self.classify_severity(composite_score, dev_ratio, is_sensor_breach)

                explanation_text = "; ".join(reasons)
                recommendation = "Investigate immediately; alert on-call engineer." if severity in ["CRITICAL", "HIGH"] else "Monitor metric trend during next review cycle."

                anomalies_list.append({
                    "id": str(uuid.uuid4())[:8],
                    "index": i,
                    "timestamp": str(timestamps.iloc[i]),
                    "metric_name": primary_col,
                    "actual": round(val, 2),
                    "expected": round(float(res_df["expected"].iloc[i]), 2),
                    "deviation": round(float(res_df["deviation"].iloc[i]), 2),
                    "anomaly_score": round(composite_score, 3),
                    "severity": severity,
                    "detection_methods": list(set([r.split()[0] for r in reasons])),
                    "explanation": explanation_text,
                    "recommendation": recommendation
                })

        # Sort anomalies by severity rank then timestamp
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        anomalies_list.sort(key=lambda x: (severity_rank.get(x["severity"], 4), -x["anomaly_score"]))

        # Breakdown counts
        summary = {
            "total_observations": n_rows,
            "total_anomalies": len(anomalies_list),
            "critical_count": sum(1 for a in anomalies_list if a["severity"] == "CRITICAL"),
            "high_count": sum(1 for a in anomalies_list if a["severity"] == "HIGH"),
            "medium_count": sum(1 for a in anomalies_list if a["severity"] == "MEDIUM"),
            "low_count": sum(1 for a in anomalies_list if a["severity"] == "LOW"),
            "anomaly_percentage": round((len(anomalies_list) / max(1, n_rows)) * 100, 2),
            "change_points_count": len(change_points)
        }

        # Timeline visualization points
        timeline_points = []
        anomaly_index_map = {a["index"]: a for a in anomalies_list}
        for idx in range(n_rows):
            point = {
                "timestamp": str(timestamps.iloc[idx]),
                "actual": round(float(actual_series.iloc[idx]), 2),
                "expected": round(float(res_df["expected"].iloc[idx]), 2),
                "is_anomaly": idx in anomaly_index_map
            }
            if idx in anomaly_index_map:
                point["severity"] = anomaly_index_map[idx]["severity"]
                point["score"] = anomaly_index_map[idx]["anomaly_score"]
            timeline_points.append(point)

        return {
            "dataset_summary": summary,
            "anomalies": anomalies_list,
            "change_points": change_points,
            "timeline": timeline_points,
            "analyzed_metric": primary_col,
            "feature_columns": eval_features
        }
