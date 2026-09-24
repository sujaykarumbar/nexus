"""
NEXUS Data Agent
Specialized in automated schema inference, data profiling,
deterministic quality audits, missingness strategies, and hygiene recommendations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from services.profiling.profiler import DataProfiler
from services.quality.quality_engine import DataQualityEngine


class DataAgent:
    """
    Autonomous Data Agent providing dataset hygiene, profiling, and quality scoring.
    """

    def __init__(self):
        self.profiler = DataProfiler()
        self.quality_engine = DataQualityEngine()

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Compute full column profiles, inferred types, and memory metrics."""
        return self.profiler.profile_dataset(df)

    def audit_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Compute rigorous quality score (0-100), missingness maps, and outlier statistics."""
        return self.quality_engine.evaluate_quality(df)

    def run_hygiene_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tool: Comprehensive hygiene audit returning quality score, critical issues,
        and concrete remediation actions.
        """
        profile = self.profile_dataset(df)
        quality = self.audit_quality(df)

        overall_score = float(quality.get("overall_score") or quality.get("score", 100.0))
        grade = quality.get("grade", "A")
        issues = quality.get("issues", [])
        recommendations = quality.get("recommendations", [])

        # Categorize readiness for downstream modeling
        is_ready = overall_score >= 60.0
        readiness_status = "READY" if overall_score >= 80.0 else ("WARNING" if overall_score >= 60.0 else "UNFIT")

        if isinstance(issues, list):
            issues_slice = issues[:10]
            issues_count = len(issues)
        elif isinstance(issues, dict):
            issues_count = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in issues.values())
            issues_slice = issues
        else:
            issues_count = 0
            issues_slice = []

        return {
            "overall_score": overall_score,
            "grade": grade,
            "readiness_status": readiness_status,
            "is_ready_for_modeling": is_ready,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues_count": issues_count,
            "issues": issues_slice,
            "recommendations": recommendations,
            "summary": f"Dataset hygiene score: {overall_score}/100 (Grade {grade}). {issues_count} data quality issues detected."
        }
