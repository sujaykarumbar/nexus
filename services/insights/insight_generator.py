from typing import List, Dict, Any
import pandas as pd
import numpy as np
from services.profiling.schema_detector import SchemaDetector


class InsightGenerator:
    """Deterministic, mathematically grounded analytical insight generator."""

    @classmethod
    def generate_insights(cls, df: pd.DataFrame, eda_results: Dict[str, Any], quality_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate structured analytical insights with underlying evidence metrics."""
        insights = []
        total_rows = len(df)
        schema_info = SchemaDetector.detect_schema(df)

        # 1. Correlation Insights
        strong_pairs = eda_results.get("correlations", {}).get("strong_pairs", [])
        for pair in strong_pairs[:3]:
            r_val = pair["correlation"]
            feat_a = pair["feature_a"]
            feat_b = pair["feature_b"]
            insights.append({
                "category": "CORRELATION",
                "title": f"Strong Co-dependency: {feat_a} ↔ {feat_b}",
                "description": f"Features '{feat_a}' and '{feat_b}' exhibit a {pair['relationship'].lower()} correlation (r = {r_val}). Changes in one feature are strongly coupled with the other.",
                "importance": "HIGH" if abs(r_val) >= 0.75 else "MEDIUM",
                "evidence": {
                    "correlation_coefficient": r_val,
                    "sample_size": total_rows
                },
                "verification_status": "SUPPORTED"
            })

        # 2. Skewness & Concentration Insights for Numerical Columns
        for col, col_info in schema_info["columns"].items():
            if col_info["inferred_type"] == "NUMERICAL":
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) >= 10:
                    skew = float(series.skew())
                    if skew > 1.5 and series.min() >= 0 and series.sum() > 0:
                        top_5_pct_idx = int(max(1, len(series) * 0.05))
                        top_5_sum = float(series.nlargest(top_5_pct_idx).sum())
                        total_sum = float(series.sum())
                        concentration_pct = round((top_5_sum / total_sum * 100), 1)

                        insights.append({
                            "category": "DISTRIBUTION",
                            "title": f"High Concentration in '{col}'",
                            "description": f"Feature '{col}' is heavily right-skewed (skewness = {round(skew, 2)}). The top 5% of records represent {concentration_pct}% of the total volume.",
                            "importance": "HIGH" if concentration_pct >= 30 else "MEDIUM",
                            "evidence": {
                                "skewness": round(skew, 2),
                                "top_5_percent_concentration": concentration_pct
                            },
                            "verification_status": "SUPPORTED"
                        })

        # 3. Class Imbalance in Binary / Categorical columns
        for col, col_info in schema_info["columns"].items():
            if col_info["inferred_type"] in ["BOOLEAN", "CATEGORICAL"] and col_info["unique_count"] == 2:
                clean = df[col].dropna()
                val_counts = clean.value_counts(normalize=True)
                dominant_pct = float(val_counts.iloc[0] * 100)
                minority_pct = float(val_counts.iloc[1] * 100)
                
                if dominant_pct >= 75.0:
                    insights.append({
                        "category": "IMBALANCE",
                        "title": f"Class Imbalance in '{col}'",
                        "description": f"Binary feature '{col}' exhibits significant class imbalance with {round(dominant_pct, 1)}% vs {round(minority_pct, 1)}%. Stratified sampling and PR-AUC optimization recommended.",
                        "importance": "HIGH",
                        "evidence": {
                            "dominant_class": str(val_counts.index[0]),
                            "dominant_ratio": round(dominant_pct, 1),
                            "minority_ratio": round(minority_pct, 1)
                        },
                        "verification_status": "SUPPORTED"
                    })

        # 4. Data Quality Warnings
        score = quality_results.get("score", 100.0)
        if score < 85.0:
            insights.append({
                "category": "DATA_QUALITY",
                "title": "Data Hygiene Optimization Required",
                "description": f"Dataset quality score is {score}/100. Key anomalies include missing values and potential outlier records.",
                "importance": "MEDIUM",
                "evidence": {
                    "quality_score": score,
                    "grade": quality_results.get("grade", "B")
                },
                "verification_status": "SUPPORTED"
            })

        return insights
