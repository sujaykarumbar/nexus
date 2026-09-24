from typing import List, Dict, Any
import pandas as pd
import numpy as np


class VisualizationEngine:
    """Generates declarative JSON visualization specifications for frontend charts."""

    @classmethod
    def generate_chart_specs(cls, df: pd.DataFrame, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate structured chart specifications based on inferred feature relationships."""
        specs = []

        # 1. Correlation Matrix Heatmap
        corr_data = eda_results.get("correlations", {})
        if corr_data.get("matrix"):
            specs.append({
                "id": "correlation_heatmap",
                "type": "heatmap",
                "title": "Pearson Feature Correlation Matrix",
                "description": "Pairwise linear relationship coefficients between numerical features.",
                "x_axis": "features",
                "y_axis": "features",
                "data": corr_data["matrix"],
                "columns": corr_data["columns"]
            })

        # 2. Key Numerical Feature Distribution Histograms
        distributions = eda_results.get("distributions", {})
        for col, dist in list(distributions.items())[:4]:
            if dist["type"] == "numerical":
                specs.append({
                    "id": f"dist_{col}",
                    "type": "histogram",
                    "title": f"Distribution of '{col}'",
                    "description": f"Frequency distribution across binned ranges for {col}.",
                    "x_key": "range",
                    "y_key": "count",
                    "data": dist["bins"],
                    "color": "#38BDF8"
                })
            elif dist["type"] == "categorical":
                specs.append({
                    "id": f"bar_{col}",
                    "type": "bar",
                    "title": f"Category Breakdown: '{col}'",
                    "description": f"Most frequent categories in {col}.",
                    "x_key": "label",
                    "y_key": "count",
                    "data": dist["categories"],
                    "color": "#818CF8"
                })

        # 3. Scatter Plot for Strongest Correlation Pair
        strong_pairs = corr_data.get("strong_pairs", [])
        if strong_pairs:
            top_pair = strong_pairs[0]
            feat_a, feat_b = top_pair["feature_a"], top_pair["feature_b"]
            
            # Sample up to 100 points for scatter rendering
            clean_df = df[[feat_a, feat_b]].dropna()
            sample_df = clean_df.sample(min(100, len(clean_df)), random_state=42)
            scatter_points = [
                {"x": round(float(r[feat_a]), 2), "y": round(float(r[feat_b]), 2)}
                for _, r in sample_df.iterrows()
            ]

            specs.append({
                "id": f"scatter_{feat_a}_{feat_b}",
                "type": "scatter",
                "title": f"Bivariate Relationship: {feat_a} vs {feat_b}",
                "description": f"Scatter plot demonstrating correlation (r = {top_pair['correlation']}).",
                "x_key": "x",
                "y_key": "y",
                "x_label": feat_a,
                "y_label": feat_b,
                "data": scatter_points,
                "color": "#34D399"
            })

        return specs
