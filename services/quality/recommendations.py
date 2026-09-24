from typing import List, Dict, Any


class CleaningRecommendationEngine:
    """Generates deterministic, non-destructive data cleaning and preprocessing recommendations."""

    @classmethod
    def generate_recommendations(
        cls,
        missing_issues: List[Dict[str, Any]],
        constant_columns: List[str],
        duplicate_rows_count: int,
        outlier_issues: List[Dict[str, Any]],
        skewed_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate structured data cleaning recommendations."""
        recommendations = []

        # 1. Missing Value Imputation Recommendations
        for issue in missing_issues:
            col = issue["column"]
            pct = issue["percentage"]
            dtype = issue["inferred_type"]

            if pct > 60.0:
                recommendations.append({
                    "column": col,
                    "issue_type": "HIGH_MISSINGNESS",
                    "severity": "HIGH",
                    "action": "DROP_COLUMN",
                    "recommendation": f"Column '{col}' has {pct}% missing values. Consider dropping this column to prevent distortion."
                })
            elif dtype == "NUMERICAL":
                recommendations.append({
                    "column": col,
                    "issue_type": "NUMERICAL_MISSINGNESS",
                    "severity": "MEDIUM",
                    "action": "MEDIAN_IMPUTE",
                    "recommendation": f"Impute {pct}% missing values in '{col}' using the median to maintain robustness against outliers."
                })
            elif dtype in ["CATEGORICAL", "TEXT"]:
                recommendations.append({
                    "column": col,
                    "issue_type": "CATEGORICAL_MISSINGNESS",
                    "severity": "LOW",
                    "action": "MODE_IMPUTE",
                    "recommendation": f"Impute missing values in '{col}' with the mode category or assign a dedicated 'Unknown' category token."
                })

        # 2. Constant Column Removal
        for col in constant_columns:
            recommendations.append({
                "column": col,
                "issue_type": "ZERO_VARIANCE",
                "severity": "MEDIUM",
                "action": "DROP_COLUMN",
                "recommendation": f"Column '{col}' is constant (zero variance). Drop this column as it provides zero predictive information."
            })

        # 3. Duplicate Rows
        if duplicate_rows_count > 0:
            recommendations.append({
                "column": None,
                "issue_type": "DUPLICATE_RECORDS",
                "severity": "MEDIUM",
                "action": "DEDUPLICATE",
                "recommendation": f"Found {duplicate_rows_count} exact duplicate rows. Remove duplicates to avoid biasing model training."
            })

        # 4. Outlier Handling
        for out in outlier_issues:
            col = out["column"]
            out_pct = out["percentage"]
            if out_pct > 2.0:
                recommendations.append({
                    "column": col,
                    "issue_type": "NUMERICAL_OUTLIERS",
                    "severity": "LOW",
                    "action": "INVESTIGATE_AND_CLIP",
                    "recommendation": f"Column '{col}' contains {out['count']} outliers ({out_pct}%). Consider winsorizing or robust scaling during feature engineering."
                })

        return recommendations
