import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from services.profiling.schema_detector import SchemaDetector
from .scoring import DataQualityScorer
from .recommendations import CleaningRecommendationEngine


class DataQualityEngine:
    """Production Data Quality Analysis Engine."""

    EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

    @classmethod
    def evaluate_quality(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform full multi-dimensional quality audit on dataset."""
        total_rows = len(df)
        total_columns = len(df.columns)
        total_cells = total_rows * total_columns if (total_rows * total_columns) > 0 else 1
        
        schema_info = SchemaDetector.detect_schema(df)
        
        # 1. Missingness Analysis
        missing_issues = []
        total_nulls = 0
        for col, col_info in schema_info["columns"].items():
            null_count = col_info["null_count"]
            null_pct = col_info["null_percentage"]
            total_nulls += null_count
            
            if null_count > 0:
                severity = "CRITICAL" if null_pct > 30 else ("HIGH" if null_pct > 10 else ("MEDIUM" if null_pct > 2 else "LOW"))
                missing_issues.append({
                    "column": col,
                    "count": null_count,
                    "percentage": null_pct,
                    "severity": severity,
                    "inferred_type": col_info["inferred_type"]
                })

        # 2. Uniqueness & Duplicate Rows
        duplicate_rows = int(df.duplicated().sum())
        id_cols = [
            col for col, col_info in schema_info["columns"].items() 
            if col_info["inferred_type"] == "IDENTIFIER" or str(col).lower() in ["id", "uuid", "guid"]
        ]
        if duplicate_rows == 0 and id_cols:
            duplicate_rows = max(int(df[c].duplicated().sum()) for c in id_cols)
        duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

        # 3. Constant Columns
        constant_columns = []
        near_constant_columns = []
        for col in df.columns:
            clean = df[col].dropna()
            if len(clean) > 0:
                nunique = clean.nunique()
                if nunique == 1:
                    constant_columns.append(col)
                elif nunique > 1:
                    top_freq_pct = clean.value_counts(normalize=True).iloc[0] * 100
                    if top_freq_pct >= 99.0:
                        near_constant_columns.append(col)

        # 4. Outlier Analysis (IQR & Z-Score)
        outlier_issues = []
        total_outlier_cells = 0
        skewed_columns = []

        for col, col_info in schema_info["columns"].items():
            if col_info["inferred_type"] == "NUMERICAL" or pd.api.types.is_numeric_dtype(df[col]):
                num_series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(num_series) >= 4:
                    q25 = num_series.quantile(0.25)
                    q75 = num_series.quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    iqr_outliers = num_series[(num_series < lower_bound) | (num_series > upper_bound)]
                    out_count = len(iqr_outliers)
                    total_outlier_cells += out_count

                    if out_count > 0:
                        out_pct = round((out_count / total_rows * 100), 2)
                        outlier_issues.append({
                            "column": col,
                            "count": out_count,
                            "percentage": out_pct,
                            "lower_bound": round(float(lower_bound), 3),
                            "upper_bound": round(float(upper_bound), 3),
                            "method": "IQR_1.5"
                        })

                    # Skewness check
                    skew = num_series.skew()
                    if abs(skew) > 1.0:
                        skewed_columns.append({
                            "column": col,
                            "skewness": round(float(skew), 2),
                            "direction": "right" if skew > 0 else "left"
                        })

        # 5. Invalid Values Check
        invalid_cells = 0
        invalid_issues = []
        for col, col_info in schema_info["columns"].items():
            col_lower = str(col).lower()
            clean = df[col].dropna()
            
            # Age check: negative age or age > 130
            if "age" in col_lower and (col_info["inferred_type"] == "NUMERICAL" or pd.api.types.is_numeric_dtype(df[col])):
                num_age = pd.to_numeric(clean, errors="coerce")
                neg_ages = int((num_age < 0).sum() + (num_age > 130).sum())
                if neg_ages > 0:
                    invalid_cells += neg_ages
                    invalid_issues.append({
                        "column": col,
                        "issue": f"Detected {neg_ages} implausible age values (< 0 or > 130)."
                    })
            
            # Email check
            if "email" in col_lower and col_info["inferred_type"] in ["IDENTIFIER", "TEXT", "CATEGORICAL"]:
                str_emails = clean.astype(str)
                malformed_count = sum(not bool(re.match(cls.EMAIL_REGEX, email.strip())) for email in str_emails)
                if malformed_count > 0:
                    invalid_cells += malformed_count
                    invalid_issues.append({
                        "column": col,
                        "issue": f"Detected {malformed_count} malformed email addresses."
                    })

        # Calculate Deterministic Score
        score_breakdown = DataQualityScorer.calculate_score(
            total_cells=total_cells,
            total_nulls=total_nulls,
            total_rows=total_rows,
            duplicate_rows=duplicate_rows,
            invalid_cells=invalid_cells,
            constant_columns_count=len(constant_columns),
            total_columns=total_columns,
            outlier_cells=total_outlier_cells
        )

        # Generate Recommendations
        recommendations = CleaningRecommendationEngine.generate_recommendations(
            missing_issues=missing_issues,
            constant_columns=constant_columns,
            duplicate_rows_count=duplicate_rows,
            outlier_issues=outlier_issues,
            skewed_columns=skewed_columns
        )

        return {
            "score": score_breakdown["overall_score"],
            "grade": score_breakdown["grade"],
            "score_breakdown": score_breakdown,
            "issues": {
                "missing_values": missing_issues,
                "duplicates": {
                    "count": duplicate_rows,
                    "percentage": duplicate_pct
                },
                "constant_columns": constant_columns,
                "near_constant_columns": near_constant_columns,
                "outliers": outlier_issues,
                "invalid_values": invalid_issues
            },
            "recommendations": recommendations
        }
