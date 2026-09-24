from typing import Dict, Any


class DataQualityScorer:
    """
    Transparent & Deterministic Data Quality Score Calculator (0-100).
    
    Formula:
      Quality Score = (0.35 * Completeness) + (0.25 * Uniqueness) + 
                      (0.20 * Validity) + (0.10 * Consistency) + (0.10 * OutlierScore)
    """

    WEIGHTS = {
        "completeness": 0.35,
        "uniqueness": 0.25,
        "validity": 0.20,
        "consistency": 0.10,
        "outlier_score": 0.10
    }

    @classmethod
    def calculate_score(
        cls,
        total_cells: int,
        total_nulls: int,
        total_rows: int,
        duplicate_rows: int,
        invalid_cells: int,
        constant_columns_count: int,
        total_columns: int,
        outlier_cells: int
    ) -> Dict[str, Any]:
        """Compute transparent quality score breakdown."""
        # 1. Completeness Score (0-100)
        null_ratio = (total_nulls / total_cells) if total_cells > 0 else 0.0
        completeness = max(0.0, 100.0 - (null_ratio * 100.0))

        # 2. Uniqueness Score (0-100)
        dup_ratio = (duplicate_rows / total_rows) if total_rows > 0 else 0.0
        uniqueness = max(0.0, 100.0 - (dup_ratio * 100.0 * 2.0))  # penalize duplicates heavily

        # 3. Validity Score (0-100)
        invalid_ratio = (invalid_cells / total_cells) if total_cells > 0 else 0.0
        validity = max(0.0, 100.0 - (invalid_ratio * 100.0 * 3.0))

        # 4. Consistency Score (0-100)
        const_ratio = (constant_columns_count / total_columns) if total_columns > 0 else 0.0
        consistency = max(0.0, 100.0 - (const_ratio * 100.0 * 2.5))

        # 5. Outlier Score (0-100)
        outlier_ratio = (outlier_cells / total_cells) if total_cells > 0 else 0.0
        outlier_score = max(0.0, 100.0 - (outlier_ratio * 100.0 * 1.5))

        # Final Weighted Total
        final_score = (
            cls.WEIGHTS["completeness"] * completeness +
            cls.WEIGHTS["uniqueness"] * uniqueness +
            cls.WEIGHTS["validity"] * validity +
            cls.WEIGHTS["consistency"] * consistency +
            cls.WEIGHTS["outlier_score"] * outlier_score
        )

        return {
            "overall_score": round(final_score, 1),
            "grade": "A" if final_score >= 90 else ("B" if final_score >= 80 else ("C" if final_score >= 70 else "D")),
            "components": {
                "completeness": round(completeness, 1),
                "uniqueness": round(uniqueness, 1),
                "validity": round(validity, 1),
                "consistency": round(consistency, 1),
                "outlier_score": round(outlier_score, 1)
            },
            "weights": cls.WEIGHTS,
            "metrics": {
                "null_ratio_percent": round(null_ratio * 100, 2),
                "duplicate_ratio_percent": round(dup_ratio * 100, 2),
                "invalid_cells_count": invalid_cells,
                "constant_columns_count": constant_columns_count,
                "outlier_cells_count": outlier_cells
            }
        }
