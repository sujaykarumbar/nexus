import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class TargetDetector:
    """Intelligent Target Variable Detector for Supervised Learning."""

    TARGET_NAME_KEYWORDS = [
        r'(?i)^target$', r'(?i)^label$', r'(?i)^class$', r'(?i)^outcome$',
        r'(?i)churn', r'(?i)default', r'(?i)fraud', r'(?i)converted',
        r'(?i)price', r'(?i)sales', r'(?i)revenue', r'(?i)profit',
        r'(?i)cost', r'(?i)demand', r'(?i)salary', r'(?i)status_flag'
    ]

    EXCLUDE_PATTERNS = [
        r'(?i)^id$', r'(?i)_id$', r'(?i)^id_', r'(?i)uuid', r'(?i)guid',
        r'(?i)created_at', r'(?i)updated_at', r'(?i)timestamp', r'(?i)date',
        r'(?i)email', r'(?i)phone', r'(?i)name'
    ]

    @classmethod
    def detect_candidate_targets(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze columns and return ranked target candidates with confidence and suggested task.
        """
        candidates = []
        total_rows = len(df)
        if total_rows == 0:
            return []

        for col in df.columns:
            col_str = str(col)
            # Skip pure ID/metadata columns
            if any(re.search(pat, col_str) for pat in cls.EXCLUDE_PATTERNS):
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            n_unique = series.nunique()
            score = 0.0
            reasons = []

            # 1. Name keyword match (up to +0.55)
            for pat in cls.TARGET_NAME_KEYWORDS:
                if re.search(pat, col_str):
                    score += 0.55
                    reasons.append(f"Column name matches standard target convention ('{pat}')")
                    break

            # 2. Position bias: target is frequently the last or second-to-last column (+0.15)
            col_idx = list(df.columns).index(col)
            if col_idx == len(df.columns) - 1:
                score += 0.15
                reasons.append("Located at the final column index")
            elif col_idx == len(df.columns) - 2:
                score += 0.05

            # 3. Cardinality analysis
            if n_unique == 2:
                score += 0.25
                suggested_type = "binary_classification"
                reasons.append(f"Binary cardinality ({n_unique} unique classes)")
            elif 2 < n_unique <= 20:
                score += 0.15
                suggested_type = "multiclass_classification"
                reasons.append(f"Low discrete cardinality ({n_unique} unique classes)")
            elif pd.api.types.is_numeric_dtype(series) and n_unique > 20:
                score += 0.20
                suggested_type = "regression"
                reasons.append(f"Continuous numerical values ({n_unique} unique levels)")
            else:
                suggested_type = "classification"

            confidence = min(0.99, max(0.20, round(score, 2)))

            candidates.append({
                "column": col_str,
                "column_name": col_str,
                "confidence": confidence,
                "confidence_score": confidence,
                "suggested_type": suggested_type,
                "recommended_problem_type": suggested_type,
                "unique_count": n_unique,
                "null_count": int(df[col].isna().sum()),
                "reasons": reasons,
                "sample_values": [str(x) for x in series.head(4).tolist()]
            })

        # Rank candidates by descending confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates

    detect_candidates = detect_candidate_targets
