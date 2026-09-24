import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class LeakageDetector:
    """Pre-training Data Leakage and Target Contamination Auditor."""

    ID_PATTERNS = [
        r'(?i)^id$', r'(?i)_id$', r'(?i)^id_', r'(?i)uuid', r'(?i)guid',
        r'(?i)customer_id', r'(?i)user_id', r'(?i)order_id', r'(?i)ssn',
        r'(?i)account_number', r'(?i)phone', r'(?i)email'
    ]

    FUTURE_PATTERNS = [
        r'(?i)churn_date', r'(?i)cancellation_date', r'(?i)exit_interview',
        r'(?i)end_date', r'(?i)resolution_time', r'(?i)refund_amount'
    ]

    @classmethod
    def audit_leakage(cls, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Audit the dataset for data leakage risks relative to the chosen target column.
        Returns detected risks, severity, and recommendations.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is not present in dataframe.")

        warnings = []
        suspicious_features = []
        target_series = df[target_column]
        clean_target = target_series.dropna()
        target_is_numeric = pd.api.types.is_numeric_dtype(clean_target)

        feature_cols = [c for c in df.columns if c != target_column]

        for col in feature_cols:
            col_str = str(col)
            series = df[col]
            clean_feat = series.dropna()

            # 1. Target Duplicate / Derived Column Name
            target_norm = target_column.lower().strip()
            col_norm = col_str.lower().strip()
            if col_norm != target_norm and (
                col_norm.startswith(target_norm + "_") or 
                col_norm.endswith("_" + target_norm) or 
                col_norm == f"is_{target_norm}" or
                f"{target_norm}_flag" in col_norm or
                f"{target_norm}_status" in col_norm
            ):
                warnings.append({
                    "column": col_str,
                    "risk_type": "TARGET_DERIVED_FEATURE",
                    "severity": "CRITICAL",
                    "description": f"Feature '{col_str}' appears to be directly derived from target '{target_column}'.",
                    "recommendation": f"Exclude '{col_str}' from training to prevent artificial 100% predictive accuracy."
                })
                suspicious_features.append(col_str)
                continue

            # 2. Identifier as Feature
            if any(re.search(pat, col_str) for pat in cls.ID_PATTERNS):
                uniqueness_ratio = clean_feat.nunique() / len(clean_feat) if len(clean_feat) > 0 else 0
                if uniqueness_ratio > 0.85 or clean_feat.nunique() > 500:
                    warnings.append({
                        "column": col_str,
                        "risk_type": "HIGH_CARDINALITY_IDENTIFIER",
                        "severity": "HIGH",
                        "description": f"Feature '{col_str}' is an arbitrary identifier with {clean_feat.nunique()} unique values.",
                        "recommendation": f"Exclude '{col_str}' as identifiers cause overfitting and cannot generalize."
                    })
                    suspicious_features.append(col_str)
                    continue

            # 3. Future Information Feature
            if any(re.search(pat, col_str) for pat in cls.FUTURE_PATTERNS):
                warnings.append({
                    "column": col_str,
                    "risk_type": "FUTURE_INFORMATION_LEAK",
                    "severity": "HIGH",
                    "description": f"Feature '{col_str}' likely captures information generated AFTER the target event occurred.",
                    "recommendation": f"Exclude '{col_str}' to ensure features reflect historical state at decision time."
                })
                suspicious_features.append(col_str)

            # 4. Near-Perfect Correlation with Target (Numeric to Numeric or Numeric to Binary)
            if target_is_numeric and pd.api.types.is_numeric_dtype(clean_feat):
                common_idx = df[[col, target_column]].dropna().index
                if len(common_idx) >= 10:
                    try:
                        corr = float(df.loc[common_idx, col].corr(df.loc[common_idx, target_column]))
                        if not np.isnan(corr) and abs(corr) >= 0.98:
                            warnings.append({
                                "column": col_str,
                                "risk_type": "PERFECT_TARGET_CORRELATION",
                                "severity": "CRITICAL",
                                "correlation": round(corr, 4),
                                "description": f"Feature '{col_str}' has an extreme correlation ({round(corr, 3)}) with target '{target_column}'.",
                                "recommendation": f"Verify if '{col_str}' is a proxy or duplicate of the target. Highly recommend removal."
                            })
                            suspicious_features.append(col_str)
                    except Exception:
                        pass
            elif not target_is_numeric and pd.api.types.is_numeric_dtype(clean_feat):
                # Try encoding target to check correlation
                try:
                    encoded_target = pd.factorize(df[target_column])[0]
                    corr = float(pd.Series(encoded_target).corr(pd.to_numeric(df[col], errors='coerce')))
                    if not np.isnan(corr) and abs(corr) >= 0.98:
                        warnings.append({
                            "column": col_str,
                            "risk_type": "PERFECT_TARGET_CORRELATION",
                            "severity": "CRITICAL",
                            "correlation": round(corr, 4),
                            "description": f"Feature '{col_str}' has an extreme correlation ({round(corr, 3)}) with the categorical target.",
                            "recommendation": f"Column '{col_str}' is almost certainly a direct leak of the target label."
                        })
                        suspicious_features.append(col_str)
                except Exception:
                    pass

        has_critical_risk = any(w["severity"] == "CRITICAL" for w in warnings)

        return {
            "has_leakage_risk": len(warnings) > 0,
            "has_critical_risk": has_critical_risk,
            "target_column": target_column,
            "warning_count": len(warnings),
            "warnings": warnings,
            "recommended_drop_columns": list(set(suspicious_features))
        }

    audit = audit_leakage


DataLeakageDetector = LeakageDetector
