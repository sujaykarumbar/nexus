from typing import Dict, Any, List


class ClaimVerifier:
    """Strict verification engine that audits analytical claims against ground-truth computed metrics."""

    @classmethod
    def verify_insight(cls, insight: Dict[str, Any], ground_truth_metrics: Dict[str, Any]) -> str:
        """
        Verify if an analytical claim is supported by underlying computational evidence.
        Returns: 'SUPPORTED', 'PARTIALLY_SUPPORTED', or 'UNSUPPORTED'.
        """
        evidence = insight.get("evidence", {})
        if not evidence:
            return "UNSUPPORTED"

        # Check correlation verification
        if "correlation_coefficient" in evidence:
            r_claimed = evidence["correlation_coefficient"]
            # Verify that claimed correlation matches truth within tolerance
            truth_corrs = ground_truth_metrics.get("correlations", {}).get("matrix", {})
            found = False
            for col_a, row_dict in truth_corrs.items():
                for col_b, r_actual in row_dict.items():
                    if abs(r_actual - r_claimed) < 0.01:
                        found = True
                        break
                if found:
                    break
            if not found:
                return "UNSUPPORTED"

        # Check quality score verification
        if "quality_score" in evidence:
            score_claimed = evidence["quality_score"]
            actual_score = ground_truth_metrics.get("quality", {}).get("score")
            if actual_score is not None and abs(actual_score - score_claimed) > 0.5:
                return "UNSUPPORTED"

        return "SUPPORTED"

    @classmethod
    def audit_all_insights(cls, insights: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Audit all insights and attach verified status."""
        audited = []
        for item in insights:
            status = cls.verify_insight(item, ground_truth)
            item_copy = dict(item)
            item_copy["verification_status"] = status
            audited.append(item_copy)
        return audited
