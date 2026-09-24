"""
NEXUS Critic Agent (Mathematical Gatekeeper)
Strict mathematical evidence checking, metric fidelity auditing,
and anti-hallucination verification for all agent claims.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class CriticAgent:
    """
    Mathematical Gatekeeper Agent that strictly audits statements, numbers,
    and model metrics against ground-truth computational artifacts.
    """

    def __init__(self, numerical_tolerance: float = 0.03):
        """
        :param numerical_tolerance: Maximum allowable relative difference (3%)
        """
        self.numerical_tolerance = numerical_tolerance

    def audit_metric_claim(
        self,
        claim_text: str,
        metric_name: str,
        reported_val: float,
        computed_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Audit a specific numerical claim against ground truth metrics.
        """
        ground_truth = computed_metrics.get(metric_name)

        if ground_truth is None:
            # Check nested keys or case-insensitive
            for k, v in computed_metrics.items():
                if k.lower() == metric_name.lower() or metric_name.lower() in k.lower():
                    if isinstance(v, (int, float)):
                        ground_truth = v
                        break
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_k.lower() == metric_name.lower() and isinstance(sub_v, (int, float)):
                                ground_truth = sub_v
                                break

        if ground_truth is None:
            return {
                "claim": claim_text,
                "metric_name": metric_name,
                "reported_value": reported_val,
                "ground_truth_value": None,
                "verified": False,
                "status": "REJECTED",
                "delta": None,
                "confidence": 0.0,
                "reason": f"Ground-truth metric '{metric_name}' not found in computational artifacts."
            }

        gt_float = float(ground_truth)
        rep_float = float(reported_val)

        # Discrepancy calculation
        if abs(gt_float) < 1e-9:
            delta = abs(rep_float - gt_float)
        else:
            delta = abs(rep_float - gt_float) / (abs(gt_float) + 1e-9)

        if delta <= self.numerical_tolerance:
            status = "SUPPORTED"
            verified = True
            confidence = max(0.95, 1.0 - delta)
            reason = f"Verified: reported {rep_float} closely matches computed ground truth {gt_float} (delta: {delta * 100:.2f}%)."
        elif delta <= 0.10:
            status = "PARTIALLY_SUPPORTED"
            verified = True
            confidence = 0.70
            reason = f"Minor discrepancy: reported {rep_float} deviates from computed {gt_float} by {delta * 100:.2f}%."
        else:
            status = "REJECTED"
            verified = False
            confidence = 0.20
            reason = f"Mathematical violation: reported {rep_float} contradicts computed ground truth {gt_float} (delta: {delta * 100:.2f}%)."

        return {
            "claim": claim_text,
            "metric_name": metric_name,
            "reported_value": rep_float,
            "ground_truth_value": gt_float,
            "verified": verified,
            "status": status,
            "delta": round(float(delta), 4),
            "confidence": round(float(confidence), 3),
            "reason": reason
        }

    def verify_artifacts(
        self,
        claims: List[Dict[str, Any]],
        computed_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify an entire dossier of claims against computed artifacts.
        """
        audits = []
        supported_count = 0
        rejected_count = 0

        for item in claims:
            claim_text = item.get("claim", "")
            metric_name = item.get("metric_name", "")
            reported_val = item.get("reported_value", 0.0)

            audit = self.audit_metric_claim(
                claim_text=claim_text,
                metric_name=metric_name,
                reported_val=reported_val,
                computed_metrics=computed_metrics
            )
            audits.append(audit)

            if audit["status"] == "SUPPORTED":
                supported_count += 1
            elif audit["status"] == "REJECTED":
                rejected_count += 1

        total = len(audits)
        if total == 0:
            return {
                "status": "SUPPORTED",
                "audit_count": 0,
                "supported_count": 0,
                "rejected_count": 0,
                "overall_confidence": 1.0,
                "gatekeeper_verdict": "PASS",
                "audits": []
            }

        support_ratio = supported_count / total
        overall_conf = round(sum(a["confidence"] for a in audits) / total, 3)

        if rejected_count == 0 and support_ratio >= 0.8:
            status = "SUPPORTED"
            verdict = "PASS"
        elif rejected_count <= 1 and support_ratio >= 0.5:
            status = "PARTIALLY_SUPPORTED"
            verdict = "PASS_WITH_WARNINGS"
        else:
            status = "REJECTED"
            verdict = "REJECT"

        return {
            "status": status,
            "audit_count": total,
            "supported_count": supported_count,
            "rejected_count": rejected_count,
            "overall_confidence": overall_conf,
            "gatekeeper_verdict": verdict,
            "audits": audits
        }
