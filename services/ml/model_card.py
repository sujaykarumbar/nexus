from datetime import datetime, timezone
from typing import Dict, Any, List


class ModelCardGenerator:
    """Generates Standardized Engineering Model Cards (JSON & Markdown)."""

    @classmethod
    def generate_model_card(
        cls,
        model_name: str,
        version: str,
        algorithm: str,
        problem_type: str,
        dataset_name: str,
        target_column: str,
        feature_names: List[str],
        metrics: Dict[str, Any],
        cv_scores: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        dataset_rows: int
    ) -> Dict[str, Any]:
        """Produce structured model card dictionary and formatted Markdown document."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        intended_use = (
            f"Supervised {problem_type.replace('_', ' ')} for predicting '{target_column}' "
            f"using {len(feature_names)} ingested features."
        )

        limitations = [
            f"Model performance is verified against {dataset_rows} historical records; distribution drift may degrade inference.",
            "Predictions should not be used as the sole deciding factor in critical high-stakes automation without human oversight.",
            "Features outside observed ranges in training distribution will experience higher uncertainty."
        ]

        markdown_doc = f"""# Model Card: {model_name} ({version})

## Model Details
* **Algorithm**: {algorithm}
* **Task Type**: {problem_type.replace('_', ' ').title()}
* **Target Variable**: `{target_column}`
* **Dataset Used**: `{dataset_name}` ({dataset_rows} samples)
* **Training Timestamp**: {now_str}
* **Version**: {version}

## Performance Benchmarks
| Metric | Holdout Evaluation Score | Cross-Validation Score (5-fold) |
| :--- | :--- | :--- |
"""
        for k, v in metrics.items():
            cv_val = cv_scores.get(k, {}).get("formatted", "N/A") if isinstance(cv_scores, dict) else "N/A"
            markdown_doc += f"| {k.upper()} | {v} | {cv_val} |\n"

        markdown_doc += f"""
## Primary Features ({len(feature_names)})
{', '.join([f'`{f}`' for f in feature_names[:12]])}{'...' if len(feature_names) > 12 else ''}

## Intended Use
{intended_use}

## Known Limitations & Governance
"""
        for lim in limitations:
            markdown_doc += f"* {lim}\n"

        return {
            "model_name": model_name,
            "version": version,
            "algorithm": algorithm,
            "problem_type": problem_type,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "features_count": len(feature_names),
            "features": feature_names,
            "metrics": metrics,
            "cv_scores": cv_scores,
            "hyperparameters": hyperparameters,
            "intended_use": intended_use,
            "limitations": limitations,
            "created_at": now_str,
            "markdown": markdown_doc
        }
