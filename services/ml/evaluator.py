from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class ModelEvaluator:
    """Rigorous Quantitative Multi-Metric Evaluation Engine."""

    @classmethod
    def evaluate_classification(
        cls,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute full classification evaluation matrix, curves, and confusion matrix."""
        y_pred = model.predict(X_test)
        unique_classes = np.unique(y_test)
        is_binary = len(unique_classes) <= 2

        # Basic scalar metrics
        acc = float(accuracy_score(y_test, y_pred))
        avg = "binary" if is_binary else "weighted"
        prec = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        rec = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))

        # Probability-based metrics (ROC-AUC, ROC Curve)
        roc_auc = None
        roc_points = []
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                if is_binary:
                    # Binary ROC-AUC
                    pos_prob = probs[:, 1]
                    roc_auc = float(roc_auc_score(y_test, pos_prob))
                    fpr, tpr, _ = roc_curve(y_test, pos_prob)
                    # Subsample curve points for clean JSON payload
                    step = max(1, len(fpr) // 30)
                    roc_points = [
                        {"fpr": round(float(fpr[i]), 3), "tpr": round(float(tpr[i]), 3)}
                        for i in range(0, len(fpr), step)
                    ]
                else:
                    roc_auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
            except Exception:
                pass

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        formatted_cm = {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in (labels or unique_classes)]
        }

        # Classification Report dict
        clf_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None
        }

        return {
            "primary_metric_name": "f1",
            "primary_metric_value": round(f1, 4),
            "metrics": metrics,
            "confusion_matrix": formatted_cm,
            "roc_curve": roc_points,
            "classification_report": clf_rep
        }

    @classmethod
    def evaluate_regression(
        cls,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compute full regression evaluation matrix, residuals, and prediction bounds."""
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        
        try:
            mape = float(mean_absolute_percentage_error(y_test, y_pred))
        except Exception:
            mape = None

        # Residuals analysis
        residuals = (y_test - y_pred).tolist()
        sample_size = min(50, len(y_test))
        actual_vs_pred = [
            {
                "actual": round(float(y_test[i]), 2),
                "predicted": round(float(y_pred[i]), 2),
                "residual": round(float(residuals[i]), 2)
            }
            for i in range(sample_size)
        ]

        metrics = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 4) if mape is not None else None
        }

        return {
            "primary_metric_name": "rmse",
            "primary_metric_value": round(rmse, 4),
            "metrics": metrics,
            "actual_vs_pred": actual_vs_pred,
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4)
        }
