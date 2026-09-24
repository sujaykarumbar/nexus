"""NEXUS ML Engineer Agent.

Orchestrates automated target detection, problem classification, leakage audits,
baseline benchmarks, multi-model training with Optuna optimization,
model evaluation, explainability, and model card generation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from services.ml.target_detector import TargetDetector
from services.ml.problem_classifier import ProblemClassifier
from services.ml.leakage_detector import DataLeakageDetector
from services.ml.trainer import AutoMLTrainer
from services.ml.explainability import ExplainabilityEngine
from services.ml.model_card import ModelCardGenerator


class MLEngineerAgent:
    """Autonomous ML Engineer Agent capable of running full AutoML workflows."""

    def __init__(self):
        self.target_detector = TargetDetector()
        self.problem_classifier = ProblemClassifier()
        self.leakage_detector = DataLeakageDetector()
        self.trainer = AutoMLTrainer()
        self.explainability = ExplainabilityEngine()
        self.model_card_generator = ModelCardGenerator()

    def suggest_targets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest candidate target columns sorted by confidence."""
        return self.target_detector.detect_candidates(df)

    def classify_problem(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Classify ML task type (binary, multiclass, regression)."""
        return self.problem_classifier.classify(df, target_column)

    def audit_leakage(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Audit dataset for potential target leakage or identifier anomalies."""
        return self.leakage_detector.audit(df, target_column)

    def train_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: Optional[str] = None,
        models_to_train: Optional[List[str]] = None,
        cv_folds: int = 5,
        optimize_hyperparams: bool = True,
        optuna_trials: int = 10,
        primary_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full AutoML pipeline returning leaderboard, metrics, and best model."""
        return self.trainer.train(
            df=df,
            target_column=target_column,
            problem_type=problem_type,
            models_to_train=models_to_train,
            cv_folds=cv_folds,
            optimize_hyperparams=optimize_hyperparams,
            optuna_trials=optuna_trials,
            primary_metric=primary_metric,
        )

    def explain_instance(
        self,
        trained_pipeline: Any,
        feature_names: List[str],
        instance_dict: Dict[str, Any],
        preprocessor: Any,
    ) -> Dict[str, Any]:
        """Explain an individual inference prediction via feature contributions."""
        df_row = pd.DataFrame([instance_dict])
        transformed_row = preprocessor.transform(df_row)
        transformed_names = preprocessor.get_feature_names()
        prediction = trained_pipeline.predict(transformed_row)[0]
        prediction_proba = None
        if hasattr(trained_pipeline, "predict_proba"):
            try:
                proba = trained_pipeline.predict_proba(transformed_row)[0]
                prediction_proba = proba.tolist()
            except Exception:
                pass

        contributions = self.explainability.explain_prediction(
            trained_pipeline, transformed_row, transformed_names
        )

        return {
            "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
            "prediction_proba": prediction_proba,
            "contributions": contributions,
        }

    def generate_model_card(
        self,
        model_name: str,
        problem_type: str,
        target_column: str,
        metrics: Dict[str, Any],
        features: List[str],
        train_samples: int,
        test_samples: int,
        baseline_comparison: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON and Markdown model card."""
        return self.model_card_generator.generate(
            model_name=model_name,
            problem_type=problem_type,
            target_column=target_column,
            metrics=metrics,
            features=features,
            train_samples=train_samples,
            test_samples=test_samples,
            baseline_comparison=baseline_comparison,
        )
