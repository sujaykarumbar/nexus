import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from .target_detector import TargetDetector
from .problem_classifier import ProblemClassifier
from .leakage_detector import LeakageDetector
from .preprocessor import MLPreprocessor
from .baseline import BaselineModelRunner
from .models.classification import ClassificationModelFactory
from .models.regression import RegressionModelFactory
from .cross_validation import CrossValidationRunner
from .optimizer import HyperparameterOptimizer
from .evaluator import ModelEvaluator
from .explainability import ModelExplainer
from .model_card import ModelCardGenerator


class AutoMLTrainer:
    """Production Autonomous Machine Learning Engine."""

    MODEL_DIR = os.path.join(os.getcwd(), "data", "models")

    @classmethod
    def _ensure_model_dir(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

    @classmethod
    def run_automl_pipeline(
        cls,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: str = "Dataset",
        dataset_id: Optional[str] = None,
        project_id: Optional[str] = None,
        candidate_algorithms: Optional[List[str]] = None,
        excluded_columns: Optional[List[str]] = None,
        cv_splits: int = 5,
        optimize_hyperparameters: bool = False,
        optuna_trials: int = 10,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end automated machine learning pipeline with real computation:
        Leakage Check -> Preprocessing -> Baseline -> Multi-Model Benchmarking -> Optuna -> Explainability -> Model Card.
        """
        cls._ensure_model_dir()
        start_pipeline_time = time.time()

        if progress_callback:
            progress_callback(5.0, "Validating dataset and detecting problem type")

        # 1. Problem Classification
        problem_info = ProblemClassifier.classify_problem(df, target_column)
        is_classification = problem_info["is_classification"]

        # 2. Data Leakage Detection
        if progress_callback:
            progress_callback(12.0, "Auditing dataset for data leakage and identifier contamination")

        leakage_audit = LeakageDetector.audit_leakage(df, target_column)
        auto_drop_cols = set(excluded_columns or [])
        # Automatically exclude critical leakage risks (e.g. perfect correlation or derived columns)
        auto_drop_cols.update(leakage_audit["recommended_drop_columns"])

        # 3. Feature Identification & Split
        if progress_callback:
            progress_callback(20.0, "Partitioning train/test splits and building preprocessing pipeline")

        clean_df = df.dropna(subset=[target_column])
        X_raw = clean_df.drop(columns=[target_column])
        y_raw = clean_df[target_column]

        feature_types = MLPreprocessor.identify_feature_types(
            clean_df, 
            target_column=target_column, 
            excluded_columns=list(auto_drop_cols)
        )

        # Stratified split for classification, standard for regression
        stratify = y_raw if is_classification and problem_info["class_count"] >= 2 else None
        try:
            X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
                X_raw, y_raw, test_size=0.20, random_state=42, stratify=stratify
            )
        except Exception:
            # Fallback to unstratified if any class has too few instances
            X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
                X_raw, y_raw, test_size=0.20, random_state=42
            )

        # 4. Fit Preprocessor ONLY on X_train (Zero Leakage)
        preprocessor = MLPreprocessor.build_preprocessor(feature_types)
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)

        # Extract generated feature names
        try:
            transformed_feature_names = preprocessor.get_feature_names_out().tolist()
            # Clean up prefixes like 'num__', 'cat__' for readable display
            clean_feature_names = [
                f.split("__")[-1] for f in transformed_feature_names
            ]
        except Exception:
            clean_feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        label_encoder = None
        if is_classification:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train_s.astype(str))
            y_test = label_encoder.transform(y_test_s.astype(str))
        else:
            y_train = pd.to_numeric(y_train_s, errors="coerce").fillna(0.0).values
            y_test = pd.to_numeric(y_test_s, errors="coerce").fillna(0.0).values

        # 5. Baseline Benchmark
        if progress_callback:
            progress_callback(30.0, "Training and evaluating deterministic dummy baseline")

        baseline_result = BaselineModelRunner.train_and_evaluate_baseline(
            X_train, y_train, X_test, y_test, is_classification
        )
        baseline_primary = baseline_result["primary_metric_value"]

        # 6. Candidate Algorithm Selection
        if is_classification:
            supported = ClassificationModelFactory.get_supported_algorithms()
            default_candidates = ClassificationModelFactory.get_default_candidates(len(df))
            norm_func = ClassificationModelFactory.normalize_algorithm
        else:
            supported = RegressionModelFactory.get_supported_algorithms()
            default_candidates = RegressionModelFactory.get_default_candidates(len(df))
            norm_func = RegressionModelFactory.normalize_algorithm

        if candidate_algorithms:
            normalized_requested = [norm_func(a) for a in candidate_algorithms]
            algorithms_to_run = [a for a in normalized_requested if a in supported]
        else:
            algorithms_to_run = default_candidates

        if not algorithms_to_run:
            algorithms_to_run = default_candidates

        # 7. Model Training & Benchmarking Loop
        leaderboard = []
        trained_models = {}
        total_algos = len(algorithms_to_run)

        for idx, algo in enumerate(algorithms_to_run):
            step_pct = 35.0 + (idx / total_algos) * 40.0
            if progress_callback:
                progress_callback(step_pct, f"Training candidate model: {algo.replace('_', ' ').title()}")

            algo_start = time.time()
            best_params = {}
            optuna_log = None

            # Hyperparameter Tuning with Optuna if enabled
            if optimize_hyperparameters:
                try:
                    opt_res = HyperparameterOptimizer.optimize_model(
                        algo, is_classification, X_train, y_train, n_trials=optuna_trials
                    )
                    best_params = opt_res["best_params"]
                    optuna_log = opt_res["trials_summary"]
                except Exception:
                    pass

            # Instantiate model
            if is_classification:
                model_inst = ClassificationModelFactory.create_model(algo, best_params)
            else:
                model_inst = RegressionModelFactory.create_model(algo, best_params)

            # Fit on training data
            model_inst.fit(X_train, y_train)
            train_duration = round(time.time() - algo_start, 3)

            # Inference latency measurement
            inf_start = time.time()
            _ = model_inst.predict(X_test[:min(10, len(X_test))])
            inf_latency_ms = round((time.time() - inf_start) * 1000, 2)

            # 5-fold Cross-Validation on training data
            cv_results = CrossValidationRunner.evaluate_cv(
                model_inst, X_train, y_train, is_classification, n_splits=cv_splits
            )

            # Rigorous evaluation on holdout test set
            if is_classification:
                eval_matrix = ModelEvaluator.evaluate_classification(
                    model_inst, X_test, y_test, labels=problem_info.get("classes")
                )
            else:
                eval_matrix = ModelEvaluator.evaluate_regression(
                    model_inst, X_test, y_test
                )

            # Compute actual relative improvement over dummy baseline
            model_primary = eval_matrix["primary_metric_value"]
            higher_better = is_classification  # F1 higher is better, RMSE lower is better
            delta_improvement = BaselineModelRunner.calculate_improvement(
                baseline_primary, model_primary, eval_matrix["primary_metric_name"], higher_better
            )

            model_id = str(uuid.uuid4())
            trained_models[algo] = {
                "id": model_id,
                "model_instance": model_inst,
                "eval_matrix": eval_matrix,
                "cv_results": cv_results,
                "hyperparameters": best_params,
                "optuna_log": optuna_log
            }

            leaderboard.append({
                "model_id": model_id,
                "algorithm": algo,
                "display_name": algo.replace("_", " ").title(),
                "primary_metric_name": eval_matrix["primary_metric_name"],
                "primary_metric_value": model_primary,
                "metrics": eval_matrix["metrics"],
                "cv_summary": cv_results["summary"],
                "delta_improvement_pct": delta_improvement,
                "training_duration_seconds": train_duration,
                "inference_latency_ms": inf_latency_ms,
                "is_best": False
            })

        # 8. Model Selection (Highest F1 / Lowest RMSE)
        if is_classification:
            leaderboard.sort(key=lambda x: x["primary_metric_value"], reverse=True)
        else:
            leaderboard.sort(key=lambda x: x["primary_metric_value"], reverse=False)

        best_candidate = leaderboard[0]
        best_candidate["is_best"] = True
        best_algo = best_candidate["algorithm"]
        best_model_data = trained_models[best_algo]
        best_model_inst = best_model_data["model_instance"]

        # 9. Explainability & Feature Importance
        if progress_callback:
            progress_callback(85.0, "Computing feature importance and attribution metrics")

        feature_importances = ModelExplainer.compute_global_feature_importance(
            best_model_inst, clean_feature_names, X_test, y_test
        )

        # 10. Generate Model Card
        if progress_callback:
            progress_callback(92.0, "Compiling standardized Model Card documentation")

        model_card = ModelCardGenerator.generate_model_card(
            model_name=f"{dataset_name} — {best_candidate['display_name']}",
            version="v1.0",
            algorithm=best_candidate["display_name"],
            problem_type=problem_info["problem_type"],
            dataset_name=dataset_name,
            target_column=target_column,
            feature_names=clean_feature_names,
            metrics=best_candidate["metrics"],
            cv_scores=best_candidate["cv_summary"],
            hyperparameters=best_model_data["hyperparameters"],
            dataset_rows=len(df)
        )

        # 11. Serialize All Trained Pipelines to Disk
        for algo_name, data in trained_models.items():
            m_id = data["id"]
            m_path = os.path.join(cls.MODEL_DIR, f"{m_id}.joblib")
            pipe_art = {
                "model_id": m_id,
                "algorithm": algo_name,
                "problem_type": problem_info["problem_type"],
                "target_column": target_column,
                "preprocessor": preprocessor,
                "model": data["model_instance"],
                "feature_names": clean_feature_names,
                "active_features": feature_types["active_features"],
                "classes": problem_info.get("classes", []),
                "label_encoder": label_encoder,
                "created_at": time.time()
            }
            joblib.dump(pipe_art, m_path)
            data["artifact_path"] = m_path

        artifact_path = trained_models[best_algo]["artifact_path"]

        total_duration = round(time.time() - start_pipeline_time, 2)
        if progress_callback:
            progress_callback(100.0, "AutoML Pipeline completed successfully")

        # Compile full models list
        all_models_list = []
        for item in leaderboard:
            algo_key = item["algorithm"]
            t_data = trained_models.get(algo_key, {})
            all_models_list.append({
                "id": item["model_id"],
                "algorithm": algo_key,
                "display_name": item["display_name"],
                "primary_metric_name": item["primary_metric_name"],
                "primary_metric_value": item["primary_metric_value"],
                "metrics": item["metrics"],
                "cv_summary": item["cv_summary"],
                "delta_improvement_pct": item["delta_improvement_pct"],
                "confusion_matrix": t_data.get("eval_matrix", {}).get("confusion_matrix"),
                "roc_curve": t_data.get("eval_matrix", {}).get("roc_curve"),
                "actual_vs_pred": t_data.get("eval_matrix", {}).get("actual_vs_pred"),
                "hyperparameters": t_data.get("hyperparameters", {}),
                "feature_importance": feature_importances if item["is_best"] else ModelExplainer.compute_global_feature_importance(t_data["model_instance"], clean_feature_names, X_test, y_test),
                "feature_names": clean_feature_names,
                "model_card": model_card if item["is_best"] else ModelCardGenerator.generate_model_card(
                    model_name=f"{dataset_name} — {item['display_name']}",
                    version="v1.0",
                    algorithm=item["display_name"],
                    problem_type=problem_info["problem_type"],
                    dataset_name=dataset_name,
                    target_column=target_column,
                    feature_names=clean_feature_names,
                    metrics=item["metrics"],
                    cv_scores=item["cv_summary"],
                    hyperparameters=t_data.get("hyperparameters", {}),
                    dataset_rows=len(df)
                ),
                "artifact_path": t_data.get("artifact_path", artifact_path),
                "version": "v1.0",
                "lifecycle_stage": "STAGING" if item["is_best"] else "CANDIDATE",
                "is_best": item["is_best"]
            })

        return {
            "status": "COMPLETED",
            "dataset_id": dataset_id,
            "project_id": project_id,
            "target_column": target_column,
            "problem_type": problem_info["problem_type"],
            "problem_info": problem_info,
            "leakage_audit": leakage_audit,
            "baseline": baseline_result,
            "leaderboard": leaderboard,
            "all_models": all_models_list,
            "best_model": {
                "id": best_candidate["model_id"],
                "algorithm": best_algo,
                "display_name": best_candidate["display_name"],
                "primary_metric_name": best_candidate["primary_metric_name"],
                "primary_metric_value": best_candidate["primary_metric_value"],
                "metrics": best_candidate["metrics"],
                "cv_summary": best_candidate["cv_summary"],
                "delta_improvement_pct": best_candidate["delta_improvement_pct"],
                "confusion_matrix": best_model_data["eval_matrix"].get("confusion_matrix"),
                "roc_curve": best_model_data["eval_matrix"].get("roc_curve"),
                "actual_vs_pred": best_model_data["eval_matrix"].get("actual_vs_pred"),
                "hyperparameters": best_model_data["hyperparameters"],
                "feature_importance": feature_importances,
                "feature_names": clean_feature_names,
                "model_card": model_card,
                "artifact_path": artifact_path,
                "version": "v1.0",
                "lifecycle_stage": "STAGING"
            },
            "total_duration_seconds": total_duration
        }

    @classmethod
    def load_pipeline_artifact(cls, artifact_path: str) -> Dict[str, Any]:
        """Safely load trained model pipeline from disk."""
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Model artifact not found at '{artifact_path}'.")
        return joblib.load(artifact_path)

    @classmethod
    def predict_instance(
        cls, 
        artifact_path: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform real-time prediction on a single record with feature attribution."""
        artifact = cls.load_pipeline_artifact(artifact_path)
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        is_classification = "classification" in artifact["problem_type"]

        input_df = pd.DataFrame([input_data])
        X_trans = preprocessor.transform(input_df)

        prediction_val = model.predict(X_trans)[0]
        probabilities = None
        if is_classification and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_trans)[0]
                classes = artifact.get("classes", [])
                if len(classes) == len(probs):
                    probabilities = {str(classes[i]): round(float(probs[i]), 4) for i in range(len(probs))}
                else:
                    probabilities = {f"class_{i}": round(float(p), 4) for i, p in enumerate(probs)}
            except Exception:
                pass

        explanations = ModelExplainer.explain_instance_prediction(
            model, X_trans, feature_names
        )

        label_enc = artifact.get("label_encoder")
        if label_enc is not None:
            try:
                display_pred = label_enc.inverse_transform([prediction_val])[0]
                display_pred = display_pred.item() if hasattr(display_pred, "item") else display_pred
                # If it's a numeric string like "0" or "1", convert to int
                if isinstance(display_pred, str) and display_pred.isdigit():
                    display_pred = int(display_pred)
            except Exception:
                display_pred = int(prediction_val) if isinstance(prediction_val, (np.integer, int)) else (float(prediction_val) if isinstance(prediction_val, (np.floating, float)) else str(prediction_val))
        else:
            display_pred = int(prediction_val) if isinstance(prediction_val, (np.integer, int)) else (float(prediction_val) if isinstance(prediction_val, (np.floating, float)) else str(prediction_val))

        return {
            "prediction": display_pred,
            "probability": max(probabilities.values()) if probabilities else None,
            "probabilities": probabilities,
            "contributing_features": explanations
        }

    train = run_automl_pipeline
    predict = predict_instance
