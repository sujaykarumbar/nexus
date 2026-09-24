"""
Tests for NEXUS Multi-Agent Swarm, Critic Gatekeeper, and Swarm Orchestrator
"""

import pytest
import pandas as pd
import numpy as np

from services.agents.data_agent import DataAgent
from services.agents.eda_agent import EDAAgent
from services.agents.critic_agent import CriticAgent
from services.agents.recommendation_agent import RecommendationAgent
from services.agents.report_agent import ReportAgent
from services.agents.swarm_orchestrator import SwarmOrchestrator


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 60
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "revenue": np.random.uniform(100, 500, n),
        "user_count": np.random.randint(10, 80, n),
        "churned": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })


def test_data_agent(sample_dataset):
    agent = DataAgent()
    profile = agent.profile_dataset(sample_dataset)
    assert profile["row_count"] == 60
    assert profile["column_count"] == 4

    quality = agent.audit_quality(sample_dataset)
    assert "score" in quality or "overall_score" in quality

    hygiene = agent.run_hygiene_assessment(sample_dataset)
    assert hygiene["readiness_status"] in ["READY", "WARNING", "UNFIT"]
    assert "summary" in hygiene


def test_eda_agent(sample_dataset):
    agent = EDAAgent()
    corr = agent.analyze_correlations(sample_dataset)
    assert "matrix" in corr

    dist = agent.analyze_distributions(sample_dataset)
    assert len(dist) > 0

    full = agent.run_full_eda(sample_dataset)
    assert "summary" in full
    assert "insights" in full


def test_critic_agent_verification():
    critic = CriticAgent(numerical_tolerance=0.03)

    computed_metrics = {
        "accuracy": 0.925,
        "f1_score": 0.880,
        "row_count": 1000,
        "anomaly_count": 14
    }

    # 1. Exact / Supported claim
    claim_supported = {
        "claim": "Model achieves an F1 score of 0.88",
        "metric_name": "f1_score",
        "reported_value": 0.880
    }
    res_sup = critic.audit_metric_claim(
        claim_text=claim_supported["claim"],
        metric_name=claim_supported["metric_name"],
        reported_val=claim_supported["reported_value"],
        computed_metrics=computed_metrics
    )
    assert res_sup["status"] == "SUPPORTED"
    assert res_sup["verified"] is True

    # 2. Minor discrepancy -> PARTIALLY_SUPPORTED
    res_partial = critic.audit_metric_claim(
        claim_text="Accuracy is around 96%",
        metric_name="accuracy",
        reported_val=0.965,
        computed_metrics=computed_metrics
    )
    assert res_partial["status"] in ["PARTIALLY_SUPPORTED", "REJECTED"]

    # 3. Blatant hallucination / fake metric -> REJECTED
    res_hallucinated = critic.audit_metric_claim(
        claim_text="Dataset contains 50,000 rows with 0 anomalies",
        metric_name="row_count",
        reported_val=50000,
        computed_metrics=computed_metrics
    )
    assert res_hallucinated["status"] == "REJECTED"
    assert res_hallucinated["verified"] is False
    assert res_hallucinated["delta"] > 0.5

    # 4. Unknown metric -> REJECTED
    res_missing = critic.audit_metric_claim(
        claim_text="Fabricated metric claims 99.9%",
        metric_name="quantum_advantage",
        reported_val=0.999,
        computed_metrics=computed_metrics
    )
    assert res_missing["status"] == "REJECTED"

    # Full dossier verification
    dossier_res = critic.verify_artifacts(
        claims=[claim_supported, res_hallucinated],
        computed_metrics=computed_metrics
    )
    assert dossier_res["audit_count"] == 2
    assert dossier_res["supported_count"] == 1
    assert dossier_res["rejected_count"] == 1
    assert dossier_res["gatekeeper_verdict"] in ["PASS_WITH_WARNINGS", "REJECT"]


def test_recommendation_and_report_agents():
    rec_agent = RecommendationAgent()
    rep_agent = ReportAgent()

    quality_data = {"overall_score": 88.5, "grade": "B+", "issues": [], "issues_count": 0, "total_rows": 500}
    ml_data = {
        "primary_metric": "roc_auc",
        "leaderboard": [{"model_name": "Random Forest", "cv_score": 0.912}]
    }
    forecast_data = {
        "best_model": "Holt-Winters",
        "future_forecast": [{"prediction": 120.0}, {"prediction": 145.0}]
    }

    recs = rec_agent.generate_recommendations(
        quality_results=quality_data,
        ml_results=ml_data,
        forecast_results=forecast_data
    )
    assert len(recs) >= 1
    assert "risk_level" in recs[0]
    assert "expected_impact" in recs[0]

    dossier = rep_agent.compile_dossier(
        project_name="E-Commerce Intelligence",
        dataset_name="transactions.csv",
        quality_data=quality_data,
        ml_data=ml_data,
        forecast_data=forecast_data,
        recommendations=recs
    )
    assert "# 📊 NEXUS Autonomous Analytical Dossier" in dossier
    assert "Random Forest" in dossier
    assert "transactions.csv" in dossier


def test_swarm_orchestrator_end_to_end(sample_dataset):
    orchestrator = SwarmOrchestrator()
    result = orchestrator.run_swarm(
        df=sample_dataset,
        project_id="p_test",
        dataset_id="ds_test",
        project_name="Test Project",
        dataset_name="test_data.csv",
        target_column="churned",
        time_column="timestamp",
        run_automl=True,
        run_forecasting=True,
        run_anomalies=True,
        run_rag=False
    )

    assert result["status"] in ["completed", "rejected_by_critic"]
    assert result["progress_percentage"] == 100.0
    assert len(result["trace"]) >= 5

    step_ids = [s["step_id"] for s in result["trace"]]
    assert "step_1_data_hygiene" in step_ids
    assert "step_2_statistical_discovery" in step_ids
    assert "step_3_automl_engineering" in step_ids
    assert "step_7_critic_verification" in step_ids
    assert "step_8_recommendation_and_dossier" in step_ids

    assert result["critic_verdict"] is not None
    assert len(result["recommendations"]) > 0
    assert len(result["report_markdown"]) > 100
