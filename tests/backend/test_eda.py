import pandas as pd
from services.eda.eda_engine import EDAEngine
from services.insights.insight_generator import InsightGenerator
from services.insights.verifier import ClaimVerifier
from services.visualization.vis_engine import VisualizationEngine
from services.quality.quality_engine import DataQualityEngine


def test_eda_correlations_and_insights():
    df = pd.DataFrame({
        "revenue": [100, 200, 300, 400, 500],
        "ad_spend": [10, 20, 30, 40, 50],  # perfect correlation r = 1.0
        "satisfaction": [5, 4, 3, 2, 1]     # perfect negative correlation r = -1.0
    })

    eda_res = EDAEngine.run_full_eda(df)
    corrs = eda_res["correlations"]
    
    assert len(corrs["strong_pairs"]) >= 2
    assert abs(corrs["strong_pairs"][0]["correlation"]) == 1.0

    quality_res = DataQualityEngine.evaluate_quality(df)
    insights = InsightGenerator.generate_insights(df, eda_res, quality_res)
    assert len(insights) >= 1

    audited = ClaimVerifier.audit_all_insights(insights, {"correlations": corrs, "quality": quality_res})
    for item in audited:
        assert item["verification_status"] == "SUPPORTED"

    specs = VisualizationEngine.generate_chart_specs(df, eda_res)
    assert len(specs) >= 2
    assert any(s["type"] == "heatmap" for s in specs)
