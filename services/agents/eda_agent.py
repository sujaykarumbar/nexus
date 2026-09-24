"""
NEXUS Data Scientist / EDA Agent
Automated statistical distributions, correlation maps, feature co-dependencies,
hypothesis testing, and evidence-grounded analytical discovery.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from services.eda.eda_engine import EDAEngine
from services.insights.insight_generator import InsightGenerator


class EDAAgent:
    """
    Autonomous Data Scientist Agent providing statistical discovery and automated EDA.
    """

    def __init__(self):
        self.eda_engine = EDAEngine()
        self.insight_generator = InsightGenerator()

    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Compute correlation matrix, detect multi-collinearity and strong feature pairs."""
        return self.eda_engine.calculate_correlations(df)

    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Generate binned distributions, skewness metrics, and category frequencies."""
        return self.eda_engine.analyze_distributions(df)

    def generate_statistical_insights(
        self,
        df: pd.DataFrame,
        quality_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tool: Synthesize evidence-backed analytical hypotheses and distribution insights.
        """
        correlations = self.analyze_correlations(df)
        distributions = self.analyze_distributions(df)
        eda_results = {
            "correlations": correlations,
            "distributions": distributions
        }

        insights = self.insight_generator.generate_insights(
            df=df,
            eda_results=eda_results,
            quality_results=quality_results or {}
        )

        return insights

    def run_full_eda(
        self,
        df: pd.DataFrame,
        quality_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete exploratory data analysis returning correlations, distributions,
        and statistical discoveries.
        """
        correlations = self.analyze_correlations(df)
        distributions = self.analyze_distributions(df)
        insights = self.generate_statistical_insights(df, quality_results=quality_results)

        return {
            "correlations": correlations,
            "distributions": distributions,
            "insights": insights,
            "strong_pairs_count": len(correlations.get("strong_pairs", [])),
            "summary": f"Identified {len(correlations.get('strong_pairs', []))} strong correlations and generated {len(insights)} statistical insights."
        }
