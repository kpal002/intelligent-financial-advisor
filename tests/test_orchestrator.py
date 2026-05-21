"""
Unit tests for the LangGraph orchestrator.

Node functions are tested in isolation with mocked tools so no ML models
or Claude API calls are made — tests run in milliseconds.

The integration test (test_graph_invoke_runs_all_five_nodes) mocks all
network-touching code and verifies the full execution trace.
"""

import operator
from unittest.mock import MagicMock, patch

from src.llm.orchestrator import (
    AgentState,
    FinancialAdvisorGraph,
    market_research_agent,
    recommendation_agent,
    risk_analysis_agent,
    synthesize_with_llm,
    validate_recommendation,
)


# ============================================================================
# TEST FIXTURES / HELPERS
# ============================================================================

def make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState dict, with optional overrides."""
    base: AgentState = {
        "user_query":          "Should I rebalance?",
        "portfolio_symbols":   ["AAPL", "MSFT"],
        "current_allocation":  {"AAPL": 0.6, "MSFT": 0.4},
        "market_data":         None,
        "risk_metrics":        None,
        "recommendations":     None,
        "final_advice":        None,
        "confidence_score":    None,
        "timestamp":           "2024-01-01T00:00:00",
        "execution_trace":     [],
    }
    base.update(overrides)
    return base


# Canonical mock payloads reused across tests
MOCK_MARKET_DATA = {
    "AAPL": {
        "forecast_30d": 160.0, "confidence_interval_low": 150.0,
        "confidence_interval_high": 170.0, "trend": "bullish",
        "volatility_estimate": 0.20, "mape": 5.0,
        "technical_indicators": {"RSI": 55.0, "SMA_20": 158.0, "MACD": 1.5},
    },
    "MSFT": {
        "forecast_30d": 320.0, "confidence_interval_low": 305.0,
        "confidence_interval_high": 335.0, "trend": "neutral",
        "volatility_estimate": 0.18, "mape": 4.0,
        "technical_indicators": {"RSI": 48.0, "SMA_20": 318.0, "MACD": -0.5},
    },
}

MOCK_RISK_METRICS = {
    "var_95": -0.012, "cvar_95": -0.018, "sharpe_ratio": 0.9,
    "sortino_ratio": 1.2, "max_drawdown": -0.10,
    "portfolio_volatility": 0.15, "portfolio_return_expected": 0.10,
    "recommended_allocation": {"AAPL": 0.55, "MSFT": 0.45},
    "efficient_frontier": {},
    "stress_test_results": {
        "rate_hike_2pct": -0.036, "market_crash_20pct": -0.05, "inflation_spike": -0.024,
    },
    "portfolio_risk_level": "medium",
}

MOCK_RECOMMENDATIONS = {
    "AAPL": {
        "symbol": "AAPL", "action": "buy", "confidence": 0.80,
        "target_allocation": 0.55, "opportunity_score": 0.75,
        "risk_factors": ["High vol"], "rationale": "Strong momentum",
    },
    "MSFT": {
        "symbol": "MSFT", "action": "hold", "confidence": 0.65,
        "target_allocation": 0.45, "opportunity_score": 0.55,
        "risk_factors": ["Flat trend"], "rationale": "Neutral momentum",
    },
}


# ============================================================================
# AGENT STATE STRUCTURE
# ============================================================================

class TestAgentState:

    def test_agent_state_is_dict(self):
        """
        AgentState is a TypedDict — at runtime it must behave as a plain dict.
        (Before the fix it was a @dataclass; LangGraph requires dict-like state.)
        """
        state = make_state()
        assert isinstance(state, dict)

    def test_agent_state_has_required_keys(self):
        state = make_state()
        required = [
            "user_query", "portfolio_symbols", "current_allocation",
            "market_data", "risk_metrics", "recommendations",
            "final_advice", "confidence_score", "timestamp", "execution_trace",
        ]
        for key in required:
            assert key in state, f"Missing required key: {key}"

    def test_execution_trace_reducer_concatenates(self):
        """
        execution_trace uses Annotated[list, operator.add].
        Verify the reducer concatenates rather than overwrites — the property
        that makes parallel nodes safe.
        """
        merged = operator.add(["step_a", "step_b"], ["step_c"])
        assert merged == ["step_a", "step_b", "step_c"]


# ============================================================================
# INDIVIDUAL NODE FUNCTIONS
# ============================================================================

class TestMarketResearchAgent:

    def test_returns_market_data_key(self):
        with patch("src.llm.orchestrator.fetch_market_data") as mock_tool:
            mock_tool.invoke.return_value = MOCK_MARKET_DATA
            result = market_research_agent(make_state())
        assert "market_data" in result
        assert result["market_data"] == MOCK_MARKET_DATA

    def test_appends_to_execution_trace(self):
        with patch("src.llm.orchestrator.fetch_market_data") as mock_tool:
            mock_tool.invoke.return_value = MOCK_MARKET_DATA
            result = market_research_agent(make_state())
        assert "market_research_complete" in result["execution_trace"]

    def test_returns_only_changed_keys(self):
        """Node should return a partial dict, not the full state."""
        with patch("src.llm.orchestrator.fetch_market_data") as mock_tool:
            mock_tool.invoke.return_value = MOCK_MARKET_DATA
            result = market_research_agent(make_state())
        # Should only contain updated keys, not the full state
        assert "user_query" not in result

    def test_error_is_captured_in_trace(self):
        with patch("src.llm.orchestrator.fetch_market_data") as mock_tool:
            mock_tool.invoke.side_effect = RuntimeError("network error")
            result = market_research_agent(make_state())
        assert any("market_research_error" in t for t in result["execution_trace"])


class TestRiskAnalysisAgent:

    def test_returns_risk_metrics_key(self):
        with patch("src.llm.orchestrator.calculate_risk_metrics") as mock_tool:
            mock_tool.invoke.return_value = MOCK_RISK_METRICS
            result = risk_analysis_agent(make_state())
        assert "risk_metrics" in result
        assert result["risk_metrics"]["sharpe_ratio"] == 0.9

    def test_appends_to_execution_trace(self):
        with patch("src.llm.orchestrator.calculate_risk_metrics") as mock_tool:
            mock_tool.invoke.return_value = MOCK_RISK_METRICS
            result = risk_analysis_agent(make_state())
        assert "risk_analysis_complete" in result["execution_trace"]

    def test_error_is_captured_in_trace(self):
        with patch("src.llm.orchestrator.calculate_risk_metrics") as mock_tool:
            mock_tool.invoke.side_effect = RuntimeError("calc error")
            result = risk_analysis_agent(make_state())
        assert any("risk_analysis_error" in t for t in result["execution_trace"])


class TestRecommendationAgent:

    def test_returns_recommendations_key(self):
        with patch("src.llm.orchestrator.score_investment_opportunities") as mock_tool:
            mock_tool.invoke.return_value = MOCK_RECOMMENDATIONS
            result = recommendation_agent(make_state())
        assert "recommendations" in result
        assert "AAPL" in result["recommendations"]

    def test_appends_to_execution_trace(self):
        with patch("src.llm.orchestrator.score_investment_opportunities") as mock_tool:
            mock_tool.invoke.return_value = MOCK_RECOMMENDATIONS
            result = recommendation_agent(make_state())
        assert "recommendation_complete" in result["execution_trace"]


class TestSynthesizeNode:

    def _make_full_state(self):
        return make_state(
            market_data=MOCK_MARKET_DATA,
            risk_metrics=MOCK_RISK_METRICS,
            recommendations=MOCK_RECOMMENDATIONS,
        )

    def test_returns_final_advice_and_confidence(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Buy AAPL, hold MSFT.")
        result = synthesize_with_llm(self._make_full_state(), mock_llm)
        assert "final_advice"    in result
        assert "confidence_score" in result
        assert result["final_advice"] == "Buy AAPL, hold MSFT."

    def test_confidence_is_mean_of_recommendation_confidences(self):
        """Confidence score = mean of individual recommendation confidences."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="advice")
        result = synthesize_with_llm(self._make_full_state(), mock_llm)
        expected = (0.80 + 0.65) / 2  # from MOCK_RECOMMENDATIONS
        assert abs(result["confidence_score"] - expected) < 1e-4

    def test_missing_agent_data_returns_safe_fallback(self):
        """If any upstream agent failed, synthesiser should not crash."""
        mock_llm = MagicMock()
        result = synthesize_with_llm(make_state(), mock_llm)  # all data is None
        assert "final_advice"    in result
        assert result["confidence_score"] == 0.0

    def test_appends_to_execution_trace(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="advice")
        result = synthesize_with_llm(self._make_full_state(), mock_llm)
        assert "synthesize_complete" in result["execution_trace"]


class TestValidateNode:

    def test_passes_on_high_confidence(self):
        result = validate_recommendation(make_state(confidence_score=0.85))
        assert "validation_complete" in result["execution_trace"]

    def test_passes_on_low_confidence_without_crashing(self):
        """Low confidence should log a warning but not raise."""
        result = validate_recommendation(make_state(confidence_score=0.30))
        assert "validation_complete" in result["execution_trace"]

    def test_handles_none_confidence_without_crashing(self):
        """None confidence_score (synthesis failed) must not cause a TypeError."""
        result = validate_recommendation(make_state(confidence_score=None))
        assert "validation_complete" in result["execution_trace"]


# ============================================================================
# FULL GRAPH (integration — all network calls mocked)
# ============================================================================

def _make_advisor_with_mock_llm(llm_response: str = "Test advice") -> tuple:
    """
    Helper: create a FinancialAdvisorGraph with ChatAnthropic replaced by a
    MagicMock at class-construction time (before _build_and_compile captures
    self.llm in a closure).

    ChatAnthropic is a Pydantic model — patch.object on an instance fails.
    Patching the class at import level sidesteps that restriction.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=llm_response)

    with patch("src.llm.orchestrator.ChatAnthropic", return_value=mock_llm):
        advisor = FinancialAdvisorGraph()

    return advisor, mock_llm


class TestFinancialAdvisorGraph:

    def test_graph_compiles_without_error(self):
        advisor, _ = _make_advisor_with_mock_llm()
        assert advisor._compiled is not None

    def test_all_five_nodes_fire_in_order(self):
        """
        Run the full graph with all external calls mocked.
        Execution trace must contain exactly 5 entries in the correct order.
        """
        advisor, _ = _make_advisor_with_mock_llm("advice")

        with (
            patch("src.llm.orchestrator.fetch_market_data")             as mm,
            patch("src.llm.orchestrator.calculate_risk_metrics")        as rm,
            patch("src.llm.orchestrator.score_investment_opportunities") as rec,
        ):
            mm.invoke.return_value  = MOCK_MARKET_DATA
            rm.invoke.return_value  = MOCK_RISK_METRICS
            rec.invoke.return_value = MOCK_RECOMMENDATIONS

            result = advisor.invoke(
                user_query="Test",
                portfolio_symbols=["AAPL", "MSFT"],
                current_allocation={"AAPL": 0.6, "MSFT": 0.4},
            )

        expected_trace = [
            "market_research_complete",
            "risk_analysis_complete",
            "recommendation_complete",
            "synthesize_complete",
            "validation_complete",
        ]
        assert result["execution_trace"] == expected_trace

    def test_final_advice_comes_from_llm(self):
        advisor, _ = _make_advisor_with_mock_llm("Mocked Claude advice")

        with (
            patch("src.llm.orchestrator.fetch_market_data")             as mm,
            patch("src.llm.orchestrator.calculate_risk_metrics")        as rm,
            patch("src.llm.orchestrator.score_investment_opportunities") as rec,
        ):
            mm.invoke.return_value  = MOCK_MARKET_DATA
            rm.invoke.return_value  = MOCK_RISK_METRICS
            rec.invoke.return_value = MOCK_RECOMMENDATIONS

            result = advisor.invoke("Test", ["AAPL", "MSFT"], {"AAPL": 0.6, "MSFT": 0.4})

        assert result["final_advice"] == "Mocked Claude advice"

    def test_output_contains_all_expected_keys(self):
        advisor, _ = _make_advisor_with_mock_llm()

        with (
            patch("src.llm.orchestrator.fetch_market_data")             as mm,
            patch("src.llm.orchestrator.calculate_risk_metrics")        as rm,
            patch("src.llm.orchestrator.score_investment_opportunities") as rec,
        ):
            mm.invoke.return_value  = MOCK_MARKET_DATA
            rm.invoke.return_value  = MOCK_RISK_METRICS
            rec.invoke.return_value = MOCK_RECOMMENDATIONS

            result = advisor.invoke("Test", ["AAPL", "MSFT"], {"AAPL": 0.6, "MSFT": 0.4})

        for key in ["user_query", "final_advice", "confidence_score",
                    "market_data", "risk_metrics", "recommendations",
                    "execution_trace", "timestamp"]:
            assert key in result, f"Missing key in output: {key}"
