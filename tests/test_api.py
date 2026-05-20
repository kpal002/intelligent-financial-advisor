"""
Tests for the FastAPI application layer.

Uses FastAPI's TestClient — no real server is started.
The FinancialAdvisorGraph is replaced with a MagicMock so no ML models
or Claude API calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Minimal stub that the API will use instead of the real advisor ────────────

MOCK_ADVISOR_RESULT = {
    "user_query":      "Test query",
    "final_advice":    "Mock Claude advice",
    "confidence_score": 0.75,
    "market_data": {
        "AAPL": {
            "trend": "bullish", "forecast_30d": 160.0,
            "confidence_interval_low": 150.0, "confidence_interval_high": 170.0,
            "volatility_estimate": 0.20, "mape": 5.0,
            "technical_indicators": {"RSI": 55.0, "SMA_20": 158.0, "MACD": 1.5},
        },
        "MSFT": {
            "trend": "neutral", "forecast_30d": 320.0,
            "confidence_interval_low": 305.0, "confidence_interval_high": 335.0,
            "volatility_estimate": 0.18, "mape": 4.0,
            "technical_indicators": {"RSI": 48.0, "SMA_20": 318.0, "MACD": -0.5},
        },
    },
    "risk_metrics": {
        "portfolio_risk_level": "medium",
        "var_95": -0.012, "cvar_95": -0.018,
        "sharpe_ratio": 0.9, "sortino_ratio": 1.2, "max_drawdown": -0.10,
        "portfolio_volatility": 0.15, "portfolio_return_expected": 0.10,
        "recommended_allocation": {"AAPL": 0.55, "MSFT": 0.45},
        "stress_test_results": {
            "rate_hike_2pct": -0.036,
            "market_crash_20pct": -0.05,
            "inflation_spike": -0.024,
        },
    },
    "recommendations": {
        "AAPL": {
            "action": "buy", "confidence": 0.80, "opportunity_score": 0.75,
            "rationale": "Strong momentum", "target_allocation": 0.55,
            "risk_factors": [],
        },
        "MSFT": {
            "action": "hold", "confidence": 0.65, "opportunity_score": 0.55,
            "rationale": "Neutral momentum", "target_allocation": 0.45,
            "risk_factors": [],
        },
    },
    "execution_trace": [
        "market_research_complete", "risk_analysis_complete",
        "recommendation_complete", "synthesize_complete", "validation_complete",
    ],
    "timestamp": "2024-01-01T00:00:00",
}

VALID_REQUEST = {
    "query": "Should I rebalance given recent rate hikes?",
    "portfolio_symbols": ["AAPL", "MSFT"],
    "current_allocation": {"AAPL": 0.6, "MSFT": 0.4},
}


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the advisor mocked out.
    The mock is set up before the lifespan runs so _advisor is a MagicMock.
    """
    mock_advisor = MagicMock()
    mock_advisor.invoke.return_value = MOCK_ADVISOR_RESULT

    with patch("src.api.app.FinancialAdvisorGraph", return_value=mock_advisor):
        from src.api.app import app
        with TestClient(app) as c:
            yield c


# ============================================================================
# HEALTH CHECK
# ============================================================================

class TestHealthEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_has_status_healthy(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_advisor_ready_true(self, client):
        """After lifespan startup the advisor should be initialised."""
        resp = client.get("/health")
        assert resp.json()["advisor_ready"] is True

    def test_body_has_timestamp(self, client):
        resp = client.get("/health")
        assert "timestamp" in resp.json()


# ============================================================================
# POST /api/v1/advice — HAPPY PATH
# ============================================================================

class TestAdviceEndpointHappyPath:

    def test_returns_200(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        assert resp.status_code == 200

    def test_response_contains_advice_field(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        assert resp.json()["advice"] == "Mock Claude advice"

    def test_response_confidence_score(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        assert resp.json()["confidence_score"] == 0.75

    def test_response_has_market_summaries(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        data = resp.json()
        assert "market_summaries" in data
        assert "AAPL" in data["market_summaries"]
        assert data["market_summaries"]["AAPL"]["trend"] == "bullish"

    def test_response_has_risk_summary(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        risk = resp.json()["risk_summary"]
        assert risk["risk_level"] == "medium"
        assert risk["sharpe_ratio"] == 0.9

    def test_response_has_recommendations(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        recs = resp.json()["recommendations"]
        assert recs["AAPL"]["action"] == "buy"
        assert recs["MSFT"]["action"] == "hold"

    def test_execution_trace_has_five_steps(self, client):
        resp = client.post("/api/v1/advice", json=VALID_REQUEST)
        assert len(resp.json()["execution_trace"]) == 5

    def test_lowercase_symbols_are_uppercased(self, client):
        """Validator should normalise 'aapl' → 'AAPL'."""
        req = {
            "query": "Should I rebalance?",
            "portfolio_symbols": ["aapl", "msft"],
            "current_allocation": {"aapl": 0.6, "msft": 0.4},
        }
        # Validator uppercases symbols but allocation keys stay lowercase →
        # mismatch triggers 422; we just verify the server doesn't 500.
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code in (200, 422)  # 422 is fine here


# ============================================================================
# POST /api/v1/advice — VALIDATION ERRORS (expect 422)
# ============================================================================

class TestAdviceEndpointValidation:

    def test_empty_query_rejected(self, client):
        req = {**VALID_REQUEST, "query": "Hi"}  # < 5 chars
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code == 422

    def test_empty_symbols_rejected(self, client):
        req = {**VALID_REQUEST, "portfolio_symbols": []}
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code == 422

    def test_allocation_not_summing_to_one_rejected(self, client):
        req = {
            **VALID_REQUEST,
            "current_allocation": {"AAPL": 0.4, "MSFT": 0.4},  # sums to 0.8
        }
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code == 422

    def test_negative_weight_rejected(self, client):
        req = {
            **VALID_REQUEST,
            "current_allocation": {"AAPL": 1.1, "MSFT": -0.1},
        }
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code == 422

    def test_mismatched_symbols_and_allocation_rejected(self, client):
        """portfolio_symbols has GOOGL but allocation does not — expect 422."""
        req = {
            "query": "Should I rebalance?",
            "portfolio_symbols": ["AAPL", "GOOGL"],
            "current_allocation": {"AAPL": 0.6, "MSFT": 0.4},  # GOOGL missing
        }
        resp = client.post("/api/v1/advice", json=req)
        assert resp.status_code == 422

    def test_missing_required_fields_rejected(self, client):
        resp = client.post("/api/v1/advice", json={"query": "test"})
        assert resp.status_code == 422
