"""
FastAPI application for the Intelligent Financial Advisor Agent.

Endpoints:
    GET  /health           — liveness probe
    POST /api/v1/advice    — generate investment recommendation

Run locally:
    uvicorn src.api.app:app --reload

The FinancialAdvisorGraph is initialised once at startup via FastAPI's
lifespan context manager, so the LangGraph is compiled and the LLM client
is ready before the first request arrives.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.llm.orchestrator import FinancialAdvisorGraph

logger = logging.getLogger(__name__)

# Single advisor instance shared across all requests
_advisor: Optional[FinancialAdvisorGraph] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: initialise heavy resources on startup, release on shutdown.
    Using lifespan (not deprecated @app.on_event) so startup errors are visible.
    """
    global _advisor
    logger.info("Startup — compiling FinancialAdvisorGraph...")
    _advisor = FinancialAdvisorGraph()
    logger.info("FinancialAdvisorGraph ready.")
    yield
    logger.info("Shutdown — releasing advisor.")
    _advisor = None


app = FastAPI(
    title="Intelligent Financial Advisor Agent",
    description=(
        "Multi-agent LLM system combining ARIMA forecasting, "
        "Markowitz portfolio optimisation, Isolation Forest anomaly detection, "
        "and Claude-powered synthesis into a single investment advisory workflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class AdviceRequest(BaseModel):
    """Input schema validated before the advisor workflow runs."""

    query: str = Field(
        ..., min_length=5,
        description="User's investment question or objective",
        examples=["Should I rebalance given recent Fed rate hikes?"],
    )
    portfolio_symbols: List[str] = Field(
        ..., min_length=1,
        description="Ticker symbols in the portfolio",
        examples=[["AAPL", "MSFT", "JPM", "JNJ"]],
    )
    current_allocation: Dict[str, float] = Field(
        ...,
        description="Current portfolio weights. Must sum to 1.0 and match portfolio_symbols.",
        examples=[{"AAPL": 0.30, "MSFT": 0.30, "JPM": 0.20, "JNJ": 0.20}],
    )

    @field_validator("portfolio_symbols")
    @classmethod
    def symbols_uppercased(cls, v: List[str]) -> List[str]:
        """Normalise tickers to uppercase (user might pass 'aapl')."""
        return [s.strip().upper() for s in v]

    @field_validator("current_allocation")
    @classmethod
    def weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"Weights must sum to 1.0 (got {total:.4f}). "
                "Adjust allocations or let the system normalise them."
            )
        return v

    @field_validator("current_allocation")
    @classmethod
    def weights_non_negative(cls, v: Dict[str, float]) -> Dict[str, float]:
        negatives = {k: w for k, w in v.items() if w < 0}
        if negatives:
            raise ValueError(f"Negative weights not allowed: {negatives}")
        return v


class MarketSummary(BaseModel):
    """Per-symbol output from the Market Research Agent."""
    trend: str                   = Field(..., description="bullish / bearish / neutral")
    forecast_30d: float          = Field(..., description="30-day ARIMA price forecast")
    confidence_interval_low: float
    confidence_interval_high: float
    annual_volatility: float
    forecast_mape: float         = Field(..., description="Validation MAPE (%)")
    rsi: float
    macd: float


class RiskSummary(BaseModel):
    """Portfolio-level output from the Risk Analysis Agent."""
    risk_level: str              = Field(..., description="low / medium / high")
    var_95_daily: float          = Field(..., description="Value at Risk (95%, daily)")
    cvar_95_daily: float         = Field(..., description="Conditional VaR (daily)")
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    recommended_allocation: Dict[str, float] = Field(
        ..., description="Markowitz max-Sharpe optimal weights"
    )
    stress_test_results: Dict[str, float]


class RecommendationSummary(BaseModel):
    """Per-symbol output from the Recommendation Agent."""
    action: str                  = Field(..., description="buy / hold / sell")
    confidence: float            = Field(..., description="0.0–1.0 signal confidence")
    opportunity_score: float
    rationale: str


class AdviceResponse(BaseModel):
    """Full output returned to the caller."""
    advice: str                  = Field(..., description="Claude-generated advisory report")
    confidence_score: float      = Field(..., description="Mean confidence across all signals")
    market_summaries: Dict[str, MarketSummary]
    risk_summary: RiskSummary
    recommendations: Dict[str, RecommendationSummary]
    execution_trace: List[str]
    timestamp: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", tags=["ops"])
async def health_check() -> Dict[str, Any]:
    """
    Liveness probe.  Returns 200 when the server is up and the advisor is ready.
    Use as a readiness check before sending traffic.
    """
    return {
        "status":        "healthy",
        "advisor_ready": _advisor is not None,
        "timestamp":     datetime.now().isoformat(),
        "version":       app.version,
    }


@app.post("/api/v1/advice", response_model=AdviceResponse, tags=["advisor"])
async def get_investment_advice(request: AdviceRequest) -> AdviceResponse:
    """
    Generate a comprehensive investment recommendation.

    **Workflow (5 steps)**:
    1. **Market Research** — ARIMA forecasts + RSI/MACD technical indicators per symbol
    2. **Risk Analysis**   — VaR, Sharpe/Sortino, Markowitz optimisation, stress tests
    3. **Recommendations** — Isolation Forest anomaly detection + momentum Z-scores
    4. **LLM Synthesis**   — Claude produces a 6-section advisory report
    5. **Validation**      — Sanity checks on confidence score

    **Typical latency**: 60–120 s (ARIMA fitting dominates for 4+ symbols).
    """
    if _advisor is None:
        raise HTTPException(status_code=503, detail="Advisor not initialised yet.")

    # Extra cross-field validation: allocation keys must match symbols
    alloc_keys = set(request.current_allocation)
    sym_keys   = set(request.portfolio_symbols)
    if alloc_keys != sym_keys:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Allocation keys {sorted(alloc_keys)} must exactly match "
                f"portfolio_symbols {sorted(sym_keys)}."
            ),
        )

    try:
        result = _advisor.invoke(
            user_query=request.query,
            portfolio_symbols=request.portfolio_symbols,
            current_allocation=request.current_allocation,
        )
    except Exception as exc:
        logger.exception("Advisor workflow failed")
        raise HTTPException(status_code=500, detail=f"Advisor error: {exc}") from exc

    # ── Build structured response ────────────────────────────────────────────
    market_raw  = result.get("market_data")  or {}
    risk_raw    = result.get("risk_metrics") or {}
    rec_raw     = result.get("recommendations") or {}

    market_summaries: Dict[str, MarketSummary] = {}
    for sym, data in market_raw.items():
        if "error" in data:
            continue
        ti = data.get("technical_indicators", {})
        market_summaries[sym] = MarketSummary(
            trend=data.get("trend", "unknown"),
            forecast_30d=data.get("forecast_30d", 0.0),
            confidence_interval_low=data.get("confidence_interval_low", 0.0),
            confidence_interval_high=data.get("confidence_interval_high", 0.0),
            annual_volatility=data.get("volatility_estimate", 0.0),
            forecast_mape=data.get("mape", 0.0),
            rsi=ti.get("RSI", 0.0),
            macd=ti.get("MACD", 0.0),
        )

    risk_summary = RiskSummary(
        risk_level=risk_raw.get("portfolio_risk_level", "unknown"),
        var_95_daily=risk_raw.get("var_95", 0.0),
        cvar_95_daily=risk_raw.get("cvar_95", 0.0),
        sharpe_ratio=risk_raw.get("sharpe_ratio", 0.0),
        sortino_ratio=risk_raw.get("sortino_ratio", 0.0),
        max_drawdown=risk_raw.get("max_drawdown", 0.0),
        recommended_allocation=risk_raw.get("recommended_allocation", {}),
        stress_test_results=risk_raw.get("stress_test_results", {}),
    )

    recommendations = {
        sym: RecommendationSummary(
            action=rec.get("action", "hold"),
            confidence=rec.get("confidence", 0.0),
            opportunity_score=rec.get("opportunity_score", 0.0),
            rationale=rec.get("rationale", ""),
        )
        for sym, rec in rec_raw.items()
    }

    return AdviceResponse(
        advice=result.get("final_advice") or "No advice generated.",
        confidence_score=result.get("confidence_score") or 0.0,
        market_summaries=market_summaries,
        risk_summary=risk_summary,
        recommendations=recommendations,
        execution_trace=result.get("execution_trace") or [],
        timestamp=result.get("timestamp") or datetime.now().isoformat(),
    )
