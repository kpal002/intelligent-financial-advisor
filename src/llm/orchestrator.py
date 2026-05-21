"""
LangGraph multi-agent orchestrator for the financial advisor.

Workflow (sequential):
    START
      → market_research_agent   (fetch prices, run ARIMA, compute technicals)
      → risk_analysis_agent     (VaR, Sharpe, Markowitz optimisation)
      → recommendation_agent    (anomaly detection, score candidates)
      → synthesize_with_llm     (Claude synthesises all outputs)
      → validate_recommendation (sanity checks)
    END

Each node receives the full AgentState dict and returns ONLY the fields it
updated. LangGraph merges those updates into the shared state automatically.

State uses TypedDict (required by LangGraph).
execution_trace uses Annotated[list, operator.add] so parallel nodes can
both append without one overwriting the other.
"""

import logging
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Resolve .env relative to this file (src/llm/orchestrator.py → ../../.env)
# so the path works regardless of where python3 is invoked from.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)  # override=True so .env wins over empty shell vars

import numpy as np
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

from src.ml_pipeline import (
    TimeSeriesForecaster,
    PortfolioRiskAnalyzer,
    PortfolioOptimizer,
    AnomalyDetector,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state passed between every node in the graph.

    TypedDict (not @dataclass) because LangGraph needs dict-like access.
    Nodes return only the keys they changed; LangGraph merges the rest.

    execution_trace uses Annotated + operator.add so that if two nodes run
    in parallel, their trace entries are concatenated instead of one
    overwriting the other.
    """
    # --- inputs ---
    user_query: str
    portfolio_symbols: List[str]
    current_allocation: Dict[str, float]

    # --- filled in by agents ---
    market_data: Optional[Dict[str, Any]]
    risk_metrics: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]

    # --- filled in by synthesizer ---
    final_advice: Optional[str]
    confidence_score: Optional[float]

    # --- metadata ---
    timestamp: str
    execution_trace: Annotated[List[str], operator.add]


# ============================================================================
# PYDANTIC OUTPUT SCHEMAS
# ============================================================================

class TimeSeriesForecastOutput(BaseModel):
    """Structured output from Market Research Agent."""
    symbol: str
    forecast_30d: float = Field(..., description="30-day price forecast")
    confidence_interval_low: float
    confidence_interval_high: float
    trend: str = Field(..., description="'bullish', 'bearish', or 'neutral'")
    volatility_estimate: float
    mape: float = Field(..., description="Validation MAPE (%)")
    technical_indicators: Dict[str, float]


class RiskAssessment(BaseModel):
    """Structured output from Risk Analysis Agent."""
    var_95: float = Field(..., description="Value at Risk at 95% confidence (daily)")
    cvar_95: float = Field(..., description="Conditional VaR — average loss beyond VaR")
    sharpe_ratio: float
    max_drawdown: float
    recommended_allocation: Dict[str, float]
    stress_test_results: Dict[str, float]
    portfolio_risk_level: str = Field(..., description="'low', 'medium', or 'high'")


class InvestmentRecommendation(BaseModel):
    """Structured output from Recommendation Agent."""
    symbol: str
    action: str = Field(..., description="'buy', 'hold', or 'sell'")
    confidence: float = Field(..., description="0.0–1.0")
    target_allocation: float
    rationale: str
    risk_factors: List[str]
    opportunity_score: float


class FinalAdvice(BaseModel):
    """Structured final output from the LLM synthesiser."""
    summary_recommendation: str
    allocation_strategy: Dict[str, float]
    expected_return: float
    expected_volatility: float
    key_risks: List[str]
    key_opportunities: List[str]
    confidence_interval: Tuple[float, float]
    reasoning_summary: str


# ============================================================================
# TOOL DEFINITIONS  (mock data — replaced with real ML in Step 4)
# ============================================================================

@tool
def fetch_market_data(symbols: List[str], lookback_days: int = 252) -> Dict[str, Any]:
    """
    Market Research Agent Tool: fetch prices and run ARIMA forecasting.

    For each symbol:
      - Downloads historical OHLCV data via yfinance
      - Fits auto-ARIMA and generates a 30-day forecast with 95% CI
      - Computes technical indicators (RSI, SMA_20, MACD)
      - Derives a trend label (bullish / bearish / neutral) from those indicators
    """
    logger.info(f"[Market Agent] Fetching real data for {symbols}")

    result = {}
    for symbol in symbols:
        try:
            forecaster = TimeSeriesForecaster(symbol, lookback_days)
            forecaster.fetch_data()
            forecast = forecaster.fit_arima(forecast_horizon=30)
            technicals = forecaster.calculate_technical_indicators().ffill().dropna()

            last = technicals.iloc[-1]
            current_price = float(forecaster.data['Close'].iloc[-1].item())
            rsi   = float(last['RSI'].item()   if 'RSI'   in last.index else 50.0)
            macd  = float(last['MACD'].item()  if 'MACD'  in last.index else 0.0)
            sma20 = float(last['SMA_20'].item()if 'SMA_20'in last.index else current_price)

            # Simple 3-factor trend rule
            if rsi > 60 and macd > 0 and current_price > sma20:
                trend = 'bullish'
            elif rsi < 40 and macd < 0 and current_price < sma20:
                trend = 'bearish'
            else:
                trend = 'neutral'

            annual_vol = float(forecaster.data['Returns'].std() * np.sqrt(252))

            result[symbol] = {
                "forecast_30d":            round(float(forecast.forecast_values[0]), 2),
                "confidence_interval_low": round(float(forecast.confidence_interval_lower[0]), 2),
                "confidence_interval_high":round(float(forecast.confidence_interval_upper[0]), 2),
                "trend":                   trend,
                "volatility_estimate":     round(annual_vol, 4),
                "mape":                    round(float(forecast.mape), 2),
                "technical_indicators": {
                    "RSI":   round(rsi,   1),
                    "SMA_20":round(sma20, 2),
                    "MACD":  round(macd,  3),
                },
            }
        except Exception as e:
            logger.error(f"[Market Agent] Error on {symbol}: {e}")
            result[symbol] = {"error": str(e), "forecast_30d": 0.0, "trend": "unknown"}

    return result


@tool
def calculate_risk_metrics(symbols: List[str],
                            allocation: Dict[str, float]) -> Dict[str, Any]:
    """
    Risk Analysis Agent Tool: compute VaR, Sharpe, and Markowitz optimisation.

    - Fetches historical returns via PortfolioRiskAnalyzer
    - Computes VaR (parametric), CVaR, Sharpe, Sortino, Max Drawdown
    - Shares the fetched data with PortfolioOptimizer to avoid a second download
    - Returns Markowitz-optimal allocation (max Sharpe) as recommended_allocation
    - Derives simple stress-test estimates from the actual risk numbers
    """
    logger.info(f"[Risk Agent] Calculating real risk metrics for {symbols}")

    # Normalise weights in case of float rounding (e.g. 0.333... × 3 ≠ 1.0)
    total = sum(allocation.values())
    weights = {k: v / total for k, v in allocation.items()}

    analyzer = PortfolioRiskAnalyzer(symbols)
    analyzer.fetch_returns()

    var     = analyzer.calculate_var(weights, method='parametric')
    cvar    = analyzer.calculate_cvar(weights)
    sharpe  = analyzer.calculate_sharpe_ratio(weights)
    sortino = analyzer.calculate_sortino_ratio(weights)
    max_dd  = analyzer.calculate_max_drawdown(weights)

    port_returns = analyzer._portfolio_returns(weights)
    annual_vol = float(port_returns.std() * np.sqrt(252))
    annual_ret = float(port_returns.mean() * 252)

    risk_level = 'low' if annual_vol < 0.12 else ('high' if annual_vol > 0.22 else 'medium')

    # Reuse already-fetched data in the optimizer (avoids second yfinance call)
    optimizer = PortfolioOptimizer(symbols)
    optimizer.returns_df       = analyzer.returns_df
    optimizer.expected_returns = analyzer.returns_df.mean() * 252
    optimizer.cov_matrix       = analyzer.returns_df.cov() * 252
    optimal = optimizer.optimize_for_max_sharpe()

    # Stress-test estimates: scale actual risk metrics by scenario multipliers
    stress = {
        "rate_hike_2pct":     round(var * 3.0,   4),   # 3× daily VaR ≈ rate shock
        "market_crash_20pct": round(max_dd * 0.5, 4),  # half of worst historical drawdown
        "inflation_spike":    round(var * 2.0,   4),   # 2× daily VaR ≈ inflation shock
    }

    return {
        "var_95":                    round(var,       4),
        "cvar_95":                   round(cvar,      4),
        "sharpe_ratio":              round(sharpe,    4),
        "sortino_ratio":             round(sortino,   4),
        "max_drawdown":              round(max_dd,    4),
        "portfolio_volatility":      round(annual_vol,4),
        "portfolio_return_expected": round(annual_ret,4),
        "recommended_allocation":    {k: round(v, 4) for k, v in optimal.weights.items()},
        "efficient_frontier":        {},
        "stress_test_results":       stress,
        "portfolio_risk_level":      risk_level,
    }


@tool
def score_investment_opportunities(symbols: List[str]) -> Dict[str, Any]:
    """
    Recommendation Agent Tool: generate buy/hold/sell signals using ML.

    Process:
      1. AnomalyDetector engineers rolling features (momentum, vol, skew, kurtosis)
      2. Isolation Forest flags anomalous dates portfolio-wide (market stress signal)
      3. Per-symbol momentum is Z-scored across the lookback period
      4. Signal = buy if momentum Z > 1 and low anomaly count
                  sell if momentum Z < -1 or high anomaly count
                  hold otherwise
    """
    logger.info(f"[Rec Agent] Scoring real signals for {symbols}")

    detector = AnomalyDetector(symbols)
    detector.fetch_data()
    detector.engineer_features()
    anomaly_labels = detector.detect_anomalies_isolation_forest()

    # Portfolio-wide stress: how many of the last 10 days were flagged?
    recent_anomaly_count = int((anomaly_labels.iloc[-10:] == -1).sum())

    result = {}
    equal_weight = round(1 / len(symbols), 4)

    for symbol in symbols:
        momentum_col = f'{symbol}_momentum'
        vol_col      = f'{symbol}_volatility'

        if momentum_col not in detector.features_df.columns:
            result[symbol] = {
                "symbol": symbol, "action": "hold", "confidence": 0.5,
                "target_allocation": equal_weight, "opportunity_score": 0.5,
                "risk_factors": ["Insufficient data"], "rationale": "No features available",
            }
            continue

        momentum_series = detector.features_df[momentum_col]
        recent_momentum = float(momentum_series.iloc[-1])
        z_momentum = float(
            (recent_momentum - momentum_series.mean()) / (momentum_series.std() + 1e-8)
        )
        recent_vol = float(detector.features_df[vol_col].iloc[-1])

        # Signal logic
        if z_momentum > 1.0 and recent_anomaly_count < 3:
            action     = 'buy'
            confidence = round(min(0.55 + z_momentum * 0.10, 0.92), 3)
        elif z_momentum < -1.0 or recent_anomaly_count >= 7:
            action     = 'sell'
            confidence = round(min(0.55 + abs(z_momentum) * 0.10, 0.92), 3)
        else:
            action     = 'hold'
            confidence = 0.65

        opportunity_score = round(float(np.clip(0.5 + z_momentum * 0.15, 0.05, 0.95)), 3)

        result[symbol] = {
            "symbol":            symbol,
            "action":            action,
            "confidence":        confidence,
            "target_allocation": equal_weight,
            "opportunity_score": opportunity_score,
            "risk_factors": [
                f"30-day volatility: {recent_vol:.3f}",
                f"Market anomaly days (last 10): {recent_anomaly_count}",
            ],
            "rationale": (
                f"Momentum z-score: {z_momentum:+.2f} | "
                f"market anomalies: {recent_anomaly_count}/10 days"
            ),
        }

    return result


# ============================================================================
# AGENT NODE FUNCTIONS
# ============================================================================

def market_research_agent(state: AgentState) -> Dict[str, Any]:
    """
    Node 1 — Market Research.

    Calls the market data tool and stores the result in state.
    Returns only the keys it changed; LangGraph merges the rest.
    """
    logger.info(f"[MARKET AGENT] Query: {state['user_query']}")

    try:
        market_data = fetch_market_data.invoke({
            "symbols": state["portfolio_symbols"],
            "lookback_days": 252,
        })
        return {
            "market_data": market_data,
            "execution_trace": ["market_research_complete"],
        }
    except Exception as e:
        logger.error(f"Market agent error: {e}")
        return {"execution_trace": [f"market_research_error: {e}"]}


def risk_analysis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Node 2 — Risk Analysis.

    Calls the risk metrics tool and stores the result in state.
    """
    logger.info("[RISK AGENT] Calculating portfolio risk")

    try:
        risk_metrics = calculate_risk_metrics.invoke({
            "symbols": state["portfolio_symbols"],
            "allocation": state["current_allocation"],
        })
        logger.info(f"Risk level: {risk_metrics['portfolio_risk_level']}, "
                    f"Sharpe: {risk_metrics['sharpe_ratio']:.2f}")
        return {
            "risk_metrics": risk_metrics,
            "execution_trace": ["risk_analysis_complete"],
        }
    except Exception as e:
        logger.error(f"Risk agent error: {e}")
        return {"execution_trace": [f"risk_analysis_error: {e}"]}


def recommendation_agent(state: AgentState) -> Dict[str, Any]:
    """
    Node 3 — Recommendations.

    Scores candidates and stores ranked recommendations in state.
    """
    logger.info("[RECOMMENDATION AGENT] Scoring candidates")

    try:
        recommendations = score_investment_opportunities.invoke({
            "symbols": state["portfolio_symbols"],
        })
        high_conf = [k for k, v in recommendations.items() if v["confidence"] > 0.75]
        logger.info(f"High-confidence signals: {high_conf}")
        return {
            "recommendations": recommendations,
            "execution_trace": ["recommendation_complete"],
        }
    except Exception as e:
        logger.error(f"Recommendation agent error: {e}")
        return {"execution_trace": [f"recommendation_error: {e}"]}


def synthesize_with_llm(state: AgentState, llm: ChatAnthropic) -> Dict[str, Any]:
    """
    Node 4 — LLM Synthesiser.

    Formats all three agents' outputs into a structured prompt and calls Claude.
    Claude returns a 6-section advisory report covering:
      Executive Summary / Portfolio Assessment / Specific Actions /
      Key Risks / Suggested Allocation / Confidence Level

    The LLM instance is injected from FinancialAdvisorGraph (created after
    load_dotenv() so the API key is guaranteed to be in the environment).
    """
    logger.info("[SYNTHESIZER] Calling Claude to synthesise agent outputs")

    if not (state.get("market_data") and state.get("risk_metrics") and state.get("recommendations")):
        return {
            "final_advice": "Insufficient data to generate a recommendation.",
            "confidence_score": 0.0,
            "execution_trace": ["synthesize_error: missing agent outputs"],
        }

    try:
        market = state["market_data"]
        risk   = state["risk_metrics"]
        recs   = state["recommendations"]

        # ── Format market section ─────────────────────────────────────────────
        market_lines = []
        for sym, data in market.items():
            if "error" in data:
                market_lines.append(f"{sym}: ERROR — {data['error']}")
                continue
            ti = data["technical_indicators"]
            market_lines.append(
                f"{sym}: {data['trend'].upper()} | "
                f"30d forecast ${data['forecast_30d']:.2f} "
                f"(95% CI ${data['confidence_interval_low']:.2f}–${data['confidence_interval_high']:.2f}) | "
                f"MAPE {data['mape']:.1f}% | "
                f"Ann.vol {data['volatility_estimate']*100:.1f}% | "
                f"RSI {ti['RSI']:.0f} MACD {ti['MACD']:+.3f}"
            )

        # ── Format risk section ───────────────────────────────────────────────
        alloc_lines = "\n".join(
            f"    {s}: {w*100:.1f}%"
            for s, w in risk["recommended_allocation"].items()
        )
        stress_lines = "\n".join(
            f"    {k.replace('_',' ').title()}: {v*100:.1f}%"
            for k, v in risk["stress_test_results"].items()
        )

        # ── Format recommendation section ─────────────────────────────────────
        rec_lines = []
        for sym, rec in recs.items():
            rec_lines.append(
                f"{sym}: {rec['action'].upper()} "
                f"(confidence {rec['confidence']*100:.0f}%, "
                f"opportunity {rec['opportunity_score']:.2f}) — "
                f"{rec['rationale']}"
            )

        current_alloc_str = "\n".join(
            f"  {s}: {w*100:.1f}%"
            for s, w in state["current_allocation"].items()
        )

        prompt = f"""You are an expert financial advisor AI. Three specialized ML agents have analysed this portfolio. Synthesise their outputs into a clear, actionable investment report.

USER QUERY
{state['user_query']}

CURRENT PORTFOLIO
{current_alloc_str}

═══ MARKET RESEARCH AGENT (ARIMA forecasts + technicals) ═══
{chr(10).join(market_lines)}

═══ RISK ANALYSIS AGENT (Markowitz + VaR) ═══
Risk level: {risk['portfolio_risk_level'].upper()}
Ann. volatility: {risk['portfolio_volatility']*100:.1f}%   Expected return: {risk['portfolio_return_expected']*100:.1f}%
Sharpe: {risk['sharpe_ratio']:.2f}   Sortino: {risk['sortino_ratio']:.2f}
VaR 95% (daily): {risk['var_95']*100:.2f}%   CVaR (daily): {risk['cvar_95']*100:.2f}%
Max drawdown: {risk['max_drawdown']*100:.1f}%

Markowitz optimal allocation (max Sharpe):
{alloc_lines}

Stress tests:
{stress_lines}

═══ RECOMMENDATION AGENT (momentum Z-score + Isolation Forest) ═══
{chr(10).join(rec_lines)}

─────────────────────────────────────────────────────────────
Provide a professional advisory report with EXACTLY these six sections:

1. EXECUTIVE SUMMARY
   2–3 sentences. Overall recommendation and single most important action.

2. PORTFOLIO ASSESSMENT
   Current risk/return profile. Is it appropriate for a typical investor?

3. SPECIFIC ACTIONS
   For each holding: action (Buy/Hold/Sell), reasoning, and suggested weight.

4. KEY RISKS
   Top 3 risks and a mitigation strategy for each.

5. SUGGESTED ALLOCATION
   Final recommended weights. Must sum to 100%.

6. CONFIDENCE LEVEL
   Your overall confidence (%) and the main source of uncertainty.

Be concise, professional, and directly actionable."""

        response = llm.invoke([HumanMessage(content=prompt)])

        confidence = float(np.mean([v["confidence"] for v in recs.values()]))

        return {
            "final_advice":   response.content,
            "confidence_score": round(confidence, 4),
            "execution_trace": ["synthesize_complete"],
        }

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return {"execution_trace": [f"synthesize_error: {e}"]}


def validate_recommendation(state: AgentState) -> Dict[str, Any]:
    """
    Node 5 — Validation.

    Sanity-checks the final recommendation before returning to the caller.
    Flags low-confidence results for human review.
    """
    logger.info("[VALIDATOR] Checking recommendation")

    confidence = state.get("confidence_score")
    if confidence is not None and confidence < 0.50:
        logger.warning(f"Low confidence ({confidence:.2%}) — flagging for human review")

    return {"execution_trace": ["validation_complete"]}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

class FinancialAdvisorGraph:
    """
    LangGraph orchestrator: builds and runs the multi-agent workflow.

    Usage:
        advisor = FinancialAdvisorGraph()
        result = advisor.invoke(
            user_query="Should I rebalance given rate hikes?",
            portfolio_symbols=["AAPL", "MSFT", "JPM"],
            current_allocation={"AAPL": 0.4, "MSFT": 0.4, "JPM": 0.2},
        )
        print(result["final_advice"])
    """

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        self._compiled = self._build_and_compile()

    def _build_and_compile(self):
        """Construct the graph once at init time, not on every invoke() call."""
        graph = StateGraph(AgentState)

        # Inject self.llm into the synthesiser via closure so the LLM instance
        # created after load_dotenv() is reused on every workflow run.
        llm = self.llm
        def _synthesize_node(state: AgentState) -> Dict[str, Any]:
            return synthesize_with_llm(state, llm)

        graph.add_node("market_research", market_research_agent)
        graph.add_node("risk_analysis", risk_analysis_agent)
        graph.add_node("recommendation", recommendation_agent)
        graph.add_node("synthesize", _synthesize_node)
        graph.add_node("validate", validate_recommendation)

        # Sequential: START → market → risk → recommendation → synthesize → validate → END
        graph.add_edge(START, "market_research")
        graph.add_edge("market_research", "risk_analysis")
        graph.add_edge("risk_analysis", "recommendation")
        graph.add_edge("recommendation", "synthesize")
        graph.add_edge("synthesize", "validate")
        graph.add_edge("validate", END)

        return graph.compile()

    def invoke(self,
               user_query: str,
               portfolio_symbols: List[str],
               current_allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Run the full agent workflow and return the final state.

        Args:
            user_query: The user's investment question
            portfolio_symbols: List of ticker symbols
            current_allocation: Current weights, must sum to 1.0

        Returns:
            Dict with final_advice, confidence_score, market_data,
            risk_metrics, recommendations, execution_trace, timestamp
        """
        initial_state: AgentState = {
            "user_query": user_query,
            "portfolio_symbols": portfolio_symbols,
            "current_allocation": current_allocation,
            "market_data": None,
            "risk_metrics": None,
            "recommendations": None,
            "final_advice": None,
            "confidence_score": None,
            "timestamp": datetime.now().isoformat(),
            "execution_trace": [],
        }

        logger.info(f"[ADVISOR] Starting workflow: {user_query!r}")
        final_state = self._compiled.invoke(initial_state)
        conf = final_state.get("confidence_score")
        logger.info(f"[ADVISOR] Done. Confidence: {f'{conf:.2%}' if conf is not None else 'N/A'}")

        return dict(final_state)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    advisor = FinancialAdvisorGraph()

    result = advisor.invoke(
        user_query="Should I rebalance my portfolio given recent rate hikes?",
        portfolio_symbols=["AAPL", "MSFT", "JPM", "JNJ"],
        current_allocation={"AAPL": 0.30, "MSFT": 0.30, "JPM": 0.20, "JNJ": 0.20},
    )

    print("\n" + "=" * 70)
    print("ADVISOR RESPONSE")
    print("=" * 70)
    print(f"Query:      {result['user_query']}")
    conf = result['confidence_score']
    print(f"Confidence: {f'{conf:.2%}' if conf is not None else 'N/A'}")
    print(f"\nAdvice:\n{result['final_advice']}")
    print(f"\nExecution trace:")
    for step in result["execution_trace"]:
        print(f"  → {step}")
    print("=" * 70)
