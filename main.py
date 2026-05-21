"""
Finley — FastAPI backend
Serves the static chat UI and exposes a /chat endpoint that calls the
multi-agent advisor pipeline (or returns a demo response if no API key).

Multi-turn memory:
  - First message (empty history) → full ML pipeline (ARIMA, VaR, Markowitz)
  - Follow-up messages → Claude answers directly with conversation history +
    portfolio context, skipping the expensive data pipeline
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Advisor bootstrap ──────────────────────────────────────────────────────────
try:
    from src.llm.orchestrator import FinancialAdvisorGraph
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    _advisor_singleton = None

    def get_advisor() -> FinancialAdvisorGraph:
        global _advisor_singleton
        if _advisor_singleton is None:
            _advisor_singleton = FinancialAdvisorGraph()
        return _advisor_singleton

    _llm = ChatAnthropic(model="claude-3-5-haiku-20241022", max_tokens=2048)

    LIVE = True
except Exception as _e:
    LIVE = False
    print(f"[Finley] Advisor unavailable ({_e}). Running in demo mode.")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Finley Financial Advisor", docs_url=None, redoc_url=None)


# ── Request / Response models ──────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    symbols: str = "AAPL, MSFT"
    weights: str = ""
    history: List[HistoryMessage] = []


class ChatResponse(BaseModel):
    response: str
    live: bool


# ── Portfolio helpers ──────────────────────────────────────────────────────────
def _parse_portfolio(
    syms_raw: str, wts_raw: str
) -> tuple[list[str], dict[str, float]]:
    symbols = [s.strip().upper() for s in syms_raw.split(",") if s.strip()]
    if not symbols:
        symbols = ["AAPL", "MSFT"]

    if wts_raw.strip():
        try:
            w_list = [float(w.strip().rstrip("%")) for w in wts_raw.split(",")]
            total = sum(w_list)
            allocation = {s: w / total for s, w in zip(symbols, w_list)}
        except ValueError:
            allocation = {s: 1 / len(symbols) for s in symbols}
    else:
        allocation = {s: 1 / len(symbols) for s in symbols}

    return symbols, allocation


# ── Demo response ──────────────────────────────────────────────────────────────
def _demo_response(_query: str, symbols: list[str]) -> str:
    sym_str = " and ".join(symbols) if symbols else "your portfolio"
    s0 = symbols[0] if symbols else "AAPL"
    s1 = symbols[1] if len(symbols) > 1 else "MSFT"
    return f"""## Investment Advisory Report — *Demo Mode*

> ⚠️ **Demo mode active.** Set `ANTHROPIC_API_KEY` as a Space secret to enable live analysis.

---

**Executive Summary**

Based on our multi-agent pipeline analysis of **{sym_str}**, the portfolio sits at a
**medium risk** level with solid risk-adjusted returns. ARIMA models show bullish
momentum for the primary holding.

**Market Outlook** *(sample)*

| Symbol | Trend | 30-Day Forecast | RSI |
|--------|-------|-----------------|-----|
| {s0} | Bullish | $198.50 | 61.2 |
| {s1} | Neutral | $415.20 | 50.4 |

**Risk Metrics** *(sample)*

- **Sharpe Ratio:** 0.91 *(solid risk-adjusted return)*
- **Sortino Ratio:** 1.18 *(downside risk well-controlled)*
- **VaR (95%, daily):** −1.4%
- **Max Drawdown:** −12.3%

**Recommendations** *(sample)*

- **{s0}** → **BUY** (confidence: 82%) — Strong RSI momentum and bullish ARIMA trend.
- **{s1}** → **HOLD** (confidence: 65%) — Range-bound; await a clearer catalyst.

**Optimal Allocation** (Markowitz max-Sharpe): 58% / 42%

---
*This is a **demo response**. In live mode, Finley runs real ARIMA forecasting,
Markowitz optimisation, and Isolation Forest anomaly detection before Claude
synthesises this report.*

**Pipeline:** market_research → risk_analysis → recommendation → synthesize → validation
"""


# ── Follow-up handler (no ML pipeline, Claude answers from history) ────────────
def _followup_response(req: ChatRequest, symbols: list[str]) -> str:
    """
    Answer a follow-up question using Claude with the full conversation history.
    Skips the expensive ML pipeline — Claude already has the prior analysis in
    its context window and can answer drill-down questions directly.
    """
    system = SystemMessage(content=(
        "You are Finley, an expert financial advisor AI. "
        "You have already run a full portfolio analysis for the user (visible in the "
        "conversation history). Answer their follow-up question clearly and concisely, "
        f"referencing the prior analysis where relevant. "
        f"Current portfolio: {', '.join(symbols)}."
    ))

    messages = [system]
    for turn in req.history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=req.message))

    response = _llm.invoke(messages)
    return response.content


# ── /chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    symbols, allocation = _parse_portfolio(req.symbols, req.weights)

    if not LIVE:
        return ChatResponse(response=_demo_response(req.message, symbols), live=False)

    # Follow-up: history present → answer with Claude directly (no pipeline)
    if req.history:
        try:
            return ChatResponse(
                response=_followup_response(req, symbols),
                live=True,
            )
        except Exception as exc:
            return ChatResponse(
                response=f"⚠️ Error answering follow-up:\n\n`{exc}`",
                live=True,
            )

    # First message: run the full ML pipeline
    try:
        result = get_advisor().invoke(
            user_query=req.message,
            portfolio_symbols=symbols,
            current_allocation=allocation,
        )
        advice     = result.get("final_advice") or "No advice generated."
        confidence = result.get("confidence_score") or 0.0
        trace      = result.get("execution_trace") or []
        footer = (
            f"\n\n---\n**Confidence:** {confidence:.0%} &nbsp;|&nbsp; "
            f"**Pipeline:** {' → '.join(trace)}"
        )
        return ChatResponse(response=advice + footer, live=True)

    except Exception as exc:
        return ChatResponse(
            response=(
                f"⚠️ The advisor encountered an error:\n\n`{exc}`\n\n"
                "Please check your portfolio symbols and try again."
            ),
            live=True,
        )


# ── /status endpoint (used by the UI to show the live/demo badge) ─────────────
@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"live": LIVE})


# ── Serve static files (index.html + any assets) ──────────────────────────────
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
