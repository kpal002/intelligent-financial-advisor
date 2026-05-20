# Intelligent Financial Advisor Agent

> A production-grade multi-agent system that combines classical quantitative
> finance with LLM-powered synthesis to generate structured investment
> recommendations via a REST API.

**Stack:** LangGraph · Claude (Anthropic) · ARIMA · Markowitz · VaR · Isolation Forest · FastAPI · pytest

---

## What it does

You send a portfolio (tickers + current weights) and a question.  
The system runs a five-node agent pipeline, then returns a structured JSON response with a Claude-written advisory report.

```
POST /api/v1/advice
{
  "query": "Should I rebalance given recent Fed rate hikes?",
  "portfolio_symbols": ["AAPL", "MSFT", "JPM", "JNJ"],
  "current_allocation": {"AAPL": 0.30, "MSFT": 0.30, "JPM": 0.20, "JNJ": 0.20}
}
```

```json
{
  "advice": "## Investment Advisory Report\n\n**Executive Summary** ...",
  "confidence_score": 0.78,
  "market_summaries": {
    "AAPL": { "trend": "bullish", "forecast_30d": 198.4, "rsi": 61.2, "macd": 1.8 }
  },
  "risk_summary": {
    "risk_level": "medium", "sharpe_ratio": 0.91, "var_95_daily": -0.014,
    "recommended_allocation": { "AAPL": 0.55, "MSFT": 0.45 }
  },
  "recommendations": {
    "AAPL": { "action": "buy", "confidence": 0.82, "rationale": "Strong momentum..." }
  },
  "execution_trace": [
    "market_research_complete", "risk_analysis_complete",
    "recommendation_complete", "synthesize_complete", "validation_complete"
  ]
}
```

---

## Architecture

```
POST /api/v1/advice
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│               LangGraph Orchestrator                     │
│                                                          │
│  ┌─────────────────┐                                     │
│  │ Market Research │  yfinance → ARIMA forecast          │
│  │     Agent       │  RSI · MACD · SMA-20 · 95% CI      │
│  └────────┬────────┘                                     │
│           │                                              │
│  ┌────────▼────────┐                                     │
│  │  Risk Analysis  │  Markowitz optimisation             │
│  │     Agent       │  VaR · CVaR · Sharpe · Sortino      │
│  └────────┬────────┘  stress tests (rate hike / crash)  │
│           │                                              │
│  ┌────────▼────────┐                                     │
│  │  Recommendation │  Isolation Forest anomaly scores   │
│  │     Agent       │  momentum Z-scores → buy/hold/sell │
│  └────────┬────────┘                                     │
│           │                                              │
│  ┌────────▼────────┐                                     │
│  │ LLM Synthesizer │  Claude claude-sonnet-4-6           │
│  │    (Claude)     │  6-section advisory report          │
│  └────────┬────────┘                                     │
│           │                                              │
│  ┌────────▼────────┐                                     │
│  │   Validation    │  confidence-score sanity check      │
│  └─────────────────┘                                     │
└──────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| LangGraph `TypedDict` state | LangGraph requires dict-like state; `TypedDict` gives type hints without breaking graph serialisation |
| `Annotated[List[str], operator.add]` on `execution_trace` | Append-only reducer — parallel-node safe without extra locking |
| LLM injected via closure | `ChatAnthropic` is a Pydantic model; instance-level patching breaks. Capturing `self.llm` in a closure during `_build_and_compile()` keeps the graph testable |
| FastAPI lifespan context | Graph compiled once at startup, ready before the first request |
| `load_dotenv(override=True)` with absolute path | Shell may export an empty `ANTHROPIC_API_KEY`; `override=True` wins; absolute path resolves regardless of working directory |

---

## Project structure

```
financial-advisor/
├── src/
│   ├── ml_pipeline/
│   │   ├── time_series.py           # TimeSeriesForecaster  — ARIMA, RSI/MACD/SMA
│   │   ├── risk_metrics.py          # PortfolioRiskAnalyzer — VaR (3 methods), Sharpe, drawdown
│   │   ├── portfolio_optimization.py# PortfolioOptimizer    — Markowitz max-Sharpe / min-vol
│   │   ├── anomaly_detection.py     # AnomalyDetector       — Isolation Forest + Z-score
│   │   ├── rebalancing.py           # RebalancingDetector   — drift classifier
│   │   ├── models.py                # Shared dataclasses    — TimeSeriesForecast, RiskMetrics, …
│   │   └── __init__.py
│   ├── llm/
│   │   └── orchestrator.py          # FinancialAdvisorGraph — 5-node LangGraph workflow
│   └── api/
│       └── app.py                   # FastAPI app           — GET /health, POST /api/v1/advice
├── tests/
│   ├── test_ml_pipeline.py          # 36 tests — synthetic data, zero network calls
│   ├── test_orchestrator.py         # 15 tests — all Claude/tool calls mocked
│   ├── test_api.py                  # 18 tests — TestClient, advisor mocked
│   ├── eval_ml.py                   # ML evaluation — ARIMA backtest, Kupiec VaR, optimizer
│   └── eval_llm.py                  # LLM grounding eval + Claude-as-judge
├── .env.example                     # Template — copy to .env and add your key
├── .gitignore
└── requirements.txt
```

---

## Quick start

### 1 · Clone & install

```bash
git clone https://github.com/kpal002/intelligent-financial-advisor.git
cd intelligent-financial-advisor
pip install -r requirements.txt
```

### 2 · Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

### 3 · Run the API server

```bash
uvicorn src.api.app:app --reload
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

### 4 · Make a request

```bash
curl -s -X POST http://localhost:8000/api/v1/advice \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should I rebalance given recent Fed rate hikes?",
    "portfolio_symbols": ["AAPL", "MSFT", "JPM", "JNJ"],
    "current_allocation": {"AAPL": 0.30, "MSFT": 0.30, "JPM": 0.20, "JNJ": 0.20}
  }' | python3 -m json.tool
```

**Typical latency:** 60–120 s (ARIMA fitting dominates; roughly 15 s per symbol).

### 5 · Run the unit tests

```bash
pytest --tb=short -q   # 69 tests, ~5 s, zero network / LLM calls
```

---

## API reference

### `GET /health`

```json
{
  "status": "healthy",
  "advisor_ready": true,
  "timestamp": "2024-01-15T10:23:45.123456",
  "version": "1.0.0"
}
```

### `POST /api/v1/advice`

**Request body**

| Field | Type | Rules |
|---|---|---|
| `query` | `string` | ≥ 5 characters |
| `portfolio_symbols` | `string[]` | Non-empty; normalised to uppercase |
| `current_allocation` | `{string: float}` | Keys must match symbols; weights sum to 1.0 (±2%); all ≥ 0 |

**Response fields**

| Field | Description |
|---|---|
| `advice` | Claude-generated 6-section report (markdown) |
| `confidence_score` | Mean signal confidence across all recommendations (0–1) |
| `market_summaries` | Per-symbol ARIMA forecast + RSI/MACD/trend |
| `risk_summary` | Portfolio VaR, Sharpe, Sortino, drawdown, stress tests, optimal weights |
| `recommendations` | Per-symbol action (buy/hold/sell) + confidence + rationale |
| `execution_trace` | 5-step audit trail of which agents ran |
| `timestamp` | ISO-8601 timestamp |

**Validation errors return `422`** with a detailed message.

---

## Core ML components

### Time Series Forecaster

- **`fetch_data()`** — pulls OHLCV from yfinance
- **`test_stationarity()`** — ADF test; returns `(is_stationary, p_value)`
- **`fit_arima(forecast_horizon)`** — `auto_arima` selects (p,d,q) order; computes 95% CI and MAPE on a held-out 30-day validation window
- **`calculate_technical_indicators()`** — RSI-14, SMA-20, MACD(12,26,9)

```python
forecaster = TimeSeriesForecaster("AAPL")
forecaster.fetch_data()
forecast = forecaster.fit_arima(forecast_horizon=30)
# forecast.forecast_values, forecast.confidence_interval_lower/upper, forecast.mape
```

### Portfolio Risk Analyzer

Three VaR methods over a shared `returns_df`:

| Method | Description |
|---|---|
| `"historical"` | 5th percentile of actual daily returns |
| `"parametric"` | μ − 1.645σ assuming normal distribution |
| `"monte_carlo"` | 10,000 simulated paths from empirical μ/σ |

Also: CVaR, Sharpe, Sortino, max drawdown, correlation matrix, beta vs S&P 500.

### Portfolio Optimizer (Markowitz)

Optimisation via `scipy.optimize.minimize` (SLSQP):
- `optimize_for_max_sharpe()` — maximises (μ − r_f) / σ
- `optimize_for_min_volatility()` — minimises portfolio variance
- Constraints: weights ∈ [0, 1], Σweights = 1 (long-only, fully invested)

### Anomaly Detector

- **`engineer_features()`** — per-symbol 30-day momentum + rolling volatility
- **`detect_anomalies_isolation_forest(contamination)`** — labels {−1, +1}; ~5% anomaly rate at `contamination=0.05`
- **`detect_outliers_zscore(threshold)`** — boolean DataFrame flagging |z| > threshold

---

## Testing

```
tests/
├── test_ml_pipeline.py    36 tests   synthetic data injected — no yfinance calls
├── test_orchestrator.py   15 tests   ChatAnthropic + all tools replaced with MagicMock
└── test_api.py            18 tests   FastAPI TestClient, FinancialAdvisorGraph mocked
                           ─────────
                           69 total   ~5 s, zero network / LLM calls
```

Key patterns used:

```python
# Inject synthetic returns — bypasses yfinance entirely
analyzer = PortfolioRiskAnalyzer(["AAPL", "MSFT"])
analyzer.returns_df = synthetic_returns_dataframe

# Patch ChatAnthropic at class level (Pydantic model — instance patching fails)
with patch("src.llm.orchestrator.ChatAnthropic", return_value=mock_llm):
    advisor = FinancialAdvisorGraph()

# FastAPI TestClient with mocked advisor
with patch("src.api.app.FinancialAdvisorGraph", return_value=mock_advisor):
    from src.api.app import app
    with TestClient(app) as client:
        response = client.post("/api/v1/advice", json=valid_request)
```

---

## Evaluation

Beyond unit tests, two evaluation scripts measure real-world correctness.

### ML pipeline evaluation (`tests/eval_ml.py`)

```bash
# Default: AAPL MSFT JPM, 2-year lookback, 10 walk-forward windows (~10 min)
python -m tests.eval_ml

# Custom
python -m tests.eval_ml --symbols AAPL MSFT GOOGL --years 2 --arima-windows 8
```

| Check | What it measures |
|---|---|
| **ARIMA directional accuracy** | Walk-forward: did the 5-day forecast point the right way? Baseline: 50% (coin flip) |
| **Kupiec POF test** | Does historical 95% VaR breach ~5% of days? Tests calibration with a χ²(1) likelihood-ratio statistic |
| **Optimizer sanity** | Does max-Sharpe portfolio beat equal-weight Sharpe? Failure = bug in objective function |

### LLM grounding evaluation (`tests/eval_llm.py`)

```bash
# Fast — fixture data, no network:
python -m tests.eval_llm

# Live advisor call (~90 s):
python -m tests.eval_llm --live --symbols AAPL MSFT

# Live + Claude-as-judge (scores 1–5 on accuracy, grounding, consistency, clarity):
python -m tests.eval_llm --live --symbols AAPL MSFT --judge
```

| Check | What it catches |
|---|---|
| **Ticker grounding** | Hallucinated stock symbols — tickers Claude invented that were never analysed |
| **Number consistency** | Key ML values (Sharpe, VaR, forecasts) not cited in advice — Claude ignored the data |
| **Action alignment** | Prose says "buy" but ML recommended "sell" — contradiction between signal and text |
| **Claude-as-judge** | Independent 1–5 scores on accuracy, grounding, consistency, clarity |

Both scripts exit with code `0` (all pass) or `1` (failures found) — CI-compatible.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key — get one at console.anthropic.com |

The orchestrator loads `.env` with `override=True` so a shell-exported empty variable never blocks authentication.

Copy `.env.example` to `.env` — the example file is committed; the actual `.env` is gitignored.

---

## Tech stack

| Layer | Library / service |
|---|---|
| Orchestration | LangGraph, LangChain Anthropic |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Time series | pmdarima (auto-ARIMA), statsmodels |
| Portfolio math | scipy (SLSQP optimiser), numpy |
| Anomaly detection | scikit-learn (Isolation Forest) |
| Market data | yfinance |
| API | FastAPI, Pydantic v2, uvicorn |
| Tests | pytest, unittest.mock |
| Evaluation | scipy (chi-squared), Kupiec POF test |

---

## Possible extensions

- **Backtesting framework** — walk-forward simulation measuring Sharpe / drawdown vs S&P 500 baseline
- **GARCH volatility** — regime-sensitive vol estimates instead of rolling std
- **SHAP explainability** — per-feature contribution waterfall plots for each recommendation
- **Real-time data** — WebSocket feed from a broker API instead of daily yfinance pulls
- **Docker + CI** — containerisation and GitHub Actions pipeline

---

## License

MIT — see [LICENSE](LICENSE)
