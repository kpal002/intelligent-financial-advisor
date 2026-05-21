"""
LLM Output Grounding Evaluation
================================
Checks that the advice text produced by Claude is consistent with
the ML pipeline outputs that were fed to it.

Three programmatic checks
--------------------------
1. Ticker grounding   — advice only names tickers present in market_summaries.
                        Unknown tickers may be hallucinations.

2. Number consistency — key ML numbers (Sharpe, VaR, per-symbol forecasts)
                        appear in the advice text within a rounding tolerance.
                        Missing numbers may mean Claude ignored the data.

3. Action alignment   — the action word (buy/hold/sell) from recommendations
                        appears near each ticker's mention in the advice.
                        A mismatch means the prose contradicts the signal.

Optional: Claude-as-judge (--judge flag)
-----------------------------------------
A second Claude call scores the report on four rubric dimensions (1–5 each):
  Accuracy · Grounding · Consistency · Clarity

Usage
-----
    # Fast — no network calls; tests the checker logic on synthetic data:
    python -m tests.eval_llm

    # Live — runs the real advisor pipeline (~90 s) then evaluates its output:
    python -m tests.eval_llm --live --symbols AAPL MSFT

    # Live + Claude-as-judge (two LLM calls total):
    python -m tests.eval_llm --live --symbols AAPL MSFT --judge
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════
#  GROUNDING CHECKER
# ═══════════════════════════════════════════════════════════════════

class GroundingChecker:
    """
    Analyse a Claude-generated advice string against the ML outputs
    that were passed to it, and return a structured report.

    Design notes
    ────────────
    • Ticker detection uses a regex for 2–5 uppercase letters, filtered by a
      stopword list of English words that would false-positive as tickers.

    • Number consistency is intentionally lenient (default ±5% relative
      tolerance) because Claude may round or restate values slightly.

    • Action alignment searches a ±2-sentence window around each ticker
      mention — tight enough to catch mismatches, loose enough for natural prose.
    """

    _TICKER_RE = re.compile(r'\b([A-Z]{2,5})\b')

    # Common English words that look like ticker symbols
    _STOPWORDS = {
        "I", "A", "AT", "IN", "IS", "OR", "TO", "BE", "AS", "OF", "ON",
        "BY", "IF", "NO", "SO", "DO", "MY", "AN", "UP", "US", "IT", "GO",
        "ALL", "AND", "FOR", "THE", "BUT", "NOT", "ARE", "WAS", "HAS",
        "USE", "ITS", "VIA", "KEY", "LOW", "HIGH", "MID", "HOLD", "BUY",
        "SELL", "RSI", "SMA", "EMA", "ETF", "IPO", "CEO", "CFO", "EPS",
        "FED", "GDP", "CPI", "SEC", "ESG", "AI", "ML", "API", "USD",
        "EUR", "GBP", "YOY", "QOQ", "TTM", "CAGR", "IRR", "NPV", "DCF",
        "FCF", "EBIT", "EBITDA", "VAR", "CVAR", "PE", "PB", "ROE", "ROA",
        "ARIMA", "MACD", "ADF", "VIX", "AIC", "BIC", "SLSQP", "LLM",
    }

    def __init__(
        self,
        advice:           str,
        market_summaries: dict[str, Any],
        risk_summary:     dict[str, Any],
        recommendations:  dict[str, Any],
    ):
        self.advice  = advice
        self.market  = market_summaries
        self.risk    = risk_summary
        self.recs    = recommendations
        self._valid  = set(market_summaries.keys())
        self._issues: list[str] = []
        self._passes: list[str] = []

    # ── check 1 ──────────────────────────────────────────────────────

    def check_ticker_grounding(self) -> dict:
        """
        Scan advice for uppercase tokens that look like ticker symbols.
        Flag any that are not in market_summaries (potential hallucination).
        """
        found = {
            m.group(1)
            for m in self._TICKER_RE.finditer(self.advice)
            if m.group(1) not in self._STOPWORDS
        }
        unknown = found - self._valid

        if unknown:
            self._issues.append(
                f"Advice references unknown tickers: {sorted(unknown)} — possible hallucination"
            )
        else:
            self._passes.append(
                f"All {len(found)} ticker mention(s) are in market_summaries"
            )

        return {"found": sorted(found), "unknown": sorted(unknown), "passed": not unknown}

    # ── check 2 ──────────────────────────────────────────────────────

    def check_number_consistency(self, tolerance: float = 0.05) -> dict:
        """
        Verify that key ML output values appear (approximately) in the advice.

        Strategy:
          - Extract every decimal literal from the advice
          - For each expected ML value, look for any text number within
            `tolerance` relative difference
          - Report which values are cited and which are absent
        """
        # All decimal numbers in the advice
        text_nums = {
            round(float(m.group()), 4)
            for m in re.finditer(r'-?\d+\.\d+', self.advice)
        }

        # Key values the LLM should cite
        expected: dict[str, float | None] = {
            "sharpe_ratio":  self.risk.get("sharpe_ratio"),
            "sortino_ratio": self.risk.get("sortino_ratio"),
            "var_95_daily":  (
                self.risk.get("var_95_daily") or self.risk.get("var_95")
            ),
        }
        for sym, data in self.market.items():
            expected[f"{sym}_forecast_30d"] = data.get("forecast_30d")

        matched, missing = [], []

        for label, ml_val in expected.items():
            if ml_val is None:
                continue
            hit = any(
                abs(txt - ml_val) / max(abs(ml_val), 0.001) < tolerance
                for txt in text_nums
            )
            (matched if hit else missing).append(f"{label}={ml_val}")

        if missing:
            self._issues.append(
                f"ML values absent from advice (±{tolerance:.0%} tolerance): {missing}"
            )
        if matched:
            self._passes.append(f"ML values cited in advice: {matched}")

        return {
            "text_numbers": sorted(text_nums),
            "matched":      matched,
            "missing":      missing,
            "passed":       len(missing) == 0,
        }

    # ── check 3 ──────────────────────────────────────────────────────

    def check_action_alignment(self) -> dict:
        """
        For each ticker, the recommended action word (buy/hold/sell) must
        appear within two sentences of the ticker's mention in the advice.

        Why a window?  Claude may write "For AAPL, momentum is strong.
        We recommend buying."  The action is in the next sentence, not the same.
        """
        sentences = re.split(r'(?<=[.!?\n])\s+', self.advice)
        aligned, mismatches = [], []

        for sym, rec in self.recs.items():
            action = rec.get("action", "").lower()
            if not action:
                continue

            sym_idxs = [i for i, s in enumerate(sentences) if sym in s]
            if not sym_idxs:
                # Ticker not mentioned — cannot verify
                continue

            window_text = " ".join(
                " ".join(sentences[max(0, i - 2) : min(len(sentences), i + 3)])
                for i in sym_idxs
            ).lower()

            if action in window_text:
                aligned.append(f"{sym}:{action}")
            else:
                mismatches.append(
                    f"{sym} — ML says '{action}' but that word is absent "
                    f"near the ticker in the advice"
                )

        for m in mismatches:
            self._issues.append(f"Action mismatch: {m}")
        if aligned:
            self._passes.append(f"Action-aligned tickers: {aligned}")

        return {
            "aligned":    aligned,
            "mismatches": mismatches,
            "passed":     len(mismatches) == 0,
        }

    # ── run all checks ────────────────────────────────────────────────

    def run(self) -> dict:
        return {
            "ticker_grounding":   self.check_ticker_grounding(),
            "number_consistency": self.check_number_consistency(),
            "action_alignment":   self.check_action_alignment(),
            "issues":             self._issues,
            "passes":             self._passes,
            "overall_passed":     not self._issues,
        }


# ═══════════════════════════════════════════════════════════════════
#  CLAUDE-AS-JUDGE
# ═══════════════════════════════════════════════════════════════════

def claude_judge(advice: str, ml_context: str) -> dict:
    """
    Ask a fresh Claude instance to score the advice on four rubric dimensions.

    Keeping this separate from the grounding checker means the judge sees
    the same data the report writer did — it gives an independent read,
    not a self-assessment.

    Returns a dict with integer scores (1–5 each) and an explanation string.
    """
    from anthropic import Anthropic

    client = Anthropic()

    rubric = textwrap.dedent(f"""
        You are evaluating a financial advisory report written by an AI system.
        The report was produced by a pipeline that ran ARIMA price forecasting,
        Markowitz portfolio optimisation, and Isolation Forest anomaly detection.

        The ML outputs fed to the report writer are:
        ───────────────────────────────────────────
        {ml_context}
        ───────────────────────────────────────────

        The generated report (may be truncated):
        ───────────────────────────────────────────
        {advice[:3000]}{'  …[truncated]' if len(advice) > 3000 else ''}
        ───────────────────────────────────────────

        Score the report on each dimension from 1 (poor) to 5 (excellent).
        Respond ONLY with valid JSON in exactly this format:
        {{
          "accuracy":    <int 1-5>,
          "grounding":   <int 1-5>,
          "consistency": <int 1-5>,
          "clarity":     <int 1-5>,
          "explanation": "<one concise paragraph>"
        }}

        Scoring rubric:
        - accuracy    (1-5): Do the numbers cited match the ML outputs?
        - grounding   (1-5): Are recommendations backed by the data, not invented?
        - consistency (1-5): Does buy/hold/sell in the prose match the ML signals?
        - clarity     (1-5): Is the report readable and useful for a real investor?
    """).strip()

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": rubric}],
    )

    raw = resp.content[0].text.strip()
    # Strip markdown code fences if Claude wraps with ```json ... ```
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURE DATA  (used in fast / no-network mode)
# ═══════════════════════════════════════════════════════════════════

FIXTURE_MARKET = {
    "AAPL": {
        "trend": "bullish", "forecast_30d": 198.50,
        "confidence_interval_low": 185.0, "confidence_interval_high": 212.0,
        "annual_volatility": 0.22, "forecast_mape": 4.8,
        "rsi": 61.0, "macd": 1.8,
    },
    "MSFT": {
        "trend": "neutral", "forecast_30d": 415.20,
        "confidence_interval_low": 398.0, "confidence_interval_high": 432.0,
        "annual_volatility": 0.19, "forecast_mape": 3.9,
        "rsi": 50.0, "macd": -0.3,
    },
}

FIXTURE_RISK = {
    "risk_level":    "medium",
    "var_95_daily":  -0.014,
    "cvar_95_daily": -0.021,
    "sharpe_ratio":  0.91,
    "sortino_ratio": 1.18,
    "max_drawdown":  -0.12,
    "recommended_allocation": {"AAPL": 0.58, "MSFT": 0.42},
    "stress_test_results":    {"rate_hike_2pct": -0.038},
}

FIXTURE_RECS = {
    "AAPL": {"action": "buy",  "confidence": 0.82, "rationale": "Strong momentum"},
    "MSFT": {"action": "hold", "confidence": 0.65, "rationale": "Neutral momentum"},
}

# Synthetic advice that references the fixture numbers — used to verify the checker
FIXTURE_ADVICE = """\
## Investment Advisory Report

**Executive Summary**
Based on our multi-agent quantitative analysis, we recommend a medium-risk
portfolio with a tilt toward AAPL (buy) and a hold stance on MSFT.

**Market Outlook**
AAPL shows bullish momentum with a 30-day ARIMA forecast of $198.50.
RSI sits at 61.0 with MACD of 1.8, both confirming upward continuation.
MSFT trades near fair value with a projected price of $415.20; the neutral
RSI of 50.0 indicates no strong catalyst in either direction.

**Risk Analysis**
Portfolio Sharpe ratio of 0.91 reflects solid risk-adjusted returns.
Sortino ratio of 1.18 confirms that downside risk is well-controlled.
Daily Value-at-Risk (95%) is -0.014, meaning on a bad day you should not
expect to lose more than 1.4% of portfolio value.

**Recommendations**
AAPL — buy. Strong momentum and a bullish ARIMA trend support adding
to this position with a target allocation of 58%.
MSFT — hold. Await a clearer directional catalyst before adding.

**Allocation**
Recommended weights: AAPL 58%, MSFT 42%.
""".strip()


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════

def _divider(title: str = "") -> None:
    w = 64
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (w - pad - len(title) - 2)}")
    else:
        print("─" * w)


def print_report(report: dict) -> None:

    _divider("Check 1 — Ticker Grounding")
    r = report["ticker_grounding"]
    print(f"  Tickers found in text : {r['found'] or 'none'}")
    print(f"  Unknown tickers       : {r['unknown'] or 'none'}")
    print(f"  Result : {'PASS ✓' if r['passed'] else 'FAIL ✗'}")

    _divider("Check 2 — Number Consistency")
    r = report["number_consistency"]
    print(f"  ML values cited    : {r['matched'] or 'none'}")
    print(f"  ML values missing  : {r['missing'] or 'none'}")
    print(f"  Result : {'PASS ✓' if r['passed'] else 'FAIL ✗'}")

    _divider("Check 3 — Action Alignment")
    r = report["action_alignment"]
    print(f"  Aligned  : {r['aligned'] or 'none'}")
    print(f"  Mismatch : {r['mismatches'] or 'none'}")
    print(f"  Result : {'PASS ✓' if r['passed'] else 'FAIL ✗'}")

    _divider("Summary")
    for p in report["passes"]:
        print(f"  ✓  {p}")
    for i in report["issues"]:
        print(f"  ✗  {i}")
    overall = report["overall_passed"]
    print(f"\n  Overall : {'ALL CHECKS PASSED ✓' if overall else 'ISSUES FOUND ✗'}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM grounding evaluation")
    parser.add_argument(
        "--live",    action="store_true",
        help="Run the real advisor pipeline (needs ANTHROPIC_API_KEY, ~90 s)",
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument(
        "--judge",   action="store_true",
        help="Score advice with a second Claude call (--live only)",
    )
    args = parser.parse_args()

    print("\n" + "═" * 64)
    print("   FINANCIAL ADVISOR — LLM GROUNDING EVALUATION")
    print("═" * 64)

    if not args.live:
        # ── fixture mode — no network, no API calls ───────────────
        print("  Mode : FIXTURE  (no network / no API calls)")
        print("  Using synthetic advice + ML data to verify checker logic.\n")

        checker = GroundingChecker(
            advice           = FIXTURE_ADVICE,
            market_summaries = FIXTURE_MARKET,
            risk_summary     = FIXTURE_RISK,
            recommendations  = FIXTURE_RECS,
        )
        report = checker.run()
        print_report(report)
        _divider()
        sys.exit(0 if report["overall_passed"] else 1)

    # ── live mode — real advisor invocation ───────────────────────
    symbols = [s.upper() for s in args.symbols]
    alloc   = {s: 1 / len(symbols) for s in symbols}

    print("  Mode    : LIVE")
    print(f"  Symbols : {symbols}")
    print("  Running advisor pipeline …  (60–120 s)\n")

    from src.llm.orchestrator import FinancialAdvisorGraph

    advisor = FinancialAdvisorGraph()
    result  = advisor.invoke(
        user_query         = "Should I rebalance this portfolio?",
        portfolio_symbols  = symbols,
        current_allocation = alloc,
    )

    advice = result.get("final_advice", "")
    market = result.get("market_data", {})
    risk   = result.get("risk_metrics", {})
    recs   = result.get("recommendations", {})

    # Normalise risk keys — orchestrator uses different names than the API layer
    risk_norm = {
        "sharpe_ratio":  risk.get("sharpe_ratio"),
        "sortino_ratio": risk.get("sortino_ratio"),
        "var_95_daily":  risk.get("var_95"),
    }

    checker = GroundingChecker(
        advice           = advice,
        market_summaries = market,
        risk_summary     = risk_norm,
        recommendations  = recs,
    )
    report = checker.run()
    print_report(report)

    if args.judge:
        _divider("Claude-as-Judge")
        print("  Sending report to Claude for scoring …")

        ml_ctx = json.dumps(
            {"market_data": market, "risk_metrics": risk, "recommendations": recs},
            indent=2, default=str,
        )[:4000]

        try:
            scores = claude_judge(advice, ml_ctx)
            mean   = (
                scores["accuracy"] + scores["grounding"] +
                scores["consistency"] + scores["clarity"]
            ) / 4
            print(f"  Accuracy    : {scores['accuracy']}/5")
            print(f"  Grounding   : {scores['grounding']}/5")
            print(f"  Consistency : {scores['consistency']}/5")
            print(f"  Clarity     : {scores['clarity']}/5")
            print(f"  Mean score  : {mean:.2f}/5")
            print(f"\n  Explanation:\n  {scores['explanation']}")
        except Exception as exc:
            print(f"  Judge call failed: {exc}")

    _divider()
    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
