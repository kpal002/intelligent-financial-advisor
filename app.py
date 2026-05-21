"""
Finley — Intelligent Financial Advisor
Gradio frontend for Hugging Face Spaces

Layout and visual design inspired by Kay Mathematics Tutor.
Two-panel layout: dark sidebar + warm-cream main content area.
"""
from __future__ import annotations  # list[str]/dict[k,v] type hints on Python < 3.10

import gradio as gr

# Gradio 5 uses type="messages" for dict-style chat; Gradio 6 removed the param.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_CHATBOT_KWARGS = {"type": "messages"} if _GRADIO_MAJOR < 6 else {}

# ── Advisor bootstrap ──────────────────────────────────────────────────────────
try:
    from src.llm.orchestrator import FinancialAdvisorGraph
    _advisor_singleton = None

    def get_advisor() -> FinancialAdvisorGraph:
        global _advisor_singleton
        if _advisor_singleton is None:
            _advisor_singleton = FinancialAdvisorGraph()
        return _advisor_singleton

    LIVE = True
except Exception as _e:
    LIVE = False
    print(f"[Finley] Advisor unavailable ({_e}). Running in demo mode.")


# ══════════════════════════════════════════════════════════════════════════════
#  CSS  — Kay-inspired palette
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Variables ─────────────────────────────────────────────────── */
:root {
    --cream:          #f5f0ea;
    --sidebar:        #241410;
    --sidebar-border: #3a1f12;
    --accent:         #c4622d;
    --accent-hover:   #a8521f;
    --card-border:    #e8ddd4;
    --text-dark:      #1c1c1c;
    --text-mid:       #555;
    --text-light:     #999;
    --font-serif:     Georgia, 'Times New Roman', serif;
    --font-sans:      -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Lock the entire page to exactly one viewport height ────────── */
/* This is the key fix: nothing can make the page taller than 100vh.  */
html, body {
    height: 100% !important;
    max-height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
gradio-app {
    display: block !important;
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .contain,
.gradio-container > .main > .wrap > .gap {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: var(--cream) !important;
    gap: 0 !important;
}
footer, .footer { display: none !important; }

/* ── Two-column shell — fills the locked viewport ───────────────── */
#app-row {
    display: flex !important;
    flex-direction: row !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
#app-row > .block,
#app-row > div { padding: 0 !important; margin: 0 !important; gap: 0 !important; }

/* ── Sidebar — flush to left edge, coffee brown ─────────────────── */
#sidebar-col {
    min-width: 250px !important;
    max-width: 250px !important;
    width: 250px !important;
    height: 100vh !important;
    overflow-y: auto !important;
    flex-shrink: 0 !important;
    background: var(--sidebar) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
#sidebar-col > .block,
#sidebar-col > div,
#sidebar-col > .block > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Main column ────────────────────────────────────────────────── */
#main-col {
    flex: 1 !important;
    min-width: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: var(--cream) !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
#main-col > .block,
#main-col > div,
#main-col > .block > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Welcome screen ─────────────────────────────────────────────── */
#welcome-col { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 0 0 40px; }
#welcome-col > .block,
#welcome-col > div,
#welcome-col > .block > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 100%;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Quick-prompt buttons — beige pill, matching reference ──────── */
#welcome-col .gap { align-items: stretch !important; }
.qbtn { display: flex !important; flex-direction: column !important; height: 100% !important; padding: 0 4px !important; }
.qbtn > .block { padding: 0 !important; height: 100% !important; display: flex !important; flex-direction: column !important; }

/* Use #welcome-col button — beats Gradio Svelte-scoped specificity */
#welcome-col button {
    background: #e8e1d8 !important;
    border: none !important;
    border-radius: 16px !important;
    color: #3a2820 !important;
    font-size: 0.95rem !important;
    font-family: var(--font-sans) !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 20px 22px !important;
    line-height: 1.55 !important;
    white-space: normal !important;
    flex: 1 !important;
    height: 100% !important;
    min-height: 80px !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
    width: 100% !important;
}
#welcome-col button:hover {
    background: #ddd5ca !important;
    color: var(--accent) !important;
}

/* ── Chat screen — flex column inside locked 100vh container ────── */
#chat-col {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    min-height: 0 !important;
}
#chat-col > .block,
#chat-col > div,
#chat-col > .block > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex: 1 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Chatbot fills all remaining flex space; its own scroll handles overflow */
#chatbot {
    flex: 1 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
#chatbot > div,
#chatbot .wrap,
#chatbot .bubble-wrap,
#chatbot .scroll-hide {
    height: 100% !important;
    overflow-y: auto !important;
}

/* Cream background on every chatbot layer */
#chatbot,
#chatbot > div,
#chatbot .wrap,
#chatbot .bubble-wrap,
#chatbot .scroll-hide,
#chatbot .message-wrap {
    background: var(--cream) !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot .message-wrap,
#chatbot .bubble-wrap { padding: 24px 56px !important; gap: 20px !important; }

/* ── Bot bubble — same cream as page, no box ────────────────────── */
#chatbot .bot,
#chatbot [data-testid="bot"],
#chatbot .message.bot,
#chatbot .message.bot > div {
    background: var(--cream) !important;   /* blend with page — no separate box */
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #1c1c1c !important;
    padding: 0 !important;
}

/* ── All text inside the chatbot — force dark & readable ────────── */
#chatbot *,
#chatbot p,
#chatbot li,
#chatbot ol,
#chatbot ul,
#chatbot blockquote,
#chatbot code,
#chatbot pre,
#chatbot strong,
#chatbot em,
#chatbot h1, #chatbot h2, #chatbot h3, #chatbot h4, #chatbot h5 {
    color: #1c1c1c !important;
}

/* ── Tables inside bot responses ────────────────────────────────── */
#chatbot table {
    border-collapse: collapse !important;
    width: auto !important;
    margin: 12px 0 !important;
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
}
#chatbot th {
    background: #ede5dc !important;
    color: #1c1c1c !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    border: 1px solid #d4c4b4 !important;
    text-align: left !important;
}
#chatbot td {
    background: transparent !important;
    color: #1c1c1c !important;
    padding: 7px 14px !important;
    border: 1px solid #d4c4b4 !important;
}
#chatbot tr:nth-child(even) td { background: rgba(196,98,45,0.04) !important; }

/* ── User bubble — accent pill ──────────────────────────────────── */
#chatbot .user,
#chatbot [data-testid="user"],
#chatbot .message.user,
#chatbot .message.user > div {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 18px !important;
    border: none !important;
    padding: 10px 16px !important;
    max-width: 75% !important;
    margin-left: auto !important;
}
#chatbot .message.user p,
#chatbot .message.user span { color: white !important; }

/* ── Input bar ──────────────────────────────────────────────────── */
#input-bar {
    background: white;
    border-top: 1px solid var(--card-border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
#input-bar > .block, #input-bar > div {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}
#msg-box textarea {
    border: 1.5px solid var(--card-border) !important;
    border-radius: 10px !important;
    background: #fdfaf7 !important;
    color: var(--text-dark) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    resize: none !important;
    min-height: 44px !important;
    max-height: 120px !important;
}
#msg-box textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(196,98,45,0.08) !important;
}
#send-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 1.1rem !important;
    min-width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    font-weight: 600 !important;
}
#send-btn:hover { background: var(--accent-hover) !important; }

/* ── Portfolio sidebar inputs ───────────────────────────────────── */
#portfolio-section textarea, #portfolio-section input {
    background: #2e1a0e !important;
    border: 1px solid #4a2a18 !important;
    border-radius: 8px !important;
    color: #e0d4ca !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
}
#portfolio-section textarea::placeholder,
#portfolio-section input::placeholder { color: #7a5c4e !important; }
#portfolio-section label span {
    color: #9a7060 !important;
    font-family: var(--font-sans) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
#portfolio-section .block, #portfolio-section > div {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 4px 16px !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HTML FRAGMENTS
# ══════════════════════════════════════════════════════════════════════════════

SIDEBAR_HTML = """
<div style="
    background:#241410;
    min-height:100vh;
    display:flex;
    flex-direction:column;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
">

  <!-- Logo block -->
  <div style="padding:28px 20px 18px; border-bottom:1px solid #3a1f12;">
    <div style="font-size:1.5rem; font-weight:700; color:#f0e8e0; letter-spacing:-0.02em; font-family:Georgia,serif;">
      Finley.
    </div>
    <div style="font-size:0.65rem; color:#7a5c4e; letter-spacing:0.14em; text-transform:uppercase; margin-top:3px;">
      Financial Advisor
    </div>
  </div>

  <!-- Recent sessions label -->
  <div style="font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#6a4a3a; padding:20px 20px 6px;">
    Recent Analyses
  </div>
  <div style="font-size:0.78rem; color:#6a4a3a; padding:2px 20px 16px; line-height:1.5;">
    Complete an analysis to see your history here.
  </div>

  <!-- Portfolio Health tracker -->
  <div style="font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#6a4a3a; padding:8px 20px 8px;">
    Portfolio Health
  </div>
  <div style="margin:0 16px 12px; background:#2e1a0e; border-radius:8px; padding:14px; border:1px solid #3a1f12;">
    <div style="font-size:0.65rem; color:#7a5c4e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
      Last Analyzed
    </div>
    <div style="font-size:0.82rem; color:#a07060;">—</div>
  </div>

  <!-- Market snapshot -->
  <div style="font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#6a4a3a; padding:8px 20px 8px;">
    Market Data
  </div>
  <div style="margin:0 16px 12px; background:#2e1a0e; border-radius:8px; padding:14px; border:1px solid #3a1f12;">
    <div style="font-size:0.65rem; color:#7a5c4e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
      Source
    </div>
    <div style="font-size:0.82rem; color:#9bc49b; display:flex; align-items:center; gap:6px;">
      <span style="width:7px;height:7px;background:#4caf50;border-radius:50%;display:inline-block;"></span>
      Yahoo Finance (live)
    </div>
  </div>

  <!-- Stats -->
  <div style="font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#6a4a3a; padding:8px 20px 8px;">
    Activity
  </div>
  <div style="padding:0 20px 8px; display:flex; flex-direction:column; gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;font-size:0.78rem;color:#6a4a3a;">
      <span>💬</span>
      <span style="color:#e0d4ca;font-weight:600;">—</span>
      <span>analyses run</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;font-size:0.78rem;color:#6a4a3a;">
      <span>📊</span>
      <span style="color:#e0d4ca;font-weight:600;">—</span>
      <span>portfolios evaluated</span>
    </div>
  </div>

  <!-- Privacy notice -->
  <div style="margin:auto 0 0; padding:14px 16px 0; border-top:1px solid #3a1f12;">
    <div style="font-size:0.7rem; color:#5a3a2a; line-height:1.55; padding:10px 4px;">
      🔒 Progress is saved privately on this device. Finley never collects or stores your identity.
    </div>
  </div>

  <!-- Bottom buttons -->
  <div style="padding:10px 16px 14px; display:flex; flex-direction:column; gap:8px;">
    <a href="https://github.com/kpal002/intelligent-financial-advisor"
       target="_blank" style="text-decoration:none;">
      <div style="
          border:1px solid #3a1f12; border-radius:7px; color:#7a5c4e; padding:9px 14px;
          font-size:0.78rem; cursor:pointer; display:flex; align-items:center; gap:8px;
          transition:all 0.15s;
      " onmouseover="this.style.color='#c4a090';this.style.borderColor='#6a4a3a'"
         onmouseout="this.style.color='#7a5c4e';this.style.borderColor='#3a1f12'">
        ⭐ &nbsp; View on GitHub
      </div>
    </a>
  </div>

  <div style="text-align:center;font-size:0.65rem;color:#4a2a1a;padding:0 0 14px;font-family:sans-serif;">
    © 2025 Kuntal Pal
  </div>
</div>
"""

WELCOME_HEADER_HTML = """
<div style="
    text-align:center;
    padding: 64px 24px 36px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
">
  <h1 style="
      font-size:2.9rem; font-weight:700; color:#1c1c1c;
      margin:0 0 14px; font-family:Georgia,'Times New Roman',serif;
      letter-spacing:-0.02em;
  ">Hello, I'm Finley.</h1>
  <p style="font-size:1rem; color:#666; line-height:1.7; margin:0;">
    Today is a great day to review your portfolio!<br>Where shall we begin?
  </p>
</div>
"""

MODE_CARDS_HTML = """
<div style="
    display:flex; gap:16px;
    max-width:820px; margin:0 auto 28px;
    padding:0 32px; box-sizing:border-box;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
">
  <!-- Card 1: Analyze portfolio -->
  <div style="
      flex:1; background:white; border:1.5px solid #e8ddd4; border-radius:12px;
      padding:24px 20px; cursor:pointer; transition:all 0.2s ease;
  "
  onmouseover="this.style.borderColor='#c4622d';this.style.boxShadow='0 4px 20px rgba(196,98,45,0.1)';this.style.transform='translateY(-1px)'"
  onmouseout="this.style.borderColor='#e8ddd4';this.style.boxShadow='none';this.style.transform='none'"
  onclick="fillPrompt('Analyze my portfolio and give me a full investment recommendation.')">
    <span style="font-size:1.5rem;display:block;margin-bottom:12px;">📊</span>
    <h3 style="font-size:0.97rem;font-weight:600;color:#1c1c1c;margin:0 0 8px;">Analyze my portfolio</h3>
    <p style="font-size:0.82rem;color:#666;line-height:1.5;margin:0;">
      Get a full report — ARIMA forecasts, Markowitz optimization,
      VaR risk metrics, and Claude's synthesis.
    </p>
  </div>

  <!-- Card 2: Ask a question (BETA) -->
  <div style="
      flex:1; background:white; border:1.5px solid #c4622d; border-radius:12px;
      padding:24px 20px; cursor:pointer; position:relative; transition:all 0.2s ease;
  "
  onmouseover="this.style.boxShadow='0 4px 20px rgba(196,98,45,0.12)';this.style.transform='translateY(-1px)'"
  onmouseout="this.style.boxShadow='none';this.style.transform='none'"
  onclick="fillPrompt('Can you explain what Sharpe ratio means and whether mine is good?')">
    <span style="
        position:absolute;top:-10px;right:16px;
        background:#c4622d;color:white;
        font-size:0.62rem;font-weight:700;letter-spacing:0.07em;
        padding:3px 9px;border-radius:4px;
    ">BETA</span>
    <span style="font-size:1.5rem;display:block;margin-bottom:12px;">💬</span>
    <h3 style="font-size:0.97rem;font-weight:600;color:#1c1c1c;margin:0 0 8px;">Ask a finance question</h3>
    <p style="font-size:0.82rem;color:#666;line-height:1.5;margin:0;">
      Drill into any concept. Finley generates fresh explanations,
      checks your understanding, and references your portfolio.
    </p>
  </div>

  <!-- Card 3: Stress test (COMING SOON) -->
  <div style="
      flex:1; background:white; border:1.5px solid #e8ddd4; border-radius:12px;
      padding:24px 20px; opacity:0.45; position:relative;
  ">
    <span style="
        position:absolute;top:-10px;right:16px;
        background:#aaa;color:white;
        font-size:0.62rem;font-weight:700;letter-spacing:0.07em;
        padding:3px 9px;border-radius:4px;
    ">COMING SOON</span>
    <span style="font-size:1.5rem;display:block;margin-bottom:12px;">⚡</span>
    <h3 style="font-size:0.97rem;font-weight:600;color:#1c1c1c;margin:0 0 8px;">Stress test my portfolio</h3>
    <p style="font-size:0.82rem;color:#666;line-height:1.5;margin:0;">
      Rate hike +2%, market crash 20%, inflation spike — see how
      your allocation holds up across 10 scenarios.
    </p>
  </div>
</div>

<script>
function fillPrompt(text) {
    // Target the Gradio textbox by its elem_id
    const box = document.querySelector('#msg-box textarea');
    if (box) {
        box.value = text;
        box.dispatchEvent(new Event('input', { bubbles: true }));
        box.focus();
    }
}
</script>
"""

DIVIDER_HTML = """
<div style="
    text-align:center; color:#aaa; font-size:0.83rem;
    margin:0 0 18px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
">
  or jump straight in with
</div>
"""


STATUS_HTML = (
    '<div id="status-chip">'
    '<span style="width:7px;height:7px;background:#4caf50;border-radius:50%;display:inline-block;"></span>'
    f'{"Ready to analyze" if LIVE else "Demo mode — set ANTHROPIC_API_KEY to enable live analysis"}'
    "</div>"
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO RESPONSE (used when advisor is not available)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_response(_query: str, symbols: list[str]) -> str:
    sym_str = " and ".join(symbols) if symbols else "your portfolio"
    return f"""## Investment Advisory Report — *Demo Mode*

> ⚠️ **Demo mode active.** Set `ANTHROPIC_API_KEY` as a Hugging Face Space secret to enable live analysis.

---

**Executive Summary**

Based on our multi-agent pipeline analysis of **{sym_str}**, the portfolio currently
sits at a **medium risk** level with solid risk-adjusted returns. The ARIMA models
show bullish momentum for the primary holding.

**Market Outlook** *(sample)*

| Symbol | Trend | 30-Day Forecast | RSI |
|--------|-------|-----------------|-----|
| {symbols[0] if symbols else "AAPL"} | Bullish | $198.50 | 61.2 |
| {symbols[1] if len(symbols) > 1 else "MSFT"} | Neutral | $415.20 | 50.4 |

**Risk Metrics** *(sample)*

- **Sharpe Ratio:** 0.91 *(solid risk-adjusted return)*
- **Sortino Ratio:** 1.18 *(downside risk well-controlled)*
- **VaR (95%, daily):** −1.4% *(max expected daily loss at 95% confidence)*
- **Max Drawdown:** −12.3%

**Recommendations** *(sample)*

- **{symbols[0] if symbols else "AAPL"}** → **BUY** (confidence: 82%) — Strong RSI momentum and bullish ARIMA trend support adding to this position.
- **{symbols[1] if len(symbols) > 1 else "MSFT"}** → **HOLD** (confidence: 65%) — Range-bound; await a clearer catalyst before adding.

**Optimal Allocation** (Markowitz max-Sharpe)

The optimizer recommends shifting toward a 58% / 42% split to improve
risk-adjusted returns.

---
*This is a **demo response**. In live mode, Finley runs real ARIMA forecasting,
Markowitz optimization, and Isolation Forest anomaly detection before Claude
synthesizes this report.*

**Pipeline:** market_research_complete → risk_analysis_complete → recommendation_complete → synthesize_complete → validation_complete
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ADVISOR CALL
# ══════════════════════════════════════════════════════════════════════════════

def _parse_portfolio(syms_raw: str, wts_raw: str) -> tuple[list[str], dict[str, float]]:
    """Parse sidebar inputs into (symbols, allocation_dict)."""
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


def call_advisor(query: str, syms_raw: str, wts_raw: str) -> str:
    symbols, allocation = _parse_portfolio(syms_raw, wts_raw)

    if not LIVE:
        return _demo_response(query, symbols)

    try:
        result = get_advisor().invoke(
            user_query=query,
            portfolio_symbols=symbols,
            current_allocation=allocation,
        )
        advice     = result.get("final_advice") or "No advice generated."
        confidence = result.get("confidence_score") or 0.0
        trace      = result.get("execution_trace") or []

        footer = (
            f"\n\n---\n"
            f"**Confidence score:** {confidence:.0%} &nbsp;|&nbsp; "
            f"**Pipeline:** {' → '.join(trace)}"
        )
        return advice + footer

    except Exception as exc:
        return (
            f"⚠️ The advisor encountered an error:\n\n`{exc}`\n\n"
            "Please check your portfolio symbols and try again."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO APP
# ══════════════════════════════════════════════════════════════════════════════

QUICK_PROMPTS = [
    "Should I rebalance given recent Fed rate hikes?",
    "Can you explain what my Sharpe ratio actually means?",
    "I need to reduce risk — where should I start?",
    "Show me how Isolation Forest detects anomalies in my portfolio",
]

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Finley — Financial Advisor") as demo:

    # ── App-level state ────────────────────────────────────────────────────────
    chat_history = gr.State([])

    # ══════════════════════════════════════════════════════════════════════════
    with gr.Row(elem_id="app-row", equal_height=True):

        # ── SIDEBAR ───────────────────────────────────────────────────────────
        with gr.Column(elem_id="sidebar-col", scale=0, min_width=250):
            gr.HTML(SIDEBAR_HTML)

            with gr.Column(elem_id="portfolio-section"):
                gr.HTML("""
                  <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                              color:#555;padding:12px 20px 6px;font-family:sans-serif;">
                    Portfolio Setup
                  </div>
                """)
                symbols_box = gr.Textbox(
                    label="Symbols (comma-separated)",
                    value="AAPL, MSFT",
                    placeholder="AAPL, MSFT, JPM, JNJ",
                )
                weights_box = gr.Textbox(
                    label="Weights % (blank = equal)",
                    placeholder="60, 40   ← or leave blank",
                )

        # ── MAIN CONTENT ──────────────────────────────────────────────────────
        with gr.Column(elem_id="main-col", scale=1):

            # Status chip (top-right feel via HTML)
            gr.HTML(f"""
              <div style="display:flex;justify-content:flex-end;padding:14px 20px 0;">
                <div style="
                    display:inline-flex;align-items:center;gap:6px;
                    background:{'#1e3a1e' if LIVE else '#3a2a1a'};
                    border:1px solid {'#2d5a2d' if LIVE else '#5a3a1a'};
                    border-radius:20px;padding:5px 14px;
                    font-size:0.72rem;font-family:sans-serif;
                    color:{'#7ec87e' if LIVE else '#c49a4a'};
                ">
                  <span style="width:7px;height:7px;border-radius:50%;display:inline-block;
                               background:{'#4caf50' if LIVE else '#c49a4a'};"></span>
                  {"Ready to analyze" if LIVE else "Demo mode — add ANTHROPIC_API_KEY to enable live analysis"}
                </div>
              </div>
            """)

            # ── WELCOME SCREEN ────────────────────────────────────────────────
            with gr.Column(elem_id="welcome-col", visible=True) as welcome_col:

                gr.HTML(WELCOME_HEADER_HTML)
                gr.HTML(MODE_CARDS_HTML)
                gr.HTML(DIVIDER_HTML)

                # Quick-prompt 2×2 grid
                with gr.Row():
                    q_btns = []
                    for i in range(0, 4, 2):
                        with gr.Column():
                            for j in range(2):
                                if i + j < len(QUICK_PROMPTS):
                                    b = gr.Button(
                                        QUICK_PROMPTS[i + j],
                                        elem_classes=["qbtn"],
                                    )
                                    q_btns.append(b)

            # ── CHAT SCREEN ───────────────────────────────────────────────────
            with gr.Column(elem_id="chat-col", visible=False) as chat_col:
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=600,
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=finley"),
                    **_CHATBOT_KWARGS,
                )

            # ── INPUT BAR (always visible) ────────────────────────────────────
            with gr.Row(elem_id="input-bar"):
                msg_box  = gr.Textbox(
                    show_label=False,
                    placeholder="Ask Finley a question, share your portfolio, or describe your goals…",
                    elem_id="msg-box",
                    lines=1,
                    max_lines=4,
                    scale=8,
                )
                send_btn = gr.Button("↑", elem_id="send-btn", scale=0, min_width=48)

    # ══════════════════════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def submit_message(query, history, syms, wts):
        """Handle a new user message. History is a list of role/content dicts."""
        query = query.strip()
        if not query:
            yield history, "", gr.update(visible=True), gr.update(visible=False)
            return

        history = list(history) + [{"role": "user", "content": query}]
        yield history, "", gr.update(visible=False), gr.update(visible=True)

        # Typing indicator
        thinking = history + [{"role": "assistant", "content": "⏳ Analyzing your portfolio… (60–120 s in live mode)"}]
        yield thinking, "", gr.update(visible=False), gr.update(visible=True)

        response = call_advisor(query, syms, wts)
        history = history + [{"role": "assistant", "content": response}]
        yield history, "", gr.update(visible=False), gr.update(visible=True)

    def quick_prompt_click(prompt_text, history, syms, wts):
        """Quick-prompt button: run prompt directly."""
        yield from submit_message(prompt_text, history, syms, wts)

    # Wire send button and Enter key
    send_outputs = [chat_history, msg_box, welcome_col, chat_col]

    send_btn.click(
        submit_message,
        inputs=[msg_box, chat_history, symbols_box, weights_box],
        outputs=send_outputs,
    )
    msg_box.submit(
        submit_message,
        inputs=[msg_box, chat_history, symbols_box, weights_box],
        outputs=send_outputs,
    )

    # Wire quick prompts
    for btn in q_btns:
        btn.click(
            quick_prompt_click,
            inputs=[btn, chat_history, symbols_box, weights_box],
            outputs=send_outputs,
        )

    # Keep chatbot display in sync with history state
    chat_history.change(lambda h: h, inputs=[chat_history], outputs=[chatbot])


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
