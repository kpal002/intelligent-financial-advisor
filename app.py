"""
Finley — Intelligent Financial Advisor
Gradio frontend for Hugging Face Spaces

Clean rebuild: sidebar is a position:fixed HTML overlay; gr.ChatInterface
handles all chat logic so we never fight Gradio's internal layout wrappers.
"""
from __future__ import annotations  # list[str]/dict[k,v] hints on Python < 3.10

import gradio as gr

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
#  CSS  — minimal; sidebar is a fixed HTML overlay, not a Gradio column
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Core reset ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; background: #f5f0ea; }
footer, .footer, .svelte-footer { display: none !important; }

/* ── Push Gradio content right of the 260px fixed sidebar ── */
.gradio-container {
    margin-left: 260px !important;
    max-width: calc(100% - 260px) !important;
    padding: 0 !important;
    background: #f5f0ea !important;
}

/* ── ChatInterface wrapper ───────────────────────────────── */
.gradio-chatinterface { background: #f5f0ea !important; }
.gradio-chatinterface > div { background: #f5f0ea !important; }

/* ── Chatbot display ─────────────────────────────────────── */
#chatbot {
    background: #f5f0ea !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot .bubble-wrap,
#chatbot .message-wrap,
#chatbot .scroll-hide,
#chatbot > div {
    background: #f5f0ea !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot .message-wrap { padding: 20px 56px !important; gap: 18px !important; }

/* Bot bubble — blends with page background */
#chatbot .bot,
#chatbot [data-testid="bot"],
#chatbot .message.bot,
#chatbot .message.bot > div {
    background: #f5f0ea !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Force all chatbot text dark */
#chatbot *,
#chatbot p, #chatbot li, #chatbot ul, #chatbot ol,
#chatbot h1, #chatbot h2, #chatbot h3, #chatbot h4,
#chatbot strong, #chatbot em, #chatbot code, #chatbot blockquote {
    color: #1c1c1c !important;
}

/* User bubble — terracotta pill */
#chatbot .user,
#chatbot [data-testid="user"],
#chatbot .message.user,
#chatbot .message.user > div {
    background: #c4622d !important;
    color: white !important;
    border-radius: 18px !important;
    border: none !important;
    box-shadow: none !important;
    padding: 10px 16px !important;
    max-width: 75% !important;
    margin-left: auto !important;
}
#chatbot .message.user p,
#chatbot .message.user span,
#chatbot .message.user * { color: white !important; }

/* Tables in bot responses */
#chatbot table { border-collapse: collapse !important; font-size: 0.88rem !important; margin: 10px 0 !important; }
#chatbot th { background: #ede5dc !important; color: #1c1c1c !important; padding: 8px 14px !important; border: 1px solid #d4c4b4 !important; font-weight: 600 !important; text-align: left !important; }
#chatbot td { color: #1c1c1c !important; padding: 7px 14px !important; border: 1px solid #d4c4b4 !important; }
#chatbot tr:nth-child(even) td { background: rgba(196,98,45,0.04) !important; }

/* ── Input row ────────────────────────────────────────────── */
.chatbot-input,
[data-testid="chatbot-input"],
.gradio-chatinterface .input-row,
.gradio-chatinterface > div > div:last-child {
    background: white !important;
    border-top: 1px solid #e8ddd4 !important;
    padding: 10px 16px !important;
}

/* Textbox inside input row */
.gradio-chatinterface textarea {
    background: #fdfaf7 !important;
    border: 1.5px solid #e8ddd4 !important;
    border-radius: 10px !important;
    color: #1c1c1c !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    resize: none !important;
}
.gradio-chatinterface textarea:focus {
    border-color: #c4622d !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(196,98,45,0.08) !important;
}

/* Submit button */
.gradio-chatinterface button[aria-label="Submit"],
.gradio-chatinterface .submit-btn {
    background: #c4622d !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
}
.gradio-chatinterface button[aria-label="Submit"]:hover { background: #a8521f !important; }

/* ── Examples (quick-prompt chips) ───────────────────────── */
.gradio-chatinterface .examples-row { padding: 0 16px 10px !important; gap: 8px !important; }
.gradio-chatinterface .examples-row button,
.example-btn {
    background: #e8e1d8 !important;
    border: none !important;
    border-radius: 20px !important;
    color: #3a2820 !important;
    font-size: 0.84rem !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}
.gradio-chatinterface .examples-row button:hover { background: #ddd5ca !important; color: #c4622d !important; }

/* ── Additional inputs (portfolio settings) ───────────────── */
.gradio-chatinterface .additional-inputs,
.gradio-chatinterface .additional-inputs-accordion {
    background: #f5f0ea !important;
    border: 1px solid #e8ddd4 !important;
    border-radius: 8px !important;
    margin: 0 16px 8px !important;
}
.gradio-chatinterface .additional-inputs input,
.gradio-chatinterface .additional-inputs textarea {
    background: white !important;
    border: 1px solid #e8ddd4 !important;
    border-radius: 6px !important;
    color: #1c1c1c !important;
    font-size: 0.88rem !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR HTML  — position:fixed overlay, zero Gradio involvement
# ══════════════════════════════════════════════════════════════════════════════

SIDEBAR_HTML = """
<div style="
    position:fixed; top:0; left:0; bottom:0; width:260px;
    background:#241410; z-index:1000; overflow-y:auto;
    display:flex; flex-direction:column;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    border-right:1px solid #3a1f12;
">

  <!-- Logo -->
  <div style="padding:28px 20px 18px; border-bottom:1px solid #3a1f12; flex-shrink:0;">
    <div style="font-size:1.5rem;font-weight:700;color:#f0e8e0;letter-spacing:-0.02em;font-family:Georgia,serif;">
      Finley.
    </div>
    <div style="font-size:0.65rem;color:#7a5c4e;letter-spacing:0.14em;text-transform:uppercase;margin-top:3px;">
      Financial Advisor
    </div>
  </div>

  <!-- Recent analyses -->
  <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#6a4a3a;padding:20px 20px 6px;">
    Recent Analyses
  </div>
  <div style="font-size:0.78rem;color:#6a4a3a;padding:2px 20px 16px;line-height:1.5;">
    Complete an analysis to see your history here.
  </div>

  <!-- Portfolio Health -->
  <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#6a4a3a;padding:8px 20px 8px;">
    Portfolio Health
  </div>
  <div style="margin:0 16px 12px;background:#2e1a0e;border-radius:8px;padding:14px;border:1px solid #3a1f12;">
    <div style="font-size:0.65rem;color:#7a5c4e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
      Last Analyzed
    </div>
    <div style="font-size:0.82rem;color:#a07060;">—</div>
  </div>

  <!-- Market data -->
  <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#6a4a3a;padding:8px 20px 8px;">
    Market Data
  </div>
  <div style="margin:0 16px 12px;background:#2e1a0e;border-radius:8px;padding:14px;border:1px solid #3a1f12;">
    <div style="font-size:0.65rem;color:#7a5c4e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
      Source
    </div>
    <div style="font-size:0.82rem;color:#9bc49b;display:flex;align-items:center;gap:6px;">
      <span style="width:7px;height:7px;background:#4caf50;border-radius:50%;display:inline-block;"></span>
      Yahoo Finance (live)
    </div>
  </div>

  <!-- Activity -->
  <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#6a4a3a;padding:8px 20px 8px;">
    Activity
  </div>
  <div style="padding:0 20px 8px;display:flex;flex-direction:column;gap:8px;">
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

  <!-- Spacer -->
  <div style="flex:1;"></div>

  <!-- Privacy notice -->
  <div style="padding:14px 16px 0;border-top:1px solid #3a1f12;flex-shrink:0;">
    <div style="font-size:0.7rem;color:#5a3a2a;line-height:1.55;padding:10px 4px;">
      🔒 Progress is saved privately on this device. Finley never collects or stores your identity.
    </div>
  </div>

  <!-- GitHub link -->
  <div style="padding:10px 16px 14px;flex-shrink:0;">
    <a href="https://github.com/kpal002/intelligent-financial-advisor"
       target="_blank" style="text-decoration:none;">
      <div style="
          border:1px solid #3a1f12;border-radius:7px;color:#7a5c4e;
          padding:9px 14px;font-size:0.78rem;cursor:pointer;
          display:flex;align-items:center;gap:8px;
      ">
        ⭐ &nbsp;View on GitHub
      </div>
    </a>
  </div>

  <div style="text-align:center;font-size:0.65rem;color:#4a2a1a;padding:0 0 14px;">
    © 2025 Kuntal Pal
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN  — gr.HTML component (renders onclick/script correctly)
#  Chatbot placeholder is plain text; gr.Chatbot sanitises arbitrary HTML.
# ══════════════════════════════════════════════════════════════════════════════

_badge_color  = ("#1e3a1e", "#2d5a2d", "#4caf50", "#7ec87e", "Ready to analyze") if LIVE \
    else        ("#3a2a1a", "#5a3a1a", "#c49a4a", "#c49a4a", "Demo mode — add ANTHROPIC_API_KEY")

WELCOME_HTML = f"""
<div id="finley-welcome" style="
    background:#f5f0ea;
    display:flex; flex-direction:column; align-items:center;
    padding:48px 32px 28px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
">

  <!-- Status badge -->
  <div style="margin-bottom:28px;">
    <span style="display:inline-flex;align-items:center;gap:6px;
                 background:{_badge_color[0]};border:1px solid {_badge_color[1]};
                 border-radius:20px;padding:4px 14px;font-size:0.72rem;color:{_badge_color[3]};">
      <span style="width:7px;height:7px;border-radius:50%;background:{_badge_color[2]};display:inline-block;"></span>
      {_badge_color[4]}
    </span>
  </div>

  <!-- Headline -->
  <h1 style="font-size:2.5rem;font-weight:700;color:#1c1c1c;margin:0 0 10px;
             font-family:Georgia,serif;letter-spacing:-0.02em;text-align:center;">
    Hello, I'm Finley.
  </h1>
  <p style="font-size:1rem;color:#666;line-height:1.7;text-align:center;
            margin:0 0 32px;max-width:460px;">
    Today is a great day to review your portfolio!<br>Where shall we begin?
  </p>

  <!-- Mode cards -->
  <div style="display:flex;gap:14px;max-width:720px;width:100%;flex-wrap:wrap;">

    <div style="flex:1;min-width:180px;background:white;border:1.5px solid #e8ddd4;
                border-radius:12px;padding:22px 18px;cursor:pointer;transition:all 0.2s;"
         onmouseover="this.style.borderColor='#c4622d';this.style.transform='translateY(-2px)'"
         onmouseout="this.style.borderColor='#e8ddd4';this.style.transform='none'"
         onclick="finleyFill('Analyze my portfolio and give me a full investment recommendation.')">
      <span style="font-size:1.4rem;display:block;margin-bottom:10px;">📊</span>
      <h3 style="font-size:0.94rem;font-weight:600;color:#1c1c1c;margin:0 0 6px;">Analyze my portfolio</h3>
      <p style="font-size:0.81rem;color:#666;line-height:1.5;margin:0;">
        Full ARIMA, Markowitz &amp; VaR analysis with Claude's synthesis.
      </p>
    </div>

    <div style="flex:1;min-width:180px;background:white;border:1.5px solid #c4622d;
                border-radius:12px;padding:22px 18px;cursor:pointer;position:relative;transition:all 0.2s;"
         onmouseover="this.style.transform='translateY(-2px)'"
         onmouseout="this.style.transform='none'"
         onclick="finleyFill('Can you explain what Sharpe ratio means and whether mine is good?')">
      <span style="position:absolute;top:-10px;right:14px;background:#c4622d;color:white;
                   font-size:0.6rem;font-weight:700;letter-spacing:0.07em;
                   padding:3px 8px;border-radius:4px;">BETA</span>
      <span style="font-size:1.4rem;display:block;margin-bottom:10px;">💬</span>
      <h3 style="font-size:0.94rem;font-weight:600;color:#1c1c1c;margin:0 0 6px;">Ask a finance question</h3>
      <p style="font-size:0.81rem;color:#666;line-height:1.5;margin:0;">
        Drill into any concept with fresh explanations and examples.
      </p>
    </div>

    <div style="flex:1;min-width:180px;background:white;border:1.5px solid #e8ddd4;
                border-radius:12px;padding:22px 18px;opacity:0.45;position:relative;">
      <span style="position:absolute;top:-10px;right:14px;background:#aaa;color:white;
                   font-size:0.6rem;font-weight:700;letter-spacing:0.07em;
                   padding:3px 8px;border-radius:4px;">COMING SOON</span>
      <span style="font-size:1.4rem;display:block;margin-bottom:10px;">⚡</span>
      <h3 style="font-size:0.94rem;font-weight:600;color:#1c1c1c;margin:0 0 6px;">Stress test my portfolio</h3>
      <p style="font-size:0.81rem;color:#666;line-height:1.5;margin:0;">
        Rate hike, crash, inflation — see how your allocation holds up.
      </p>
    </div>

  </div>

  <p style="font-size:0.82rem;color:#bbb;margin:20px 0 0;">or use the quick prompts below ↓</p>

</div>

<script>
// Fill the ChatInterface textarea when a card is clicked
function finleyFill(text) {{
  const ta = Array.from(document.querySelectorAll('textarea'))
               .find(t => t.offsetParent !== null && !t.readOnly && !t.disabled);
  if (!ta) return;
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  setter.call(ta, text);
  ta.dispatchEvent(new Event('input', {{bubbles: true}}));
  ta.focus();
}}

// Hide welcome section once the first chat message appears
(function() {{
  function hideWelcome() {{
    const el = document.getElementById('finley-welcome');
    if (el) el.style.display = 'none';
  }}
  const obs = new MutationObserver(function() {{
    const bot = document.querySelector('#chatbot .message, #chatbot [data-testid="bot"], #chatbot [data-testid="user"]');
    if (bot) {{ hideWelcome(); obs.disconnect(); }}
  }});
  document.addEventListener('DOMContentLoaded', function() {{
    const root = document.getElementById('chatbot') || document.body;
    obs.observe(root, {{childList: true, subtree: true}});
  }});
}})();
</script>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def _demo_response(_query: str, symbols: list[str]) -> str:
    sym_str = " and ".join(symbols) if symbols else "your portfolio"
    s0 = symbols[0] if symbols else "AAPL"
    s1 = symbols[1] if len(symbols) > 1 else "MSFT"
    return f"""## Investment Advisory Report — *Demo Mode*

> ⚠️ **Demo mode active.** Set `ANTHROPIC_API_KEY` as a Hugging Face Space secret to enable live analysis.

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
Markowitz optimization, and Isolation Forest anomaly detection before Claude
synthesizes this report.*

**Pipeline:** market_research_complete → risk_analysis_complete → recommendation_complete → synthesize_complete → validation_complete
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ADVISOR CALL
# ══════════════════════════════════════════════════════════════════════════════

def _parse_portfolio(syms_raw: str, wts_raw: str) -> tuple[list[str], dict[str, float]]:
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

def respond(
    message: str,
    _history: list,   # gr.ChatInterface passes history; we don't need it
    symbols: str,
    weights: str,
) -> str:
    """Chat handler for gr.ChatInterface."""
    return call_advisor(message.strip(), symbols, weights)


# When additional_inputs are present, gr.ChatInterface requires examples to be
# a list of lists: [message, <value for each additional_input>, ...]
# Here: [message, symbols, weights]
EXAMPLES = [
    ["Should I rebalance given recent Fed rate hikes?",        "AAPL, MSFT", ""],
    ["Can you explain what my Sharpe ratio actually means?",   "AAPL, MSFT", ""],
    ["I need to reduce risk — where should I start?",          "AAPL, MSFT", ""],
    ["Show me how Isolation Forest detects anomalies in my portfolio", "AAPL, MSFT", ""],
]

with gr.Blocks(
    css=CSS,
    theme=gr.themes.Base(),
    title="Finley — Financial Advisor",
    analytics_enabled=False,
) as demo:

    # ── Fixed sidebar overlay — rendered at DOM root, styled with inline CSS ──
    gr.HTML(SIDEBAR_HTML)

    # ── Welcome screen (rendered via gr.HTML so onclick/script are preserved) ──
    gr.HTML(WELCOME_HTML)

    # ── Main chat interface ────────────────────────────────────────────────────
    gr.ChatInterface(
        fn=respond,
        type="messages",          # set on ChatInterface to silence UserWarning
        chatbot=gr.Chatbot(
            elem_id="chatbot",
            placeholder="Type a question or click an example below…",
            show_label=False,
            avatar_images=(
                None,
                "https://api.dicebear.com/7.x/bottts/svg?seed=finley",
            ),
        ),
        additional_inputs=[
            gr.Textbox(
                label="Portfolio Symbols (comma-separated)",
                value="AAPL, MSFT",
                placeholder="AAPL, MSFT, JPM, JNJ",
            ),
            gr.Textbox(
                label="Weights % (leave blank for equal weight)",
                placeholder="60, 40   ← or leave blank",
            ),
        ],
        additional_inputs_accordion=gr.Accordion(
            "⚙️  Portfolio Settings", open=False
        ),
        examples=EXAMPLES,
        cache_examples=False,     # prevent running full ARIMA pipeline at startup
        title="",
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)
