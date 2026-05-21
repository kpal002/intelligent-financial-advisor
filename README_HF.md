---
title: Intelligent Financial Advisor
emoji: 📈
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# Intelligent Financial Advisor — Finley

A production-grade multi-agent financial advisor powered by:

- **LangGraph** orchestration (5-node pipeline)
- **Claude** (Anthropic) for synthesis and natural-language reports
- **ARIMA** time-series forecasting with 95% confidence intervals
- **Markowitz** portfolio optimisation (max-Sharpe)
- **VaR / CVaR / Sharpe / Sortino** risk metrics
- **Isolation Forest** anomaly detection → buy/hold/sell signals
- **FastAPI** REST backend + **Gradio** frontend

## How to use

1. Enter your portfolio symbols (e.g. `AAPL, MSFT, JPM`) in the sidebar
2. Optionally set weights (e.g. `40, 35, 25`) — blank = equal weight
3. Click a mode card, use a quick prompt, or type your own question
4. Finley runs the full 5-agent pipeline and returns a structured advisory report

## Running locally

```bash
git clone https://github.com/kpal002/intelligent-financial-advisor
cd intelligent-financial-advisor
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python app.py
```

## Environment secrets

Set `ANTHROPIC_API_KEY` as a Space secret in Settings → Variables and secrets.
Without it the Space runs in **demo mode** (shows sample output).

## Source

GitHub: [kpal002/intelligent-financial-advisor](https://github.com/kpal002/intelligent-financial-advisor)
