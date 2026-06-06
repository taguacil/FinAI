# FinAI Advisor

Read-only agentic trading advisor that consumes the FinAI MCP server.

It does **not** mutate the portfolio. It can refresh market data, analyze
positions, build strategies, and produce recommendations.

## Run

```bash
uv run python -m advisor.run
```

By default the advisor launches FinAI's MCP server over stdio as a subprocess.

## Layout

- `config/`      — settings + read-only tool allowlist
- `mcp_client/`  — MCP session wrapper, LangChain tool adapter, guardrails
- `agents/`      — orchestrator + specialist agents (research, strategy, risk, recommender)
- `core/`        — LangGraph wiring, shared state, session memory
- `console/`     — REPL and rich rendering

## Slash commands

- `/portfolio <name>` — select active portfolio
- `/portfolios`       — list portfolios
- `/refresh`          — refresh market data
- `/analyze`          — run full analysis on current portfolio
- `/exit`             — quit
