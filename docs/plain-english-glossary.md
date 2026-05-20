# Plain-English glossary

Shared vocabulary for operators, reviewers, and AI assistants. Prefer these
definitions in docs and PRs; link here instead of redefining terms inline.

**See also:** [entry-points.md](entry-points.md) · [codebase-layout.md](codebase-layout.md) · [operations/independent-mode.md](operations/independent-mode.md) · [engineering.md](engineering.md)

---

## Processes and deployment

| Term | Meaning |
|------|---------|
| **Research host** | Long-running process on the research machine: `python -m research` → `research/host.py`. Runs the signal scorer and template evolution. Writes scores and **heartbeat** to Postgres. Does **not** call Grok or place IBKR orders. |
| **Trader** | Long-running process on the trader machine: `python __main__.py` → `core/agent.py`. Grok ReAct loop, tools, and IBKR execution. Reads scores and WM from Postgres each cycle. |
| **Split-host** | Production layout: one research machine + one trader machine, **same Postgres** (`DATABASE_URL`). See [operations/deployment.md](operations/deployment.md). |
| **In-process scorer** | Trader starts `signals/scorer` inside its own process when the research host heartbeat is stale. OK for dev single-box; **avoid in production** (duplicate scoring, double-writes). Block with `--require-research-host` or `TRADER_IN_PROCESS_SCORER=never`. |
| **Heartbeat** | Timestamp in `research_config` (`research_host_heartbeat_ts`; legacy read `daemon_heartbeat_ts`) updated each scoring round. Trader uses `is_research_host_operational()` (fresh heartbeat **and** status not `SHUTTING_DOWN` / `CAP_STOPPED`) to skip in-process scoring. |
| **Research host status** | Numeric codes in `research_config`: `research_host_status` (running, scoring, shutting down, cap stopped), `research_host_round`, `research_host_usage_pct`. |

---

## Quality, context, and working memory

| Term | Meaning |
|------|---------|
| **QualityMatrix** | In-process policy object (`core/quality/quality_matrix.py`): overall quality tier, risk multiplier, blocked tool categories, LLM temperature/token hints. Populated from Postgres feedback and tool logs; **enforced by the host** (agent/executor), not by individual tools. |
| **QualityMatrix learning** | Optional offline/live adjustment of execution-scoring weights from closed-trade history (`learn_from_history` / `QUALITY_MATRIX_LEARN_FROM_HISTORY`). Outcomes land in `quality_matrix_trade_outcomes` and `data/quality_matrix_learned.json`; weights stay within ±15% of the active profile base by default and **cannot move in a riskier direction** than the active ProfitConfig profile (profile risk guard). Backtests train via `simulate_backtest`; live trades via `execution_repo.record_trade`; batch replay via `scripts/train_quality_matrix.py` (cycle logs → evolved profile loop patches). |
| **Operating context** | Singleton (`core/runtime/operating_context.py`) holding mode flags: whether the researcher is available, WM source, risk multiplier, prompt-facing status. Synced from heartbeat each cycle. |
| **WM routing** | Choosing Postgres vs local JSON for working memory via `get_active_working_memory()` (`core/runtime/working_memory_access.py`). **One writer per mode** — no dual-write. See [operations/independent-mode.md](operations/independent-mode.md). |
| **Working memory (WM)** | Short-lived theses and notes in five sections: `open_theses`, `recent_verdicts`, `watching_for`, `regime_notes`, `lessons_today`. Postgres when healthy; local file `data/local_working_memory.json` in Independent Mode. |
| **Independent Mode** | Trader posture when the research host heartbeat is stale and/or WM uses local fallback: conservative risk, reduced trust in research tools, prompts show degraded context. `OperatingContext.is_independent_mode`. |
| **Full mode** | Heartbeat fresh and WM on Postgres: normal risk and research tool access. |
| **WM recovery** | When heartbeat returns, Postgres becomes WM authority again; local JSON is **not** auto-merged (Option A). Operator may merge manually after reviewing logs. |

---

## Data and external systems

| Term | Meaning |
|------|---------|
| **MDA** | MarketData.app — primary market-data API on the **research host** (quotes, candles, options). Startup health probe: `SPY` quote must return a usable `last`. |
| **Researcher token cap** | Daily budget in `research_config` (`researcher_daily_usage_YYYY-MM-DD` vs `RESEARCHER_DAILY_TOKEN_CAP`). Research host refuses start at 100%; warns above ~85%. |
| **IBKR** | Interactive Brokers — **trader host only** for orders, positions, streaming quotes, auction imbalance. |
| **Grok / xAI** | LLM used only on the trader (`core/grok_llm.py`). Zero LLM spend on the research host. |

---

## Engineering and safety

| Term | Meaning |
|------|---------|
| **Boundary extraction** | Splitting a large module into smaller pieces without changing behavior. |
| **State builder** | Assembles account, positions, market, and policy text for the agent (`StateContextBuilder`). |
| **Refactor** | Restructure code; behavior unchanged for the same inputs. |
| **Contract** | Expected shape between components (e.g. tool results: `success`, `data`, `error`). |
| **Invariant** | Rule that must stay true (e.g. safety rails fire at threshold). |
| **Regression** | Behavior that worked before and broke after a change. |
| **Blast radius** | How far a change can affect the rest of the system. |
| **Parity / characterization test** | Locks current behavior so refactors cannot silently change semantics. |
| **Hotspot file** | High-churn, high-risk file (`core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`). |
| **Cycle loop** | Trader: read state → Grok reasoning → tools → wait or trade. |
| **Cooldown / wake** | Wait between cycles unless a wake event fires early. |
| **Safety rails** | Daily loss, intraday drawdown, EOD flatten, LLM dollar and token caps (`core/runtime/safety.py`). |

---

## Profitability configuration and simulation

| Term | Meaning |
|------|---------|
| **Master ProfitConfig** | Singleton (`get_profit_config()` in `core/central_profit_config.py`) composing risk, loop, memory, prompt, and tool registry. Call `.reload()` after `PROFIT_PROFILE` or `.env` changes. |
| **ProfitConfig profile** | Named preset: built-in `conservative`, `balanced`, `aggressive`, or evolved entry in `data/evolved_profiles.json`. Set via `PROFIT_PROFILE` or `--profit-profile`. |
| **Centralized levers** | The five sub-config modules above — avoid duplicating their fields in `core.config` or `core/agent.py`. |
| **ComposedProfitConfig** | Immutable snapshot of the five levers; used by optimizer candidates and thread-local parallel workers without mutating the singleton. |
| **ReplayDataProvider** | Loads historical bars once per backtest window; shared across optimizer candidates (`data/archives/`). |
| **`--simulate`** | Trader CLI historical backtest: real ReAct loop, `BacktestLLM`, no live xAI/IBKR. See [simulation-and-optimization.md](simulation-and-optimization.md). |
| **Cycle log** | `logs/profit_cycles_YYYY-MM-DD.json` entry per trader cycle with P&L, QualityMatrix, and ProfitConfig snapshot (`log_cycle()`). |
| **ProfitConfig A/B test** | Paper-only mode (`--ab-test PROFILE_A,PROFILE_B`): one trader process rotates active profile per cycle or per session day, logs each arm via cycle logger (`ab_test` metadata), prints live comparative P&L, and writes daily winner reports (`logs/ab_test_daily_*.json`). |
| **Ops dashboard** | Password-protected web UI at `GET /dashboard` on the status API (`infra/status_api`, port 8790). Shows active profile, cycle P&L charts, QualityMatrix, research heartbeat, and simulation vs live comparison. Set `DASHBOARD_PASSWORD` in env. |
| **Strategy evolution** | Optional daily meta-review (`scripts/evolve_strategy.py`): MULTI_AGENT_MODEL analyzes ~7d cycle/trade logs and outputs a **review-only** `.patch` diff for `prompt_config.py` / tool docstrings — never auto-applied. Gate with `EVOLVE_STRATEGY_ENABLED=1`. |
| **Composite score (optimizer)** | Weighted ranking: Sharpe, profit factor, win rate; normalized when `cycles_per_day` &lt; 4. |
| **`get_research_settings()`** | Profile-aware view of memory, loop, risk, and tool levers for the research host (scorer, combiner, caches). |
| **`research_host_profit_profile`** | Postgres `research_config` key: active ProfitConfig profile label written on heartbeat sync. |
| **Profile rollback** | In **live** mode, if a new trial `PROFIT_PROFILE` draws down ≥10% from its adoption peak, the trader reverts to the previous known-good profile (`data/profile_rollback_state.json`, logs in `logs/profile_rollback_events.jsonl`). |

---

## Health scripts (exit codes)

| Code | Meaning |
|------|---------|
| **0** | Healthy |
| **1** | Degraded (warnings only) |
| **2** | Unhealthy (one or more checks failed) |

Used by `scripts/health.py`, `scripts/verify_trader_db.py`, and platform checks in `scripts/smoke_tools.py`.
