# Simulation and profile optimization

This guide covers **historical backtests**, **profile grid/genetic search**, and **live profitability dashboards**. All paths share one configuration system: the master [`ProfitConfig`](../core/central_profit_config.py) singleton composed from five sub-configs and selected by **`PROFIT_PROFILE`**.

**See also:** [entry-points.md](entry-points.md) · [plain-english-glossary.md](plain-english-glossary.md#profitability-configuration-and-simulation) · [README.md](../README.md)

---

## Terminology (use consistently)

| Term | Meaning |
|------|---------|
| **Master ProfitConfig** | Process-wide singleton from `get_profit_config()`; call `.reload()` after env/CLI profile changes. Used by trader, simulation, and cycle logger. |
| **ProfitConfig profile** | Named preset: built-in `conservative` \| `balanced` \| `aggressive`, or an **evolved** name in `data/evolved_profiles.json`. Set via `PROFIT_PROFILE` or `--profit-profile`. |
| **Centralized levers** | The five sub-configs (risk, loop, memory, prompt, tools) that compose ProfitConfig — not scattered constants in `core.config`. |
| **ComposedProfitConfig** | Immutable snapshot of the five levers (optimizer cache, patches). Parallel grid workers push these via `core.profit_config_context` without mutating the singleton. |
| **ReplayDataProvider** | Loads `data/archives/` bars once per date window; each backtest session gets a lightweight `spawn_session()` view. |

---

## Centralized config (read this first)

At process startup, ABC loads profitability settings through:

| Piece | Role |
|-------|------|
| [`core/central_profit_config.py`](../core/central_profit_config.py) | **`ProfitConfig`** singleton + `simulate_backtest()` / `ReplayDataProvider` |
| [`core/profit_profiles.py`](../core/profit_profiles.py) | Built-in presets and evolved profiles in `data/evolved_profiles.json` |
| [`core/profit_profile_cache.py`](../core/profit_profile_cache.py) | In-memory cache of composed configs per profile (optimizer grid) |
| [`core/profit_config_context.py`](../core/profit_config_context.py) | Thread-local composed config for `--parallel` optimizer workers |
| [`core/risk_execution_config.py`](../core/risk_execution_config.py) | Risk rails, LLM spend caps, IBKR-related limits |
| [`core/loop_config.py`](../core/loop_config.py) | ReAct turns, gap guard, posture caps |
| [`core/memory_config.py`](../core/memory_config.py) | Working-memory and attention budgets |
| [`core/prompt_config.py`](../core/prompt_config.py) | Model tokens, R:R floors, confidence gates |
| [`core/tool_registry.py`](../core/tool_registry.py) | Tool enablement and profitability weights |

**How a profile is applied**

1. Sub-configs load from `.env` (and `TRADING_MODE` defaults).
2. If `PROFIT_PROFILE` is set (`--profit-profile` or env), overlays from `core/profit_profiles.py` patch risk / loop / memory / prompt / tools.
3. Evolved names (e.g. `evolved_v1`) load patches from `data/evolved_profiles.json`.
4. `get_profit_config().reload()` rebuilds the singleton and syncs exports into `core.config` so the live agent and `--simulate` share one instance.

**Inspect the active config**

```bash
# After setting PROFIT_PROFILE or --profit-profile
python __main__.py --config-summary

# Or only set env then summary (no trader loop)
# Windows PowerShell:
$env:PROFIT_PROFILE = "conservative"
python __main__.py --config-summary
```

Research host: `python -m research --config-summary --profit-profile balanced`.

**Profitability impact:** Profiles trade off **LLM spend**, **loss/drawdown rails**, and **tool/context budgets**. Conservative caps cost and risk; aggressive raises caps but still runs QualityMatrix and SafetyController in sim and live.

---

## Built-in `PROFIT_PROFILE` presets

| Profile | Intent | Risk & spend (high level) | Loop & gates | Memory & prompt |
|---------|--------|---------------------------|--------------|-----------------|
| **`conservative`** | Capital preservation | Lower daily loss cap (10%), tighter intraday drawdown (2%), **$3/day** LLM cap, multi-agent research off | Fewer ReAct turns, stricter quality multiplier gates, smaller tool feedback | Smaller WM/attention caps, higher min R:R on paper |
| **`balanced`** | Default | Uses `.env` / mode defaults (no profile overlay) | Standard loop and posture behavior | Standard memory and prompt budgets |
| **`aggressive`** | Seek edge (still bounded) | Higher loss/drawdown rails, **$6.50/day** LLM cap, multi-agent research on | More turns, looser failure limits, wider gap guard | Larger context, lower confidence floor, research tools weighted up |

Simulation and optimization **do not call live xAI** — they use [`BacktestLLM`](../core/simulation/backtest_llm.py) with the **real** `TradingAgent.run_cycle()` path, QualityMatrix, and SafetyController. Token lines in reports are **estimated** live-equivalent cost (same pricing table as [`data/cost_tracker.py`](../data/cost_tracker.py)), not an API bill.

**Evolved profiles** — After genetic search, save with `--save-profile NAME` and run live or in sim with:

```bash
# Windows PowerShell
$env:PROFIT_PROFILE = "evolved_v1"
python __main__.py --profit-profile evolved_v1 --simulate --sim-start 2024-06-03 --sim-end 2024-06-28
```

(macOS/Linux: `export PROFIT_PROFILE=evolved_v1`.)

---

## Historical backtest: `--simulate`

Entry point: **`python __main__.py`**. Simulation sets `ABC_SIMULATION=1`, uses archived daily bars under `data/archives/`, and never places IBKR orders.

### Required flags

| Flag | Meaning |
|------|---------|
| `--simulate [PROFILES]` | Run backtest; optional comma-separated profiles (default: `balanced`) |
| `--sim-start YYYY-MM-DD` | Inclusive start |
| `--sim-end YYYY-MM-DD` | Inclusive end |

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--sim-cash` | `100000` | Starting equity |
| `--sim-cycles-per-day` / `--cycles-per-day` | `1` | ReAct cycles per session (max 4 ET slots) |
| `--sim-csv [PATH]` | auto path under `data/sim_exports/` | Export per-trade CSV |
| `--profit-profile` | env | Sets `PROFIT_PROFILE` before load (built-in or evolved) |

**Profitability impact:** More cycles per day increases simulated token use and trade attempts; default `1` matches production-style cadence. Compare profiles on the **same** window and cycle count.

### Example commands

```bash
# Single profile, one month (balanced preset)
python __main__.py --simulate balanced --sim-start 2024-06-03 --sim-end 2024-06-28

# Compare all three built-ins on the same window
python __main__.py --simulate conservative,balanced,aggressive --sim-start 2024-06-03 --sim-end 2024-06-28

# Shorter window, one cycle per session (faster)
python __main__.py --simulate balanced --sim-start 2024-06-03 --sim-end 2024-06-14 --sim-cycles-per-day 1

# With explicit profile flag + CSV export
python __main__.py --profit-profile aggressive --simulate aggressive --sim-start 2024-06-03 --sim-end 2024-06-28 --sim-csv

# Programmatic (tests, notebooks)
python -c "from core.central_profit_config import simulate_backtest; r=simulate_backtest('balanced','2024-06-03','2024-06-14'); print(r.total_profit, r.sharpe_ratio)"
```

### What runs under the hood

1. `validate_backtest_date_range()` — date format and max calendar span (`ABC_MAX_BACKTEST_CALENDAR_DAYS`, default 400).
2. `get_profit_config().reload()` — refreshes the master ProfitConfig singleton for each profile in a compare run.
3. `ReplayDataProvider` / `get_shared_replay_data()` — loads historical bars **once** per date window; each session uses `spawn_session()`.
4. `run_backtest_async()` — replays bars, runs cycles with `BacktestLLM` (thread-local `composed=` when passed from optimizer).
5. Markdown report via `core.simulation.report` (printed to stdout).

Logs include timing and token estimates, for example:

```text
simulate_backtest complete in 12.4s | balanced: 11.8s total (archives 0.3s, cycles 11.2s) | 15 cycles / 5 sessions | tokens in=… out=… est_live=$0.01 (grok-4.3)
```

### Interpreting backtest output

**Single-profile report** (`format_backtest_report`):

| Field | Meaning |
|-------|---------|
| **Total profit** | Final equity minus initial cash |
| **Win rate** | % of closed sim fills with positive P&L |
| **Max drawdown** | Peak-to-trough on EOD equity curve |
| **Sharpe (daily, ann.)** | Annualized Sharpe from daily returns |
| **Avg R:R** | Mean realized risk/reward from trade log |
| **LLM cost (est.)** | What live Grok would have cost for the same token counts (sim spends **$0**) |
| **Notes** | Profile label, safety gates, missing-data warnings, per-cycle failures |

**Multi-profile comparison** — Markdown table ranked by metrics; **highest total profit** called out. Use this to pick a preset before paper trading — not as a guarantee of future results.

**Guards**

- `ABC_SIM_MAX_ESTIMATED_LLM_USD` — abort if estimated live-equivalent cost exceeds cap (default `25` per backtest).
- `ABC_MAX_BACKTEST_CALENDAR_DAYS` — reject oversized windows.

---

## Profile optimization: `scripts/optimize_profiles.py`

Grid-searches **built-in profiles plus perturbations** (or runs a **genetic algorithm** over key levers). Each candidate calls `evaluate_optimizer_candidate()` → `simulate_backtest(..., composed=..., replay_data=shared)` — same centralized levers as `--simulate`, without mutating the singleton when parallel.

### Grid mode (default)

```bash
# Last 30 calendar days, all candidates, JSON → data/profile_optimization.json
python scripts/optimize_profiles.py --days 30

# Quick: only conservative / balanced / aggressive (no perturbations)
python scripts/optimize_profiles.py --days 14 --quick

# Explicit one cycle per session (default; documented for clarity)
python scripts/optimize_profiles.py --days 30 --cycles-per-day 1 -o data/my_grid.json

# Parallel grid (explicit flag; bare --parallel uses 2 workers; max 4)
python scripts/optimize_profiles.py --days 14 --quick --parallel
python scripts/optimize_profiles.py --days 14 --parallel 4
```

Parallel runs use **thread-local** `ComposedProfitConfig` (`core.profit_config_context`) so the master ProfitConfig singleton is not mutated per worker. Genetic search remains **sequential** (generation loop).

**Profitability impact:** `--parallel` speeds sweeps but increases CPU/RAM; values above 2 log a warning. Composite scores normalize profit factor and win rate when `cycles_per_day` &lt; 4 (reference cadence).

### Genetic mode

```bash
python scripts/optimize_profiles.py --genetic --generations 25 --days 14

# Save winner as evolved ProfitConfig profile for PROFIT_PROFILE
python scripts/optimize_profiles.py --genetic --generations 25 --days 30 --save-profile evolved_v1 -o data/ga_result.json
```

### CLI reference

| Flag | Purpose |
|------|---------|
| `--days N` | Calendar lookback ending today (or `--end`) |
| `--end YYYY-MM-DD` | Inclusive end date |
| `-o PATH` | Output JSON path |
| `--initial-cash` | Sim starting equity |
| `--cycles-per-day` | Cycles per session (default `1`, max 4; composite normalized vs reference 4) |
| `--parallel [N]` | Optional parallel backtests (1–4 workers; bare `--parallel` → 2). Omit for sequential. |
| `--quick` | Grid: 3 built-in profiles only |
| `--baseline` | Profile for “recommended changes” diff (default `balanced`) |
| `--top N` | Rankings entries in JSON |
| `--genetic` | Enable GA |
| `--generations`, `--population`, `--elite`, `--mutation-rate`, `--seed` | GA tuning |
| `--save-profile NAME` | Write `data/evolved_profiles.json` entry |
| `--emit-snippet PATH` | Optional Python snippet for review |

### Interpreting optimizer output

**Console (per candidate)**

- `composite` — `0.4×Sharpe_norm + 0.3×profit_factor_norm + 0.3×win_rate_norm` (pf/win scaled when `cycles_per_day` &lt; 4)
- `sharpe`, `pf`, `win%`, `pnl` — raw backtest metrics
- `est=$…` — estimated live LLM USD for that run
- Wall time and token counts when `run_stats` is present

**JSON (`data/profile_optimization.json` or `data/genetic_optimization.json`)**

| Key | Meaning |
|-----|---------|
| `best` | Top candidate / genome and metrics |
| `rankings` / `top_genomes` | Leaderboard |
| `recommended_config_changes` | Diff vs `--baseline` (risk/loop/memory/prompt fields) |
| `errors` | Failed candidates (grid) |
| `runtime` | Prefetch timing, optimizer wall time, estimated LLM totals, `parallel_workers` |
| `saved_profile` | Present if `--save-profile` used |

**Pre-flight**

- Archives prefetched once per run (shared `ReplayDataProvider`).
- `ABC_OPTIMIZER_MAX_ESTIMATED_LLM_USD` (default `200`) blocks sweeps whose upper-bound estimate is too high.

Apply recommendations by setting env fields, using `--profit-profile`, or adopting an evolved profile — then confirm with `--config-summary` and a short `--simulate` window.

---

## Live profitability dashboard

After paper/live trading, cycles are logged to **`logs/profit_cycles_YYYY-MM-DD.json`** (and optionally Postgres when `DATABASE_URL` is set). Each entry snapshots the active **ProfitConfig profile** and lever subset via `log_cycle()`.

**Profitability impact:** Compare **real** LLM cost and cycle P&L to sim estimates; if live underperforms sim, prefer `conservative` or fewer cycles before raising risk.

### Terminal dashboard

```bash
# Today's session
python scripts/dashboard.py

# Specific date
python scripts/dashboard.py --date 2026-05-19

# Rolling 7-day window (terminal and HTML)
python scripts/dashboard.py --days 7

# HTML report + open in browser
python scripts/dashboard.py --days 7 --html logs/profit_report.html --open
```

**Terminal fields**

| Field | Meaning |
|-------|---------|
| Window | Session date or `last N days` |
| Cycle sum | Sum of per-cycle realized P&L deltas in the window |
| Cumulative | Running realized P&L for the window |
| LLM cost | From cost tracker (real spend on live runs) |
| Max drawdown / Win rate | Derived from logged cycles |
| Top profiles | Which `PROFIT_PROFILE` label produced best cumulative P&L |
| Outcomes | Counts of `done`, `market_hours`, halts, etc. |
| Quality | Last cycle’s QualityMatrix snapshot |

### HTTP API

If the API app is running:

```bash
# Default: last 24h (PROFIT_SUMMARY_WINDOW_HOURS)
curl http://127.0.0.1:8000/profit_summary
```

Response includes aggregated P&L, per-profile breakdown, and a config snapshot from `get_profit_config()`.

### Live profile suggestion (no backtest)

Uses **real** cycle logs only — cheap nightly check:

```bash
python __main__.py --live-optimize --live-optimize-days 7
# JSON default: data/live_profile_suggestion.json
```

When logs are empty, suggests `balanced` with confidence `none`.

### Daily summary (dashboard + optimizer + tomorrow)

End-of-day report combining today's dashboard, live log ranking, and a quick simulation grid (`--quick` by default). Writes `data/daily_summary_YYYY-MM-DD.json` and updates `data/live_profile_suggestion.json`.

```bash
python scripts/daily_summary.py
python scripts/daily_summary.py --notify          # Slack / email when env set
python scripts/daily_summary.py --skip-sim        # live logs only (fast)
python scripts/daily_summary.py --dashboard-days 7 --sim-days 14
```

Notifications (optional):

| Env | Purpose |
|-----|---------|
| `DAILY_SUMMARY_SLACK_WEBHOOK` or `ALERT_WEBHOOK_URL` | Slack incoming webhook |
| `DAILY_SUMMARY_EMAIL_TO` | Comma-separated recipients |
| `DAILY_SUMMARY_SMTP_HOST` / `PORT` / `USER` / `PASSWORD` | SMTP delivery |

---

## How to improve profitability

Use this loop; always validate with simulation before changing production env.

1. **Establish a baseline**  
   Run `python __main__.py --config-summary` with `PROFIT_PROFILE=balanced`. Note `max_daily_llm_cost`, loss rails, and min R:R.

2. **Backtest the window you care about**  
   ```bash
   python __main__.py --simulate conservative,balanced,aggressive --sim-start 2024-06-03 --sim-end 2024-06-28
   ```  
   Prefer comparable metrics: Sharpe, profit factor, drawdown, not profit alone.

3. **Search perturbations**  
   ```bash
   python scripts/optimize_profiles.py --days 30 --quick
   ```  
   Read `recommended_config_changes` in the JSON. For deeper search: `--genetic --save-profile evolved_v1`.

4. **Apply deliberately**  
   - Switch preset: `PROFIT_PROFILE=conservative` or `--profit-profile conservative` on trader start.  
   - Or adopt evolved profile from `data/evolved_profiles.json`.  
   - Re-run `--config-summary` to verify overlays.

5. **Paper trade and monitor**  
   ```bash
   python scripts/dashboard.py --days 7
   python scripts/daily_summary.py --notify
   ```  
   Compare **real** LLM cost and cycle P&L to sim estimates; use the daily summary for tomorrow's `PROFIT_PROFILE`.

6. **Guardrails**  
   - Keep simulation windows within `ABC_MAX_BACKTEST_CALENDAR_DAYS`.  
   - Use `--sim-cycles-per-day 1` or `--quick` for fast iteration.  
   - Do not run multiple traders in parallel ([entry-points](entry-points.md)).  
   - On split-host production, trader uses master ProfitConfig; research host does not run Grok trades.

**When to choose each preset**

- **Conservative** — drawdown or LLM spend too high; quality often degraded.  
- **Balanced** — default; start here after config changes.  
- **Aggressive** — sim shows better risk-adjusted results *and* live dashboard confirms stable quality; accept higher spend variance.

---

## Related environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `PROFIT_PROFILE` | All entry points | `conservative` \| `balanced` \| `aggressive` \| evolved name |
| `ABC_SIMULATION` | Set automatically in sim | Marks simulation mode |
| `ABC_MAX_BACKTEST_CALENDAR_DAYS` | Backtest / optimizer | Max inclusive calendar span (default 400) |
| `ABC_SIM_MAX_ESTIMATED_LLM_USD` | Single backtest | Estimated live-cost cap (default 25; `0` = off) |
| `ABC_OPTIMIZER_MAX_ESTIMATED_LLM_USD` | Optimizer | Sweep cap (default 200; `0` = off) |
| `ABC_SIM_LLM_PRICE_MODEL` | Token estimates | Pricing slug (default `grok-4.3`) |
| `ABC_SIM_TRADE_SCORE_THRESHOLD` | BacktestLLM | Min composite to attempt a sim trade (optional override) |
| `ABC_SIM_DEFAULT_COMPOSITE_SCORE` | Runner / BacktestLLM | Default top-symbol composite when scorer data absent (default `0.72`; floor `COMPOSITE_TRADE_THRESHOLD`) |
| `ABC_SIM_STOP_DISTANCE_PCT` | BacktestLLM | Default stop distance % for sim sizing (default `2.0`) |
| `DATABASE_URL` | Cycle logs | Optional Postgres mirror for `profit_cycle_logs` |
| `PROFIT_SUMMARY_WINDOW_HOURS` | API `/profit_summary` | Aggregation window (default 24) |

---

## See also

- [entry-points.md](entry-points.md) — trader vs research host  
- [plain-english-glossary.md](plain-english-glossary.md) — QualityMatrix, safety rails, ProfitConfig terms  
- [engineering.md](engineering.md) — change checklist when editing `core/agent.py` or config hotspots  
- Tests: `tests/test_simulation.py`, `tests/test_optimizer.py`, `tests/test_logger.py`, `tests/test_profit_config_context.py`
