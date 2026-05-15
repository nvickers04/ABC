# ABC Grok Autonomous Trader — Production Launch Checklist

**Use this before every live deployment and after any code change.**

## 1. Pre-Flight Environment
- [ ] `cp .env.template .env` and fill: `GROK_API_KEY`, `IBKR_ACCOUNT_ID`, `DATABASE_URL` (or PG* vars), `TRADING_MODE=live` (or paper for soak).
- [ ] `python -c "from core.config import validate_config; print(validate_config() or 'OK')"` — zero errors.
- [ ] Postgres reachable: `python -c "from memory import init_db; init_db(); print('DB OK')"` (runs migrations).
- [ ] TWS/Gateway running + API enabled (port 7496 for live, 7497 paper). Test connection: `python -c "from data.broker_gateway import create_gateway; g=create_gateway(); print(g.get_account_summary())"`.
- [ ] Research daemon heartbeat fresh (or intentionally use `--force-in-process` for dev).
- [ ] Docker (optional but recommended): `docker --version && docker compose version`.

## 2. Research Daemon (Separate Process/Host Recommended)
- [ ] `python research_daemon.py` (or Docker: `docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build`).
- [ ] Confirm heartbeat: `python -c "from core.runtime.heartbeat import is_daemon_alive, heartbeat_age_s; print(is_daemon_alive(), heartbeat_age_s())"`.
- [ ] Template evolution thread running (logs "Template evolution loop started").
- [ ] No LLM spend in daemon process (confirm via xAI dashboard or cost_tracker).

## 3. Trader Smoke in Paper First (Mandatory Soak)
- [ ] `TRADING_MODE=paper python __main__.py --test` (Grok connection).
- [ ] `TRADING_MODE=aggressive_paper python -m pytest tests/test_aggressive_paper_smoke.py -q` (or manual `python tools/trader_smoke.py`).
- [ ] Observe 1-2 cycles: all cheap tools, research() cap respected, complex orders (vertical_spread, iron_condor, covered_call, etc.) exercised with tiny size, safety not triggered, "done" with checklist.
- [ ] Cash-only: confirm "TotalCashValue" used, no margin attempts.
- [ ] Logs clean, no "CASH-ONLY", "FLATTEN INCOMPLETE", or repeated failures.

## 4. Safety & Risk Verification (Live Only After Paper Soak ≥7 days)
- [ ] Daily loss cap 15%, intraday 3%, LLM $4.5 (or .env), token ceilings active.
- [ ] `risk_ramp_approved` flag only set by successful daily_review (positive expectancy).
- [ ] EOD flatten 5min before close, gap guard 2%/15min.
- [ ] Turn limit 8-nudge / 10-hard.
- [ ] `MAX_DAILY_MULTI_AGENT_RESEARCH_USD=0.75` (or lower for live).
- [ ] Circuit breakers: consecutive failures, persistent across restart (research_config).

## 5. Docker / Ops (Production)
- [ ] Research: `docker compose -f infra/runtime/docker-compose.research.yml up -d`.
- [ ] Trader: `docker compose -f infra/runtime/docker-compose.trader.yml up -d` (or `--require-daemon`).
- [ ] Healthchecks passing: `docker ps` (healthy), logs show "Trader health OK".
- [ ] Log rotation: `logs/agent.log` + `research_daemon.log` (7-day TimedRotating).
- [ ] DB backup scheduled (`scripts/backup_postgres.ps1` or cron).
- [ ] Monitoring: tail logs for "EMERGENCY FLATTEN", "halted", "LLM cost", "daemon stale".
- [ ] TWS auto-restart handling documented for operator (midnight ET common).

## 6. Live Mode Gate
- [ ] `TRADING_MODE=live` + `RISK_PER_TRADE` ≤0.5% (or ramp-approved 1%).
- [ ] `--require-daemon` (or `TRADER_IN_PROCESS_SCORER=never`).
- [ ] First cycle: pre-scan cheap tools only (unless PRESCAN_PROMPT_EXPENSIVE_RESEARCH=1).
- [ ] Hypothesis emitted to `trader_hypotheses` table; daily_review incorporates.
- [ ] Open orders/positions monitored via `open_orders`, `positions`, `execution_status`.
- [ ] No auto-close — Grok decides via ReAct (overnight OK if conviction).

## 7. Post-Launch (First 48h)
- [ ] Daily: review `open_hypotheses()`, `review_trades()`, `execution_status()`.
- [ ] Incorporate feedback: `mark_hypothesis_incorporated`.
- [ ] xAI usage dashboard vs `cost_tracker` (within 10%).
- [ ] IBKR fills vs simulated (execution_gap in feedback_repo).
- [ ] Restart both processes cleanly; verify heartbeat + handoff flags in `research_config`.

## 8. Rollback
- `docker compose ... down`, revert to previous image/tag, `git checkout <last-good>`.
- Kill TWS API connections if needed (client_id leak).

**Sign-off**: Operator + review of last 3 cycles in logs + `open_hypotheses` count + daily P&L vs risk.

When all green: `TRADING_MODE=live python __main__.py --require-daemon`

See also: README.md (Production Deployment section), docs/RESEARCH_HOST_TRADER_SETUP.md, docs/data-sources/ibkr.md, stabilization-pr-checklist.md.