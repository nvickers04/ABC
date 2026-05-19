# Production launch checklist

**Use before every live deployment and after any code change.**

**Entry points:** [../entry-points.md](../entry-points.md) — research host `python -m research`, trader `python __main__.py --require-research-host`.

## 1. Pre-flight environment
- [ ] `cp .env.template .env` — `GROK_API_KEY`, `IBKR_ACCOUNT_ID`, `DATABASE_URL`, `TRADING_MODE`
- [ ] `python -c "from core.config import validate_config; print(validate_config() or 'OK')"`
- [ ] Postgres: `python -c "from memory import init_db; init_db(); print('DB OK')"`
- [ ] TWS/Gateway API enabled (7497 paper, 7496 live)
- [ ] Research heartbeat fresh (or `--force-in-process` for dev only)
- [ ] Docker available if using compose

## 2. Research host
- [ ] `python -m research` or `docker compose -f infra/runtime/docker-compose.research.yml up -d`
- [ ] Heartbeat: `python scripts/health.py researcher`
- [ ] Template evolution running; no LLM spend on research host

## 3. Trader paper soak (mandatory before live)
- [ ] `TRADING_MODE=paper python __main__.py --test`
- [ ] `pytest tests/test_aggressive_paper_smoke.py -q` or `python tools/trader_smoke.py`
- [ ] `python scripts/smoke_tools.py --client-id 11` (safe tools)
- [ ] 1–2 cycles clean: cash-only, no repeated safety trips

## 4. Safety (live only after ≥7 days paper)
- [ ] Daily loss 15%, intraday 3%, LLM budget, turn limits 8/10
- [ ] EOD flatten, gap guard, circuit breakers in `research_config`

## 5. Docker / ops
- [ ] Compose stacks healthy; log rotation; `scripts/backup_postgres.ps1` scheduled
- [ ] Monitor for `EMERGENCY FLATTEN`, `halted`, `research host stale` (heartbeat)

## 6. Live gate
- [ ] `TRADING_MODE=live`, `RISK_PER_TRADE` ≤0.5% (or ramp-approved)
- [ ] `python __main__.py --require-research-host --verbose`
- [ ] First cycles: cheap tools; hypotheses and execution_status reviewed

## 7. Post-launch (48h)
- [ ] Daily: `open_hypotheses`, `review_trades`, cost vs xAI dashboard
- [ ] Restart test: heartbeat + WM recovery per [independent-mode.md](independent-mode.md)

## 8. Rollback
- `docker compose ... down`, revert image/tag, `git checkout <last-good>`

When green: `TRADING_MODE=live python __main__.py --require-research-host`

See [entry-points.md](../entry-points.md), [deployment.md](deployment.md), [postgres.md](postgres.md), [../data-sources.md](../data-sources.md), [../engineering.md](../engineering.md).
