# Production launch checklist

**Use before every live deployment and after any code change.**

**See also:** [../entry-points.md](../entry-points.md) · [deployment.md](deployment.md) · [postgres.md](postgres.md) · [../plain-english-glossary.md](../plain-english-glossary.md)

**Commands:** research host `python -m research`; trader `python __main__.py --require-research-host`.

## 1. Pre-flight environment
- [ ] `cp .env.template .env` — `GROK_API_KEY`, `IBKR_ACCOUNT_ID`, `DATABASE_URL`, `TRADING_MODE`
- [ ] `python -c "from core.config import validate_config; print(validate_config() or 'OK')"`
- [ ] Postgres: `python scripts/verify_trader_db.py` (exit 0)
- [ ] TWS/Gateway API enabled (7497 paper, 7496 live)
- [ ] Research heartbeat fresh: `python scripts/health.py researcher` (exit 0–1; not 2). Dev-only fallback: `--force-in-process`
- [ ] Docker available if using compose

## 2. Research host
- [ ] `python -m research` or `docker compose -f infra/runtime/docker-compose.research.yml up -d`
- [ ] `python scripts/health.py researcher` (heartbeat, MDA, token cap)
- [ ] Template evolution running; no LLM spend on research host

## 3. Trader paper soak (mandatory before live)
- [ ] `TRADING_MODE=paper python __main__.py --test`
- [ ] `pytest tests/test_aggressive_paper_smoke.py -q` or `python tools/trader_smoke.py`
- [ ] `python scripts/smoke_tools.py --preflight --client-id 11` then safe sweep
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

**Related:** [independent-mode.md](independent-mode.md) (WM recovery after restart) · [../data-sources.md](../data-sources.md) · [../engineering.md](../engineering.md)
