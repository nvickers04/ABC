# ABC — TODO

## Next
- [ ] Run bot with `python __main__.py` and verify IBKR connection + Grok cycle
- [ ] Swap model to `grok-4-20` when available (change `DEFAULT_MODEL` in `core/grok_llm.py`)
- [ ] Test paper trades end-to-end: quote → calculate_size → plan_order → stop placement

## Backlog
- [ ] Add MarketData.app real-time streaming (currently polling)
- [ ] Consider adding simple trade journal (append-only JSON log)
- [ ] Tune RISK_PER_TRADE for live account (start at 0.5%, increase after validation)
- [ ] Add Telegram/Discord alerting for trade fills

## Completed
- [x] Remove LiveState entirely — direct broker queries everywhere
- [x] Delete profit_metrics.py (dead after LiveState removal)
- [x] Strip data_provider.py (screener, orphan dataclasses, log stubs)
- [x] Gut tools_stats.py (625 → 118 lines — dead decision tracking removed)
- [x] Simplify tools_executor.py (delete _enrich_symbol, _prepare_session)
- [x] Simplify tools_instruments.py (delete _build_playbook)
- [x] Trim market_hours.py (delete unused convenience functions)
- [x] Fix tools_sizing.py (remove broken LiveState references, use gateway)
- [x] Clean requirements.txt (remove newsapi-python, fredapi)
- [x] Clean .env.template (remove unused API keys, add MARKETDATA_TOKEN)
