# ABC — TODO

## Next

## Backlog
- [ ] Add MarketData.app real-time streaming (currently polling)
- [ ] Tune RISK_PER_TRADE for live account (start at 0.5%, increase after validation)
- [ ] Swap model to `grok-4-20` when available (change `DEFAULT_MODEL` in `core/grok_llm.py`)

## Completed
- [x] **Remove market_scan auto-injection** — Grok now calls market_scan() manually each cycle
- [x] **Position management nudge** — prompt instructs Grok to monitor open positions, set stops/targets
- [x] **Super aggressive paper mode** — PAPER_AGGRESSIVE=true, 5% risk, 1.5:1 R:R, 50% confidence
- [x] **Trade execution on FINAL_DECISION** — TRADE decisions now continue the loop and place real orders
- [x] **First live paper trade** — BOT 28x SMCI $31C 2/20 @ $0.72 ($2,016)
- [x] **Tool aliases** — 20+ aliases (options_chain→option_chain, bull_call_spread→vertical_spread, etc.)
- [x] **OCC symbol redirect** — auto-parse OCC options symbols in stock order tools → buy_option
- [x] **BUY_TO_OPEN normalization** — strip _TO_OPEN/_TO_CLOSE suffixes for IBKR compatibility
- [x] **Market scan tool** — 35 liquid symbols scanned per cycle (tools/tools_scan.py)
- [x] **VIX fallback** — UVXY proxy when VIX quote unavailable
- [x] **Force LIVE market data** — reqMarketDataType(1) on IBKR connect, no more delayed data
- [x] **Fix MarketData.app auth** — Token→Bearer header fix
- [x] **Fix is_realtime check** — startswith('marketdata') instead of exact match
- [x] **Clean logging** — silenced httpx/httpcore/openai/ib_insync, think at DEBUG
- [x] **Think-loop breaker** — nudge after 3 consecutive thinks
- [x] **Strict FINAL_DECISION prompt** — JSON format required, cycles complete in 3-7 turns
- [x] **5-cycle rolling snapshots** — context continuity across cycles
- [x] **dotenv override** — load_dotenv(override=True) prevents stale shell vars
- [x] Remove LiveState entirely — direct broker queries everywhere
- [x] Lean refactor — -1,506 lines across 13 files
- [x] Clean .env.template, requirements.txt
