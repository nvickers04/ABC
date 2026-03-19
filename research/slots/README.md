# Strategy Slots

Each file in this directory is one independently evolving strategy slot.

## Contract

- Filename format: `strategy_XX.py`
- Must define `scan(candles: pd.DataFrame, symbol: str) -> list[dict]`
- Must follow the signal schema enforced by `research/config.py`
- Must stay inside the sandbox allowlist

## Operational Notes

- The research loop evaluates all slots on the same shared candle universe.
- The selector may clone a stronger slot into a weaker slot before the next mutation step.
- Git history is recorded per slot file so evolution is reviewable after each round.