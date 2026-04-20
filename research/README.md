# Research Package

Configuration, simulation, and market-environment classification for the signal combination engine.

## Layout

- `config.py`: research universe, signal combination parameters, option schemas, and API budget settings
- `environment.py`: market-regime classification from shared candle data
- `simulator.py`: mechanical trade simulation and aggregate scoring

## Related Packages

- `signals/`: 50 base signal generators, combiner, templates, template evolution, scorer, and briefing

## Per-Signal Forward-Return Cadence

Every signal declares its own return cadence on the `Signal` subclass:

| Attribute             | Meaning                                  | Example          |
|-----------------------|------------------------------------------|------------------|
| `return_resolution`   | Bar resolution used for forward returns  | `"1min"`, `"D"`  |
| `return_horizon`      | Number of bars to look ahead             | `5` (5 bars)     |
| `return_lookback_days`| History fetched per round for that res   | `30`             |

Defaults are derived per category (`signals/base.py:CATEGORY_FORWARD_DEFAULTS`):

- microstructure → `5min`, h=6, lb=5d
- price → `1h`, h=4, lb=30d
- volatility → `1h`, h=24, lb=30d
- macro → `D`, h=1, lb=60d
- fundamental → `D`, h=5, lb=90d

Individual signals may override (e.g. `spread_dynamics` uses `1min/30/2`,
`earnings_momentum` uses `D/10/120`).

The scorer (`signals/scorer.py`):
1. Fetches one candle bundle per `(resolution, lookback_days)` combination
   per round (daily is always fetched; sub-daily added based on active signals).
2. For each prior-round score, looks up the **bar at score time** in that
   signal's resolution-specific candle map and writes the forward return
   keyed by **entry-bar timestamp** — multiple intraday rounds resolving to
   the same bar collapse via `INSERT OR REPLACE` (latest score wins).
3. Drops orphan-symbol scores and any scores older than 30 days so the
   per-resolution backlog budget keeps draining fresh data into IC.

The combiner (`signals/combiner.py`) gates IC trust per cadence via
`_min_obs_for(sig)` — sub-daily signals need more rows for stable IC
because their bars are tightly autocorrelated:

| Resolution | Min observations |
|-----------:|-----------------:|
| 1min       | 200              |
| 5min       | 100              |
| 15min      | 75               |
| 1h         | 50               |
| D          | 30               |

This eliminates the historical class of bug where intraday score rounds
mapping to a single daily bar wrote 3-5× duplicate forward returns,
inflating IC magnitudes and `n` simultaneously and producing fake
high-confidence signals.