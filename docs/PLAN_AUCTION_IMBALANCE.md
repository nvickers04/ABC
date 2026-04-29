# Plan: Auction Imbalance Signal

Status: **SHIPPED** (commit `bb8afde`, 2026-04-29).
Final test count: **778 passed / 4 skipped** (was 742 before this work).

## Decisions made (resolves §6 open questions)

1. Tick list = `"225"` only.  Did not bundle `233` (volume rate) or keep
   the spurious `588` — they weren't needed for the imbalance feed.
2. **No** config gate.  The change is the new default.  Auction ticks
   piggyback on the existing stock subscription and consume zero
   additional lines, so there's no operational cost to leaving them on.
3. Use the wider 09:25–09:30 ET open window unconditionally (covers
   NASDAQ).  ~3 minutes of low-confidence scores for NYSE-listed names
   that don't publish until 09:28 is acceptable; not worth the
   `primaryExchange` lookup complexity.

## What actually shipped (vs original plan)

- Data layer fixes landed exactly as specified.
- **Bug caught during implementation:** `_ticker_value` filtered values
  ≤ 0 as "unset", but auction imbalance is **signed** (negative = sell-side).
  Introduced `_signed_ticker_value` to preserve negatives.  Without this
  fix, every sell-side imbalance would have been silently dropped.  The
  original plan said "use the existing `_ticker_value` helper" — that
  was wrong.
- Public `Quote` dataclass in `data/data_provider.py` was extended too
  (the plan only mentioned `IBKRQuote`).  `_ibkr_quote_to_quote`
  propagates the four new fields so signals reading
  `data["quote"].auction_imbalance` get the right values.
- ADV is computed inside the signal from the existing `candles_daily`
  rather than via a new `data["adv_20"]` key — fewer scorer changes.
- `compute_auction_score(...)` is exposed at module scope as a pure
  function so tests drive it without reconstructing a full data dict
  (mirrors the `core/runtime/cadence.py` pattern).
- Tests: **21** in `tests/test_auction_imbalance.py` and **4** new in
  `tests/test_ibkr_quote_source.py` (covering field extraction,
  NaN→None, zero preserved, and missing-attr defaults).  Plan estimated
  ~10 tests.

## Files touched (final)

| File | Change |
|---|---|
| `data/ibkr_quote_source.py` | `DEFAULT_GENERIC_TICK_LIST` `"588"`→`"225"`; `_signed_ticker_value` helper; `_read_ticker` extracts 4 fields; `IBKRQuote` extended. |
| `data/data_provider.py` | `Quote` dataclass extended with the 4 fields; `_ibkr_quote_to_quote` propagates them. |
| `signals/auction_imbalance.py` | NEW. Microstructure, tier 1, every round, 1-min × 30 horizon. |
| `tests/test_auction_imbalance.py` | NEW. 21 tests. |
| `tests/test_ibkr_quote_source.py` | Updated existing assertion to `"225"`; +4 tests for `_read_ticker` auction extraction. |

---

## Original plan (preserved below for context)

---

## 0. Why

NYSE / NASDAQ publish order-imbalance feeds in the minutes leading into the
opening cross (09:28→09:30 ET) and closing cross (15:50→16:00 ET).  The
imbalance number — net buy vs sell shares unable to match at the indicative
price — is one of the most causally tight short-horizon signals available
to a retail trader: it precedes the auction print by minutes, the print
itself is a measurable forward return, and it costs **zero IBKR data lines**
because it piggybacks on the existing stock subscription.

We currently subscribe to genericTickList `"588"` in
[data/ibkr_quote_source.py](../data/ibkr_quote_source.py#L46) but `"588"`
is **not** the imbalance code.  IBKR's documented generic tick for auction
imbalance is **`225`**, which delivers ticks:

| Tick ID | Field                  | ib_insync attr           |
| ------- | ---------------------- | ------------------------ |
| 34      | Auction volume         | `ticker.auctionVolume`   |
| 35      | Auction price          | `ticker.auctionPrice`    |
| 36      | Auction imbalance      | `ticker.auctionImbalance`|
| 61      | Regulatory imbalance   | `ticker.regulatoryImbalance` |

So step 1 is fixing the data layer.  Step 2 is the signal.

---

## 1. Data layer fix

**File:** [data/ibkr_quote_source.py](../data/ibkr_quote_source.py)

- Change `DEFAULT_GENERIC_TICK_LIST = "588"` → `"225"`.
  - (If we want both volume-rate ticks and imbalance, use `"225,233"`.
    Confirm before merging — `588` may have been intentional for a different
    purpose; grep history.)
- Update the comment block accordingly.
- Extend the `IBKRQuote` dataclass:
  ```python
  auction_imbalance: Optional[float] = None     # signed shares, + = buy-side
  auction_volume:    Optional[int]   = None     # paired shares
  auction_price:     Optional[float] = None     # indicative cross price
  regulatory_imbalance: Optional[float] = None  # NYSE-only, signed shares
  ```
- In `_read_ticker`, read `getattr(ticker, "auctionImbalance", None)` etc.
  Use the existing `_ticker_value` helper (it already maps NaN/0/-1 → None).
  **Do not** require these to be present — they are absent outside auction
  windows; current quote validity check (`last` or `bid&ask`) is unchanged.
- Update [tests/test_ibkr_quote_source.py](../tests/test_ibkr_quote_source.py)
  `test_imbalance_generic_tick_passed_to_reqmktdata` to assert `"225"` (or
  `"225,…"`).  Add a new test for `_read_ticker` populating the four new
  fields when the mock ticker exposes them.

## 2. Signal

**File:** `signals/auction_imbalance.py` (new)

```python
class AuctionImbalanceSignal(Signal):
    name = "auction_imbalance"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1
    # Microstructure default: 5-min bars, 6 bars ahead, 5 days.
    # Auction print resolves within minutes, so override to 1-min × 30.
    return_resolution = "1min"
    return_horizon = 30
    return_lookback_days = 5
```

### Active windows (ET, weekdays only)

| Window | Start  | End    | Notes                              |
| ------ | ------ | ------ | ---------------------------------- |
| open   | 09:28  | 09:30  | NYSE opens publication at 09:28; NASDAQ at 09:25 — use the wider 09:25 if NASDAQ symbols are present. |
| close  | 15:50  | 16:00  | NYSE & NASDAQ both publish from 15:50. |

Outside both windows: `score = 0.0, confidence = 0.0`.

### Score formula

```python
imb     = quote.auction_imbalance          # signed shares; None outside window
paired  = quote.auction_volume or 0        # paired (matched) shares
adv     = data["adv_20"]                   # 20-day avg daily volume from candles_daily

if imb is None or adv <= 0:
    return SignalResult(0.0, 0.0, {"window": window or None})

ratio = imb / adv                          # signed; typical magnitudes 1e-4 .. 5e-2
score = math.tanh(ratio * 50.0)            # squashes ~2% of ADV → ~0.76
```

### Confidence

```python
# Ramps with proximity to the cross AND with magnitude.
mins_to_cross = max(0, (cross_ts - now_ts).total_seconds() / 60.0)
proximity   = 1.0 - min(mins_to_cross / window_minutes, 1.0)   # 0 → 1
magnitude   = min(abs(ratio) * 50.0, 1.0)                      # 0 → 1
confidence  = 0.5 * proximity + 0.5 * magnitude
```

### Components (returned to scorer for transparency)

```python
{
    "window": "open" | "close",
    "imbalance_shares": int,
    "paired_shares": int,
    "imbalance_pct_adv": float,        # rounded %, signed
    "auction_price": float,            # indicative cross price
    "regulatory_imbalance": int,       # NYSE-only, may be None
    "minutes_to_cross": float,
}
```

### Failure modes (all → score=0, confidence=0, component "abstain": reason)

- No `quote` in data dict.
- `auction_imbalance` is None (outside window or feed not flowing).
- `adv` missing or ≤ 0.
- `candles_daily` < 20 bars.

## 3. Tests

**File:** `tests/test_auction_imbalance.py` (new)

- `test_outside_window_returns_zero` — Tue 11:00 ET → score=0, conf=0.
- `test_open_window_in_range` — 09:29 ET, imbalance positive → score>0.
- `test_close_window_in_range` — 15:55 ET, imbalance negative → score<0.
- `test_weekend_inactive` — Sat 09:29 ET → score=0, conf=0.
- `test_score_sign_matches_imbalance_sign` — parametrize ±.
- `test_score_monotone_in_magnitude` — larger |imb| ⇒ larger |score|.
- `test_confidence_increases_toward_cross` — 09:28 vs 09:29:50.
- `test_abstains_when_adv_missing`.
- `test_abstains_when_quote_missing_imbalance_field`.
- `test_components_dict_keys` — locks the component schema.

Use `freezegun` (already in dev deps — confirm) or pass an injected `now`
parameter into the helper function (mirrors the pattern used by
[core/runtime/cadence.py](../core/runtime/cadence.py) — accept `dt=None`,
default to `datetime.now(_ET)`).

## 4. Wiring

- `signals/__init__.py`: `from signals.auction_imbalance import AuctionImbalanceSignal`
  (subclass auto-registers via `Signal.__init_subclass__`).
- `signals/scorer.py`: nothing required; the round loop already enumerates
  `SIGNAL_REGISTRY`.
- `signals/scorer.py` data prep (search for `data["quote"]` or
  `_build_signal_data`): add `data["adv_20"]` from existing
  `candles_daily` if not already present.

## 5. Definition of done

- `data/ibkr_quote_source.py` reads four imbalance fields.
- `_read_ticker` populates them; existing tests still pass.
- `AuctionImbalanceSignal` registered, scored every round, persists to
  `signal_scores`, included in IC attribution, included in per-symbol IC.
- New test file: ~10 tests, all green.
- Full suite: 742 + ~12 new ≈ 754 passed / 4 skipped.
- One PR-style commit `Auction imbalance signal: data layer fix + signal + tests`.

## 6. Open questions to resolve before coding

1. Confirm `225` (vs `225,233` or `225,588`) is the intended generic tick list.
2. Are we OK changing the live default tick list, or do we want it gated
   behind a new `core.config.IBKR_AUCTION_IMBALANCE_ENABLED` flag (default
   True)?  Gating preserves the option to revert without a redeploy.
3. NYSE-listed vs NASDAQ-listed symbols may need different window starts
   (09:25 vs 09:28).  Use the wider window unconditionally, or look up
   primary exchange via `Stock.primaryExchange`?  Wider is simpler and
   merely costs ~3 minutes of low-confidence scores at the open.
