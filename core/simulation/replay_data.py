"""Replay historical quotes/candles as-of simulation time."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from core.simulation.archive import bars_by_date, ensure_archive
from data.data_provider import Candles, DataProvider, Quote

logger = logging.getLogger(__name__)

_DEFAULT_UNIVERSE = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT")


class ReplaySessionDataProvider(DataProvider):
    """Per-backtest view over bars loaded by :class:`core.central_profit_config.ReplayDataProvider`."""

    def __init__(
        self,
        *,
        start_date: str,
        end_date: str,
        symbols: tuple[str, ...] = _DEFAULT_UNIVERSE,
        fallback: Optional[DataProvider] = None,
        bars: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        # Avoid live MDA client init — replay uses archives only.
        self._mda_client = None
        self._cache = {}
        self._ttl_map = {}
        self._default_ttl = 60
        self._executor = None
        self._start = start_date
        self._end = end_date
        self._symbols = tuple(s.upper() for s in symbols)
        self._fallback = fallback
        self._session_date = start_date
        self._now_utc: datetime = datetime.now(timezone.utc)
        self._bars: dict[str, dict[str, dict[str, Any]]] = (
            bars if bars is not None else {}
        )
        if bars is None:
            self._load_archives()

    def load_bars(self, symbol: str, payload: dict) -> None:
        self._bars[symbol.upper()] = bars_by_date(payload)

    def _load_archives(self) -> None:
        for sym in self._symbols:
            payload = ensure_archive(sym, self._start, self._end)
            self._bars[sym] = bars_by_date(payload)

    def set_sim_context(self, session_date: str, now_utc: datetime) -> None:
        self._session_date = session_date
        self._now_utc = now_utc

    def _bar_for(self, symbol: str, date: str | None = None) -> dict[str, Any] | None:
        sym = symbol.upper()
        if sym not in self._bars:
            try:
                payload = ensure_archive(sym, self._start, self._end)
                self._bars[sym] = bars_by_date(payload)
            except Exception as e:
                logger.debug("replay_archive_miss %s: %s", sym, e)
                return None
        d = date or self._session_date
        return self._bars[sym].get(d)

    def _prior_close(self, symbol: str) -> float | None:
        sym = symbol.upper()
        dates = sorted(self._bars.get(sym, {}).keys())
        try:
            idx = dates.index(self._session_date)
        except ValueError:
            return None
        if idx <= 0:
            return None
        prev = self._bars[sym].get(dates[idx - 1])
        return float(prev["close"]) if prev else None

    def get_quote(self, symbol: str, **kwargs: Any) -> Quote | dict[str, Any]:
        bar = self._bar_for(symbol)
        if not bar:
            if self._fallback:
                return self._fallback.get_quote(symbol, **kwargs)
            return Quote(
                symbol=symbol.upper(),
                last=None,
                bid=None,
                ask=None,
                volume=0,
                source="replay",
            )
        close = float(bar["close"])
        prev = self._prior_close(symbol) or close
        change_pct = ((close - prev) / prev * 100.0) if prev else 0.0
        q = Quote(
            symbol=symbol.upper(),
            last=close,
            bid=close * 0.9999,
            ask=close * 1.0001,
            volume=int(bar.get("volume") or 0),
            change_pct=change_pct,
            source="replay",
            timestamp=self._now_utc,
        )
        # Gap guard reads previous_close on dict or object
        out: dict[str, Any] = {
            "symbol": q.symbol,
            "last": q.last,
            "bid": q.bid,
            "ask": q.ask,
            "volume": q.volume,
            "change_pct": q.change_pct,
            "previous_close": prev,
            "close": close,
            "source": "replay",
        }
        return out

    def get_candles(
        self,
        symbol: str,
        days_back: int = 30,
        resolution: str = "D",
        **kwargs: Any,
    ) -> Candles:
        sym = symbol.upper()
        dates = sorted(self._bars.get(sym, {}).keys())
        end_idx = dates.index(self._session_date) if self._session_date in dates else len(dates) - 1
        start_idx = max(0, end_idx - max(days_back, 1) + 1)
        window = dates[start_idx : end_idx + 1]
        o, h, l, c, v, t = [], [], [], [], [], []
        for d in window:
            b = self._bars[sym][d]
            o.append(float(b["open"]))
            h.append(float(b["high"]))
            l.append(float(b["low"]))
            c.append(float(b["close"]))
            v.append(int(b.get("volume") or 0))
            t.append(int(b.get("timestamp") or 0))
        return Candles(symbol=sym, open=o, high=h, low=l, close=c, volume=v, timestamps=t, source="replay")


# Backward-compatible alias (shared loader lives in central_profit_config).
ReplayDataProvider = ReplaySessionDataProvider
