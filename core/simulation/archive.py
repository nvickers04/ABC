"""Historical bar archive under ``data/archives/`` (MDA or yfinance fallback)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = _REPO_ROOT / "data" / "archives"


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in ("429", "rate limit", "too many requests", "ratelimit")
    )


def _archive_path(symbol: str, start: str, end: str) -> Path:
    sym = symbol.upper().strip()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_ROOT / sym / f"{start}_{end}.json"


def load_archive(symbol: str, start: str, end: str) -> dict[str, Any] | None:
    path = _archive_path(symbol, start, end)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("archive_read_failed", extra={"path": str(path), "error": str(e)})
        return None


def save_archive(symbol: str, start: str, end: str, payload: dict[str, Any]) -> Path:
    path = _archive_path(symbol, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _scalar(row, key: str, default: float = 0.0) -> float:
    val = row[key] if key in row.index else default
    if hasattr(val, "iloc"):
        val = val.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _fetch_yfinance_daily(symbol: str, start: str, end: str) -> dict[str, Any]:
    import yfinance as yf

    # yfinance end is exclusive
    end_excl = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.download(symbol, start=start, end=end_excl, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.warning(
            "yfinance_archive_fetch_failed",
            extra={"symbol": symbol, "error": str(exc)},
        )
        return {"symbol": symbol.upper(), "source": "yfinance", "bars": [], "error": str(exc)}
    if df is None or df.empty:
        return {"symbol": symbol.upper(), "source": "yfinance", "bars": []}
    bars: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        ts = int(idx.to_pydatetime().replace(tzinfo=None).timestamp())
        bars.append(
            {
                "open": _scalar(row, "Open"),
                "high": _scalar(row, "High"),
                "low": _scalar(row, "Low"),
                "close": _scalar(row, "Close"),
                "volume": int(_scalar(row, "Volume", 0)),
                "timestamp": ts,
                "date": idx.strftime("%Y-%m-%d"),
            }
        )
    return {"symbol": symbol.upper(), "source": "yfinance", "bars": bars}


async def fetch_mda_daily_async(symbol: str, start: str, end: str) -> dict[str, Any] | None:
    """Fetch daily bars from MarketData.app (async-safe)."""
    import os

    if not (os.getenv("MARKETDATA_API_KEY") or os.getenv("MDA_API_KEY")):
        return None
    try:
        from data.marketdata_client import MarketDataClient

        client = MarketDataClient()
        raw = await client.get_candles(
            symbol,
            resolution="D",
            from_date=start,
            to_date=end,
        )
        bars: list[dict[str, Any]] = []
        if raw and getattr(raw, "c", None):
            for i in range(len(raw.c)):
                bars.append(
                    {
                        "open": float(raw.o[i]),
                        "high": float(raw.h[i]),
                        "low": float(raw.l[i]),
                        "close": float(raw.c[i]),
                        "volume": int(raw.v[i]) if raw.v else 0,
                        "timestamp": int(raw.t[i]) if raw.t else 0,
                        "date": datetime.utcfromtimestamp(int(raw.t[i])).strftime("%Y-%m-%d")
                        if raw.t
                        else "",
                    }
                )
        return {"symbol": symbol.upper(), "source": "marketdata", "bars": bars}
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(
                "mda_rate_limit_fallback",
                extra={"symbol": symbol, "start": start, "end": end, "error": str(e)},
            )
        else:
            logger.debug("mda_archive_fetch_failed: %s", e)
        return None


async def ensure_archive_async(symbol: str, start: str, end: str) -> dict[str, Any]:
    """Load cache or fetch MDA (async) then yfinance fallback."""
    cached = load_archive(symbol, start, end)
    if cached and cached.get("bars"):
        return cached
    payload = await fetch_mda_daily_async(symbol, start, end)
    if not payload or not payload.get("bars"):
        payload = _fetch_yfinance_daily(symbol, start, end)
    if not payload.get("bars"):
        logger.warning(
            "archive_no_bars",
            extra={"symbol": symbol, "start": start, "end": end, "source": payload.get("source")},
        )
    save_archive(symbol, start, end, payload)
    return payload


def ensure_archive(symbol: str, start: str, end: str) -> dict[str, Any]:
    """Load cached archive or fetch from yfinance (sync). Use :func:`ensure_archive_async` in async runners."""
    cached = load_archive(symbol, start, end)
    if cached and cached.get("bars"):
        return cached
    payload = _fetch_yfinance_daily(symbol, start, end)
    save_archive(symbol, start, end, payload)
    return payload


async def prefetch_archives_async(
    start: str,
    end: str,
    symbols: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Warm archives via :class:`core.central_profit_config.ReplayDataProvider` (load once)."""
    from core.central_profit_config import get_shared_replay_data

    shared = get_shared_replay_data(start, end, symbols=symbols)
    stats = await shared.load()
    return {
        "symbol_count": stats.get("symbol_count", 0),
        "elapsed_sec": stats.get("elapsed_sec", 0.0),
        "per_symbol_sec": stats.get("per_symbol_sec", {}),
        "missing_symbols": stats.get("missing_symbols", []),
    }


def bars_by_date(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bar in payload.get("bars") or []:
        d = bar.get("date") or ""
        if d:
            out[d] = bar
    return out
