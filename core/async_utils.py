"""Shared async helpers — safe wrappers for known CPython bugs."""

import asyncio
import time as _time_mod


async def safe_sleep(seconds: float) -> None:
    """asyncio.sleep wrapper — falls back to time.sleep on Python 3.13 deque bug."""
    try:
        await asyncio.sleep(seconds)
    except IndexError:
        # Python 3.13 known bug: "pop from an empty deque" in _run_once
        _time_mod.sleep(seconds)
