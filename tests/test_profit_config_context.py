"""Thread-local ProfitConfig overrides for parallel optimizer backtests."""

from __future__ import annotations

import os
import threading

import pytest

from core.central_profit_config import build_profit_config, clear_profit_profile_cache
from core.profit_config_context import (
    clear_thread_config,
    pop_composed_config,
    push_composed_config,
)
from core.profit_profiles import PROFIT_PROFILE_ENV
from core.risk_execution_config import get_risk_execution_config


@pytest.fixture(autouse=True)
def _clean():
    clear_thread_config()
    clear_profit_profile_cache()
    saved = os.environ.get(PROFIT_PROFILE_ENV)
    yield
    clear_thread_config()
    clear_profit_profile_cache()
    if saved is None:
        os.environ.pop(PROFIT_PROFILE_ENV, None)
    else:
        os.environ[PROFIT_PROFILE_ENV] = saved


def _composed_for(profile: str):
    os.environ[PROFIT_PROFILE_ENV] = profile
    clear_profit_profile_cache()
    return build_profit_config()


def test_thread_local_config_isolated_between_threads():
    conservative = _composed_for("conservative")
    aggressive = _composed_for("aggressive")
    caps: dict[str, float] = {}

    def worker(label: str, composed) -> None:
        push_composed_config(composed)
        try:
            caps[label] = get_risk_execution_config().max_daily_llm_cost
        finally:
            pop_composed_config()

    t1 = threading.Thread(target=worker, args=("conservative", conservative))
    t2 = threading.Thread(target=worker, args=("aggressive", aggressive))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert caps["conservative"] == pytest.approx(3.0)
    assert caps["aggressive"] == pytest.approx(6.5)
