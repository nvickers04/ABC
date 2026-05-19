"""Master composition of all profitability-related configuration.

Use :func:`get_profit_config` for the process-wide :class:`ProfitConfig` singleton
(trader, research host, simulation, cycle logger). After CLI or env changes, call
:meth:`ProfitConfig.reload` on that instance (or :func:`load_profit_config` /
:func:`reload_profit_config`). Optimizer grid search uses :class:`ComposedProfitConfig`
via :func:`load_cached_profit_profile` and thread-local overrides
(:mod:`core.profit_config_context`) so parallel workers do not mutate the live singleton.
Historical bars load once per date window in :class:`ReplayDataProvider`.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from typing import Any, ClassVar

_PROFIT_CONFIG_LOCK = threading.RLock()

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from core.loop_config import LoopConfig, install_loop_config, reload_loop_config
from core.memory_config import MemoryConfig, install_memory_config, reload_memory_config
from core.profit_profiles import (
    PROFIT_PROFILE_ENV,
    ProfitProfile,
    apply_loop_profile,
    apply_memory_profile,
    apply_prompt_profile,
    apply_risk_profile,
    apply_tool_profile,
    is_evolved_profile,
    load_evolved_profile_registry,
    normalize_profit_profile,
    profile_change_lines,
    profile_note,
)
from core.prompt_config import PromptConfig, install_prompt_config, reload_prompt_config
from core.risk_execution_config import (
    RiskExecutionConfig,
    install_risk_execution_config,
    reload_risk_execution_config,
)
from core.profile_optimization import DEFAULT_CYCLES_PER_DAY
from core.tool_registry import (
    ToolRegistry,
    get_tool_registry,
    install_tool_registry,
    reset_tool_registry_for_tests,
)


def _profitability_note(field_info: FieldInfo | None) -> str:
    if field_info is None:
        return ""
    raw = field_info.description or ""
    for prefix in ("Profitability: ", "profitability: "):
        if raw.startswith(prefix):
            return raw[len(prefix) :].strip()
    return raw.strip()


def _format_value(val: Any, *, max_len: int = 120) -> str:
    if isinstance(val, dict):
        if len(val) <= 6:
            return repr(val)
        return f"<dict len={len(val)}>"
    if isinstance(val, (list, tuple)):
        if len(val) <= 8:
            return repr(val)
        return f"<{type(val).__name__} len={len(val)}>"
    s = repr(val)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _print_model_section(title: str, model: BaseModel) -> None:
    print(f"## {title}")
    for name, field_info in model.model_fields.items():
        val = getattr(model, name)
        note = _profitability_note(field_info)
        print(f"  {name} = {_format_value(val)}")
        if note:
            print(f"    -> P&L: {note}")
    print()


def _print_tool_registry_section(registry: ToolRegistry) -> None:
    print("## Tool registry")
    tools = list(registry.tools.values())
    enabled = [t for t in tools if t.enabled]
    mutating = [t for t in tools if t.mutates_broker]
    print(f"  tools_registered = {len(tools)}")
    print(f"  tools_enabled = {len(enabled)}")
    print(f"  tools_mutating_broker = {len(mutating)}")
    print("    -> P&L: catalog of agent tools; weights and blocks steer spend vs edge.")
    print()
    for spec in sorted(tools, key=lambda t: (-t.profitability_weight, t.name)):
        note_parts = [
            f"weight={spec.profitability_weight:.1f}",
            f"tier={spec.cost_tier}",
            f"cat={spec.category}",
        ]
        if spec.mutates_broker:
            note_parts.append("mutates")
        if not spec.enabled:
            note_parts.append("DISABLED")
        line = " ".join(note_parts)
        desc = (spec.schema_description or "").split("\n")[0][:80]
        print(f"  {spec.name}: {line}")
        if desc:
            print(f"    -> P&L: {desc}")
    print()


@dataclass(frozen=True)
class ComposedProfitConfig:
    """Immutable composed levers (grid cache / patches). Use :class:`ProfitConfig` at runtime."""

    prompt: PromptConfig
    tools: ToolRegistry
    memory: MemoryConfig
    loop: LoopConfig
    risk: RiskExecutionConfig

    @property
    def trading_mode(self) -> str:
        return str(self.risk.trading_mode)


class ProfitConfig:
    """Process-wide singleton for live trader, research host, simulation, and logger.

    Always obtain via :func:`get_profit_config`. Call :meth:`reload` after env or CLI
    profile changes — the same object is updated in place.
    """

    _instance: ClassVar[ProfitConfig | None] = None

    __slots__ = ("_data",)

    def __init__(self) -> None:
        raise RuntimeError("Use get_profit_config(), not ProfitConfig()")

    @classmethod
    def _bootstrap(cls) -> ProfitConfig:
        with _PROFIT_CONFIG_LOCK:
            if cls._instance is not None:
                return cls._instance
            inst = object.__new__(cls)
            object.__setattr__(inst, "_data", _build_composed_profit_config())
            cls._instance = inst
            return inst

    @property
    def composed(self) -> ComposedProfitConfig:
        return self._data

    def _set_composed(self, data: ComposedProfitConfig) -> None:
        with _PROFIT_CONFIG_LOCK:
            object.__setattr__(self, "_data", data)

    def reload(self, *, dotenv: bool = True) -> ProfitConfig:
        """Rebuild sub-configs from ``os.environ`` and refresh ``core.config`` exports."""
        with _PROFIT_CONFIG_LOCK:
            if dotenv:
                try:
                    from dotenv import load_dotenv

                    load_dotenv(override=True)
                except ImportError:
                    pass
            reset_tool_registry_for_tests()
            object.__setattr__(self, "_data", _build_composed_profit_config())
            sync_research_host_from_profit_config(publish_heartbeat=False)
            try:
                from core.quality.quality_learning import sync_bases_for_current_profile

                sync_bases_for_current_profile(force=True)
            except Exception:
                pass
            return self

    def get_research_settings(self) -> "ResearchSettings":
        """Research-host view of active memory, loop, risk, and tool levers."""
        from core.research_settings import build_research_settings_from_composed

        return build_research_settings_from_composed(self.composed)

    @property
    def learn_from_history(self) -> bool:
        """When True, QualityMatrix adjusts scoring weights from trade outcome history."""
        return bool(self.memory.quality_matrix_learn_from_history)

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        return getattr(self._data, name)

    def summary(self) -> None:
        """Print every configured lever with value and one-line profitability note."""
        print("=" * 72)
        print("ProfitConfig — all profitability levers")
        print(f"  trading_mode = {self.trading_mode!r}")
        print(f"  ibkr_account_type = {self.risk.ibkr_account_type!r}")
        print(f"  tools_registered = {len(self.tools.tools)}")
        print("=" * 72)
        print()
        _print_model_section("Risk & execution", self.risk)
        _print_model_section("Agent loop (ReAct / scheduler / gates)", self.loop)
        _print_model_section("Memory & prompt budget", self.memory)
        _print_model_section("LLM prompts & quality sampling", self.prompt)
        _print_tool_registry_section(self.tools)
        print(
            "Env tips: TRADING_MODE, RISK_PER_TRADE, MAX_DAILY_LLM_COST, CASH_ONLY, "
            "CYCLE_* caps, TOOL_REGISTRY_DISABLE / TOOL_REGISTRY_WEIGHTS, PROFIT_PROFILE."
        )

    def optimize_for_profit(
        self,
        profile: str = "balanced",
        *,
        verbose: bool = True,
    ) -> ProfitConfig:
        """Apply a validated profitability profile across all five sub-configs.

        Profiles: ``conservative``, ``balanced`` (env defaults only), ``aggressive``.
        Persists ``PROFIT_PROFILE`` in the environment and reloads singletons.
        """
        from core.profit_config_state import get_active_profile_label

        norm = normalize_profit_profile(profile)
        previous = get_active_profile_label()
        os.environ[PROFIT_PROFILE_ENV] = norm
        refreshed = self.reload(dotenv=False)
        try:
            from core.profile_rollback import on_profile_applied

            on_profile_applied(previous, norm)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("profile_rollback on optimize_for_profit: %s", exc)
        if verbose:
            print("=" * 72)
            print(f"ProfitConfig.optimize_for_profit({norm!r})")
            print(f"  {profile_note(norm)}")
            print("=" * 72)
            for line in profile_change_lines(
                norm,
                risk=refreshed.risk,
                loop=refreshed.loop,
                memory=refreshed.memory,
                prompt=refreshed.prompt,
                tools=refreshed.tools,
            ):
                print(line)
            print()
        return refreshed

    def log_cycle(
        self,
        *,
        cycle_id: int,
        outcome: str,
        cooldown_seconds: int,
        session: str = "unknown",
        cycle_summary: str = "",
        cycle_actions: list[str] | None = None,
        agent: Any | None = None,
        profile_label: str | None = None,
    ):
        """Record this cycle's config snapshot and P&L / quality metrics."""
        from core.profit_cycle_logger import append_profit_cycle_log

        return append_profit_cycle_log(
            self,
            cycle_id=cycle_id,
            outcome=outcome,
            cooldown_seconds=cooldown_seconds,
            session=session,
            cycle_summary=cycle_summary,
            cycle_actions=cycle_actions,
            agent=agent,
            profile_label=profile_label,
        )


def _active_evolved_profile() -> str | None:
    raw = os.getenv(PROFIT_PROFILE_ENV, "").strip().lower()
    if raw and is_evolved_profile(raw):
        return raw
    return None


def _active_profit_profile() -> ProfitProfile | None:
    raw = os.getenv(PROFIT_PROFILE_ENV, "").strip().lower()
    if not raw or raw == "balanced" or is_evolved_profile(raw):
        return None
    return normalize_profit_profile(raw)  # type: ignore[return-value]


def _apply_profile_to_profit_config(
    cfg: ComposedProfitConfig, profile: ProfitProfile
) -> ComposedProfitConfig:
    risk = apply_risk_profile(cfg.risk, profile)
    memory = apply_memory_profile(cfg.memory, profile)
    loop = apply_loop_profile(cfg.loop, profile)
    prompt = apply_prompt_profile(cfg.prompt, profile)
    tools = cfg.tools
    apply_tool_profile(tools, profile)
    install_risk_execution_config(risk)
    install_memory_config(memory)
    install_loop_config(loop)
    install_prompt_config(prompt)
    install_tool_registry(tools)
    return replace(cfg, risk=risk, memory=memory, loop=loop, prompt=prompt, tools=tools)


def _clear_profile_installs() -> None:
    install_risk_execution_config(None)
    install_memory_config(None)
    install_loop_config(None)
    install_prompt_config(None)
    install_tool_registry(None)


def _sync_core_config_exports(risk: RiskExecutionConfig) -> None:
    """Refresh :mod:`core.config` module-level aliases after sub-config reload."""
    import core.config as cfg

    cfg._sync_risk_module_exports(risk)
    cfg.rebuild_prompt_exports()


def _build_composed_profit_config() -> ComposedProfitConfig:
    """Load all sub-config singletons (call after env / CLI overrides are set)."""
    _clear_profile_installs()
    risk = reload_risk_execution_config()
    memory = reload_memory_config()
    loop = reload_loop_config()
    prompt = reload_prompt_config()
    tools = get_tool_registry(reload=True)
    cfg = ComposedProfitConfig(
        prompt=prompt,
        tools=tools,
        memory=memory,
        loop=loop,
        risk=risk,
    )
    evolved = _active_evolved_profile()
    profile = _active_profit_profile()
    if evolved is not None:
        from core.profile_optimization import apply_config_patches

        entry = load_evolved_profile_registry()[evolved]
        cfg = _apply_profile_to_profit_config(cfg, entry.base_profile)
        cfg = apply_config_patches(cfg, entry.patches, profile_label=evolved)
        risk = cfg.risk
    elif profile is not None:
        cfg = _apply_profile_to_profit_config(cfg, profile)
        risk = cfg.risk
    _sync_core_config_exports(risk)
    from core.profit_config_state import set_active_profile_label

    if evolved is not None:
        set_active_profile_label(evolved)
    elif profile is not None:
        set_active_profile_label(profile)
    else:
        set_active_profile_label("balanced")
    return cfg


def build_profit_config() -> ComposedProfitConfig:
    """Build a fresh composed config without updating the process singleton (grid cache)."""
    return _build_composed_profit_config()


def refresh_profit_config_cache(cfg: ComposedProfitConfig | ProfitConfig) -> ProfitConfig:
    """Pin the active composed config on the process singleton (patches / simulation)."""
    data = cfg.composed if isinstance(cfg, ProfitConfig) else cfg
    with _PROFIT_CONFIG_LOCK:
        inst = get_profit_config()
        inst._set_composed(data)
        return inst


def get_profit_config() -> ProfitConfig:
    """Return the process-wide :class:`ProfitConfig` singleton (thread-safe)."""
    with _PROFIT_CONFIG_LOCK:
        if ProfitConfig._instance is None:
            return ProfitConfig._bootstrap()
        return ProfitConfig._instance


def reload_profit_config() -> ProfitConfig:
    """Reload the singleton from ``os.environ`` (profile grid cache uses env fingerprint)."""
    return get_profit_config().reload(dotenv=False)


def get_research_settings() -> "ResearchSettings":
    """Profile-aware research-host settings (respects thread-local composed config)."""
    from core.profit_config_context import (
        get_thread_loop_config,
        get_thread_memory_config,
        get_thread_risk_config,
        get_thread_tool_registry,
    )
    from core.loop_config import get_loop_config
    from core.memory_config import get_memory_config
    from core.research_settings import ResearchSettings, build_research_settings
    from core.risk_execution_config import get_risk_execution_config
    from core.tool_registry import get_tool_registry

    mem = get_thread_memory_config() or get_memory_config()
    loop = get_thread_loop_config() or get_loop_config()
    risk = get_thread_risk_config() or get_risk_execution_config()
    tools = get_thread_tool_registry() or get_tool_registry()
    return build_research_settings(memory=mem, loop=loop, risk=risk, tools=tools)


def sync_research_host_from_profit_config(*, publish_heartbeat: bool = False) -> "ResearchSettings":
    """Align research caches and Postgres metadata with the active ProfitConfig profile."""
    from core.research_settings import RESEARCH_HOST_PROFILE_KEY, ResearchSettings

    settings = get_research_settings()
    clear_profit_profile_cache()
    try:
        from core.research_topics import invalidate_research_topic_caches

        invalidate_research_topic_caches()
    except Exception:
        pass
    try:
        from memory import set_research_config

        set_research_config(
            RESEARCH_HOST_PROFILE_KEY,
            settings.profile_label,
            reason="profit_config_sync",
            log=False,
        )
    except Exception:
        pass
    if publish_heartbeat:
        try:
            from core.runtime.heartbeat import publish_research_host_heartbeat

            publish_research_host_heartbeat()
        except Exception:
            pass
    return settings


def reset_profit_config_for_tests() -> None:
    """Drop the singleton instance (unit tests only)."""
    with _PROFIT_CONFIG_LOCK:
        ProfitConfig._instance = None
    from core.profit_config_context import clear_thread_config

    clear_thread_config()


def clear_profit_profile_cache() -> None:
    """Clear in-memory per-profile ProfitConfig cache (grid search / optimizer)."""
    from core.profit_profile_cache import clear_profit_profile_cache as _clear

    _clear()


def load_cached_profit_profile(profile: str, *, dotenv: bool = False) -> ComposedProfitConfig:
    """Load a profile-scoped ProfitConfig using the grid-search cache."""
    from core.profit_profile_cache import load_cached_profit_profile as _load

    return _load(profile, dotenv=dotenv)


def get_profit_profile_cache_stats():
    """Hit/miss stats for :func:`load_cached_profit_profile`."""
    from core.profit_profile_cache import get_profit_profile_cache_stats as _stats

    return _stats()


def load_profit_config(*, dotenv: bool = True) -> ProfitConfig:
    """Load ``.env`` if requested and reload the process :class:`ProfitConfig` singleton."""
    return get_profit_config().reload(dotenv=dotenv)


_replay_data_cache: dict[tuple[str, str, tuple[str, ...]], ReplayDataProvider] = {}
_replay_cache_lock = threading.Lock()


class ReplayDataProvider:
    """Load historical archives once per date window; share across backtest candidates.

    Use :func:`get_shared_replay_data` for optimizer/compare runs, then pass the same
    instance into :func:`simulate_backtest` / :func:`run_backtest_async`.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        *,
        symbols: tuple[str, ...] | None = None,
    ) -> None:
        from core.simulation.replay_data import _DEFAULT_UNIVERSE

        self.start_date = start_date
        self.end_date = end_date
        self.symbols = tuple(s.upper() for s in (symbols or _DEFAULT_UNIVERSE))
        self._bars: dict[str, dict[str, dict[str, Any]]] = {}
        self._missing_symbols: list[str] = []
        self._loaded = False
        self.load_stats: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def missing_symbols(self) -> list[str]:
        return list(self._missing_symbols)

    async def load(self) -> dict[str, Any]:
        """Fetch archives for all symbols (no-op if already loaded; thread-safe)."""
        with self._lock:
            if self._loaded:
                return self.load_stats

        import logging
        import time

        from core.simulation.archive import bars_by_date, ensure_archive_async

        log = logging.getLogger(__name__)
        t0 = time.perf_counter()
        per_symbol: dict[str, float] = {}
        missing: list[str] = []
        bars: dict[str, dict[str, dict[str, Any]]] = {}

        for sym in self.symbols:
            sym_t0 = time.perf_counter()
            payload = await ensure_archive_async(sym, self.start_date, self.end_date)
            bars[sym] = bars_by_date(payload)
            per_symbol[sym] = round(time.perf_counter() - sym_t0, 3)
            if not payload.get("bars"):
                missing.append(sym)

        elapsed = time.perf_counter() - t0
        stats = {
            "symbol_count": len(self.symbols),
            "elapsed_sec": round(elapsed, 3),
            "per_symbol_sec": per_symbol,
            "missing_symbols": missing,
        }

        with self._lock:
            if self._loaded:
                return self.load_stats
            self._bars = bars
            self._missing_symbols = missing
            self._loaded = True
            self.load_stats = stats
            log.info(
                "ReplayDataProvider loaded %d symbols %s..%s in %.2fs",
                len(self.symbols),
                self.start_date,
                self.end_date,
                elapsed,
            )
            return self.load_stats

    def spawn_session(self) -> Any:
        """Create a per-candidate :class:`~core.simulation.replay_data.ReplaySessionDataProvider`."""
        from core.simulation.replay_data import ReplaySessionDataProvider

        with self._lock:
            if not self._loaded:
                raise RuntimeError(
                    "ReplayDataProvider.load() must complete before spawn_session()"
                )
            return ReplaySessionDataProvider(
                start_date=self.start_date,
                end_date=self.end_date,
                symbols=self.symbols,
                bars=self._bars,
            )


def get_shared_replay_data(
    start_date: str,
    end_date: str,
    *,
    symbols: tuple[str, ...] | None = None,
    reuse: bool = True,
) -> ReplayDataProvider:
    """Return a cached :class:`ReplayDataProvider` for ``start_date``..``end_date``."""
    from core.simulation.replay_data import _DEFAULT_UNIVERSE

    syms = tuple(s.upper() for s in (symbols or _DEFAULT_UNIVERSE))
    key = (start_date, end_date, syms)
    with _replay_cache_lock:
        if reuse and key in _replay_data_cache:
            return _replay_data_cache[key]
        inst = ReplayDataProvider(start_date, end_date, symbols=syms)
        if reuse:
            _replay_data_cache[key] = inst
        return inst


def clear_shared_replay_data() -> None:
    """Drop cached replay providers (optimizer window change / tests)."""
    with _replay_cache_lock:
        _replay_data_cache.clear()


def simulate_backtest_compare(
    profile_names: list[str],
    start_date: str,
    end_date: str,
    *,
    initial_cash: float = 100_000.0,
    cycles_per_day: int = DEFAULT_CYCLES_PER_DAY,
    replay_data: ReplayDataProvider | None = None,
):
    """Run :func:`simulate_backtest` for each profile on the same historical window."""
    import asyncio

    from core.profit_profiles import normalize_profit_profile

    shared = replay_data or get_shared_replay_data(start_date, end_date)
    if not shared.is_loaded:
        asyncio.run(shared.load())

    results = []
    for raw in profile_names:
        name = normalize_profit_profile(raw)
        results.append(
            simulate_backtest(
                name,
                start_date,
                end_date,
                initial_cash=initial_cash,
                cycles_per_day=cycles_per_day,
                replay_data=shared,
                reload_dotenv=False,
            )
        )
    return results


def simulate_backtest(
    profile_name: str,
    start_date: str,
    end_date: str,
    *,
    initial_cash: float = 100_000.0,
    cycles_per_day: int = DEFAULT_CYCLES_PER_DAY,
    config_patches: dict[str, dict[str, Any]] | None = None,
    candidate_id: str | None = None,
    reload_dotenv: bool = True,
    replay_data: ReplayDataProvider | None = None,
    composed: ComposedProfitConfig | None = None,
):
    """Run historical simulation with a profitability profile and return P&L stats.

    Loads ``profile_name`` via the :class:`ProfitConfig` singleton (or ``composed=`` for
    parallel optimizer workers), replays archived or
    freshly fetched MarketData.app / yfinance daily bars, and executes
    :meth:`core.agent.TradingAgent.run_cycle` with QualityMatrix and safety gates
    enabled (simulated broker + BacktestLLM).

    Args:
        profile_name: ``conservative``, ``balanced``, or ``aggressive``.
        start_date: Inclusive session start ``YYYY-MM-DD``.
        end_date: Inclusive session end ``YYYY-MM-DD``.
        initial_cash: Starting equity for the simulated account.
        cycles_per_day: ReAct cycles per trading day (max 4 ET slots).

    Returns:
        :class:`core.simulation.types.BacktestResult`
    """
    import asyncio
    import time

    from core.profit_config_state import (
        BacktestWindowError,
        InvalidProfitProfileError,
        SimulationDataError,
        log_active_profit_config,
        validate_backtest_date_range,
    )
    from core.simulation.llm_cost_estimate import SimulationBudgetError
    from core.simulation.runner import run_backtest_async

    from core.simulation.llm_cost_estimate import format_run_summary

    logger = __import__("logging").getLogger(__name__)
    t0 = time.perf_counter()
    try:
        start_date, end_date = validate_backtest_date_range(start_date, end_date)
        logger.info(
            "simulate_backtest cycles_per_day=%d profile=%s window=%s..%s",
            cycles_per_day,
            profile_name,
            start_date,
            end_date,
        )
        shared_replay = replay_data or get_shared_replay_data(start_date, end_date)
        result = asyncio.run(
            run_backtest_async(
                profile_name,
                start_date,
                end_date,
                initial_cash=initial_cash,
                cycles_per_day=cycles_per_day,
                config_patches=None if composed is not None else config_patches,
                candidate_id=candidate_id,
                reload_dotenv=reload_dotenv,
                replay_data=shared_replay,
                composed=composed,
            )
        )
        if result.run_stats is not None:
            logger.info(
                "simulate_backtest complete in %.2fs | cycles_per_day=%d cycles_run=%d | %s",
                time.perf_counter() - t0,
                result.cycles_per_day,
                result.cycles_run,
                format_run_summary(
                    label=result.profile,
                    stats=result.run_stats,
                    trading_days=result.trading_days,
                    cycles_run=result.cycles_run,
                ),
            )
        else:
            logger.info(
                "simulate_backtest complete in %.2fs profile=%s days=%s "
                "cycles_per_day=%d cycles_run=%d",
                time.perf_counter() - t0,
                result.profile,
                result.trading_days,
                result.cycles_per_day,
                result.cycles_run,
            )
        try:
            profit_cfg = get_profit_config()
            if profit_cfg.learn_from_history:
                from core.quality.quality_learning import ingest_backtest_trades_and_refit

                learn_summary = ingest_backtest_trades_and_refit(
                    result,
                    candidate_id or profile_name,
                )
                if learn_summary.get("refitted"):
                    result.notes.append(
                        f"QualityMatrix learned weights: reward={learn_summary.get('reward')} "
                        f"samples={learn_summary.get('samples')}"
                    )
        except Exception as learn_exc:
            logger.debug("simulate_backtest quality learning skipped: %s", learn_exc)

        return result
    except (
        BacktestWindowError,
        InvalidProfitProfileError,
        SimulationDataError,
        SimulationBudgetError,
        ValueError,
    ) as exc:
        log_active_profit_config(
            logger,
            "simulate_backtest failed",
            exc=exc,
            profile_name=profile_name,
            start_date=start_date,
            end_date=end_date,
            candidate_id=candidate_id,
        )
        raise
    except Exception as exc:
        log_active_profit_config(
            logger,
            "simulate_backtest unexpected error",
            exc=exc,
            profile_name=profile_name,
            start_date=start_date,
            end_date=end_date,
            candidate_id=candidate_id,
        )
        raise


__all__ = [
    "DEFAULT_CYCLES_PER_DAY",
    "PROFIT_PROFILE_ENV",
    "ComposedProfitConfig",
    "ProfitConfig",
    # ResearchSettings: import from core.research_settings
    "build_profit_config",
    "clear_profit_profile_cache",
    "get_profit_profile_cache_stats",
    "get_profit_config",
    "get_research_settings",
    "load_cached_profit_profile",
    "load_profit_config",
    "refresh_profit_config_cache",
    "ReplayDataProvider",
    "clear_shared_replay_data",
    "get_shared_replay_data",
    "reload_profit_config",
    "reset_profit_config_for_tests",
    "sync_research_host_from_profit_config",
    "simulate_backtest",
    "simulate_backtest_compare",
]
