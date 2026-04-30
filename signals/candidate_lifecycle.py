"""Shadow candidate horizon-variant lifecycle.

This module provides a conservative, config-gated scaffold to evaluate and
promote/demote candidate signals (typically horizon variants) without forcing
them into live weighting immediately.
"""

from __future__ import annotations

from typing import Iterable


def _get_cfg(key: str, default: float) -> float:
    try:
        from memory import get_research_config
        return float(get_research_config(key, default))
    except Exception:
        return float(default)


def _set_cfg(key: str, value: float, reason: str) -> None:
    try:
        from memory import set_research_config
        set_research_config(key, float(value), reason=reason)
    except Exception:
        pass


def is_candidate_signal(name: str) -> bool:
    """Candidate naming convention for auto-lifecycle management."""
    return name.startswith("cand_") or "__cand__" in name


def lifecycle_enabled() -> bool:
    return bool(int(_get_cfg("candidate_lifecycle_enabled", 0.0)))


def shadow_only_mode() -> bool:
    return bool(int(_get_cfg("candidate_shadow_only", 1.0)))


def is_live_candidate(name: str) -> bool:
    if not is_candidate_signal(name):
        return True
    return bool(int(_get_cfg(f"candidate_live:{name}", 0.0)))


def filter_live_signal_names(signal_names: Iterable[str]) -> list[str]:
    """Remove non-promoted candidates from live weighting when enabled."""
    names = list(signal_names)
    if not lifecycle_enabled():
        return names
    if shadow_only_mode():
        return [n for n in names if not is_candidate_signal(n)]
    out: list[str] = []
    for n in names:
        if not is_candidate_signal(n) or is_live_candidate(n):
            out.append(n)
    return out


def apply_candidate_weight_caps(weights: dict[str, float]) -> dict[str, float]:
    """Cap live candidate absolute weights to control blast radius."""
    if not lifecycle_enabled() or not weights:
        return weights

    max_abs = float(_get_cfg("candidate_max_live_abs_weight", 0.10))
    out = dict(weights)
    touched = False
    for name, w in list(out.items()):
        if not is_candidate_signal(name):
            continue
        if not is_live_candidate(name):
            out[name] = 0.0
            touched = True
            continue
        if abs(w) > max_abs:
            out[name] = max_abs if w > 0 else -max_abs
            touched = True

    if not touched:
        return out
    abs_sum = sum(abs(v) for v in out.values())
    if abs_sum > 1e-12:
        out = {k: (v / abs_sum) for k, v in out.items()}
    return out


def update_candidate_lifecycle(
    ic_stats: dict[str, dict[str, float]],
    *,
    evaluated_signal_names: Iterable[str] | None = None,
) -> None:
    """Promote/demote candidate variants from rolling IC evidence.

    Promotion/demotion is streak-based and config-driven:
    - candidate_min_obs
    - candidate_min_abs_ic
    - candidate_min_abs_t
    - candidate_promote_streak
    - candidate_demote_streak
    """
    if not lifecycle_enabled():
        return
    eval_set = set(evaluated_signal_names or ic_stats.keys())

    min_obs = int(_get_cfg("candidate_min_obs", 120.0))
    min_abs_ic = float(_get_cfg("candidate_min_abs_ic", 0.03))
    min_abs_t = float(_get_cfg("candidate_min_abs_t", 1.5))
    promote_streak = int(_get_cfg("candidate_promote_streak", 3.0))
    demote_streak = int(_get_cfg("candidate_demote_streak", 2.0))
    shadow_only = shadow_only_mode()

    for name in eval_set:
        if not is_candidate_signal(name):
            continue
        d = ic_stats.get(name)
        if not d:
            continue
        n = int(d.get("n", 0))
        ic = float(d.get("ic", 0.0))
        t = float(d.get("t", 0.0))
        if n < min_obs:
            continue

        good = abs(ic) >= min_abs_ic and abs(t) >= min_abs_t
        p_key = f"candidate_promote_streak:{name}"
        d_key = f"candidate_demote_streak:{name}"
        live_key = f"candidate_live:{name}"
        p_prev = int(_get_cfg(p_key, 0.0))
        d_prev = int(_get_cfg(d_key, 0.0))
        live_prev = int(_get_cfg(live_key, 0.0))

        if good:
            p_new = p_prev + 1
            d_new = 0
            _set_cfg(p_key, float(p_new), reason=f"candidate good ic={ic:.4f} t={t:.2f} n={n}")
            if d_prev != 0:
                _set_cfg(d_key, 0.0, reason="candidate demote streak reset")
            if p_new >= promote_streak and not shadow_only and live_prev == 0:
                _set_cfg(live_key, 1.0, reason=f"candidate promoted ic={ic:.4f} t={t:.2f} n={n}")
        else:
            d_new = d_prev + 1
            _set_cfg(d_key, float(d_new), reason=f"candidate weak ic={ic:.4f} t={t:.2f} n={n}")
            if p_prev != 0:
                _set_cfg(p_key, 0.0, reason="candidate promote streak reset")
            if d_new >= demote_streak and live_prev == 1:
                _set_cfg(live_key, 0.0, reason=f"candidate demoted ic={ic:.4f} t={t:.2f} n={n}")
