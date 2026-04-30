"""Tests for shadow candidate lifecycle scaffold."""

from __future__ import annotations


def _set_cfg(key: str, value: float, reason: str = "test"):
    from memory import set_research_config
    set_research_config(key, value, reason=reason)


def _get_cfg(key: str, default: float = 0.0) -> float:
    from memory import get_research_config
    return float(get_research_config(key, default))


def test_filter_live_signal_names_respects_shadow_mode():
    from signals.candidate_lifecycle import filter_live_signal_names

    _set_cfg("candidate_lifecycle_enabled", 1.0)
    _set_cfg("candidate_shadow_only", 1.0)

    names = ["momentum", "cand_momentum_h4", "cand_meanrev_h8"]
    out = filter_live_signal_names(names)
    assert out == ["momentum"]


def test_candidate_promotes_after_streak_when_not_shadow_only():
    from signals.candidate_lifecycle import update_candidate_lifecycle

    name = "cand_momentum_h4"
    _set_cfg("candidate_lifecycle_enabled", 1.0)
    _set_cfg("candidate_shadow_only", 0.0)
    _set_cfg("candidate_min_obs", 10.0)
    _set_cfg("candidate_min_abs_ic", 0.02)
    _set_cfg("candidate_min_abs_t", 1.0)
    _set_cfg("candidate_promote_streak", 2.0)
    _set_cfg("candidate_demote_streak", 2.0)
    _set_cfg(f"candidate_live:{name}", 0.0)
    _set_cfg(f"candidate_promote_streak:{name}", 0.0)
    _set_cfg(f"candidate_demote_streak:{name}", 0.0)

    good = {name: {"ic": 0.05, "t": 2.0, "n": 20}}
    update_candidate_lifecycle(good, evaluated_signal_names=[name])
    assert int(_get_cfg(f"candidate_live:{name}", 0.0)) == 0
    update_candidate_lifecycle(good, evaluated_signal_names=[name])
    assert int(_get_cfg(f"candidate_live:{name}", 0.0)) == 1


def test_candidate_demotes_after_weak_streak():
    from signals.candidate_lifecycle import update_candidate_lifecycle

    name = "cand_momentum_h4"
    _set_cfg("candidate_lifecycle_enabled", 1.0)
    _set_cfg("candidate_shadow_only", 0.0)
    _set_cfg("candidate_min_obs", 10.0)
    _set_cfg("candidate_min_abs_ic", 0.02)
    _set_cfg("candidate_min_abs_t", 1.0)
    _set_cfg("candidate_promote_streak", 2.0)
    _set_cfg("candidate_demote_streak", 2.0)
    _set_cfg(f"candidate_live:{name}", 1.0)
    _set_cfg(f"candidate_promote_streak:{name}", 0.0)
    _set_cfg(f"candidate_demote_streak:{name}", 0.0)

    weak = {name: {"ic": 0.001, "t": 0.1, "n": 20}}
    update_candidate_lifecycle(weak, evaluated_signal_names=[name])
    assert int(_get_cfg(f"candidate_live:{name}", 0.0)) == 1
    update_candidate_lifecycle(weak, evaluated_signal_names=[name])
    assert int(_get_cfg(f"candidate_live:{name}", 0.0)) == 0


def test_apply_candidate_weight_caps_caps_and_renormalizes():
    from signals.candidate_lifecycle import apply_candidate_weight_caps

    name = "cand_momentum_h4"
    _set_cfg("candidate_lifecycle_enabled", 1.0)
    _set_cfg("candidate_shadow_only", 0.0)
    _set_cfg("candidate_max_live_abs_weight", 0.1)
    _set_cfg(f"candidate_live:{name}", 1.0)

    w = {"momentum": 0.4, name: 0.6}
    out = apply_candidate_weight_caps(w)
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9
    assert abs(out[name]) <= 0.21  # post-renorm cap retains bounded influence
