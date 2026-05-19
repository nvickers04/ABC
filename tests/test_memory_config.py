"""MemoryConfig singleton and P&L report."""

from __future__ import annotations

from core.memory_config import MemoryConfig, get_memory_config, reload_memory_config


def test_memory_config_singleton():
    a = get_memory_config()
    b = get_memory_config()
    assert a is b
    assert a.wm_policy == "postgres_wins_no_merge"
    assert a.section_caps["open_theses"] == 8


def test_legacy_risk_multiplier_tiers():
    mem = MemoryConfig()
    assert mem.legacy_risk_multiplier_for_quality(
        researcher_available=True, overall_quality="minimal"
    ) == 1.0
    assert mem.legacy_risk_multiplier_for_quality(
        researcher_available=False, overall_quality="minimal"
    ) == mem.legacy_risk_multiplier_minimal
    assert mem.legacy_risk_multiplier_for_quality(
        researcher_available=False, overall_quality="limited"
    ) == mem.legacy_risk_multiplier_limited


def test_optimize_for_profit_prints(capsys):
    reload_memory_config().optimize_for_profit()
    out = capsys.readouterr().out
    assert "MemoryConfig" in out
    assert "cycle_wm_max_chars" in out
    assert "P&L:" in out


def test_sector_of_unknown():
    mem = MemoryConfig()
    assert mem.sector_of("ZZZZ") == "other"
    assert mem.sector_of("NVDA") == "tech"
