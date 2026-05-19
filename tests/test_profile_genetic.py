"""Genetic profile search (unit tests, no backtest)."""

from __future__ import annotations

import json

import pytest

from core.profile_genetic import (
    GeneticSearchConfig,
    crossover,
    genome_from_profile,
    genome_to_patches,
    mutate,
    run_genetic_search,
    seed_population,
    validate_genome,
)
from core.profit_profiles import (
    EvolvedProfileEntry,
    load_evolved_profile_registry,
    save_evolved_profile,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_genome_from_profile_has_all_genes():
    g = genome_from_profile("balanced")
    assert "risk.risk_per_trade_pct" in g.genes
    assert "memory.cycle_wm_max_chars" in g.genes
    patches = genome_to_patches(g)
    assert "risk" in patches and "memory" in patches


def test_mutate_and_crossover_stay_in_bounds():
    a = validate_genome(genome_from_profile("conservative"))
    b = validate_genome(genome_from_profile("aggressive"))
    child = validate_genome(crossover(a, b))
    assert child.genes["risk.risk_per_trade_pct"] >= 0.25
    m = validate_genome(mutate(child, rate=1.0))
    assert m.generation >= child.generation


def test_run_genetic_search_mock_fitness():
    pop = seed_population(population_size=6)

    def evaluate(genome):
        # Higher risk % -> higher fake fitness (deterministic)
        r = genome.genes.get("risk.risk_per_trade_pct", 1.0)
        return {
            "composite_score": r / 5.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 1.0,
            "win_rate_pct": 50.0,
        }

    result = run_genetic_search(
        evaluate,
        GeneticSearchConfig(generations=3, population_size=6, elite_count=1, seed=42),
    )
    assert result.best.fitness is not None
    assert len(result.history) == 3


def test_save_evolved_profile_roundtrip(tmp_path, monkeypatch):
    from core import profit_profiles as pp

    path = tmp_path / "evolved_profiles.json"
    monkeypatch.setattr(pp, "evolved_profiles_path", lambda: path)

    entry = EvolvedProfileEntry(
        base_profile="balanced",
        patches={"risk": {"risk_per_trade_pct": 1.5}},
        note="test evolved",
        genes={"risk.risk_per_trade_pct": 1.5},
    )
    save_evolved_profile("evolved_test", entry)
    reg = load_evolved_profile_registry()
    assert "evolved_test" in reg
    assert reg["evolved_test"].patches["risk"]["risk_per_trade_pct"] == 1.5
