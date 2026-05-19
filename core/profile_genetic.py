"""Genetic search over profitability profile levers (historical simulation)."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from core.profile_optimization import (
    PatchDict,
    ProfileCandidate,
    apply_config_patches,
    _load_profile_config,
)
from core.profit_profiles import ProfitProfile

# Mutable genes: (section, field, min, max, is_int)
GENE_SPECS: tuple[tuple[str, str, float, float, bool], ...] = (
    ("risk", "risk_per_trade_pct", 0.25, 5.0, False),
    ("prompt", "high_conviction_confidence_floor", 0.65, 0.92, False),
    ("prompt", "wm_default_entry_confidence", 0.70, 0.95, False),
    ("memory", "cycle_wm_max_chars", 1200, 3600, True),
    ("memory", "cycle_attention_max_chars", 600, 2000, True),
    ("memory", "cycle_last_summary_chars", 80, 400, True),
)


@dataclass
class Genome:
    """Evolvable lever set applied as patches on a built-in base profile."""

    base_profile: ProfitProfile
    genes: dict[str, float] = field(default_factory=dict)
    generation: int = 0
    genome_id: str = ""
    fitness: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.genome_id:
            self.genome_id = self._hash()[:12]

    def _hash(self) -> str:
        payload = json.dumps(
            {"base": self.base_profile, "genes": self.genes},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def candidate_id(self) -> str:
        return f"ga_{self.base_profile}_{self.genome_id}"


def _clamp_gene(section: str, fname: str, value: float) -> float:
    for sec, name, lo, hi, is_int in GENE_SPECS:
        if sec == section and name == fname:
            v = max(lo, min(hi, value))
            return float(int(round(v))) if is_int else round(v, 4)
    return value


def genome_to_patches(genome: Genome) -> PatchDict:
    """Map flat gene dict to risk / prompt / memory patch sections."""
    patches: PatchDict = {}
    for section, fname, _, _, _ in GENE_SPECS:
        key = f"{section}.{fname}"
        if key not in genome.genes:
            continue
        patches.setdefault(section, {})[fname] = _clamp_gene(section, fname, genome.genes[key])
    return patches


def genome_from_profile(profile: ProfitProfile) -> Genome:
    """Seed genome from an applied built-in profile."""
    cfg = _load_profile_config(profile)
    genes: dict[str, float] = {}
    for section, fname, _, _, is_int in GENE_SPECS:
        model = getattr(cfg, section)
        val = float(getattr(model, fname))
        if is_int:
            val = float(int(val))
        genes[f"{section}.{fname}"] = val
    return Genome(base_profile=profile, genes=genes, generation=0)


def validate_genome(genome: Genome) -> Genome:
    """Ensure patches pass Pydantic validators (loop/risk constraints)."""
    cfg = _load_profile_config(genome.base_profile)
    apply_config_patches(cfg, genome_to_patches(genome))
    return genome


def genome_to_candidate(genome: Genome) -> ProfileCandidate:
    return ProfileCandidate(
        candidate_id=genome.candidate_id(),
        base_profile=genome.base_profile,
        patches=genome_to_patches(genome),
    )


def mutate(genome: Genome, *, rate: float = 0.25, rng: random.Random | None = None) -> Genome:
    """Perturb genes with probability ``rate`` per locus."""
    r = rng or random.Random()
    child_genes = dict(genome.genes)
    for section, fname, lo, hi, is_int in GENE_SPECS:
        key = f"{section}.{fname}"
        if key not in child_genes or r.random() > rate:
            continue
        cur = child_genes[key]
        span = hi - lo
        delta = r.uniform(-0.12, 0.12) * span
        child_genes[key] = _clamp_gene(section, fname, cur + delta)
    return Genome(
        base_profile=genome.base_profile,
        genes=child_genes,
        generation=genome.generation + 1,
    )


def crossover(
    a: Genome,
    b: Genome,
    *,
    rng: random.Random | None = None,
) -> Genome:
    """Uniform crossover; child base profile from fitter parent."""
    r = rng or random.Random()
    fit_a = a.fitness if a.fitness is not None else 0.0
    fit_b = b.fitness if b.fitness is not None else 0.0
    base = a.base_profile if fit_a >= fit_b else b.base_profile
    if a.base_profile != b.base_profile and r.random() < 0.5:
        base = r.choice([a.base_profile, b.base_profile])
    genes: dict[str, float] = {}
    for section, fname, _, _, _ in GENE_SPECS:
        key = f"{section}.{fname}"
        genes[key] = a.genes[key] if r.random() < 0.5 else b.genes[key]
    gen = max(a.generation, b.generation) + 1
    return Genome(base_profile=base, genes=genes, generation=gen)


def seed_population(
    *,
    rng: random.Random | None = None,
    population_size: int = 12,
) -> list[Genome]:
    """Three built-in seeds plus mutated variants to fill the population."""
    r = rng or random.Random()
    seeds = [genome_from_profile(p) for p in ("conservative", "balanced", "aggressive")]  # type: ignore[arg-type]
    pop: list[Genome] = [validate_genome(g) for g in seeds]
    while len(pop) < population_size:
        parent = r.choice(seeds)
        child = mutate(parent, rate=0.35, rng=r)
        try:
            pop.append(validate_genome(child))
        except Exception:
            continue
    return pop[:population_size]


def tournament_select(
    population: list[Genome],
    *,
    k: int = 3,
    rng: random.Random | None = None,
) -> Genome:
    r = rng or random.Random()
    contenders = r.sample(population, min(k, len(population)))
    return max(contenders, key=lambda g: g.fitness if g.fitness is not None else -1.0)


@dataclass
class GeneticSearchConfig:
    generations: int = 25
    population_size: int = 12
    elite_count: int = 2
    mutation_rate: float = 0.25
    tournament_k: int = 3
    seed: int | None = None


@dataclass
class GeneticSearchResult:
    best: Genome
    history: list[dict[str, Any]]
    population_final: list[Genome]
    evaluated_count: int


def format_profit_profiles_snippet(
    profile_name: str,
    genome: Genome,
    *,
    note: str = "",
    metrics: dict[str, Any] | None = None,
) -> str:
    """Python snippet to register an evolved profile (merge into profit_profiles.py)."""
    patches = genome_to_patches(genome)
    metrics_block = ""
    if metrics:
        metrics_block = f"\n    # backtest metrics: {json.dumps(metrics, default=str)}"
    return f'''
# --- Evolved profile "{profile_name}" (genetic search; safe to paste into profit_profiles.py) ---
# Register via data/evolved_profiles.json or call register_evolved_profile() at startup.
EVOLVED_PROFILE_SNIPPET_{profile_name.upper().replace("-", "_")} = {{
    "base_profile": "{genome.base_profile}",
    "note": {note!r},{metrics_block}
    "patches": {json.dumps(patches, indent=4)},
    "genes": {json.dumps(genome.genes, indent=4)},
}}
'''.strip()


def run_genetic_search(
    evaluate_fn: Any,
    config: GeneticSearchConfig | None = None,
) -> GeneticSearchResult:
    """Run GA; ``evaluate_fn(genome) -> metrics dict`` must set fitness on genome."""
    cfg = config or GeneticSearchConfig()
    rng = random.Random(cfg.seed)
    pop = seed_population(rng=rng, population_size=cfg.population_size)
    cache: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    evaluated = 0

    def _eval(genome: Genome) -> None:
        nonlocal evaluated
        key = genome._hash()
        if key in cache:
            genome.fitness = cache[key]["composite_score"]
            genome.metrics = cache[key]
            return
        metrics = evaluate_fn(genome)
        cache[key] = metrics
        genome.fitness = float(metrics["composite_score"])
        genome.metrics = metrics
        evaluated += 1

    for gen in range(cfg.generations):
        for g in pop:
            if g.fitness is None:
                _eval(g)
        pop.sort(key=lambda g: g.fitness or -1.0, reverse=True)
        best = pop[0]
        history.append(
            {
                "generation": gen,
                "best_id": best.genome_id,
                "best_fitness": best.fitness,
                "best_base": best.base_profile,
            }
        )

        if gen >= cfg.generations - 1:
            break

        next_pop: list[Genome] = [validate_genome(Genome(
            base_profile=g.base_profile,
            genes=dict(g.genes),
            generation=g.generation,
            fitness=g.fitness,
            metrics=dict(g.metrics),
        )) for g in pop[: cfg.elite_count]]

        while len(next_pop) < cfg.population_size:
            p1 = tournament_select(pop, k=cfg.tournament_k, rng=rng)
            p2 = tournament_select(pop, k=cfg.tournament_k, rng=rng)
            child = crossover(p1, p2, rng=rng)
            child = mutate(child, rate=cfg.mutation_rate, rng=rng)
            try:
                next_pop.append(validate_genome(child))
            except Exception:
                next_pop.append(validate_genome(mutate(p1, rate=0.4, rng=rng)))

        pop = next_pop

    for g in pop:
        if g.fitness is None:
            _eval(g)
    pop.sort(key=lambda g: g.fitness or -1.0, reverse=True)
    return GeneticSearchResult(
        best=pop[0],
        history=history,
        population_final=pop,
        evaluated_count=evaluated,
    )


__all__ = [
    "GENE_SPECS",
    "GeneticSearchConfig",
    "GeneticSearchResult",
    "Genome",
    "crossover",
    "format_profit_profiles_snippet",
    "genome_from_profile",
    "genome_to_candidate",
    "genome_to_patches",
    "mutate",
    "run_genetic_search",
    "seed_population",
    "validate_genome",
]
