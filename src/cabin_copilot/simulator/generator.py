"""Seeded, stratified scenario generation.

Counts are allocated proportionally to cell weights (largest-remainder rounding so the
total is exact), then each scenario is drawn from its cell's sampler with a per-cell
RNG seeded from (seed, cell name) — adding a cell never perturbs the draws of existing
cells, which keeps datasets diffable across taxonomy changes.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from cabin_copilot.schemas import Scenario
from cabin_copilot.simulator.taxonomy import TAXONOMY, TaxonomyCell


def _allocate(n: int) -> list[tuple[TaxonomyCell, int]]:
    total_w = sum(c.weight for c in TAXONOMY)
    raw = [(c, n * c.weight / total_w) for c in TAXONOMY]
    counts = {c.name: int(x) for c, x in raw}
    remainder = n - sum(counts.values())
    for c, _x in sorted(raw, key=lambda t: t[1] - int(t[1]), reverse=True)[:remainder]:
        counts[c.name] += 1
    return [(c, counts[c.name]) for c in TAXONOMY]


def generate_scenarios(n: int, seed: int, id_prefix: str = "scn") -> Iterator[Scenario]:
    for cell, count in _allocate(n):
        rng = random.Random(f"{seed}:{cell.name}")
        for i in range(count):
            fields = cell.sampler(rng)
            yield Scenario(
                scenario_id=f"{id_prefix}-{cell.name}-{seed}-{i:04d}",
                taxonomy=cell.name,
                expected_severity_min=cell.band[0],
                expected_severity_max=cell.band[1],
                **fields,
            )
