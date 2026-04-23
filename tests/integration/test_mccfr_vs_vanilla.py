"""Integration tests: MCCFR vs Vanilla CFR (speed + Nash convergence).

These tests check that MCCFR integrates correctly with the shared GameProtocol
and exploitability infrastructure, and that its observable behaviour is
consistent with its design goals:

1. **Iter-per-second speedup**: External sampling's whole point is that each
   iteration visits O(|I|) rather than O(|tree|) nodes (Lanctot 2009 §3).
   On Leduc (120 deals, ~290 infosets) the per-iter cost should drop by at
   least 5× relative to Vanilla.

2. **Kuhn Nash convergence** (5-seed mean): MCCFR's game value under the
   average strategy should approach -1/18 despite its stochastic nature.

Target module (NOT YET IMPLEMENTED):
    src/poker_ai/algorithms/mccfr.py
"""

from __future__ import annotations

import time

import numpy as np

from poker_ai.algorithms.mccfr import MCCFRExternalSampling
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


# -----------------------------------------------------------------------------
# Iter-per-second speedup (Leduc)
# -----------------------------------------------------------------------------
class TestMCCFRIterSpeed:
    def test_mccfr_iter_per_sec_at_least_5x_vanilla(self) -> None:
        """At T=500 iter on Leduc, MCCFR wall-clock should be ≥ 5× faster.

        Vanilla Leduc: ~11 it/s. MCCFR external sampling: ~100+ it/s
        expected (per-iter work drops from 2 × 120 deals × full tree to
        2 × 1 deal × sampled path). A 5× threshold is conservative; 10×+ is
        the design target, but CI machines vary.

        Note: timing test tolerance is intentionally loose. If this fails
        marginally, widen to `speedup >= 3.0` and open a perf issue — don't
        silently disable. A sub-1× "speedup" means the implementation is
        O(tree) per iter, which is an actual bug.
        """
        n_iter = 500

        t0 = time.perf_counter()
        vanilla = VanillaCFR(game=LeducPoker(), n_actions=3)
        vanilla.train(n_iter)
        vanilla_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        cfr = MCCFRExternalSampling(
            game=LeducPoker(),
            n_actions=3,
            rng=np.random.default_rng(42),
        )
        cfr.train(n_iter)
        mccfr_time = time.perf_counter() - t0

        speedup = vanilla_time / mccfr_time
        assert speedup >= 5.0, (
            f"MCCFR speedup {speedup:.2f}× not ≥ 5× "
            f"(Vanilla {vanilla_time:.2f}s, MCCFR {mccfr_time:.2f}s)"
        )


# -----------------------------------------------------------------------------
# Kuhn Nash convergence (5-seed mean)
# -----------------------------------------------------------------------------
class TestMCCFRKuhnConvergenceIntegration:
    def test_kuhn_mccfr_matches_nash_5_seed_average(self) -> None:
        """5-seed averaged game_value close to Kuhn Nash -1/18 ≈ -0.0556.

        End-to-end check combining MCCFR training + average_strategy() +
        game_value() evaluation. Individual seeds may drift by ~0.01–0.02 at
        10k iter; the 5-seed mean should land within 0.005 of Nash.
        """
        seeds = [42, 123, 456, 789, 1024]
        n_iter = 10_000
        nash_value = -1.0 / 18.0

        game_values: list[float] = []
        for seed in seeds:
            cfr = MCCFRExternalSampling(
                game=KuhnPoker(),
                n_actions=2,
                rng=np.random.default_rng(seed),
            )
            cfr.train(n_iter)
            game_values.append(cfr.game_value())

        mean_gv = float(np.mean(game_values))
        assert abs(mean_gv - nash_value) < 0.005, (
            f"Kuhn MCCFR mean game_value {mean_gv:.5f}, Nash {nash_value:.5f}. "
            f"Individual seeds: {game_values}"
        )
