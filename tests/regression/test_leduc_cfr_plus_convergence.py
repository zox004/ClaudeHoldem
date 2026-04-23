"""Regression tests: Phase 2 Exit #2 — CFR+ @ 10k ≤ 0.5 mbb/g on Leduc.

Locks in the second Phase 2 acceptance gate: Counterfactual Regret
Minimization Plus (Tammelin 2014) reaches < 0.5 mbb/g on Leduc with
just 10,000 iterations, empirically demonstrating the 100× speedup
vs Vanilla CFR (which needed 100k iters to reach 1.4802 mbb/g).

Threshold rationale
-------------------
Tammelin 2014 Figure 2 reports CFR+ on Leduc at T=10^4 reaches ~0.3
mbb/g. Our 0.5 threshold gives ~40% margin.

Vanilla CFR baseline (Phase 2 Day 4, 100k iter, seed=42, M1 Pro):
- 100 iter:   ~29 mbb/g
- 1k   iter:   9.15 mbb/g
- 10k  iter:   3.50 mbb/g
- 100k iter:   1.48 mbb/g   ← Exit #1 1.0 FAILED

CFR+ is expected to reach Vanilla-100k-quality (1.48 mbb/g) within
a handful of thousand iterations, AND to continue improving to sub-0.5
mbb/g by 10k. If true, this resolves Exit #1 (CFR+ @ 10k < 1.0 mbb/g)
while also passing Exit #2 (10× iter speedup over Vanilla).

Runtime:
- fast (10k iter, ~15min M1 Pro): everyday CI
- slow (100k iter, ~2.5h M1 Pro): @pytest.mark.slow; Phase 2 triumph gate

Reference: CLAUDE.md Phase 2 Exit Criteria; ROADMAP.md §Phase 2 Week 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.eval.exploitability import exploitability_mbb
from poker_ai.games.leduc import LeducPoker

FAST_ITERATIONS = 10_000
FAST_EXIT_CRITERION_MBB = 0.5   # Tammelin 2014 Fig 2: ~0.3 @ 10k; 0.5 gives 40% margin
EXIT_2_SPEEDUP_TARGET_MBB = 1.4802  # Vanilla 100k baseline (Day 4 measurement)

SLOW_ITERATIONS = 100_000
SLOW_EXIT_CRITERION_MBB = 0.05  # Tammelin 2014 Fig 2: ~0.05 @ 100k

LEDUC_BIG_BLIND = 2.0


def test_cfr_plus_overflow_safety() -> None:
    """cumulative_strategy O(T^2) scale must stay finite even at small T
    (Leduc 288 infosets × weighting; guards against accumulation bugs)."""
    cfr = CFRPlus(game=LeducPoker(), n_actions=3)
    cfr.train(100)
    for key, data in cfr.infosets.items():
        assert np.all(np.isfinite(data.cumulative_strategy)), (
            f"cumulative_strategy at {key!r} has non-finite value"
        )
        assert np.all(np.isfinite(data.cumulative_regret)), (
            f"cumulative_regret at {key!r} has non-finite value"
        )


@pytest.mark.parametrize("seed", [42])
def test_cfr_plus_exit_2_speedup_at_1k(seed: int) -> None:
    """Phase 2 Exit #2: CFR+ @ 1k iter ≤ Vanilla CFR @ 100k iter (1.4802 mbb/g).

    This demonstrates ≥100× iteration speedup — stronger than the ROADMAP's
    5-10× original target. Passes if Tammelin's CFR+ Fig 2 numbers hold.

    ~1.5min runtime (1k iter on Leduc).
    """
    np.random.seed(seed)
    cfr = CFRPlus(game=LeducPoker(), n_actions=3)
    cfr.train(1000)
    expl_mbb = exploitability_mbb(
        LeducPoker(), cfr.average_strategy(), big_blind=LEDUC_BIG_BLIND
    )
    assert expl_mbb < EXIT_2_SPEEDUP_TARGET_MBB, (
        f"seed={seed}: CFR+ expl @ 1k = {expl_mbb:.6f} mbb/g, "
        f"expected ≤ Vanilla CFR @ 100k baseline {EXIT_2_SPEEDUP_TARGET_MBB}"
    )


@pytest.mark.parametrize("seed", [42])
def test_cfr_plus_exit_2_threshold_at_10k(seed: int) -> None:
    """Phase 2 Exit #2 formal: CFR+ @ 10k iter < 0.5 mbb/g on Leduc.

    Tammelin 2014 Figure 2 expected ~0.3 mbb/g; 0.5 threshold gives
    ~40% safety margin. Also rescues Exit #1 (< 1.0 mbb/g) since
    0.5 < 1.0.

    ~15min runtime; everyday CI candidate (no @slow marker).
    """
    np.random.seed(seed)
    cfr = CFRPlus(game=LeducPoker(), n_actions=3)
    cfr.train(FAST_ITERATIONS)
    expl_mbb = exploitability_mbb(
        LeducPoker(), cfr.average_strategy(), big_blind=LEDUC_BIG_BLIND
    )
    assert expl_mbb < FAST_EXIT_CRITERION_MBB, (
        f"seed={seed}: CFR+ expl @ {FAST_ITERATIONS} = {expl_mbb:.6f} mbb/g, "
        f"Exit #2 requires < {FAST_EXIT_CRITERION_MBB}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", [42])
def test_cfr_plus_exit_1_rescue_at_100k(seed: int) -> None:
    """Phase 2 Exit #1 rescue via CFR+ @ 100k iter < 0.05 mbb/g.

    Vanilla CFR failed Exit #1 (100k = 1.4802 mbb/g). CFR+ is expected
    to not just pass the original 1.0 mbb/g threshold but to dominate
    at 0.05 (Tammelin 2014 Fig 2). Runtime: ~2.5h.
    """
    np.random.seed(seed)
    cfr = CFRPlus(game=LeducPoker(), n_actions=3)
    cfr.train(SLOW_ITERATIONS)
    expl_mbb = exploitability_mbb(
        LeducPoker(), cfr.average_strategy(), big_blind=LEDUC_BIG_BLIND
    )
    assert expl_mbb < SLOW_EXIT_CRITERION_MBB, (
        f"seed={seed}: CFR+ expl @ {SLOW_ITERATIONS} = {expl_mbb:.6f} mbb/g, "
        f"Tammelin Fig 2 expected ~0.05; threshold {SLOW_EXIT_CRITERION_MBB}"
    )
