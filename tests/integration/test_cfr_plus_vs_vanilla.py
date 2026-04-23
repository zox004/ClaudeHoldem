"""Integration: CFR+ vs Vanilla CFR convergence comparison.

Tammelin 2014's core empirical claim: at the same iteration count T on
the same game with the same seed, CFR+ reaches a lower exploitability
than Vanilla CFR. This file verifies that claim at small (fast) scales;
:mod:`tests.regression.test_leduc_cfr_plus_convergence` verifies the
same claim at the Phase 2 Exit-criterion scale (10k / 100k iter).

TDD note
--------
These tests import :class:`CFRPlus` which does not yet exist; they fail
at collection with ``ModuleNotFoundError`` until the implementation
lands in ``src/poker_ai/algorithms/cfr_plus.py``.
"""

from __future__ import annotations

import numpy as np

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import exploitability_mbb
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


class TestCFRPlusOutperformsVanilla:
    def test_kuhn_cfr_plus_beats_vanilla_at_1k(self) -> None:
        """At iter=1000 on Kuhn, CFR+ exploitability < Vanilla exploitability.

        Both trainers use the same seed (42) so any difference is due to
        the three CFR+ components (clipping + linear averaging +
        alternating), NOT random sampling.
        """
        # Vanilla baseline
        np.random.seed(42)
        vanilla = VanillaCFR(game=KuhnPoker(), n_actions=2)
        vanilla.train(1000)
        v_expl = exploitability_mbb(
            KuhnPoker(), vanilla.average_strategy(), big_blind=1.0
        )

        # CFR+
        np.random.seed(42)
        plus = CFRPlus(game=KuhnPoker(), n_actions=2)
        plus.train(1000)
        p_expl = exploitability_mbb(
            KuhnPoker(), plus.average_strategy(), big_blind=1.0
        )

        assert p_expl < v_expl, (
            f"CFR+ ({p_expl:.6f} mbb/g) should beat Vanilla "
            f"({v_expl:.6f} mbb/g) at 1k iter on Kuhn"
        )

    def test_leduc_cfr_plus_beats_vanilla_at_500(self) -> None:
        """At iter=500 on Leduc, CFR+ expl < Vanilla expl.

        500 iter ≈ 50s on M1 Pro (Day 4 baseline: ~100 iter / 10s on
        Leduc). Fast enough for everyday CI. Same-seed comparison
        eliminates sampling noise.
        """
        np.random.seed(42)
        vanilla = VanillaCFR(game=LeducPoker(), n_actions=3)
        vanilla.train(500)
        v_expl = exploitability_mbb(
            LeducPoker(), vanilla.average_strategy(), big_blind=2.0
        )

        np.random.seed(42)
        plus = CFRPlus(game=LeducPoker(), n_actions=3)
        plus.train(500)
        p_expl = exploitability_mbb(
            LeducPoker(), plus.average_strategy(), big_blind=2.0
        )

        assert p_expl < v_expl, (
            f"CFR+ ({p_expl:.6f} mbb/g) should beat Vanilla "
            f"({v_expl:.6f} mbb/g) at 500 iter on Leduc"
        )


class TestCFRPlusMonotonicImprovement:
    def test_kuhn_cfr_plus_monotonic_100_to_1k(self) -> None:
        """CFR+ exploitability at 1000 iter < at 100 iter on Kuhn.

        CFR+ enjoys an empirical ~O(1/T) exploitability decay (Tammelin
        2014, Fig 1-2), much faster than Vanilla's O(1/√T). A 10×
        iteration increase must reduce exploitability strictly.
        """
        np.random.seed(42)
        short = CFRPlus(game=KuhnPoker(), n_actions=2)
        short.train(100)
        short_expl = exploitability_mbb(
            KuhnPoker(), short.average_strategy(), big_blind=1.0
        )

        np.random.seed(42)
        long = CFRPlus(game=KuhnPoker(), n_actions=2)
        long.train(1000)
        long_expl = exploitability_mbb(
            KuhnPoker(), long.average_strategy(), big_blind=1.0
        )

        assert long_expl < short_expl, (
            f"CFR+ expl(1k)={long_expl:.6f} should be < "
            f"expl(100)={short_expl:.6f}"
        )
