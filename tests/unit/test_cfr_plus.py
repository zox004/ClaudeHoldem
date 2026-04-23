"""Unit tests for CFR+ (Tammelin 2014).

CFR+ = regret clipping + linear averaging + alternating updates, all three
components. Inherits from :class:`VanillaCFR`, overriding only the regret
and strategy accumulation hooks. This file asserts the mathematical contract
of those two hooks (clipping non-negativity, O(T^2) cumulative strategy
growth) and the Kuhn Nash convergence at small iteration counts where CFR+
is expected to dominate Vanilla.

Reference
---------
Tammelin (2014), "Solving Large Imperfect Information Games Using CFR+".
See also Tammelin et al. (2015, Science), "Solving heads-up limit Texas
hold'em".

TDD note
--------
These tests are written BEFORE the CFR+ implementation exists. They are
expected to fail at collection time with ``ModuleNotFoundError`` until
:mod:`poker_ai.algorithms.cfr_plus` is created.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import exploitability_mbb
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


# =============================================================================
# Instantiation & class contract
# =============================================================================


class TestCFRPlusInstantiation:
    def test_instantiate_with_kuhn(self) -> None:
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        assert cfr is not None
        assert cfr.iteration == 0
        assert cfr.infosets == {}

    def test_instantiate_with_leduc(self) -> None:
        cfr = CFRPlus(game=LeducPoker(), n_actions=3)
        assert cfr is not None
        assert cfr.iteration == 0
        assert cfr.infosets == {}

    def test_inherits_vanilla_cfr(self) -> None:
        """CFR+ should be a subclass of Vanilla CFR (sharing the alternating
        A-pattern traversal and the ``average_strategy`` / ``game_value``
        machinery)."""
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        assert isinstance(cfr, VanillaCFR)

    def test_class_docstring_mentions_tammelin_three_components(self) -> None:
        """The CFR+ class must document the three-component definition from
        Tammelin 2014 so future maintainers do not confuse it with the
        partial 'Regret Matching+' variant (clipping-only)."""
        doc = CFRPlus.__doc__ or ""
        assert "Tammelin 2014" in doc, (
            "CFR+ docstring must cite Tammelin 2014"
        )
        assert "regret clipping" in doc, (
            "CFR+ docstring must mention regret clipping component"
        )
        assert "linear averaging" in doc, (
            "CFR+ docstring must mention linear averaging component"
        )
        assert "alternating" in doc, (
            "CFR+ docstring must mention alternating updates component"
        )
        assert "all three components" in doc, (
            "CFR+ docstring must explicitly state that all three components "
            "are included (distinguishes from partial RM+ variant)"
        )


# =============================================================================
# Regret clipping (CFR+ component #1)
# =============================================================================


class TestCFRPlusRegretClipping:
    def test_negative_regret_gets_clipped_to_zero(self) -> None:
        """After a single iteration, every cumulative_regret slot must be
        non-negative (positive-part at storage, Tammelin 2014 Eq. 3)."""
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr.train(1)
        for key, data in cfr.infosets.items():
            assert np.all(data.cumulative_regret >= 0.0), (
                f"CFR+ cumulative_regret at {key!r} has negative value: "
                f"{data.cumulative_regret}"
            )

    def test_regret_stays_non_negative_over_10_iter(self) -> None:
        """Clipping invariant holds after EVERY iteration, not just the first."""
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        for t in range(10):
            cfr.train(1)
            for key, data in cfr.infosets.items():
                assert np.all(data.cumulative_regret >= 0.0), (
                    f"iter {t + 1}: cumulative_regret at {key!r} negative: "
                    f"{data.cumulative_regret}"
                )

    def test_vanilla_allows_negative_regret(self) -> None:
        """Control test: Vanilla CFR stores RAW regret (no clipping), so at
        least one infoset slot is expected to go negative within 100 iter on
        Kuhn. If this fails the Vanilla reference behaviour has changed."""
        np.random.seed(42)
        cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
        cfr.train(100)
        has_negative = any(
            np.any(data.cumulative_regret < 0.0)
            for data in cfr.infosets.values()
        )
        assert has_negative, (
            "Expected at least one negative cumulative_regret slot in "
            "Vanilla CFR on Kuhn after 100 iter — Vanilla must NOT clip "
            "at storage time. If this fails, Vanilla may have been "
            "silently refactored to clip."
        )


# =============================================================================
# Linear averaging (CFR+ component #2)
# =============================================================================


class TestCFRPlusLinearAveraging:
    def test_cumulative_strategy_weighted_by_iteration(self) -> None:
        """CFR+ linear averaging: s(I, a) += t * pi_i * sigma(I, a).

        Under linear averaging cumulative_strategy grows roughly as
        sum_{t=1..T} t = T(T+1)/2 = O(T^2). Doubling T should therefore
        multiply the total magnitude by ~4× (upper bound; strategy values
        vary so growth is loose). We require ratio > 2.0 so that *something
        faster than O(T)* is demonstrably present.
        """
        np.random.seed(42)
        cfr_short = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr_short.train(10)
        total_10 = sum(
            data.cumulative_strategy.sum()
            for data in cfr_short.infosets.values()
        )

        np.random.seed(42)
        cfr_long = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr_long.train(20)
        total_20 = sum(
            data.cumulative_strategy.sum()
            for data in cfr_long.infosets.values()
        )

        ratio = total_20 / max(total_10, 1e-12)
        assert ratio > 2.0, (
            f"Linear averaging should grow faster than O(T) — doubling T "
            f"should give ratio > 2.0, saw {ratio:.2f}. "
            f"(total_10={total_10:.4f}, total_20={total_20:.4f})"
        )

    def test_vanilla_cumulative_strategy_grows_linearly(self) -> None:
        """Control test: Vanilla uses unweighted averaging, cumulative_strategy
        grows O(T), so doubling T yields ratio ~2."""
        np.random.seed(42)
        cfr_short = VanillaCFR(game=KuhnPoker(), n_actions=2)
        cfr_short.train(10)
        total_10 = sum(
            data.cumulative_strategy.sum()
            for data in cfr_short.infosets.values()
        )

        np.random.seed(42)
        cfr_long = VanillaCFR(game=KuhnPoker(), n_actions=2)
        cfr_long.train(20)
        total_20 = sum(
            data.cumulative_strategy.sum()
            for data in cfr_long.infosets.values()
        )

        ratio = total_20 / max(total_10, 1e-12)
        assert 1.5 < ratio < 3.5, (
            f"Vanilla should grow O(T): doubling T gives ~2× ratio, "
            f"expected [1.5, 3.5], saw {ratio:.2f}. "
            f"(total_10={total_10:.4f}, total_20={total_20:.4f})"
        )


# =============================================================================
# Kuhn Nash convergence (small-scale smoke)
# =============================================================================


class TestCFRPlusKuhnConvergence:
    def test_kuhn_nash_value_converges(self) -> None:
        """Kuhn Poker Nash game value = -1/18 ≈ -0.0556 (Neller & Lanctot 2013
        Section 4.1). CFR+ must converge to this at 1k iter on Kuhn."""
        np.random.seed(42)
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr.train(1000)
        v = cfr.game_value()
        assert abs(v - (-1.0 / 18.0)) < 0.005, (
            f"CFR+ Kuhn game value after 1k iter = {v:.6f}, "
            f"expected {-1.0 / 18.0:.6f} ± 0.005"
        )

    def test_kuhn_expl_below_threshold_at_10k(self) -> None:
        """CFR+ @ 10k iter Kuhn exploitability < 0.1 mbb/g.

        Tammelin 2014 Fig 2 reports ~0.01-0.05 mbb/g at this scale on
        Kuhn-sized games; threshold 0.1 gives 2-10× safety margin.
        """
        np.random.seed(42)
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr.train(10_000)
        expl = exploitability_mbb(
            KuhnPoker(), cfr.average_strategy(), big_blind=1.0
        )
        assert expl < 0.1, (
            f"CFR+ Kuhn @ 10k expl = {expl:.6f} mbb/g, expected < 0.1"
        )


# =============================================================================
# Overflow safety (linear averaging is O(T^2), guard against fp issues)
# =============================================================================


class TestCFRPlusOverflowSafety:
    def test_cumulative_strategy_finite_after_100_iter(self) -> None:
        """Linear averaging's cumulative_strategy is O(T^2); at Leduc 100k
        iter this reaches ~10^10, well within float64 precision (~10^15).
        This test guards against accidental fp accumulation bugs that
        introduce NaN/inf under the weighting."""
        cfr = CFRPlus(game=KuhnPoker(), n_actions=2)
        cfr.train(100)
        for key, data in cfr.infosets.items():
            assert np.all(np.isfinite(data.cumulative_strategy)), (
                f"cumulative_strategy at {key!r} has non-finite value: "
                f"{data.cumulative_strategy}"
            )
            assert np.all(np.isfinite(data.cumulative_regret)), (
                f"cumulative_regret at {key!r} has non-finite value: "
                f"{data.cumulative_regret}"
            )
