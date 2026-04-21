"""Integration tests: Vanilla CFR × Kuhn Poker engine.

Verify that the CFR trainer correctly wires together the Kuhn game engine and
the regret-matching machinery after a small number of iterations. Focus on
invariants that should hold regardless of Nash convergence (those are in the
regression suite).

Target modules:
    src/poker_ai/algorithms/vanilla_cfr.py
    src/poker_ai/games/kuhn.py  (already implemented)

References: Zinkevich et al. 2007 Eq. (3)–(9).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.kuhn import KuhnPoker

N_ACTIONS = 2

# The 12 Kuhn infosets that Vanilla CFR must reach.
EXPECTED_INFOSET_KEYS: frozenset[str] = frozenset(
    {
        # P1 decision points (non-terminal histories "", "pb")
        "J|", "Q|", "K|",
        "J|pb", "Q|pb", "K|pb",
        # P2 decision points (non-terminal histories "p", "b")
        "J|p", "Q|p", "K|p",
        "J|b", "Q|b", "K|b",
    }
)


@pytest.fixture
def cfr_after_one_iter() -> VanillaCFR:
    """Deterministic seed for future-proofing (Vanilla CFR itself is deterministic,
    but pinning the seed documents intent and guards against future edits)."""
    np.random.seed(42)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=N_ACTIONS)
    cfr.train(1)
    return cfr


class TestOneIterSnapshot:
    def test_exactly_twelve_infosets_populated(
        self, cfr_after_one_iter: VanillaCFR
    ) -> None:
        """Eq. (3)–(4): full tree traversal from all 6 deals × 2 players reaches
        every one of Kuhn's 12 infosets in a single training iteration."""
        assert len(cfr_after_one_iter.infosets) == 12

    def test_infoset_keys_match_expected_set(
        self, cfr_after_one_iter: VanillaCFR
    ) -> None:
        """The populated infoset keys must exactly equal the 12 Kuhn infosets."""
        assert set(cfr_after_one_iter.infosets.keys()) == EXPECTED_INFOSET_KEYS

    def test_all_cumulative_strategy_non_negative(
        self, cfr_after_one_iter: VanillaCFR
    ) -> None:
        """Eq. (6): reach-weighted strategy accumulation is never negative."""
        for key, data in cfr_after_one_iter.infosets.items():
            assert (data.cumulative_strategy >= 0).all(), (
                f"{key}: {data.cumulative_strategy}"
            )


class TestAverageStrategyIntegration:
    def test_every_distribution_valid_after_small_training(self) -> None:
        """Eq. (6): after 50 iter, average_strategy() gives 12 valid distributions."""
        cfr = VanillaCFR(game=KuhnPoker(), n_actions=N_ACTIONS)
        cfr.train(50)
        avg = cfr.average_strategy()
        assert len(avg) == 12
        for key, dist in avg.items():
            assert dist.shape == (N_ACTIONS,)
            assert dist.sum() == pytest.approx(1.0, abs=1e-6), (
                f"{key} sum = {dist.sum()}"
            )
            assert (dist >= 0).all(), f"{key} has negative entry: {dist}"


class TestDeterminism:
    def test_two_trainers_with_identical_setup_produce_identical_results(
        self,
    ) -> None:
        """Vanilla CFR is deterministic (full tree traversal, no sampling).
        Two trainers trained identically must produce identical numerical state.
        """
        np.random.seed(42)
        cfr_a = VanillaCFR(game=KuhnPoker(), n_actions=N_ACTIONS)
        cfr_a.train(20)

        np.random.seed(42)
        cfr_b = VanillaCFR(game=KuhnPoker(), n_actions=N_ACTIONS)
        cfr_b.train(20)

        assert set(cfr_a.infosets.keys()) == set(cfr_b.infosets.keys())
        for key in cfr_a.infosets:
            np.testing.assert_array_equal(
                cfr_a.infosets[key].cumulative_regret,
                cfr_b.infosets[key].cumulative_regret,
                err_msg=f"regret mismatch at {key}",
            )
            np.testing.assert_array_equal(
                cfr_a.infosets[key].cumulative_strategy,
                cfr_b.infosets[key].cumulative_strategy,
                err_msg=f"strategy mismatch at {key}",
            )
