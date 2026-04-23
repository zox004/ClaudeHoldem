"""Integration smoke test: VanillaCFR runs on Leduc after Day 3 C2 refactor.

Target refactor (Phase 2 Week 1 Day 3 C2):
    src/poker_ai/algorithms/vanilla_cfr.py — ``for a_idx in range(n_actions)``
    loop replaced with ``for a in state.legal_actions()``. Uses
    ``regret_matching(cumulative_regret, legal_mask=state.legal_action_mask())``
    so illegal action slots stay exactly 0 in both cumulative_regret and
    strategy outputs.

This file verifies:
  1. VanillaCFR can be instantiated against LeducPoker (game-agnostic type).
  2. ``train(1)`` and ``train(10)`` execute without raising.
  3. ``len(cfr.infosets) > 0`` after training (tree is actually being visited).
  4. ``cfr.game_value()`` is finite.
  5. **Key math invariant**: at every visited infoset, the strategy produced
     by ``current_strategy`` and ``average_strategy`` assigns probability
     **exactly zero** to any action illegal at that infoset. This is the
     downstream guarantee that Day 3 refactor must uphold.

Does NOT verify convergence (<1 mbb/g @ 100k iter). That is Day 4's
``tests/regression/test_leduc_vanilla_cfr_convergence.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.leduc import LeducAction, LeducPoker, LeducState


def _find_infoset_state(game: LeducPoker, target_key: str) -> LeducState | None:
    """DFS to find any non-terminal state with a given infoset_key.

    Needed to recover the ``legal_actions()`` for a given infoset_key when
    we only have the key string from ``cfr.infosets.keys()``.
    """
    for deal in game.all_deals():
        stack: list[LeducState] = [game.state_from_deal(deal)]
        while stack:
            state = stack.pop()
            if state.is_terminal:
                continue
            if state.infoset_key == target_key:
                return state
            for a in state.legal_actions():
                stack.append(state.next_state(a))
    return None


class TestLeducCFRInstantiation:
    def test_instantiate_with_leduc(self) -> None:
        """VanillaCFR accepts LeducPoker instance (GameProtocol conformance)."""
        cfr = VanillaCFR(game=LeducPoker(), n_actions=LeducPoker.NUM_ACTIONS)
        assert cfr.iteration == 0
        assert cfr.infosets == {}

    def test_n_actions_default_does_not_crash(self) -> None:
        """Explicit ``n_actions=3`` works; default (2) from Kuhn is a caller
        responsibility and may crash, but the explicit-3 path must work."""
        cfr = VanillaCFR(game=LeducPoker(), n_actions=3)
        # Sanity: attribute round-trip
        assert cfr.n_actions == 3


class TestLeducCFRTrainSmoke:
    def test_train_one_iter_does_not_raise(self) -> None:
        cfr = VanillaCFR(game=LeducPoker(), n_actions=3)
        cfr.train(1)
        assert cfr.iteration == 1

    def test_train_ten_iter_populates_infosets(self) -> None:
        """After 10 iterations, tree visitation should populate many infosets.

        Full Leduc has 288 infosets; even a single iteration should touch
        most (every deal's root at minimum = 6 infosets for round-1 roots
        × plus descendants). With 10 iterations the visit set is bounded
        above by 288 and should be a healthy fraction.
        """
        cfr = VanillaCFR(game=LeducPoker(), n_actions=3)
        cfr.train(10)
        assert len(cfr.infosets) > 0
        assert len(cfr.infosets) <= 288  # sanity upper bound

    def test_game_value_finite_after_train(self) -> None:
        cfr = VanillaCFR(game=LeducPoker(), n_actions=3)
        cfr.train(5)
        v = cfr.game_value()
        assert math.isfinite(v)


class TestLeducCFRStrategyMaskingInvariant:
    """Core math invariant of Day 3 C2: illegal slots must be exactly 0 in
    every strategy output, for every visited infoset.
    """

    def test_current_strategy_zero_on_illegal_slots(self) -> None:
        """After training, ``current_strategy(key)`` must give 0 probability
        to any action illegal at a state in that infoset."""
        game = LeducPoker()
        cfr = VanillaCFR(game=game, n_actions=3)
        cfr.train(10)

        for key in cfr.infosets:
            sample_state = _find_infoset_state(game, key)
            assert sample_state is not None, f"infoset {key!r} not reachable"
            legal = {int(a) for a in sample_state.legal_actions()}
            strategy = cfr.current_strategy(key)
            for a_idx in range(3):
                if a_idx not in legal:
                    assert strategy[a_idx] == 0.0, (
                        f"illegal slot {a_idx} at {key!r} has prob "
                        f"{strategy[a_idx]:.6f} (expected 0)"
                    )

    def test_average_strategy_zero_on_illegal_slots(self) -> None:
        """Same invariant for ``average_strategy()`` output."""
        game = LeducPoker()
        cfr = VanillaCFR(game=game, n_actions=3)
        cfr.train(10)
        avg = cfr.average_strategy()

        for key, strategy in avg.items():
            sample_state = _find_infoset_state(game, key)
            assert sample_state is not None, f"infoset {key!r} not reachable"
            legal = {int(a) for a in sample_state.legal_actions()}
            for a_idx in range(3):
                if a_idx not in legal:
                    assert strategy[a_idx] == 0.0, (
                        f"illegal slot {a_idx} at {key!r} has avg prob "
                        f"{strategy[a_idx]:.6f} (expected 0)"
                    )

    def test_all_strategies_sum_to_one(self) -> None:
        """Probability measure integrity: every strategy sums to 1.0 (not 2/3
        or any other value that would indicate illegal probability mass)."""
        game = LeducPoker()
        cfr = VanillaCFR(game=game, n_actions=3)
        cfr.train(10)
        for key in cfr.infosets:
            strategy = cfr.current_strategy(key)
            assert strategy.sum() == pytest.approx(1.0), (
                f"strategy at {key!r} sums to {strategy.sum():.6f}, not 1.0"
            )
