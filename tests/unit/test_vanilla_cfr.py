"""Unit tests for Vanilla CFR trainer (Zinkevich et al. 2007).

Target module (NOT YET IMPLEMENTED — these tests must fail with ModuleNotFoundError):
    src/poker_ai/algorithms/vanilla_cfr.py

Target API:
    @dataclass
    class InfosetData:
        cumulative_regret: np.ndarray       # shape (n_actions,)
        cumulative_strategy: np.ndarray     # shape (n_actions,), reach-weighted

    class VanillaCFR:
        def __init__(self, game: KuhnPoker, n_actions: int = 2) -> None: ...
        infosets: dict[str, InfosetData]   # lazy-initialised on first visit
        iteration: int                      # one cycle = both players updated

        def train(self, iterations: int) -> None: ...
        def current_strategy(self, infoset_key: str) -> np.ndarray:
            '''Eq. (5): regret matching on cumulative regret.'''
        def average_strategy(self) -> dict[str, np.ndarray]:
            '''Eq. (6): reach-weighted time-averaged strategy per infoset.'''
        def game_value(self) -> float:
            '''Game value from P1 perspective using average_strategy().'''

References:
    Zinkevich et al. 2007, "Regret Minimization in Games with Incomplete Information".
    Neller & Lanctot 2013, "An Introduction to CFR", Algorithm 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import InfosetData, VanillaCFR
from poker_ai.games.kuhn import KuhnPoker

N_ACTIONS = 2  # Kuhn: PASS, BET


@pytest.fixture
def fresh_cfr() -> VanillaCFR:
    """Brand-new trainer, no training performed."""
    return VanillaCFR(game=KuhnPoker(), n_actions=N_ACTIONS)


# -----------------------------------------------------------------------------
# Trainer bookkeeping
# -----------------------------------------------------------------------------
class TestTrainerBookkeeping:
    def test_empty_trainer_has_no_infosets(self, fresh_cfr: VanillaCFR) -> None:
        """Fresh trainer has empty infoset dict (lazy init)."""
        assert fresh_cfr.infosets == {}
        assert fresh_cfr.iteration == 0

    def test_train_zero_iterations_is_noop(self, fresh_cfr: VanillaCFR) -> None:
        """train(0) leaves the trainer unchanged."""
        fresh_cfr.train(0)
        assert fresh_cfr.infosets == {}
        assert fresh_cfr.iteration == 0

    def test_iteration_counter_increments_once_per_cycle(
        self, fresh_cfr: VanillaCFR
    ) -> None:
        """Each train() iteration = both players updated once (per user spec)."""
        fresh_cfr.train(3)
        assert fresh_cfr.iteration == 3


# -----------------------------------------------------------------------------
# Lazy InfosetData initialization
# -----------------------------------------------------------------------------
class TestLazyInfosetInit:
    def test_one_iter_populates_exactly_12_infosets(
        self, fresh_cfr: VanillaCFR
    ) -> None:
        """Kuhn has 12 infosets (3 cards × 4 non-terminal histories).

        Lazy init must create exactly 12 — no duplicates, no missing.
        """
        fresh_cfr.train(1)
        assert len(fresh_cfr.infosets) == 12

    def test_infoset_data_regret_shape(self, fresh_cfr: VanillaCFR) -> None:
        """Each InfosetData holds per-action regret (shape (n_actions,))."""
        fresh_cfr.train(1)
        for key, data in fresh_cfr.infosets.items():
            assert data.cumulative_regret.shape == (N_ACTIONS,), (
                f"infoset {key} has wrong regret shape"
            )

    def test_infoset_data_strategy_shape(self, fresh_cfr: VanillaCFR) -> None:
        """Each InfosetData holds per-action strategy sum (shape (n_actions,))."""
        fresh_cfr.train(1)
        for key, data in fresh_cfr.infosets.items():
            assert data.cumulative_strategy.shape == (N_ACTIONS,)

    def test_cumulative_strategy_is_non_negative(
        self, fresh_cfr: VanillaCFR
    ) -> None:
        """Eq. (6): reach-weighted σ accumulation; each σ(a) is already in [0,1]
        and reach probability is non-negative, so the sum is always ≥ 0."""
        fresh_cfr.train(1)
        for key, data in fresh_cfr.infosets.items():
            assert (data.cumulative_strategy >= 0).all(), (
                f"{key} has negative cumulative_strategy: {data.cumulative_strategy}"
            )


# -----------------------------------------------------------------------------
# current_strategy — Eq. (5) regret matching
# -----------------------------------------------------------------------------
class TestCurrentStrategyRegretMatching:
    def test_uniform_on_unseen_infoset(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (5) base case: infoset with all-zero (implicit) cumulative regret
        returns uniform distribution.

        Lazy init means an unseen infoset has no InfosetData, but its implicit
        regret is zero → positive-part sum is zero → uniform fallback.
        """
        strategy = fresh_cfr.current_strategy("J|")
        np.testing.assert_allclose(strategy, [1.0 / N_ACTIONS] * N_ACTIONS, atol=1e-9)

    def test_positive_regret_concentration(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (5): single dominant positive regret → that action."""
        fresh_cfr.train(1)  # populate the dict first
        fresh_cfr.infosets["J|"].cumulative_regret = np.array([1.0, 0.0])
        strategy = fresh_cfr.current_strategy("J|")
        assert strategy[0] > 0.99
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)

    def test_all_negative_regret_returns_uniform(
        self, fresh_cfr: VanillaCFR
    ) -> None:
        """Eq. (5): if all positive parts are zero, fall back to uniform."""
        fresh_cfr.train(1)
        fresh_cfr.infosets["J|"].cumulative_regret = np.array([-5.0, -3.0])
        strategy = fresh_cfr.current_strategy("J|")
        np.testing.assert_allclose(strategy, [0.5, 0.5], atol=1e-9)

    def test_mixed_regret_clips_negatives(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (5): negative regrets are treated as 0, positives are normalised."""
        fresh_cfr.train(1)
        fresh_cfr.infosets["J|"].cumulative_regret = np.array([-5.0, 2.0])
        strategy = fresh_cfr.current_strategy("J|")
        np.testing.assert_allclose(strategy, [0.0, 1.0], atol=1e-9)


# -----------------------------------------------------------------------------
# Cumulative regret table: raw storage (NOT clipped) — Eq. (4) vs Eq. (5)
# -----------------------------------------------------------------------------
class TestCumulativeRegretStorage:
    def test_table_can_hold_negative_values(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (4) accumulates raw instantaneous regrets (no positive-part clip).
        Positive part is applied only in Eq. (5) at strategy-computation time.

        Guards against a common bug where the table is clipped eagerly (which
        turns Vanilla CFR into something closer to CFR+).
        """
        fresh_cfr.train(1)
        fresh_cfr.infosets["J|"].cumulative_regret = np.array([-3.0, 2.0])
        # Round-trip: negative value must survive in the table.
        stored = fresh_cfr.infosets["J|"].cumulative_regret
        assert stored[0] == -3.0

        # But current_strategy applies positive part (Eq. 5).
        strategy = fresh_cfr.current_strategy("J|")
        np.testing.assert_allclose(strategy, [0.0, 1.0], atol=1e-9)


# -----------------------------------------------------------------------------
# average_strategy — Eq. (6)
# -----------------------------------------------------------------------------
class TestAverageStrategy:
    def test_returns_dict_keyed_by_infoset(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (6) output type: dict[str, np.ndarray]."""
        fresh_cfr.train(5)
        avg = fresh_cfr.average_strategy()
        assert isinstance(avg, dict)
        assert all(isinstance(k, str) for k in avg)
        assert all(isinstance(v, np.ndarray) for v in avg.values())

    def test_every_infoset_has_entry(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (6) must produce an entry for every visited infoset."""
        fresh_cfr.train(5)
        avg = fresh_cfr.average_strategy()
        assert set(avg.keys()) == set(fresh_cfr.infosets.keys())

    def test_distributions_sum_to_one(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (6) output is a probability distribution per infoset."""
        fresh_cfr.train(5)
        for key, dist in fresh_cfr.average_strategy().items():
            assert dist.sum() == pytest.approx(1.0, abs=1e-6), (
                f"{key} sum = {dist.sum()}"
            )

    def test_distributions_are_non_negative(self, fresh_cfr: VanillaCFR) -> None:
        """Eq. (6) output has no negative entries."""
        fresh_cfr.train(5)
        for key, dist in fresh_cfr.average_strategy().items():
            assert (dist >= 0).all(), f"{key} has negative: {dist}"

    def test_distribution_shape_matches_n_actions(
        self, fresh_cfr: VanillaCFR
    ) -> None:
        """Shape consistency with Kuhn's 2 actions."""
        fresh_cfr.train(5)
        for dist in fresh_cfr.average_strategy().values():
            assert dist.shape == (N_ACTIONS,)


# -----------------------------------------------------------------------------
# game_value
# -----------------------------------------------------------------------------
class TestGameValue:
    def test_returns_float(self, fresh_cfr: VanillaCFR) -> None:
        """game_value() returns a scalar Python float (not np.float64, etc.)."""
        fresh_cfr.train(10)
        v = fresh_cfr.game_value()
        assert isinstance(v, float)
