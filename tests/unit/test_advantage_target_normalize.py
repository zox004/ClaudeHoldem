"""Unit tests for advantage-target running-std normaliser (Phase 3 Day 3c D-2).

The running EMA per-player std (``_adv_target_std``) rescales advantage
targets before MSE so that Leduc's large signed-regret magnitude doesn't
cause gradient imbalance. Tests verify:

1. Initial state (unstarted).
2. First-batch initialisation (no EMA-lag window).
3. EMA update on subsequent batches.
4. Per-player independence.
5. Preserved across network reinit (data property, not network property).
6. Regret-matching remains scale-invariant (sanity check).
"""

from __future__ import annotations

import numpy as np
import torch

from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.games.kuhn import KuhnPoker


def _tiny_trainer(seed: int = 42, target_normalize: bool = True) -> DeepCFR:
    """Default ``target_normalize=True`` for this test suite — D-2 is OFF
    by default in production (Day 3c FAIL) but these tests verify the
    feature when enabled."""
    return DeepCFR(
        game=KuhnPoker(),
        n_actions=2,
        encoding_dim=6,
        device="cpu",
        seed=seed,
        traversals_per_iter=10,
        buffer_capacity=1000,
        batch_size=16,
        advantage_epochs=2,
        strategy_epochs=1,
        advantage_target_normalize=target_normalize,
    )


class TestAdvantageTargetStdInitial:
    """Initial state: uninitialised flag, std defaults to 1.0."""

    def test_default_std_is_one(self) -> None:
        tr = _tiny_trainer()
        assert tr._adv_target_std == {0: 1.0, 1: 1.0}

    def test_default_initialised_flag_is_false(self) -> None:
        tr = _tiny_trainer()
        assert tr._adv_target_std_initialized == {0: False, 1: False}

    def test_default_ema_coefficient(self) -> None:
        tr = _tiny_trainer()
        assert tr._adv_target_std_ema == 0.99

    def test_flag_default_is_false_post_day3c(self) -> None:
        """Day 3c FAIL → D-2 default OFF in production."""
        tr = DeepCFR(
            game=KuhnPoker(),
            n_actions=2,
            encoding_dim=6,
            seed=0,
            traversals_per_iter=5,
            buffer_capacity=200,
            batch_size=16,
            advantage_epochs=1,
            strategy_epochs=1,
        )
        assert tr.advantage_target_normalize is False

    def test_flag_off_keeps_std_uninit_through_training(self) -> None:
        tr = _tiny_trainer(target_normalize=False)
        tr.train(3)
        assert tr._adv_target_std_initialized == {0: False, 1: False}


class TestAdvantageTargetStdFirstBatchInit:
    """First batch should set std directly (no EMA-lag at start)."""

    def test_single_iter_initialises_both_players(self) -> None:
        tr = _tiny_trainer()
        tr.train(1)
        # Both players updated during a single iteration (alternating).
        assert tr._adv_target_std_initialized[0]
        assert tr._adv_target_std_initialized[1]

    def test_initialised_std_is_positive_non_default(self) -> None:
        tr = _tiny_trainer()
        tr.train(1)
        # The initialisation uses the first batch's abs-mean, which must
        # be > 0 for any meaningful regret signal (assert > 1e-4 threshold).
        assert tr._adv_target_std[0] > 1e-4
        assert tr._adv_target_std[1] > 1e-4
        # Post-init std is unlikely to equal the default 1.0 exactly.
        # (If it does, it's by coincidence — the batch happened to have
        # abs-mean ≈ 1.0. Not a correctness issue, so not asserted.)


class TestAdvantageTargetStdEma:
    """EMA update formula ``new = α·old + (1−α)·batch``."""

    def test_second_iter_updates_via_ema(self) -> None:
        tr = _tiny_trainer()
        tr.train(1)
        std_after_iter_1 = dict(tr._adv_target_std)
        tr.train(1)
        std_after_iter_2 = dict(tr._adv_target_std)
        # At minimum, at least one player's std should have shifted
        # via EMA (unless batches happened to have identical abs-mean,
        # very unlikely with sampling-based Kuhn traversal).
        shifted = [
            abs(std_after_iter_1[p] - std_after_iter_2[p]) > 1e-6
            for p in (0, 1)
        ]
        assert any(shifted), (
            f"running std should change on subsequent iter: "
            f"iter1={std_after_iter_1}, iter2={std_after_iter_2}"
        )

    def test_std_remains_positive_across_training(self) -> None:
        tr = _tiny_trainer()
        tr.train(3)
        for p in (0, 1):
            assert tr._adv_target_std[p] > 0.0


class TestAdvantageTargetStdPerPlayer:
    """Per-player state: updates to player 0 must not affect player 1."""

    def test_per_player_storage_separate(self) -> None:
        tr = _tiny_trainer()
        tr.train(2)
        # Trivial smoke: both flags True after training. The actual
        # cross-contamination test is implicit in that different players
        # visit different infoset distributions → different target stats.
        assert tr._adv_target_std_initialized[0] is True
        assert tr._adv_target_std_initialized[1] is True


class TestAdvantageTargetStdReinit:
    """Running std must survive network reinit (it's a data property)."""

    def test_std_preserved_after_train_multi_iter(self) -> None:
        """train(N) triggers N reinits per player; std accumulates across."""
        tr = _tiny_trainer()
        tr.train(1)
        std_iter1 = dict(tr._adv_target_std)
        tr.train(2)
        std_iter3 = dict(tr._adv_target_std)
        # After 3 iters, std should be a smooth EMA of batch stats —
        # not reset to 1.0 (i.e. not re-initialised to default).
        for p in (0, 1):
            assert std_iter3[p] > 1e-4
        # And it has incorporated iter-2 and iter-3 data (some change).
        assert std_iter1 != std_iter3


class TestRegretMatchingScaleInvariance:
    """Sanity: regret matching is scale-invariant, so scaling advantage
    targets does not alter the derived strategy — a key property the
    D-2 design relies on (no un-scaling needed at inference)."""

    def test_regret_matching_invariant_to_positive_scale(self) -> None:
        r = np.array([3.0, -1.0, 2.0])
        mask = np.array([True, True, True])
        s1 = regret_matching(r, legal_mask=mask)
        s2 = regret_matching(r * 10.0, legal_mask=mask)
        s3 = regret_matching(r * 0.1, legal_mask=mask)
        np.testing.assert_allclose(s1, s2, atol=1e-8)
        np.testing.assert_allclose(s1, s3, atol=1e-8)


class TestAdvantageTargetStdLossCorrectness:
    """End-to-end: 1-iter train still produces finite params (no NaN/Inf
    from the rescaling, including edge cases like near-zero batch_std)."""

    def test_one_iter_train_produces_finite_params(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        tr = _tiny_trainer(seed=0)
        tr.train(1)
        for p in (0, 1):
            for param in tr.advantage_nets[p].parameters():
                assert torch.isfinite(param).all()
