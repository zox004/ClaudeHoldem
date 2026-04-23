"""Unit tests for the strategy-net cross-entropy training path (Phase 3 Day 2 fix).

Phase 3 Day 2 fix (2026-04-24): Strategy net was originally trained with
MSE, which fails to learn pure-strategy (Nash-extreme) targets. Cross-entropy
with legal-action masking (``-inf`` on illegal logits) is the replacement.

These tests verify:
1. Inference post-softmax produces a valid simplex on legal actions.
2. Illegal actions get exactly zero probability.
3. A pure target (e.g. [1.0, 0.0]) is learnable — output reaches > 0.95.
4. A mixed target (e.g. [0.7, 0.3]) is learnable — output approximates.

Tests 3-4 run a tiny stand-alone optimisation loop on a single-infoset
synthetic batch (no CFR traversal). This isolates the loss pathway from
the rest of Deep CFR and lets us verify the fix independently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.eval.deep_cfr_correlation import _strategy_net_output_masked
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.networks.strategy_net import StrategyNet


def _tiny_trainer(seed: int = 42) -> DeepCFR:
    """A DeepCFR instance small enough to train in a unit test."""
    return DeepCFR(
        game=KuhnPoker(),
        n_actions=2,
        encoding_dim=6,
        device="cpu",
        seed=seed,
        traversals_per_iter=10,
        buffer_capacity=1000,
        batch_size=32,
        advantage_epochs=1,
        strategy_epochs=1,
    )


class TestStrategyNetInferenceSimplex:
    """Post-masking softmax must produce a valid probability simplex."""

    def test_output_is_valid_simplex_kuhn(self) -> None:
        trainer = _tiny_trainer()
        trainer.train(2)
        encoding = np.zeros(6, dtype=np.float32)
        encoding[0] = 1.0  # Jack, root history
        mask = np.array([True, True])  # Kuhn: all legal

        probs = _strategy_net_output_masked(trainer, encoding, mask)
        assert probs.shape == (2,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5), (
            f"probs must sum to 1, got {probs.sum()}"
        )
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_illegal_actions_have_zero_probability(self) -> None:
        """Masking illegal action via -inf → exact 0 posterior."""
        trainer = _tiny_trainer()
        trainer.train(2)
        encoding = np.zeros(6, dtype=np.float32)
        encoding[0] = 1.0
        # Simulate a 2-action setting where one is illegal.
        mask = np.array([True, False])
        probs = _strategy_net_output_masked(trainer, encoding, mask)
        assert probs[1] == 0.0, (
            f"illegal slot must be exactly 0, got {probs[1]}"
        )
        assert np.isclose(probs[0], 1.0)


class TestStrategyNetLearnsPureTarget:
    """Standalone training loop: can the strategy net learn an extreme target?

    Uses the same loss path as ``DeepCFR._train_strategy_net`` (masked
    cross-entropy with per-sample iter weights), but on a single synthetic
    infoset so the test is deterministic and fast.
    """

    @pytest.mark.parametrize("target", [(1.0, 0.0), (0.0, 1.0)])
    def test_single_pure_target_reaches_above_095(
        self, target: tuple[float, float]
    ) -> None:
        torch.manual_seed(42)
        net = StrategyNet(input_dim=6, n_actions=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        x = torch.zeros((32, 6), dtype=torch.float32)
        x[:, 0] = 1.0  # Jack one-hot
        y = torch.tensor([list(target)] * 32, dtype=torch.float32)
        mask = torch.ones((32, 2), dtype=torch.bool)
        weights = torch.ones((32,), dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"))

        for _ in range(200):
            logits = net(x)
            masked = torch.where(mask, logits, neg_inf)
            log_probs = torch.nn.functional.log_softmax(masked, dim=-1)
            terms = torch.where(y > 0.0, y * log_probs, torch.zeros_like(y))
            per_sample = -terms.sum(dim=-1)
            loss = (per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

        with torch.no_grad():
            logits = net(x[:1])
            masked = torch.where(mask[:1], logits, neg_inf)
            probs = torch.nn.functional.softmax(masked, dim=-1)[0].numpy()

        # Primary assertion: output[target_idx] > 0.95 — cross-entropy
        # successfully pushes extreme probability onto the true action.
        target_idx = int(np.argmax(np.array(target)))
        assert probs[target_idx] > 0.95, (
            f"pure target {target} not learned: output={probs} "
            f"(target_idx={target_idx})"
        )

    def test_mixed_target_learnable_with_tolerance(self) -> None:
        """A mixed target like (0.7, 0.3) should be approximated to within ±0.05."""
        torch.manual_seed(42)
        target = (0.7, 0.3)
        net = StrategyNet(input_dim=6, n_actions=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        x = torch.zeros((32, 6), dtype=torch.float32)
        x[:, 1] = 1.0  # Queen one-hot
        y = torch.tensor([list(target)] * 32, dtype=torch.float32)
        mask = torch.ones((32, 2), dtype=torch.bool)
        weights = torch.ones((32,), dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"))

        for _ in range(300):
            logits = net(x)
            masked = torch.where(mask, logits, neg_inf)
            log_probs = torch.nn.functional.log_softmax(masked, dim=-1)
            terms = torch.where(y > 0.0, y * log_probs, torch.zeros_like(y))
            per_sample = -terms.sum(dim=-1)
            loss = (per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

        with torch.no_grad():
            logits = net(x[:1])
            masked = torch.where(mask[:1], logits, neg_inf)
            probs = torch.nn.functional.softmax(masked, dim=-1)[0].numpy()

        # Mixed target: CE's optimum is exactly target; tolerance for
        # finite-step optimisation.
        assert abs(probs[0] - target[0]) < 0.05, (
            f"mixed target {target} not approximated: output={probs}"
        )


class TestStrategyBufferStoresMasks:
    """Sanity: strategy buffer now carries legal masks end-to-end."""

    def test_strategy_buffer_has_mask_dim_equal_to_n_actions(self) -> None:
        trainer = _tiny_trainer()
        assert trainer.strategy_buffer.mask_dim == trainer.n_actions

    def test_strategy_buffer_populated_after_train(self) -> None:
        trainer = _tiny_trainer()
        trainer.train(1)
        assert trainer.strategy_buffer.total_seen > 0
        # Masks tensor should be populated for retained samples.
        _, _, _, masks = trainer.strategy_buffer.sample_all_with_masks()
        assert masks.shape == (len(trainer.strategy_buffer), trainer.n_actions)
        # For Kuhn, all mask entries should be True.
        assert masks.all(), "Kuhn has all-legal actions; mask must be all-True"

    def test_advantage_buffer_has_no_mask(self) -> None:
        """Advantage buffer is mask-free (design scope: Day 2 fix is strategy-only)."""
        trainer = _tiny_trainer()
        assert trainer.advantage_buffers[0].mask_dim == 0
        assert trainer.advantage_buffers[1].mask_dim == 0
