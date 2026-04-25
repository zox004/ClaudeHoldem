"""Smoke tests for Deep CFR trainer (Phase 3 Day 1 — Deep CFR infrastructure).

Target modules (NOT YET IMPLEMENTED — tests must fail with
``ModuleNotFoundError``):
    src/poker_ai/algorithms/deep_cfr.py
    src/poker_ai/networks/advantage_net.py
    src/poker_ai/networks/strategy_net.py

Target API:
    class DeepCFR:
        def __init__(
            self,
            game: GameProtocol,
            n_actions: int,
            encoding_dim: int,
            device: str = "cpu",
            seed: int = 0,
            traversals_per_iter: int = 1000,
            buffer_capacity: int = 1_000_000,
            batch_size: int = 256,
            advantage_epochs: int = 4,
            strategy_epochs: int = 4,
        ) -> None: ...

        # Per-player advantage networks (Brown 2019 §3 — separate head per p).
        advantage_nets: dict[int, torch.nn.Module]   # {0, 1} -> AdvantageNet
        # Shared strategy network (final output policy).
        strategy_net: torch.nn.Module
        # Per-player reservoir buffers for advantage training.
        advantage_buffers: dict[int, ReservoirBuffer]   # {0, 1}
        # One shared strategy buffer (averaged policy samples).
        strategy_buffer: ReservoirBuffer

        def train(self, iterations: int) -> None: ...

    class AdvantageNet(nn.Module):
        def __init__(self, input_dim: int, n_actions: int) -> None: ...
        # Layout: self.layers[0].weight accessible (first linear layer).
        def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # raw logits

    class StrategyNet(nn.Module):
        # Same signature / shape conventions as AdvantageNet.

Why smoke-only in Phase 3 Day 1:
    Convergence tests (Nash regression on Kuhn/Leduc, exploitability tracking)
    are deferred to Day 5+ once the trainer is stable. Day 1 scope: does it
    instantiate, does 1 iter not crash, do gradients flow? Three questions.

Reference: Brown, Lerer, Gross, Sandholm 2019, "Deep Counterfactual Regret
Minimization", ICML.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker
from poker_ai.networks.advantage_net import AdvantageNet
from poker_ai.networks.strategy_net import StrategyNet

# Smoke-test sizing (tiny — we only care that it *runs*, not that it converges).
SMOKE_TRAVERSALS = 10
SMOKE_BUFFER_CAP = 1_000
SMOKE_BATCH = 32


# -----------------------------------------------------------------------------
# Fixture helpers
# -----------------------------------------------------------------------------
def _make_kuhn_trainer(seed: int = 42) -> DeepCFR:
    return DeepCFR(
        game=KuhnPoker(),
        n_actions=2,
        encoding_dim=6,
        device="cpu",
        seed=seed,
        traversals_per_iter=SMOKE_TRAVERSALS,
        buffer_capacity=SMOKE_BUFFER_CAP,
        batch_size=SMOKE_BATCH,
        advantage_epochs=1,
        strategy_epochs=1,
    )


def _make_leduc_trainer(seed: int = 42) -> DeepCFR:
    return DeepCFR(
        game=LeducPoker(),
        n_actions=3,
        encoding_dim=13,
        device="cpu",
        seed=seed,
        traversals_per_iter=SMOKE_TRAVERSALS,
        buffer_capacity=SMOKE_BUFFER_CAP,
        batch_size=SMOKE_BATCH,
        advantage_epochs=1,
        strategy_epochs=1,
    )


# =============================================================================
# Instantiation: constructor wiring, network & buffer bookkeeping
# =============================================================================
class TestDeepCFRInstantiation:
    """Constructor should wire up per-player networks/buffers without crash."""

    def test_kuhn_instantiates(self) -> None:
        """Construction completes without exception."""
        trainer = _make_kuhn_trainer()
        assert trainer is not None

    def test_leduc_instantiates(self) -> None:
        trainer = _make_leduc_trainer()
        assert trainer is not None

    def test_advantage_nets_are_nn_module(self) -> None:
        """Brown 2019 §3 uses per-player advantage heads — 2 separate modules."""
        trainer = _make_kuhn_trainer()
        assert isinstance(trainer.advantage_nets[0], torch.nn.Module)
        assert isinstance(trainer.advantage_nets[1], torch.nn.Module)

    def test_strategy_net_is_nn_module(self) -> None:
        """A single shared strategy net (final averaged policy)."""
        trainer = _make_kuhn_trainer()
        assert isinstance(trainer.strategy_net, torch.nn.Module)

    def test_reservoir_buffers_exist(self) -> None:
        """advantage: one buffer per player; strategy: a single shared buffer."""
        trainer = _make_kuhn_trainer()
        assert 0 in trainer.advantage_buffers
        assert 1 in trainer.advantage_buffers
        assert trainer.strategy_buffer is not None

    def test_seed_fixes_network_init_weights(self) -> None:
        """Identical seed → identical initial weights (first linear layer)."""
        trainer_a = _make_kuhn_trainer(seed=123)
        trainer_b = _make_kuhn_trainer(seed=123)

        w_a = trainer_a.advantage_nets[0].layers[0].weight
        w_b = trainer_b.advantage_nets[0].layers[0].weight
        assert torch.allclose(w_a, w_b), (
            "same seed must yield same advantage-net init"
        )


# =============================================================================
# One-iteration smoke: trainer.train(1) completes + populates buffers
# =============================================================================
class TestDeepCFROneIterSmoke:
    """Does the trainer run one iter end-to-end without errors?"""

    def test_kuhn_one_iter_runs_without_error(self) -> None:
        trainer = _make_kuhn_trainer()
        # Must not raise.
        trainer.train(1)

    def test_leduc_one_iter_runs_without_error(self) -> None:
        trainer = _make_leduc_trainer()
        trainer.train(1)

    def test_one_iter_populates_advantage_buffer(self) -> None:
        """With traversals_per_iter=10, at least one advantage buffer should
        receive samples. (Which player's buffer receives depends on the
        alternating-update schedule — we assert a weak disjunction.)"""
        trainer = _make_kuhn_trainer()
        trainer.train(1)
        total = (
            trainer.advantage_buffers[0].total_seen
            + trainer.advantage_buffers[1].total_seen
        )
        assert total > 0, (
            f"no advantage samples collected in 1 iter × {SMOKE_TRAVERSALS} traversals"
        )

    def test_one_iter_populates_strategy_buffer(self) -> None:
        """Strategy buffer accumulates across both player updates."""
        trainer = _make_kuhn_trainer()
        trainer.train(1)
        assert trainer.strategy_buffer.total_seen > 0, (
            "strategy buffer must receive samples during 1 iter"
        )

    def test_network_forward_produces_correct_shape_kuhn(self) -> None:
        """Kuhn: input (6,) → output (n_actions=2,). Single-example, un-batched."""
        trainer = _make_kuhn_trainer()
        x = torch.zeros(6)
        out = trainer.advantage_nets[0](x)
        assert out.shape == (2,), f"Kuhn advantage-net output shape {out.shape}"

    def test_network_forward_produces_correct_shape_leduc(self) -> None:
        """Leduc: input (13,) → output (n_actions=3,)."""
        trainer = _make_leduc_trainer()
        x = torch.zeros(13)
        out = trainer.advantage_nets[0](x)
        assert out.shape == (3,), f"Leduc advantage-net output shape {out.shape}"

    def test_strategy_net_output_shape_same_as_advantage(self) -> None:
        """Strategy net mirrors advantage net's action head cardinality."""
        trainer = _make_kuhn_trainer()
        x = torch.zeros(6)
        adv_out = trainer.advantage_nets[0](x)
        strat_out = trainer.strategy_net(x)
        assert strat_out.shape == adv_out.shape


# =============================================================================
# Network forward/backward smoke — gradients flow, loss stays finite
# =============================================================================
class TestDeepCFRNetworkForwardBackward:
    """Basic gradient flow through the standalone network modules.

    We test the network classes directly (not via the trainer's train loop)
    so that a bug in the trainer doesn't mask a network-level issue.
    """

    def test_advantage_net_gradient_flows(self) -> None:
        """loss.backward() populates .grad on the first linear layer."""
        net = AdvantageNet(input_dim=6, n_actions=2)
        x = torch.randn(4, 6, requires_grad=False)
        y = torch.randn(4, 2)

        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        first_weight_grad = net.layers[0].weight.grad
        assert first_weight_grad is not None, "backward did not populate .grad"
        assert not torch.isnan(first_weight_grad).any(), "grad has NaN"

    def test_strategy_net_gradient_flows(self) -> None:
        net = StrategyNet(input_dim=13, n_actions=3)
        x = torch.randn(4, 13)
        y = torch.randn(4, 3)

        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        first_weight_grad = net.layers[0].weight.grad
        assert first_weight_grad is not None
        assert not torch.isnan(first_weight_grad).any()

    def test_training_loss_is_finite(self) -> None:
        """After 1 iter, no parameter contains NaN/Inf (loss stayed finite)."""
        trainer = _make_kuhn_trainer()
        trainer.train(1)

        for p in trainer.advantage_nets[0].parameters():
            assert torch.isfinite(p).all(), "advantage_nets[0] has NaN/Inf param"
        for p in trainer.advantage_nets[1].parameters():
            assert torch.isfinite(p).all(), "advantage_nets[1] has NaN/Inf param"
        for p in trainer.strategy_net.parameters():
            assert torch.isfinite(p).all(), "strategy_net has NaN/Inf param"


# =============================================================================
# H Tier 1 logging — train_history captures per-call training stats
# =============================================================================
class TestDeepCFRTrainHistory:
    """Phase 3 Day 5 H Tier 1: train_history records loss/grad/target stats."""

    def test_train_history_starts_empty(self) -> None:
        trainer = _make_kuhn_trainer()
        assert trainer.train_history == []

    def test_train_history_records_three_events_per_iter(self) -> None:
        """One iter = 2 advantage_p (p∈{0,1}) + 1 strategy = 3 events."""
        trainer = _make_kuhn_trainer()
        trainer.train(2)
        assert len(trainer.train_history) == 6
        nets_seen = [(ev["iter"], ev["net"]) for ev in trainer.train_history]
        assert nets_seen == [
            (1, "advantage"), (1, "advantage"), (1, "strategy"),
            (2, "advantage"), (2, "advantage"), (2, "strategy"),
        ]

    def test_advantage_event_has_required_keys(self) -> None:
        trainer = _make_kuhn_trainer()
        trainer.train(1)
        adv = next(e for e in trainer.train_history if e["net"] == "advantage")
        for key in (
            "iter", "net", "player", "n_samples", "loss_per_epoch",
            "loss_initial", "loss_final",
            "target_abs_mean", "target_abs_std", "grad_norm_max",
        ):
            assert key in adv, f"advantage event missing {key!r}"

    def test_strategy_event_has_no_player_key(self) -> None:
        """Strategy net is shared (Brown 2019); event has no player field."""
        trainer = _make_kuhn_trainer()
        trainer.train(1)
        strat = next(e for e in trainer.train_history if e["net"] == "strategy")
        assert "player" not in strat

    def test_loss_per_epoch_length_matches_advantage_epochs(self) -> None:
        trainer = DeepCFR(
            game=KuhnPoker(), n_actions=2, encoding_dim=6,
            traversals_per_iter=10, batch_size=8,
            advantage_epochs=3, strategy_epochs=2,
            seed=42,
        )
        trainer.train(1)
        adv = next(e for e in trainer.train_history if e["net"] == "advantage")
        strat = next(e for e in trainer.train_history if e["net"] == "strategy")
        assert len(adv["loss_per_epoch"]) == 3
        assert len(strat["loss_per_epoch"]) == 2

    def test_logged_stats_are_finite_and_nonnegative(self) -> None:
        trainer = _make_kuhn_trainer()
        trainer.train(2)
        for ev in trainer.train_history:
            assert math.isfinite(float(ev["loss_initial"]))
            assert math.isfinite(float(ev["loss_final"]))
            assert math.isfinite(float(ev["target_abs_mean"]))
            assert float(ev["target_abs_mean"]) >= 0.0
            assert math.isfinite(float(ev["grad_norm_max"]))
            assert float(ev["grad_norm_max"]) >= 0.0
            assert float(ev["n_samples"]) > 0

    def test_strategy_target_abs_mean_close_to_simplex_average(self) -> None:
        """For 2-action Kuhn, strategy targets are simplex (probs sum to 1
        with both legal). |target| mean = exact 0.5 by construction —
        sanity check that target_abs_mean wires the right tensor."""
        trainer = _make_kuhn_trainer()
        trainer.train(1)
        strat = next(e for e in trainer.train_history if e["net"] == "strategy")
        # Both Kuhn actions always legal → each prob is 0.5 on uniform
        # init, so |target| mean must be exactly 0.5.
        assert abs(float(strat["target_abs_mean"]) - 0.5) < 1e-6
