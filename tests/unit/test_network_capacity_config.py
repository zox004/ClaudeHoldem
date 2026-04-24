"""Unit tests for configurable ``hidden_dim`` / ``num_hidden_layers``
(Phase 3 Day 3b hypothesis Cap — network capacity scaling).

Ensures:
1. Default values (64, 2) exactly reproduce the original 3-Linear-layer
   architecture used in Day 1/2/2b (backward compat with 51+ existing
   tests).
2. ``hidden_dim=128`` produces the expected parameter count and
   ``layers[0].weight`` shape.
3. ``num_hidden_layers=3`` (Step 2 config: 4×128) produces one extra
   ReLU block and the correct Linear-layer count.
4. ``DeepCFR`` threads the capacity config into both advantage and
   strategy nets (including after the from-scratch reinit inside
   training loops).
"""

from __future__ import annotations

import torch

from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.networks.advantage_net import AdvantageNet
from poker_ai.networks.strategy_net import StrategyNet


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class TestDefaultCapacityBackwardCompat:
    """Defaults must exactly match the original 3-Linear-layer / 64 config."""

    def test_advantage_default_is_3_linear_layers(self) -> None:
        net = AdvantageNet(input_dim=13, n_actions=3)
        linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3, f"default should have 3 Linear layers, got {len(linears)}"

    def test_advantage_default_hidden_dim_is_64(self) -> None:
        net = AdvantageNet(input_dim=13, n_actions=3)
        assert net.layers[0].weight.shape == (64, 13)
        assert net.layers[2].weight.shape == (64, 64)

    def test_strategy_default_is_3_linear_layers(self) -> None:
        net = StrategyNet(input_dim=13, n_actions=3)
        linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3


class TestWidthScaling:
    """Step 1 config: 3-layer × 128 for Leduc Cap hypothesis."""

    def test_hidden_dim_128_weight_shape(self) -> None:
        net = AdvantageNet(input_dim=13, n_actions=3, hidden_dim=128)
        assert net.layers[0].weight.shape == (128, 13)
        assert net.layers[2].weight.shape == (128, 128)

    def test_hidden_dim_128_param_count(self) -> None:
        """Leduc 3×128: 13→128→128→3 = 1792 + 16512 + 387 = 18691."""
        net = AdvantageNet(input_dim=13, n_actions=3, hidden_dim=128)
        expected = (13 * 128 + 128) + (128 * 128 + 128) + (128 * 3 + 3)
        assert _count_params(net) == expected
        # Reference baseline 3×64 Leduc: 5,251.
        baseline = (13 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3)
        assert baseline == 5_251
        # Growth ratio ≈ 3.6× width-only (18691/5251).
        assert _count_params(net) == 18_691


class TestDepthScaling:
    """Step 2 config: 4-layer × 128 = num_hidden_layers=3."""

    def test_num_hidden_layers_3_has_4_linear(self) -> None:
        net = AdvantageNet(
            input_dim=13, n_actions=3, hidden_dim=128, num_hidden_layers=3
        )
        linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 4

    def test_num_hidden_layers_3_param_count(self) -> None:
        """Leduc 4×128: 13→128→128→128→3 = 1792 + 2*16512 + 387 = 35203."""
        net = AdvantageNet(
            input_dim=13, n_actions=3, hidden_dim=128, num_hidden_layers=3
        )
        expected = (
            (13 * 128 + 128)
            + 2 * (128 * 128 + 128)
            + (128 * 3 + 3)
        )
        assert _count_params(net) == expected
        assert expected == 35_203

    def test_forward_shape_preserved_at_higher_capacity(self) -> None:
        net = AdvantageNet(
            input_dim=13, n_actions=3, hidden_dim=128, num_hidden_layers=3
        )
        x = torch.zeros(4, 13)
        out = net(x)
        assert out.shape == (4, 3)


class TestDeepCFRThreadsCapacity:
    """DeepCFR must propagate hidden_dim / num_hidden_layers to both nets."""

    def test_capacity_applied_to_advantage_nets(self) -> None:
        trainer = DeepCFR(
            game=KuhnPoker(),
            n_actions=2,
            encoding_dim=6,
            device="cpu",
            seed=0,
            traversals_per_iter=5,
            buffer_capacity=200,
            batch_size=16,
            advantage_epochs=1,
            strategy_epochs=1,
            hidden_dim=128,
            num_hidden_layers=3,
        )
        for p in (0, 1):
            net = trainer.advantage_nets[p]
            linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
            assert len(linears) == 4
            assert net.layers[0].weight.shape == (128, 6)

    def test_capacity_applied_to_strategy_net(self) -> None:
        trainer = DeepCFR(
            game=KuhnPoker(),
            n_actions=2,
            encoding_dim=6,
            device="cpu",
            seed=0,
            traversals_per_iter=5,
            buffer_capacity=200,
            batch_size=16,
            advantage_epochs=1,
            strategy_epochs=1,
            hidden_dim=128,
            num_hidden_layers=3,
        )
        linears = [
            m for m in trainer.strategy_net.layers if isinstance(m, torch.nn.Linear)
        ]
        assert len(linears) == 4

    def test_capacity_preserved_across_reinit(self) -> None:
        """From-scratch reinit during training must keep capacity config."""
        trainer = DeepCFR(
            game=KuhnPoker(),
            n_actions=2,
            encoding_dim=6,
            device="cpu",
            seed=0,
            traversals_per_iter=5,
            buffer_capacity=200,
            batch_size=16,
            advantage_epochs=1,
            strategy_epochs=1,
            hidden_dim=128,
            num_hidden_layers=3,
        )
        trainer.train(2)  # triggers reinit twice
        for p in (0, 1):
            net = trainer.advantage_nets[p]
            linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
            assert len(linears) == 4
            assert net.layers[0].weight.shape == (128, 6)
        strat_linears = [
            m for m in trainer.strategy_net.layers if isinstance(m, torch.nn.Linear)
        ]
        assert len(strat_linears) == 4
