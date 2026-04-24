"""Advantage network for Deep CFR (Brown 2019 §3).

Per-player network ``V_p(I) -> R^{|A|}`` mapping an encoded infoset to
raw advantage (regret) logits over the action space. The current-strategy
at inference time is derived by positive-part clipping + normalisation
over legal actions (the regret-matching step).

Architecture: configurable MLP with ``num_hidden_layers`` hidden layers of
``hidden_dim`` units, ReLU, **raw-logit output** (no softmax — regret
values are unbounded reals, positive-part clipping belongs to the CFR
layer not the network).

Default ``hidden_dim=64, num_hidden_layers=2`` reproduces the Kuhn /
Phase 3 Day 2 configuration (3 Linear layers total: 2 hidden + 1 output).
Phase 3 Day 3b Leduc uses ``hidden_dim=128`` (width doubling) after Day 3
FAIL implicated under-parameterisation (hypothesis Cap).

Design note: ``self.layers`` is an ``nn.Sequential`` so that
``net.layers[0].weight`` resolves to the first Linear layer's parameters.
Seed-determinism tests rely on this access pattern.
"""

from __future__ import annotations

import torch
from torch import nn


class AdvantageNet(nn.Module):
    """Configurable MLP with ReLU, raw-logit output (Brown 2019 spec)."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        assert num_hidden_layers >= 1, "need at least one hidden layer"
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, n_actions))
        self.layers: nn.Sequential = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(*, input_dim)`` → out ``(*, n_actions)`` raw logits."""
        out: torch.Tensor = self.layers(x)
        return out
