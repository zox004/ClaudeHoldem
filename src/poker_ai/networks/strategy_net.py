"""Strategy network for Deep CFR (Brown 2019 §3).

Shared network ``Π(I) -> R^{|A|}`` trained on time-averaged strategy samples
drawn from the non-updating player's decision nodes during traversal. At
inference, softmax over legal actions gives σ̄, the time-averaged policy
whose exploitability is the Phase 3 Exit #4 metric.

Architecture is identical to :class:`AdvantageNet` by Phase 3 design lock
decision #6 — simpler deployment + symmetric hyperparameter tuning.
Configurable ``hidden_dim`` / ``num_hidden_layers`` (defaults match Kuhn
3-layer × 64 spec).
"""

from __future__ import annotations

import torch
from torch import nn


class StrategyNet(nn.Module):
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
        out: torch.Tensor = self.layers(x)
        return out
