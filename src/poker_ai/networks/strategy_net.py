"""Strategy network for Deep CFR (Brown 2019 §3).

Shared network ``Π(I) -> R^{|A|}`` trained on time-averaged strategy samples
drawn from the non-updating player's decision nodes during traversal. At
inference, softmax over legal actions gives σ̄, the time-averaged policy
whose exploitability is the Phase 3 Exit #4 metric.

Architecture is identical to :class:`AdvantageNet` by Phase 3 design lock
decision #6 — simpler deployment + symmetric hyperparameter tuning.
"""

from __future__ import annotations

import torch
from torch import nn


class StrategyNet(nn.Module):
    """3-layer MLP with ReLU, raw-logit output (Brown 2019 spec)."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out
