"""Advantage network for Deep CFR (Brown 2019 §3).

Per-player network ``V_p(I) -> R^{|A|}`` mapping an encoded infoset to
raw advantage (regret) logits over the action space. The current-strategy
at inference time is derived by positive-part clipping + normalisation
over legal actions (the regret-matching step).

Architecture (Brown 2019 Leduc config): 3 fully-connected layers, hidden
width 64, ReLU activations, **raw-logit output** (no softmax — regret
values are unbounded reals, positive-part clipping belongs to the CFR
layer not the network).

Design note: ``self.layers`` is an ``nn.Sequential`` so that
``net.layers[0].weight`` resolves to the first linear layer's parameters.
Seed-determinism tests rely on this access pattern.
"""

from __future__ import annotations

import torch
from torch import nn


class AdvantageNet(nn.Module):
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
        """``x`` shape ``(*, input_dim)`` → out ``(*, n_actions)`` raw logits."""
        out: torch.Tensor = self.layers(x)
        return out
