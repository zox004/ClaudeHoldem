"""Reservoir sampling buffer for Deep CFR (Brown 2019 §3).

Implements Vitter 1985 Algorithm R with pre-allocated torch-tensor backing:
- While ``total_seen < capacity``, every arrival is appended at the next slot.
- Once saturated, arrival ``k`` (0-indexed, ``k >= capacity``) is placed at
  slot ``j = rng.integers(0, k+1)`` iff ``j < capacity``; otherwise discarded.
  This keeps each historic arrival retained with probability
  ``capacity / total_seen`` at any future query (unbiasedness).

Reference: Vitter, J. (1985). "Random Sampling with a Reservoir", ACM TOMS
11(1):37-57.

Why torch tensors (not Python lists)
------------------------------------
Deep CFR training pulls batched features/targets/weights straight into an
``nn.Module`` on the configured device. A pre-allocated tensor gives:
1. O(1) slot writes (no list resize).
2. Zero-copy ``sample_all()`` for training (slice, not Python iteration).
3. Stable device placement — no mid-stream host↔device transfers.

Buffer sizing (Phase 3 Day 1 target): capacity=1_000_000 in production,
scaled down for tests. For Kuhn's 12 infosets and small traversal counts
the buffer never saturates; saturation path is still covered by unit tests
with capacity=100.
"""

from __future__ import annotations

import numpy as np
import torch


class ReservoirBuffer:
    """Vitter 1985 reservoir sampler backed by pre-allocated torch tensors.

    Parameters:
        capacity: maximum number of samples retained.
        feature_dim: per-sample feature vector length.
        device: torch device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        seed: RNG seed for reproducibility. A dedicated
            :class:`numpy.random.Generator` is owned by the buffer so that
            sampling does not perturb any global torch/numpy RNG state.
    """

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        device: str = "cpu",
        seed: int = 0,
        target_dim: int = 1,
    ) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.device = torch.device(device)
        self._features: torch.Tensor = torch.zeros(
            (capacity, feature_dim), dtype=torch.float32, device=self.device
        )
        # Scalar targets (``target_dim=1``) are stored as a 1D tensor for
        # backward-compat with the Vitter/bookkeeping unit tests. Vector
        # targets (``target_dim>1``), used by Deep CFR to store per-action
        # regret or strategy vectors, are stored as a 2D tensor.
        if target_dim == 1:
            self._targets: torch.Tensor = torch.zeros(
                (capacity,), dtype=torch.float32, device=self.device
            )
        else:
            self._targets = torch.zeros(
                (capacity, target_dim), dtype=torch.float32, device=self.device
            )
        self._weights: torch.Tensor = torch.zeros(
            (capacity,), dtype=torch.float32, device=self.device
        )
        self.total_seen: int = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return min(self.total_seen, self.capacity)

    def add(
        self,
        features: torch.Tensor,
        target: float | torch.Tensor,
        iter_weight: float,
    ) -> None:
        """Vitter Algorithm R insertion.

        Under capacity: write to slot ``total_seen``. Over capacity: roll
        ``j ∈ [0, total_seen]``; if ``j < capacity``, evict slot ``j``.

        ``target`` is a ``float`` when ``target_dim == 1`` and a 1D
        ``torch.Tensor`` of length ``target_dim`` otherwise.
        """
        k = self.total_seen
        if k < self.capacity:
            slot = k
        else:
            slot = int(self._rng.integers(0, k + 1))
            if slot >= self.capacity:
                self.total_seen += 1
                return

        # Ensure incoming features live on the buffer's device (caller may
        # pass a CPU tensor when buffer is on MPS/CUDA).
        self._features[slot] = features.to(dtype=torch.float32, device=self.device)
        if self.target_dim == 1:
            self._targets[slot] = float(target) if not isinstance(target, torch.Tensor) else target.item()
        else:
            assert isinstance(target, torch.Tensor), (
                "vector-target buffer requires a torch.Tensor target"
            )
            self._targets[slot] = target.to(dtype=torch.float32, device=self.device)
        self._weights[slot] = float(iter_weight)
        self.total_seen += 1

    def sample_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all retained samples (features, targets, weights).

        For ``total_seen < capacity`` this returns the filled prefix; otherwise
        the entire pre-allocated tensor. Slices are views into the underlying
        storage (no copy).
        """
        n = len(self)
        return (
            self._features[:n],
            self._targets[:n],
            self._weights[:n],
        )
