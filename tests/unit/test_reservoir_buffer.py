"""Unit tests for ReservoirBuffer (Phase 3 Day 1 — Deep CFR infrastructure).

Target module (NOT YET IMPLEMENTED — tests must fail with
``ModuleNotFoundError: No module named 'poker_ai.algorithms.reservoir'``):
    src/poker_ai/algorithms/reservoir.py

Target API:
    class ReservoirBuffer:
        def __init__(
            self,
            capacity: int,
            feature_dim: int,
            device: str = "cpu",
            seed: int = 0,
        ) -> None: ...

        # Pre-allocated torch storage (not a Python list — memory stability
        # and GPU transfer efficiency during training).
        _features: torch.Tensor   # shape (capacity, feature_dim), float32
        _targets: torch.Tensor    # shape (capacity,), float32
        _weights: torch.Tensor    # shape (capacity,), float32
        total_seen: int

        def add(
            self,
            features: torch.Tensor,   # shape (feature_dim,)
            target: float,
            iter_weight: float,
        ) -> None: ...

        def __len__(self) -> int: ...  # min(total_seen, capacity)

        def sample_all(
            self,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

Reference: Vitter 1985, "Random Sampling with a Reservoir", ACM TOMS 11(1):37-57.

Why a reservoir buffer for Deep CFR
-----------------------------------
Brown 2019 §3 stores (infoset_features, regret_vector, iter_weight) tuples
in an infinite-in-expectation stream that cannot fit in memory for NLHE.
Reservoir sampling keeps a uniformly-random fixed-size sample of the stream
so that the advantage net is trained on an unbiased sample over iterations.
Without it, on-policy bias dominates and the network overfits recent data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poker_ai.algorithms.reservoir import ReservoirBuffer


# =============================================================================
# Basic bookkeeping: storage backing, dtype, device, len semantics
# =============================================================================
class TestReservoirBasic:
    """Core data-structure invariants (shape, dtype, device)."""

    def test_empty_buffer_len_is_0(self) -> None:
        """A freshly constructed buffer has no samples."""
        buf = ReservoirBuffer(capacity=10, feature_dim=6, device="cpu", seed=0)
        assert len(buf) == 0

    def test_add_increments_len_until_capacity(self) -> None:
        """len() grows linearly while under capacity."""
        buf = ReservoirBuffer(capacity=10, feature_dim=6, device="cpu", seed=0)
        # Add 5 entries: len == 5.
        for i in range(5):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)
        assert len(buf) == 5

        # Add 10 more (total 15, cap 10): len == 10 (saturated).
        for i in range(5, 15):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)
        assert len(buf) == 10

    def test_total_seen_tracks_all_adds_ignoring_capacity(self) -> None:
        """total_seen is the running count of add() calls, unbounded by cap."""
        buf = ReservoirBuffer(capacity=10, feature_dim=6, device="cpu", seed=0)
        for i in range(100):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)
        assert buf.total_seen == 100
        assert len(buf) == 10

    def test_preallocated_torch_tensor_storage(self) -> None:
        """Backing storage is a single pre-allocated ``torch.Tensor`` (not a
        Python list). This guarantees O(1) writes and stable device placement.
        """
        buf = ReservoirBuffer(capacity=32, feature_dim=13, device="cpu", seed=0)
        assert isinstance(buf._features, torch.Tensor)
        assert buf._features.shape == (32, 13)

    def test_device_honored(self) -> None:
        """Explicit device choice is respected for all tensor backings."""
        buf = ReservoirBuffer(capacity=8, feature_dim=6, device="cpu", seed=0)
        assert buf._features.device.type == "cpu"
        assert buf._targets.device.type == "cpu"
        assert buf._weights.device.type == "cpu"

    def test_dtype_features_float32(self) -> None:
        """Feature tensor uses float32 (network-friendly, memory-efficient)."""
        buf = ReservoirBuffer(capacity=8, feature_dim=6, device="cpu", seed=0)
        assert buf._features.dtype == torch.float32

    def test_dtype_targets_float32(self) -> None:
        buf = ReservoirBuffer(capacity=8, feature_dim=6, device="cpu", seed=0)
        assert buf._targets.dtype == torch.float32

    def test_dtype_weights_float32(self) -> None:
        buf = ReservoirBuffer(capacity=8, feature_dim=6, device="cpu", seed=0)
        assert buf._weights.dtype == torch.float32


# =============================================================================
# Vitter 1985 Algorithm R — unbiasedness property tests
# =============================================================================
class TestReservoirProperty:
    """Statistical guarantees of Vitter 1985 reservoir sampling."""

    def test_under_capacity_all_samples_retained(self) -> None:
        """If we add fewer items than capacity, every item is retained."""
        buf = ReservoirBuffer(capacity=100, feature_dim=6, device="cpu", seed=0)
        for i in range(50):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)

        _, targets, _ = buf.sample_all()
        ids = set(int(t.item()) for t in targets)
        assert ids == set(range(50)), (
            f"under-capacity: all added items must be retained, got {len(ids)}"
        )

    def test_at_capacity_exact_saturation(self) -> None:
        """Adding exactly ``capacity`` items retains all of them in order."""
        buf = ReservoirBuffer(capacity=10, feature_dim=6, device="cpu", seed=0)
        for i in range(10):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)
        assert len(buf) == 10

        _, targets, _ = buf.sample_all()
        ids = set(int(t.item()) for t in targets)
        assert ids == set(range(10))

    def test_over_capacity_no_duplicates_and_in_range(self) -> None:
        """Retained targets are a subset of the 0..total_seen range, no dups."""
        buf = ReservoirBuffer(capacity=100, feature_dim=6, device="cpu", seed=42)
        for i in range(10_000):
            buf.add(torch.zeros(6), target=float(i), iter_weight=1.0)

        _, targets, _ = buf.sample_all()
        ids = [int(t.item()) for t in targets]
        assert len(ids) == 100, f"buffer must be saturated, got {len(ids)}"
        assert len(set(ids)) == 100, f"duplicates detected: {len(set(ids))} unique"
        assert all(0 <= i < 10_000 for i in ids), (
            f"out-of-range id(s): {[i for i in ids if not 0 <= i < 10_000]}"
        )

    def test_retention_rate_matches_theory(self) -> None:
        """Vitter 1985 unbiasedness: P(item i retained) = capacity/total_seen.

        Monte-Carlo check: average retention rate over 100 seeds should be
        close to 100/10_000 = 0.01 within tolerance.
        """
        capacity = 100
        total = 10_000
        theoretical = capacity / total  # 0.01

        retention_counts = np.zeros(total, dtype=np.int64)
        n_seeds = 100
        for seed in range(n_seeds):
            buf = ReservoirBuffer(
                capacity=capacity, feature_dim=1, device="cpu", seed=seed
            )
            for i in range(total):
                buf.add(torch.zeros(1), target=float(i), iter_weight=1.0)
            _, targets, _ = buf.sample_all()
            for t in targets:
                retention_counts[int(t.item())] += 1

        # Average retention rate across all items
        avg_rate = retention_counts.mean() / n_seeds
        # ±30% tolerance (capacity=100 × n_seeds=100 — moderate MC variance).
        assert abs(avg_rate - theoretical) < theoretical * 0.30, (
            f"avg retention {avg_rate:.5f} far from theoretical {theoretical:.5f}"
        )

    def test_bucketed_retention_uniform(self) -> None:
        """Retention rate should be uniform across buckets (first / middle /
        last third of the stream) — no recency bias or staleness bias."""
        capacity = 100
        total = 10_000

        retention_counts = np.zeros(total, dtype=np.int64)
        n_seeds = 100
        for seed in range(n_seeds):
            buf = ReservoirBuffer(
                capacity=capacity, feature_dim=1, device="cpu", seed=seed
            )
            for i in range(total):
                buf.add(torch.zeros(1), target=float(i), iter_weight=1.0)
            _, targets, _ = buf.sample_all()
            for t in targets:
                retention_counts[int(t.item())] += 1

        first_rate = retention_counts[:100].sum() / (100 * n_seeds)
        mid_rate = retention_counts[5000:5100].sum() / (100 * n_seeds)
        last_rate = retention_counts[-100:].sum() / (100 * n_seeds)

        # Each bucket should sit near 0.01 with bucket-level tolerance.
        for name, rate in [("first", first_rate), ("mid", mid_rate), ("last", last_rate)]:
            assert abs(rate - 0.01) < 0.01, (
                f"{name}-bucket retention {rate:.4f} outside 0.01 ± 0.01"
            )


# =============================================================================
# Determinism — same seed, same stream, same output
# =============================================================================
class TestReservoirDeterminism:
    """Reproducibility guarantees from the injected seed."""

    def test_same_seed_same_buffer(self) -> None:
        """Two buffers with identical seed + add sequence → identical outputs."""
        buf_a = ReservoirBuffer(capacity=50, feature_dim=6, device="cpu", seed=2024)
        buf_b = ReservoirBuffer(capacity=50, feature_dim=6, device="cpu", seed=2024)

        for i in range(1000):
            feat = torch.full((6,), float(i))
            buf_a.add(feat, target=float(i), iter_weight=float(i) + 1.0)
            buf_b.add(feat, target=float(i), iter_weight=float(i) + 1.0)

        f_a, t_a, w_a = buf_a.sample_all()
        f_b, t_b, w_b = buf_b.sample_all()

        assert torch.allclose(f_a, f_b), "feature tensors diverged under same seed"
        assert torch.allclose(t_a, t_b), "target tensors diverged under same seed"
        assert torch.allclose(w_a, w_b), "weight tensors diverged under same seed"
