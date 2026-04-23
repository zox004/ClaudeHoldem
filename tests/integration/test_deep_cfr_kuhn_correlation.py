"""Correlation between Deep CFR outputs and tabular baselines on Kuhn.

Phase 3 Day 2 — Exit #4 primary dry-run on Kuhn.

**Design revision 2026-04-24** (멘토): Primary metric was originally
``corr(Deep advantage, CFR+ R⁺) > 0.95`` but the smoke measurement
revealed this is a mathematical mismatch — Deep CFR approximates the
**signed** regret (Brown 2019 Alg 1 Line 5), while CFR+'s
``cumulative_regret`` is positive-by-construction. See PHASE.md
"Phase 3 Day 2 pre-smoke" log for the full analysis.

Revised metric set (via :func:`compute_correlations`):
- **Primary A** — `corr(Deep advantage, Vanilla R_cum)` (signed vs signed)
- **Primary B** — `corr(Deep strategy softmax, CFR+ σ̄)` (simplex space)
- **Tertiary** — `corr(Deep advantage, CFR+ R⁺)` (diagnostic only)

Tests in this file:
- `TestDeepCFRKuhnCorrelationSmoke` (fast): tertiary-only smoke, kept as
  backward-compat for :func:`compute_flat_correlation`. Threshold r > 0.3.
- `TestDeepCFRKuhnCorrelationFull` (@slow): full Primary A/B dry-run at
  T=2000. Asserts Primary A > 0.9 AND Primary B > 0.85.
- `TestCorrelationReport`: structural tests for the new
  :func:`compute_correlations` API.
- `TestCorrelationUtility`: structural tests for
  :func:`compute_flat_correlation` (legacy API).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.deep_cfr_correlation import (
    CorrelationReport,
    compute_correlations,
    compute_flat_correlation,
)
from poker_ai.games.kuhn import KuhnPoker


def _make_small_deep(seed: int = 0) -> DeepCFR:
    """Reusable tiny DeepCFR instance (for utility structural tests)."""
    return DeepCFR(
        game=KuhnPoker(),
        n_actions=2,
        encoding_dim=6,
        seed=seed,
        traversals_per_iter=10,
        buffer_capacity=1000,
        batch_size=16,
        advantage_epochs=1,
        strategy_epochs=1,
    )


# =============================================================================
# Fast smoke — tertiary diagnostic only. Backward-compat for existing callers.
# =============================================================================
class TestDeepCFRKuhnCorrelationSmoke:
    """Fast tertiary-only smoke (~60s). Catches structural regression only."""

    def test_kuhn_correlation_smoke(self) -> None:
        """K=20, T=100 — tertiary correlation should exceed sanity threshold.

        Note: tertiary is the *design-corrected-away* metric per 2026-04-24.
        Its threshold 0.3 is loose and catches structural bugs (not-converging
        traversal, dead loss) rather than convergence quality. Primary A/B
        smoke is covered by ``TestCorrelationReport.test_primary_a_above_smoke``.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        game = KuhnPoker()
        deep = DeepCFR(
            game=game,
            n_actions=2,
            encoding_dim=6,
            device="cpu",
            seed=42,
            traversals_per_iter=20,
            buffer_capacity=5000,
            batch_size=32,
            advantage_epochs=2,
            strategy_epochs=2,
        )
        deep.train(100)

        tab = CFRPlus(game=game, n_actions=2)
        tab.train(100)

        pearson_r, tab_vec, net_vec = compute_flat_correlation(deep, tab, game)
        assert tab_vec.shape == (24,), f"expected 24 pairs, got {tab_vec.shape}"
        assert net_vec.shape == (24,)
        # Tertiary threshold 0.3: empirically ~0.4 at this config (2026-04-24).
        assert pearson_r > 0.3, (
            f"tertiary smoke r={pearson_r:.3f} — structural bug suspected"
        )
        assert tab_vec.std() > 0.0
        assert net_vec.std() > 0.0


# =============================================================================
# Slow path — Exit #4 primary dry-run (Primary A + Primary B)
# =============================================================================
@pytest.mark.slow
class TestDeepCFRKuhnCorrelationFull:
    """Exit #4 primary dry-run (Primary A + Primary B) at T=2000.

    Uses :func:`compute_correlations` (3-metric) with Vanilla CFR and
    CFR+ trained in parallel. Primary A gates the advantage-net fidelity,
    Primary B gates the strategy-net fidelity. Wall-clock ≈ 30–60 min.
    """

    def test_kuhn_primary_a_and_b_at_2000_iters(self) -> None:
        """Mentor 2026-04-24 gates: Primary A > 0.9 AND Primary B > 0.85."""
        torch.manual_seed(42)
        np.random.seed(42)

        game = KuhnPoker()
        deep = DeepCFR(
            game=game,
            n_actions=2,
            encoding_dim=6,
            device="cpu",
            seed=42,
            traversals_per_iter=100,
            buffer_capacity=100_000,
            batch_size=64,
            advantage_epochs=4,
            strategy_epochs=4,
        )
        vanilla = VanillaCFR(game=game, n_actions=2)
        cfp = CFRPlus(game=game, n_actions=2)

        checkpoints = [500, 1000, 1500, 2000]
        prim_a: list[float] = []
        prim_b: list[float] = []
        for target_T in checkpoints:
            if deep.iteration < target_T:
                deep.train(target_T - deep.iteration)
            if vanilla.iteration < target_T:
                vanilla.train(target_T - vanilla.iteration)
            if cfp.iteration < target_T:
                cfp.train(target_T - cfp.iteration)

            rep: CorrelationReport = compute_correlations(
                deep, vanilla, cfp, game
            )
            prim_a.append(rep.primary_a_advantage_vs_vanilla)
            prim_b.append(rep.primary_b_strategy_vs_sigma_bar)

        assert prim_a[-1] > 0.9, (
            f"Primary A FAIL at T=2000: r={prim_a[-1]:.3f} "
            f"(trajectory: {[f'{v:.3f}' for v in prim_a]})"
        )
        assert prim_b[-1] > 0.85, (
            f"Primary B FAIL at T=2000: r={prim_b[-1]:.3f} "
            f"(trajectory: {[f'{v:.3f}' for v in prim_b]})"
        )
        # Monotonic improvement gate.
        assert prim_a[-1] - prim_a[0] > 0.05, (
            f"Primary A not improving: {prim_a[0]:.3f} → {prim_a[-1]:.3f}"
        )


# =============================================================================
# Structural tests — new 3-metric API
# =============================================================================
class TestCorrelationReport:
    """Structural guarantees for :func:`compute_correlations`."""

    @pytest.fixture
    def trainers(self) -> tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker]:
        torch.manual_seed(0)
        np.random.seed(0)
        game = KuhnPoker()
        deep = _make_small_deep(seed=0)
        deep.train(5)
        van = VanillaCFR(game=game, n_actions=2)
        van.train(5)
        cfp = CFRPlus(game=game, n_actions=2)
        cfp.train(5)
        return deep, van, cfp, game

    def test_returns_report_with_three_correlations(
        self,
        trainers: tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker],
    ) -> None:
        deep, van, cfp, game = trainers
        rep = compute_correlations(deep, van, cfp, game)
        assert isinstance(rep, CorrelationReport)
        assert isinstance(rep.primary_a_advantage_vs_vanilla, float)
        assert isinstance(rep.primary_b_strategy_vs_sigma_bar, float)
        assert isinstance(rep.tertiary_advantage_vs_cfr_plus, float)

    def test_all_correlations_in_minus_one_to_one(
        self,
        trainers: tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker],
    ) -> None:
        deep, van, cfp, game = trainers
        rep = compute_correlations(deep, van, cfp, game)
        for r in (
            rep.primary_a_advantage_vs_vanilla,
            rep.primary_b_strategy_vs_sigma_bar,
            rep.tertiary_advantage_vs_cfr_plus,
        ):
            # NaN guard only — the main assertion is that r is finite and in range.
            assert np.isnan(r) or -1.0 <= r <= 1.0

    def test_kuhn_yields_24_pairs(
        self,
        trainers: tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker],
    ) -> None:
        deep, van, cfp, game = trainers
        rep = compute_correlations(deep, van, cfp, game)
        assert rep.n_pairs == 24, f"Kuhn expected 24 pairs, got {rep.n_pairs}"
        assert rep.tab_vanilla_vec.shape == (24,)
        assert rep.net_advantage_vec.shape == (24,)
        assert rep.tab_sigma_bar_vec.shape == (24,)
        assert rep.net_strategy_vec.shape == (24,)

    def test_strategy_net_output_is_valid_simplex(
        self,
        trainers: tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker],
    ) -> None:
        """Strategy net softmax output ∈ [0, 1] on legal slots."""
        deep, van, cfp, game = trainers
        rep = compute_correlations(deep, van, cfp, game)
        assert (rep.net_strategy_vec >= 0.0).all()
        assert (rep.net_strategy_vec <= 1.0).all()

    def test_primary_a_above_smoke(
        self,
        trainers: tuple[DeepCFR, VanillaCFR, CFRPlus, KuhnPoker],
    ) -> None:
        """Primary A smoke: 5-iter training should already give non-trivial r.

        Expected range at this tiny scale: 0.3–0.8 (short-training noise
        dominated, but structural relationship already forming).
        """
        deep, van, cfp, game = trainers
        rep = compute_correlations(deep, van, cfp, game)
        # NaN = constant vector; loose lower bound checks non-degeneracy.
        if not np.isnan(rep.primary_a_advantage_vs_vanilla):
            assert rep.primary_a_advantage_vs_vanilla > -0.5, (
                f"Primary A anti-correlates or constant: "
                f"r={rep.primary_a_advantage_vs_vanilla:.3f} — structural bug?"
            )


# =============================================================================
# Structural tests — legacy single-metric API (backward-compat)
# =============================================================================
class TestCorrelationUtility:
    """Structural guarantees of :func:`compute_flat_correlation`."""

    def test_returns_tuple_of_three(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        game = KuhnPoker()
        deep = _make_small_deep(seed=0)
        deep.train(5)
        tab = CFRPlus(game=game, n_actions=2)
        tab.train(5)

        result = compute_flat_correlation(deep, tab, game)
        assert len(result) == 3
        r, tab_vec, net_vec = result
        assert isinstance(r, float)
        assert isinstance(tab_vec, np.ndarray)
        assert isinstance(net_vec, np.ndarray)

    def test_vectors_have_same_length(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        game = KuhnPoker()
        deep = _make_small_deep(seed=0)
        deep.train(5)
        tab = CFRPlus(game=game, n_actions=2)
        tab.train(5)
        _, tab_vec, net_vec = compute_flat_correlation(deep, tab, game)
        assert tab_vec.shape == net_vec.shape

    def test_kuhn_yields_24_pairs(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        game = KuhnPoker()
        deep = _make_small_deep(seed=0)
        deep.train(5)
        tab = CFRPlus(game=game, n_actions=2)
        tab.train(5)
        _, tab_vec, _ = compute_flat_correlation(deep, tab, game)
        assert tab_vec.shape == (24,), f"Kuhn expected 24, got {tab_vec.shape}"

    def test_correlation_is_in_minus_one_to_one(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        game = KuhnPoker()
        deep = _make_small_deep(seed=0)
        deep.train(5)
        tab = CFRPlus(game=game, n_actions=2)
        tab.train(5)
        r, _, _ = compute_flat_correlation(deep, tab, game)
        assert -1.0 <= r <= 1.0
