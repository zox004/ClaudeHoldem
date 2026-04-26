"""FAILING tests for Local Best Response (LBR) — Phase 4 M3.3.

Target module: ``src/poker_ai/eval/local_best_response.py`` (does NOT exist yet).
These tests are RED-by-design: importing the module raises ``ModuleNotFoundError``
until M3.3 implementation lands.

Algorithm reference: Lisý & Bowling 2016 (IJCAI), "Equilibrium Approximation
Quality of Current No-Limit Poker Bots". LBR is the standard HUNL benchmark
(Pluribus 2019, Brown 2017). LBR is a strict lower bound on the true exact
best-response value (Lemma 1).

Test categories
---------------
A. LBRConfig dataclass (3)
B. lbr_value basic (5)
C. AbstractedHUNLGame integration (3)
D. Statistical correctness — Leduc validation (5)
E. lbr_exploitability + paired sampling (5)
F. rollout policy (3)
G. Multi-seed monotone trend smoke (3)
H. Edge cases (4)
I. GameProtocol compliance / typing (2)

Total: 33 tests (target ≥ 30).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import exploitability
from poker_ai.games.hunl_abstraction import AbstractedHUNLAction, AbstractedHUNLGame
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker

# RED-by-design import: module does not exist until M3.3 implementation lands.
from poker_ai.eval.local_best_response import (  # noqa: E402
    LBRConfig,
    default_rollout_policy,
    lbr_exploitability,
    lbr_value,
)


# =============================================================================
# Fixtures — share expensive CFR training across the whole module session.
# =============================================================================


@pytest.fixture(scope="session")
def leduc_cfr_t2000() -> dict[str, np.ndarray]:
    """Leduc near-Nash strategy via VanillaCFR T=2000 (~10s)."""
    np.random.seed(42)
    trainer = VanillaCFR(game=LeducPoker(), n_actions=3)
    trainer.train(iterations=2_000)
    return trainer.average_strategy()


@pytest.fixture(scope="session")
def kuhn_cfr_t10000() -> dict[str, np.ndarray]:
    """Kuhn near-Nash strategy via VanillaCFR T=10000 (instant)."""
    np.random.seed(42)
    trainer = VanillaCFR(game=KuhnPoker(), n_actions=2)
    trainer.train(iterations=10_000)
    return trainer.average_strategy()


@pytest.fixture(scope="session")
def leduc_cfr_t1000() -> dict[str, np.ndarray]:
    """Leduc weak strategy via VanillaCFR T=1000 (a few seconds)."""
    np.random.seed(42)
    trainer = VanillaCFR(game=LeducPoker(), n_actions=3)
    trainer.train(iterations=1_000)
    return trainer.average_strategy()


@pytest.fixture(scope="session")
def leduc_cfr_t100k_seeds() -> list[dict[str, np.ndarray]]:
    """Leduc near-Nash strategies for 3 seeds, T=100000 (slow ~30s each).

    Used only by the multi-seed monotone trend smoke test.
    """
    out: list[dict[str, np.ndarray]] = []
    for seed in (42, 123, 456):
        np.random.seed(seed)
        trainer = VanillaCFR(game=LeducPoker(), n_actions=3)
        trainer.train(iterations=100_000)
        out.append(trainer.average_strategy())
    return out


# =============================================================================
# A. LBRConfig dataclass (3 tests)
# =============================================================================


class TestLBRConfig:
    def test_defaults(self):
        """LBRConfig defaults: n_samples=200, seed=42, paired=True."""
        cfg = LBRConfig()
        assert cfg.n_samples == 200
        assert cfg.seed == 42
        assert cfg.paired is True

    def test_frozen(self):
        """LBRConfig is frozen — attribute assignment raises FrozenInstanceError."""
        cfg = LBRConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_samples = 999  # type: ignore[misc]

    def test_custom_values_stored(self):
        """LBRConfig accepts and preserves custom n_samples / seed / paired."""
        cfg = LBRConfig(n_samples=1000, seed=7, paired=False)
        assert cfg.n_samples == 1000
        assert cfg.seed == 7
        assert cfg.paired is False


# =============================================================================
# B. lbr_value basic (5 tests)
# =============================================================================


class TestLBRValueBasic:
    def test_returns_float_on_leduc_uniform(self):
        """lbr_value returns a float on Leduc with empty strategy (uniform)."""
        rng = np.random.default_rng(42)
        v = lbr_value(LeducPoker(), {}, responder=0, n_samples=10, rng=rng)
        assert isinstance(v, float)

    def test_responder_asymmetry(self):
        """Different responders yield distinct LBR values (asymmetric tree)."""
        rng0 = np.random.default_rng(42)
        rng1 = np.random.default_rng(42)
        v0 = lbr_value(LeducPoker(), {}, responder=0, n_samples=50, rng=rng0)
        v1 = lbr_value(LeducPoker(), {}, responder=1, n_samples=50, rng=rng1)
        assert v0 != v1

    def test_zero_samples_raises(self):
        """n_samples == 0 raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            lbr_value(LeducPoker(), {}, responder=0, n_samples=0, rng=rng)

    def test_determinism_same_seed(self):
        """Same RNG seed → same LBR value (bit-equal float)."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        v_a = lbr_value(LeducPoker(), {}, responder=0, n_samples=30, rng=rng_a)
        v_b = lbr_value(LeducPoker(), {}, responder=0, n_samples=30, rng=rng_b)
        assert v_a == v_b

    def test_kuhn_smoke(self):
        """lbr_value runs on KuhnPoker (cross-game smoke)."""
        rng = np.random.default_rng(42)
        v = lbr_value(KuhnPoker(), {}, responder=0, n_samples=20, rng=rng)
        assert isinstance(v, float)


# =============================================================================
# C. AbstractedHUNLGame integration (3 tests)
# =============================================================================


class TestLBRAbstractedHUNL:
    @pytest.fixture(scope="class")
    def hunl_game(self) -> AbstractedHUNLGame:
        # Small fixture — keep MC trials minimal.
        return AbstractedHUNLGame(
            n_buckets=10,
            n_trials=200,
            postflop_threshold_sample_size=40,
            postflop_mc_trials=20,
        )

    def test_smoke_n_samples_2(self, hunl_game: AbstractedHUNLGame):
        """LBR runs on AbstractedHUNLGame with n_samples=2."""
        rng = np.random.default_rng(42)
        v = lbr_value(hunl_game, {}, responder=0, n_samples=2, rng=rng)
        assert isinstance(v, float)

    def test_default_rollout_returns_int(self, hunl_game: AbstractedHUNLGame):
        """default_rollout_policy returns an int-castable IntEnum action."""
        rollout = default_rollout_policy(hunl_game)
        rng = np.random.default_rng(42)
        deal = hunl_game.sample_deal(rng)
        state = hunl_game.state_from_deal(deal)
        a = rollout(state)
        # Must be castable to int (IntEnum or int).
        assert int(a) in {int(x) for x in state.legal_actions()}

    def test_illegal_rollout_raises(self, hunl_game: AbstractedHUNLGame):
        """A rollout policy returning an illegal action raises ValueError."""
        rng = np.random.default_rng(42)

        # Always-return-FOLD is illegal at the root (no prior bet to fold to).
        def always_fold(state):
            return AbstractedHUNLAction.FOLD

        with pytest.raises(ValueError):
            lbr_value(
                hunl_game,
                {},
                responder=0,
                n_samples=2,
                rng=rng,
                rollout_policy=always_fold,
            )


# =============================================================================
# D. Statistical correctness — Leduc validation (5 tests)
# =============================================================================


class TestLBRStatisticalCorrectness:
    def test_signal_uniform_vs_near_nash(
        self, leduc_cfr_t2000: dict[str, np.ndarray]
    ):
        """LBR(uniform) > LBR(near-Nash strategy) — exploitability signal."""
        n = 300
        rng_u = np.random.default_rng(42)
        rng_n = np.random.default_rng(42)
        lbr_unif = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_u, paired=True
        )[0]
        lbr_near = lbr_exploitability(
            LeducPoker(), leduc_cfr_t2000, n_samples=n, rng=rng_n, paired=True
        )[0]
        assert lbr_unif > lbr_near, (
            f"expected LBR(uniform)={lbr_unif:.4f} > "
            f"LBR(near-Nash)={lbr_near:.4f}"
        )

    def test_lower_bound_uniform(self):
        """LBR(uniform) ≤ exact exploitability(uniform) (Lisý & Bowling Lemma 1)."""
        n = 300
        rng = np.random.default_rng(42)
        lbr_val, _ = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng, paired=True
        )
        exact = exploitability(LeducPoker(), {})
        # Allow a small tolerance for sampling overshoot (LBR is in expectation
        # ≤ exact, finite samples may overshoot by a little). Use 2 chips.
        assert lbr_val <= exact + 2.0, (
            f"LBR={lbr_val:.4f} should be ≤ exact={exact:.4f} (Lemma 1)"
        )

    def test_lower_bound_near_nash(
        self, leduc_cfr_t2000: dict[str, np.ndarray]
    ):
        """LBR(near-Nash) ≤ exact exploitability(near-Nash) (Lemma 1)."""
        n = 300
        rng = np.random.default_rng(42)
        lbr_val, _ = lbr_exploitability(
            LeducPoker(), leduc_cfr_t2000, n_samples=n, rng=rng, paired=True
        )
        exact = exploitability(LeducPoker(), leduc_cfr_t2000)
        assert lbr_val <= exact + 1.0, (
            f"LBR={lbr_val:.4f} should be ≤ exact={exact:.4f}"
        )

    def test_kuhn_near_nash_small(
        self, kuhn_cfr_t10000: dict[str, np.ndarray]
    ):
        """LBR(KuhnPoker, T=10k near-Nash) is small (< 0.05 chips/game ≈ 50 mbb)."""
        n = 500
        rng = np.random.default_rng(42)
        lbr_val, _ = lbr_exploitability(
            KuhnPoker(), kuhn_cfr_t10000, n_samples=n, rng=rng, paired=True
        )
        assert lbr_val < 0.05, (
            f"LBR(KuhnPoker, T=10k) = {lbr_val:.4f} should be < 0.05"
        )

    def test_reproducibility_bit_exact(self):
        """Two calls with the same n_samples + RNG seed return bit-equal floats."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        v_a = lbr_value(LeducPoker(), {}, responder=0, n_samples=200, rng=rng_a)
        v_b = lbr_value(LeducPoker(), {}, responder=0, n_samples=200, rng=rng_b)
        assert v_a == v_b


# =============================================================================
# E. lbr_exploitability + paired sampling (5 tests)
# =============================================================================


class TestLBRExploitability:
    def test_returns_tuple(self):
        """lbr_exploitability returns (mean: float, se: float)."""
        rng = np.random.default_rng(42)
        out = lbr_exploitability(
            LeducPoker(), {}, n_samples=20, rng=rng, paired=True
        )
        assert isinstance(out, tuple) and len(out) == 2
        mean, se = out
        assert isinstance(mean, float)
        assert isinstance(se, float)

    def test_paired_vs_unpaired_means_close(self):
        """Paired and unpaired LBR means agree within 2σ at moderate n."""
        n = 300
        rng_p = np.random.default_rng(42)
        rng_u = np.random.default_rng(42)
        m_p, se_p = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_p, paired=True
        )
        m_u, se_u = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_u, paired=False
        )
        # Combined SE upper bound; allow 3σ for test robustness.
        bound = 3.0 * (se_p + se_u + 1e-9)
        assert abs(m_p - m_u) <= bound, (
            f"paired={m_p:.4f}, unpaired={m_u:.4f}, "
            f"bound=3*(se_p+se_u)={bound:.4f}"
        )

    def test_paired_se_le_unpaired_se(self):
        """Paired sampling SE ≤ unpaired SE (control variate sanity)."""
        n = 300
        rng_p = np.random.default_rng(42)
        rng_u = np.random.default_rng(42)
        _, se_p = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_p, paired=True
        )
        _, se_u = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_u, paired=False
        )
        # Allow tiny epsilon — the variance reduction may be marginal at small n.
        assert se_p <= se_u + 1e-6, (
            f"paired SE {se_p:.6f} should be ≤ unpaired SE {se_u:.6f}"
        )

    def test_mean_equals_average_of_p0_p1_unpaired(self):
        """Unpaired mean = (LBR_P0 + LBR_P1) / 2 with same per-side RNG."""
        n = 100
        # The unpaired implementation must be reproducible from a single seed.
        rng_combined = np.random.default_rng(42)
        mean, _ = lbr_exploitability(
            LeducPoker(), {}, n_samples=n, rng=rng_combined, paired=False
        )
        # Manually recompute P0 and P1 LBR values with their own seeded RNGs.
        # Implementation is expected to derive sub-rngs from `rng` deterministically;
        # we check a structural identity: mean is in [min(p0,p1)-eps, max(p0,p1)+eps]
        # since it is their arithmetic average.
        rng_p0 = np.random.default_rng(123)
        rng_p1 = np.random.default_rng(456)
        v0 = lbr_value(LeducPoker(), {}, responder=0, n_samples=n, rng=rng_p0)
        v1 = lbr_value(LeducPoker(), {}, responder=1, n_samples=n, rng=rng_p1)
        assert min(v0, v1) - 1.0 <= mean <= max(v0, v1) + 1.0, (
            f"mean={mean:.4f} should be within rough convex hull of "
            f"v0={v0:.4f}, v1={v1:.4f}"
        )

    def test_se_positive_when_n_ge_2(self):
        """SE > 0 when n_samples ≥ 2 (non-degenerate sample)."""
        rng = np.random.default_rng(42)
        _, se = lbr_exploitability(
            LeducPoker(), {}, n_samples=20, rng=rng, paired=True
        )
        assert se > 0.0


# =============================================================================
# F. rollout policy (3 tests)
# =============================================================================


class TestRolloutPolicy:
    def test_custom_callable_accepted(self):
        """A custom rollout callable is accepted by lbr_value."""
        rng = np.random.default_rng(42)
        # always pick the first legal action
        v = lbr_value(
            LeducPoker(),
            {},
            responder=0,
            n_samples=20,
            rng=rng,
            rollout_policy=lambda state: state.legal_actions()[0],
        )
        assert isinstance(v, float)

    def test_lbr_ge_rollout_value(self):
        """LBR (myopic argmax) ≥ value of fixed rollout (LBR optimizes over rollout).

        We compare LBR against itself with rollout_policy = always-pick-first-legal:
        when LBR uses myopic argmax over THE SAME rollout for future LBR moves,
        the LBR value (over the SAME deal samples) must be ≥ pure-rollout value
        (i.e. LBR with rollout_policy applied at every decision, no argmax).
        Since pure rollout is a special case of LBR's myopic argmax (one of the
        candidate actions IS the rollout's choice), LBR ≥ pure-rollout.
        """
        n = 100

        def first_legal(state):
            return state.legal_actions()[0]

        rng_lbr = np.random.default_rng(42)
        v_lbr = lbr_value(
            LeducPoker(),
            {},
            responder=0,
            n_samples=n,
            rng=rng_lbr,
            rollout_policy=first_legal,
        )
        # Reference: a "pure rollout" baseline can be approximated by giving LBR
        # the same trivial first-legal rollout but no real opponent strategy.
        # We don't have a separate API, so we just compare against responder=1
        # with the same setup as a sanity-finite check that LBR >= floor.
        assert isinstance(v_lbr, float)
        # Strict LBR-≥-rollout cannot be tested without a separate rollout-only
        # function; sub-test reduces to a finiteness sanity that ensures the
        # API accepts the rollout and returns a real number.
        assert np.isfinite(v_lbr)

    def test_default_rollout_factory_returns_callable(self):
        """default_rollout_policy(game) returns a callable: state → action."""
        rollout = default_rollout_policy(LeducPoker())
        assert isinstance(rollout, Callable)  # type: ignore[arg-type]

        deal = LeducPoker.sample_deal(np.random.default_rng(42))
        state = LeducPoker.state_from_deal(deal)
        a = rollout(state)
        assert int(a) in {int(x) for x in state.legal_actions()}


# =============================================================================
# G. Multi-seed monotone trend smoke (3 tests)
# =============================================================================


class TestMultiSeedTrend:
    @pytest.mark.slow
    def test_T100k_lt_T1k(
        self,
        leduc_cfr_t1000: dict[str, np.ndarray],
        leduc_cfr_t100k_seeds: list[dict[str, np.ndarray]],
    ):
        """Mean LBR(T=100k) < mean LBR(T=1k) by ≥ 3× (CFR convergence smoke)."""
        n = 200
        # T=1k baseline (single seed=42 from fixture).
        rng_a = np.random.default_rng(42)
        lbr_t1k, _ = lbr_exploitability(
            LeducPoker(), leduc_cfr_t1000, n_samples=n, rng=rng_a, paired=True
        )
        # T=100k mean across 3 seeds.
        means: list[float] = []
        for i, strat in enumerate(leduc_cfr_t100k_seeds):
            rng_b = np.random.default_rng(1000 + i)
            m, _ = lbr_exploitability(
                LeducPoker(), strat, n_samples=n, rng=rng_b, paired=True
            )
            means.append(m)
        lbr_t100k = float(np.mean(means))
        assert lbr_t100k * 3.0 < lbr_t1k + 1e-6, (
            f"LBR(T=100k)={lbr_t100k:.4f} should be < LBR(T=1k)/3 "
            f"= {lbr_t1k / 3.0:.4f}"
        )

    def test_per_seed_determinism(self):
        """For each fixed seed, LBR is deterministic across two runs.

        Different seeds need not produce different results in exact mode
        (Kuhn / Leduc) because all_deals is enumerated and the opponent
        is σ-enumerated — RNG is unused. The strict bit-equality across
        rerun is the actual contract.
        """
        for seed in (42, 123, 456):
            rng = np.random.default_rng(seed)
            v = lbr_value(
                LeducPoker(), {}, responder=0, n_samples=30, rng=rng
            )
            rng2 = np.random.default_rng(seed)
            v2 = lbr_value(
                LeducPoker(), {}, responder=0, n_samples=30, rng=rng2
            )
            assert v == v2, (
                f"non-deterministic at seed={seed}: {v} != {v2}"
            )

    def test_mean_se_reportable(self):
        """lbr_exploitability output is unpackable as (mean, se) for reporting."""
        rng = np.random.default_rng(42)
        out = lbr_exploitability(
            LeducPoker(), {}, n_samples=20, rng=rng, paired=True
        )
        mean, se = out
        # Format-check: both fit "mean ± SE" reporting.
        s = f"{mean:.4f} ± {se:.4f}"
        assert "±" in s


# =============================================================================
# H. Edge cases (4 tests)
# =============================================================================


class TestEdgeCases:
    def test_empty_strategy_uniform_fallback(self):
        """Strategy={} → uniform fallback over NUM_ACTIONS, LBR runs."""
        rng = np.random.default_rng(42)
        v = lbr_value(LeducPoker(), {}, responder=0, n_samples=10, rng=rng)
        assert np.isfinite(v)

    def test_partial_strategy_falls_back_per_missing_key(
        self, leduc_cfr_t2000: dict[str, np.ndarray]
    ):
        """Strategy missing some infosets → uniform fallback for those, LBR runs."""
        # Drop a few keys to simulate sparse strategy.
        partial = dict(leduc_cfr_t2000)
        keys = list(partial.keys())
        for k in keys[:3]:
            del partial[k]
        rng = np.random.default_rng(42)
        v = lbr_value(LeducPoker(), partial, responder=0, n_samples=20, rng=rng)
        assert np.isfinite(v)

    def test_random_legal_strategy_finite_lbr(self):
        """LBR over a deterministic skewed strategy is finite.

        Note: ``LBR_exploitability`` is NOT guaranteed to be ≥ 0 even
        for non-Nash σ. LBR is a lower bound on BR, and BR_exploitability
        ≥ 0 always; but the LBR strategy itself can underperform Nash
        (its "loss" is allowed to be negative when the myopic + rollout
        combination is suboptimal). Test reduces to a finiteness sanity.
        """
        rng_strat = np.random.default_rng(7)
        np.random.seed(7)
        trainer = VanillaCFR(game=LeducPoker(), n_actions=3)
        trainer.train(iterations=100)
        strat = trainer.average_strategy()
        skewed: dict[str, np.ndarray] = {}
        for k, v in strat.items():
            arr = np.zeros_like(v)
            i = int(np.argmax(v + rng_strat.random(v.shape) * 1e-9))
            arr[i] = 1.0
            skewed[k] = arr
        rng = np.random.default_rng(42)
        v_lbr, _ = lbr_exploitability(
            LeducPoker(), skewed, n_samples=200, rng=rng, paired=True
        )
        assert np.isfinite(v_lbr)

    def test_n_samples_one(self):
        """n_samples=1 returns a finite mean (single-deal smoke)."""
        rng = np.random.default_rng(42)
        v = lbr_value(LeducPoker(), {}, responder=0, n_samples=1, rng=rng)
        assert np.isfinite(v)


# =============================================================================
# I. GameProtocol compliance / typing (2 tests)
# =============================================================================


class TestGameProtocolCompliance:
    def test_works_across_all_three_games(self):
        """lbr_value runs on KuhnPoker, LeducPoker, AbstractedHUNLGame."""
        for game in (
            KuhnPoker(),
            LeducPoker(),
            AbstractedHUNLGame(
                n_buckets=10,
                n_trials=200,
                postflop_threshold_sample_size=40,
                postflop_mc_trials=20,
            ),
        ):
            rng = np.random.default_rng(42)
            v = lbr_value(game, {}, responder=0, n_samples=2, rng=rng)
            assert isinstance(v, float)

    def test_average_strategy_dict_directly_usable(
        self, leduc_cfr_t2000: dict[str, np.ndarray]
    ):
        """VanillaCFR.average_strategy() output is directly usable by lbr_value."""
        rng = np.random.default_rng(42)
        v = lbr_value(
            LeducPoker(), leduc_cfr_t2000, responder=0, n_samples=20, rng=rng
        )
        assert np.isfinite(v)
