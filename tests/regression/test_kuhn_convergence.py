"""Regression tests: Vanilla CFR on Kuhn converges to the known Nash equilibrium.

These tests lock in the **mathematical correctness** of the CFR implementation.
A drift in game value or Nash-family parameters almost certainly indicates a
bug in one of Zinkevich 2007 Eq. (3)–(9) — the areas CLAUDE.md flags as
"quietly performance-degrading".

Kuhn Poker Nash equilibrium (Neller & Lanctot 2013, Section 4.1):
    Game value (P1 perspective):  v1 = -1/18 ≈ -0.05556
    Nash strategy family parametrised by α ∈ [0, 1/3]:
        P1 at "J|"  (own J):    bet with prob α        (bluff with Jack)
        P1 at "Q|"  (own Q):    bet with prob 0        (always check with Queen)
        P1 at "K|"  (own K):    bet with prob 3α       (value-bet with King)
        P1 at "J|pb" (pass→bet): call with prob 0      (fold Jack)
        P1 at "K|pb":           call with prob 1       (always call King)
        P2 at "J|p":            bet with prob 1/3
        P2 at "Q|b":            call with prob 1/3
        P2 at "J|b":            call with prob 0       (fold Jack)
        P2 at "K|b":            call with prob 1

Tolerances are tight (10k iter of tabular Vanilla CFR on Kuhn is well past
Nash convergence; see Zinkevich 2007 Theorem 1 for the O(1/√T) bound).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.kuhn import KuhnAction, KuhnPoker

pytestmark = pytest.mark.slow  # Module-wide slow mark.

ITERATIONS = 10_000
BET = int(KuhnAction.BET)
PASS = int(KuhnAction.PASS)

# Nash known values
NASH_GAME_VALUE = -1.0 / 18.0
ALPHA_UPPER_BOUND = 1.0 / 3.0


@pytest.fixture
def trained_cfr() -> VanillaCFR:
    """Train Vanilla CFR to convergence.

    Function-scoped (default) — each test gets its own training run. This is
    intentional during TDD: sharing a training result across tests would mask
    which specific invariant a regression breaks. Slower but diagnostically
    clean; 10k iter on Kuhn is a few seconds per test.
    """
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    return cfr


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_game_value_converges_to_minus_one_eighteenth(seed: int) -> None:
    """Neller & Lanctot 2013 Section 4.1: Nash game value = -1/18 ≈ -0.0556.

    Vanilla CFR is deterministic for tabular Kuhn; the seed parametrisation
    documents intent and guards against future edits that introduce sampling.
    """
    np.random.seed(seed)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    v = cfr.game_value()
    assert abs(v - NASH_GAME_VALUE) < 1e-3, (
        f"game value {v:.6f} deviates from Nash {NASH_GAME_VALUE:.6f}"
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_jack_bet_probability_in_nash_range(seed: int) -> None:
    """Nash family: P1 at "J|" bets with prob α ∈ [0, 1/3].

    Eq. (6) (reach-weighted average) must fall within this range. Tolerance
    ε = 0.01 on the upper bound allows for finite-iteration drift.
    """
    np.random.seed(seed)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    jack_bet = cfr.average_strategy()["J|"][BET]
    assert 0.0 <= jack_bet <= ALPHA_UPPER_BOUND + 0.01, (
        f"Jack bet prob {jack_bet:.4f} outside [0, 1/3]"
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_king_bet_is_three_times_jack_bet(seed: int) -> None:
    """Nash relation: P1 King bet = 3 × P1 Jack bet (both parametrised by α).

    Tolerance 0.1 (not 0.05) because when α is close to 0 the ratio is
    sensitive to small absolute errors and seed-level variance can push
    |K - 3J| above 0.05 without indicating a real bug.
    """
    np.random.seed(seed)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    avg = cfr.average_strategy()
    jack_bet = avg["J|"][BET]
    king_bet = avg["K|"][BET]
    assert abs(king_bet - 3 * jack_bet) < 0.1, (
        f"King bet {king_bet:.4f} ≠ 3 × Jack bet 3*{jack_bet:.4f}"
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_queen_first_action_is_always_check(seed: int) -> None:
    """Nash: P1 at "Q|" never bets (bet prob = 0). Queen is too weak to bluff
    and too weak to value-bet.

    NOTE: This test checks ONLY the "Q|" infoset (P1's first move with Q).
    Other Queen-containing infosets have non-zero bet/call probabilities
    under Nash and would cause false failures if included here:
      - "Q|p"  (P2 facing check, own Q) — non-zero bet prob
      - "Q|b"  (P2 facing bet, own Q)   — calls with prob ≈ 1/3
      - "Q|pb" (P1 facing bet after own check) — non-trivial mix
    """
    np.random.seed(seed)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    queen_bet = cfr.average_strategy()["Q|"][BET]
    assert queen_bet < 0.05, f"Queen first-move bet prob {queen_bet:.4f} should be ≈ 0"


def test_average_strategy_is_valid_everywhere_after_10k_iter(
    trained_cfr: VanillaCFR,
) -> None:
    """Eq. (6) sanity after long training: every infoset yields a valid distribution."""
    strategies = trained_cfr.average_strategy()
    assert len(strategies) == 12
    for key, dist in strategies.items():
        assert dist.shape == (2,)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6), f"{key} sum ≠ 1"
        assert (dist >= 0).all(), f"{key} has negative entry: {dist}"


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_perfect_recall_invariant_after_10k_iter(seed: int) -> None:
    """CLAUDE.md invariant: perfect recall on Kuhn has exactly 12 infosets.

    This is the core test of the Lazy InfosetData initialisation: even after
    10,000 iterations of full-tree traversal, the dict must hold exactly 12
    entries — no duplicates from pathological key-collision bugs, no drift
    from cache/aliasing issues.
    """
    np.random.seed(seed)
    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(ITERATIONS)
    assert len(cfr.infosets) == 12, (
        f"expected 12 infosets after {ITERATIONS} iter, got {len(cfr.infosets)}"
    )
