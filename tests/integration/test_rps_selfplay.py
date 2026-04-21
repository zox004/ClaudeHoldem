"""Integration tests: Rock-Paper-Scissors self-play with RegretMatcher.

Target module (NOT YET IMPLEMENTED — these tests must fail with ModuleNotFoundError):
    src/poker_ai/algorithms/regret_matching.py

Target API:
    class RegretMatcher:
        def __init__(self, n_actions: int, rng: np.random.Generator) -> None: ...
        def current_strategy(self) -> np.ndarray:
            \"\"\"Strategy used for sampling at this iteration (from cumulative regret).\"\"\"
        def sample_action(self) -> int:
            \"\"\"Samples an action from current_strategy using the injected RNG.\"\"\"
        def update(self, action_utilities: np.ndarray) -> None:
            \"\"\"Accumulate regret (u_a - <strat, u>) and accumulate current strategy
            (used for average_strategy computation).\"\"\"
        def average_strategy(self) -> np.ndarray:
            \"\"\"Time-averaged strategy over all updates (CFR convergence theorem).\"\"\"

Reference: Regret Matching on bandits converges in average to the minimax equilibrium
(Neller & Lanctot 2013, Section 2.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.regret_matching import RegretMatcher

ROCK, PAPER, SCISSORS = 0, 1, 2
N_ACTIONS = 3

# Payoff matrix from P1's perspective: payoff[my][opp]
#   rock beats scissors, paper beats rock, scissors beats paper.
_PAYOFF = np.array(
    [
        [0, -1, 1],  # ROCK     vs rock/paper/scissors
        [1, 0, -1],  # PAPER    vs rock/paper/scissors
        [-1, 1, 0],  # SCISSORS vs rock/paper/scissors
    ],
    dtype=np.float64,
)


def rps_utilities(opp_action: int) -> np.ndarray:
    """Utility vector I would receive for each action given a fixed opponent action.

    Returns shape (3,) where entry ``a`` is my payoff if I play action ``a``.
    """
    return _PAYOFF[:, opp_action].copy()


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_converges_to_paper_vs_always_rock(seed: int) -> None:
    """상대가 항상 ROCK을 낼 때, 내 평균 전략은 PAPER(0,1,0)로 수렴해야 한다.

    ROCK 고정 상대에 대해 PAPER는 유일한 지배 전략이므로 best response가 곧 Nash의
    one-sided 해이고, regret matching의 time-averaged strategy는 그 best response로
    수렴한다.
    """
    matcher = RegretMatcher(n_actions=N_ACTIONS, rng=np.random.default_rng(seed))
    opp_action = ROCK
    for _ in range(1000):
        matcher.update(rps_utilities(opp_action))

    avg = matcher.average_strategy()
    assert avg.shape == (N_ACTIONS,)
    np.testing.assert_allclose(avg, [0.0, 1.0, 0.0], atol=0.02)


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_two_players_selfplay_converge_to_uniform(seed: int) -> None:
    """두 플레이어의 self-play는 대칭 zero-sum이므로 양쪽 모두 (1/3,1/3,1/3)로 수렴한다.

    RPS의 유일한 Nash equilibrium은 양쪽 uniform. CFR의 수렴 정리상 self-play에서
    time-averaged strategy는 Nash로 수렴하므로 두 평균 전략 모두 uniform에 근접해야
    한다.
    """
    m1 = RegretMatcher(n_actions=N_ACTIONS, rng=np.random.default_rng(seed))
    m2 = RegretMatcher(n_actions=N_ACTIONS, rng=np.random.default_rng(seed + 1))

    for _ in range(10_000):
        a1 = m1.sample_action()
        a2 = m2.sample_action()
        m1.update(rps_utilities(a2))
        # For P2, "opponent action" is a1; same payoff layout because RPS is symmetric.
        m2.update(rps_utilities(a1))

    uniform = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
    np.testing.assert_allclose(m1.average_strategy(), uniform, atol=0.03)
    np.testing.assert_allclose(m2.average_strategy(), uniform, atol=0.03)
