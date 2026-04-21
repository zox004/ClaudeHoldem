"""Unit tests for the pure regret_matching function.

Target module (NOT YET IMPLEMENTED — these tests must fail with ModuleNotFoundError):
    src/poker_ai/algorithms/regret_matching.py

Target API:
    regret_matching(cumulative_regret: np.ndarray) -> np.ndarray
      - Returns a probability distribution over actions, same shape as input.
      - Negative regrets are clipped to 0 before normalisation.
      - If all regrets are non-positive, returns a uniform distribution.

Reference: Zinkevich et al. 2007, Regret Matching (used by Vanilla CFR).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.regret_matching import regret_matching


class TestRegretMatching:
    def test_zero_regret_returns_uniform(self) -> None:
        """regret이 모두 0이면 균등 분포 [1/3, 1/3, 1/3]를 반환해야 한다."""
        strategy = regret_matching(np.zeros(3))
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3], atol=1e-9)

    def test_single_positive_regret_concentrates(self) -> None:
        """단일 positive regret이 압도적으로 크면 해당 액션에 확률이 집중된다."""
        strategy = regret_matching(np.array([1.0, 0.0, 0.0]))
        assert strategy[0] > 0.99, f"first action prob {strategy[0]:.6f} should be > 0.99"
        assert strategy[1] == pytest.approx(0.0, abs=1e-9)
        assert strategy[2] == pytest.approx(0.0, abs=1e-9)
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)

    def test_negative_regret_is_ignored(self) -> None:
        """negative regret은 0으로 clipping된 뒤 정규화되므로 유효 확률이 0이다."""
        strategy = regret_matching(np.array([-5.0, 1.0, 1.0]))
        np.testing.assert_allclose(strategy, [0.0, 0.5, 0.5], atol=1e-9)

    def test_probabilities_sum_to_one_for_random_regrets(self) -> None:
        """임의의 regret 벡터 100개에 대해 확률 합 == 1, 모든 원소 >= 0을 유지해야 한다."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            n_actions = int(rng.integers(3, 11))  # 3..10 inclusive
            regrets = rng.standard_normal(n_actions) * 5.0  # mix of signs, various scales
            strategy = regret_matching(regrets)
            assert strategy.shape == (n_actions,)
            assert abs(strategy.sum() - 1.0) < 1e-6, (
                f"sum={strategy.sum():.9f} for regrets={regrets}"
            )
            assert (strategy >= 0.0).all(), f"negative prob in {strategy}"
