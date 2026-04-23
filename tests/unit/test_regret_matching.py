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


# -----------------------------------------------------------------------------
# TestRegretMatchingLegalMask — Phase 2 Week 1 Day 2 extension.
#
# Target API extension (NOT YET IMPLEMENTED — these tests must initially fail
# with TypeError: unexpected keyword argument 'legal_mask'):
#     regret_matching(
#         cumulative_regret: np.ndarray,
#         legal_mask: np.ndarray | None = None,
#     ) -> np.ndarray
#
# Semantics:
#   legal_mask is None → 기존 동작 완전 보존 (Kuhn 경로 불변).
#   legal_mask is bool array of same shape as cumulative_regret:
#     - positive-part regret masked elementwise (illegal positions forced 0)
#     - if total > 0 → normalize
#     - if total == 0 → uniform OVER LEGAL ONLY (illegal 유지 0)
# -----------------------------------------------------------------------------
class TestRegretMatchingLegalMask:
    """legal_mask 옵션 추가 후 기존 동작 보존 + illegal 확률 0 보장."""

    def test_legal_mask_none_matches_legacy_behavior(self) -> None:
        """legal_mask=None은 legacy regret_matching과 bit-identical 결과를 낸다."""
        cases = [
            np.array([1.0, 2.0, 3.0]),      # positive regrets
            np.zeros(4),                     # all-zero → uniform
            np.array([-1.0, -2.0, -3.0]),    # all-negative → uniform
            np.array([-1.0, 2.0, -3.0, 4.0]),  # mixed signs
        ]
        for regrets in cases:
            legacy = regret_matching(regrets)
            explicit_none = regret_matching(regrets, legal_mask=None)
            np.testing.assert_array_almost_equal(legacy, explicit_none, decimal=12)

    def test_legal_mask_zeros_illegal_probability(self) -> None:
        """mask로 illegal 처리된 액션의 확률은 정확히 0, 나머지 합 == 1."""
        regrets = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, True, False])
        strategy = regret_matching(regrets, legal_mask=mask)
        assert strategy[2] == 0.0
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)
        # Proportional to positive regrets among legal (1 and 2 → 1/3, 2/3).
        assert strategy[0] == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert strategy[1] == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_legal_mask_fallback_uniform_over_legal_only(self) -> None:
        """regret 전부 0 + legal 2개 → 해당 2개에만 1/2 분배, illegal 0."""
        regrets = np.zeros(3)
        mask = np.array([True, True, False])
        strategy = regret_matching(regrets, legal_mask=mask)
        assert strategy[0] == pytest.approx(0.5, abs=1e-9)
        assert strategy[1] == pytest.approx(0.5, abs=1e-9)
        assert strategy[2] == 0.0
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)

    def test_legal_mask_fallback_all_negative_regret(self) -> None:
        """모든 regret negative → positive-part 전부 0 → legal 중 uniform."""
        regrets = np.array([-1.0, -2.0, -3.0])
        mask = np.array([False, True, True])
        strategy = regret_matching(regrets, legal_mask=mask)
        assert strategy[0] == 0.0
        assert strategy[1] == pytest.approx(0.5, abs=1e-9)
        assert strategy[2] == pytest.approx(0.5, abs=1e-9)
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)

    def test_legal_mask_illegal_positive_regret_masked_out(self) -> None:
        """illegal 위치의 큰 positive regret은 완전히 무시되어야 한다.

        이것이 Leduc에서 FOLD illegal-but-regret-positive 시나리오의 regression
        guard. mask를 positive-part에 곱하는 순서가 틀리면 강하게 샌다.
        """
        regrets = np.array([10.0, 1.0, 1.0])  # index 0에 큰 positive, 그런데 illegal
        mask = np.array([False, True, True])
        strategy = regret_matching(regrets, legal_mask=mask)
        assert strategy[0] == 0.0
        # Legal positions have equal positive regret (1.0 each) → 0.5/0.5
        assert strategy[1] == pytest.approx(0.5, abs=1e-9)
        assert strategy[2] == pytest.approx(0.5, abs=1e-9)
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)

    def test_legal_mask_single_legal_action(self) -> None:
        """legal 액션이 하나뿐이면 그 액션에 모든 확률이 집중된다."""
        regrets = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, True])
        strategy = regret_matching(regrets, legal_mask=mask)
        assert strategy[0] == 0.0
        assert strategy[1] == 0.0
        assert strategy[2] == pytest.approx(1.0, abs=1e-9)
        assert strategy.sum() == pytest.approx(1.0, abs=1e-9)
