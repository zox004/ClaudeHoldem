"""Unit tests for KuhnState.legal_action_mask() — Phase 2 Day 3 C1 addition.

Target method (NOT YET IMPLEMENTED — these tests must fail initially):
    KuhnState.legal_action_mask() -> np.ndarray

Kuhn property: all actions (PASS, BET) legal at every non-terminal state,
so mask is uniformly ``np.array([True, True])`` of shape (2,) dtype bool.

This method is added for StateProtocol conformance once Day 3 refactor makes
VanillaCFR game-agnostic via ``regret_matching(legal_mask=...)``. Kuhn
behavior is unchanged (loop-masked with all-True is a no-op), but the
uniform interface lets the same CFR code run on Leduc and future games.

Reference: Phase 2 design decision B (add method to Kuhn, not hasattr
branching). Kuhn 193 existing tests must remain GREEN after C1.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.kuhn import KuhnAction, KuhnPoker, KuhnState


class TestKuhnLegalActionMaskShape:
    def test_mask_returns_ndarray(self) -> None:
        """Return type is np.ndarray."""
        root = KuhnPoker.state_from_deal((0, 1))
        mask = root.legal_action_mask()
        assert isinstance(mask, np.ndarray)

    def test_mask_shape_is_num_actions(self) -> None:
        """mask.shape == (KuhnPoker.NUM_ACTIONS,) == (2,)."""
        root = KuhnPoker.state_from_deal((0, 1))
        assert root.legal_action_mask().shape == (KuhnPoker.NUM_ACTIONS,)
        assert root.legal_action_mask().shape == (2,)

    def test_mask_dtype_is_bool(self) -> None:
        """dtype is np.bool_."""
        root = KuhnPoker.state_from_deal((0, 1))
        assert root.legal_action_mask().dtype == np.bool_


class TestKuhnLegalActionMaskAlwaysAllTrue:
    """Kuhn-specific invariant: at every non-terminal state, both
    PASS and BET are legal. Distinguishes Kuhn from Leduc's conditional
    legality (FOLD illegal at bets=0, RAISE illegal at cap)."""

    def test_root_state_mask_all_true(self) -> None:
        root = KuhnPoker.state_from_deal((0, 1))
        np.testing.assert_array_equal(
            root.legal_action_mask(),
            np.array([True, True]),
        )

    def test_after_pass_mask_all_true(self) -> None:
        """After P1 PASS, P2 to act — both actions still legal (they can
        check to showdown or bet to start the betting)."""
        state = KuhnPoker.state_from_deal((0, 1)).next_state(KuhnAction.PASS)
        assert not state.is_terminal
        np.testing.assert_array_equal(
            state.legal_action_mask(),
            np.array([True, True]),
        )

    def test_after_bet_mask_all_true(self) -> None:
        """After P1 BET, P2 to act — PASS (fold) and BET (call) both legal."""
        state = KuhnPoker.state_from_deal((0, 1)).next_state(KuhnAction.BET)
        assert not state.is_terminal
        np.testing.assert_array_equal(
            state.legal_action_mask(),
            np.array([True, True]),
        )

    def test_after_pass_bet_mask_all_true(self) -> None:
        """P1 PASS, P2 BET → P1 still has PASS (fold) or BET (call)."""
        state = (
            KuhnPoker.state_from_deal((0, 1))
            .next_state(KuhnAction.PASS)
            .next_state(KuhnAction.BET)
        )
        assert not state.is_terminal
        np.testing.assert_array_equal(
            state.legal_action_mask(),
            np.array([True, True]),
        )


class TestKuhnLegalActionMaskConsistentWithLegalActions:
    """legal_action_mask() must agree with legal_actions() (index-by-index)."""

    @pytest.mark.parametrize(
        "history_chain",
        [
            (),
            (KuhnAction.PASS,),
            (KuhnAction.BET,),
            (KuhnAction.PASS, KuhnAction.BET),
        ],
        ids=["root", "after_P", "after_B", "after_PB"],
    )
    def test_mask_matches_legal_actions(
        self, history_chain: tuple[KuhnAction, ...]
    ) -> None:
        state = KuhnPoker.state_from_deal((0, 1))
        for action in history_chain:
            state = state.next_state(action)
        if state.is_terminal:
            pytest.skip("terminal state — legal_actions not defined")
        mask = state.legal_action_mask()
        legal = state.legal_actions()
        expected = np.zeros(KuhnPoker.NUM_ACTIONS, dtype=bool)
        for a in legal:
            expected[int(a)] = True
        np.testing.assert_array_equal(mask, expected)
