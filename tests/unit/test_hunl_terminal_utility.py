"""Unit tests for HUNL terminal utility (Phase 4 M1.4).

Coverage (per mentor's M1.4 sentinel sentinel #1, #2, #3):
- FOLD scenarios (multiple rounds, both players)
- Showdown win/loss/tie
- Uncalled chip return correctness (chip math)
- Non-terminal state raises
"""

from __future__ import annotations

import pytest

from poker_ai.games.hunl_state import (
    BB_BLIND_CHIPS_VALUE,
    HUNLAction,
    HUNLState,
    SB_BLIND_CHIPS,
    STARTING_STACK_CHIPS,
)


_EMPTY_RH: tuple[
    tuple[HUNLAction, ...], tuple[HUNLAction, ...],
    tuple[HUNLAction, ...], tuple[HUNLAction, ...],
] = ((), (), (), ())
_EMPTY_RS: tuple[
    tuple[int, ...], tuple[int, ...],
    tuple[int, ...], tuple[int, ...],
] = ((), (), (), ())


def _root_preflop(
    private_cards: tuple[int, int, int, int] = (0, 1, 2, 3),
    pending_board: tuple[int, int, int, int, int] = (4, 5, 6, 7, 8),
) -> HUNLState:
    return HUNLState(
        private_cards=private_cards,
        pending_board=pending_board,
        board_cards=(),
        round_history=_EMPTY_RH,
        round_bet_sizes=_EMPTY_RS,
        current_round=0,
        current_player=1,
        stack_p0=STARTING_STACK_CHIPS - BB_BLIND_CHIPS_VALUE,
        stack_p1=STARTING_STACK_CHIPS - SB_BLIND_CHIPS,
        last_raise_increment=BB_BLIND_CHIPS_VALUE,
        pot=BB_BLIND_CHIPS_VALUE + SB_BLIND_CHIPS,
    )


# Card-id helper using project-wide rank * 4 + suit packing.
_R = {c: i for i, c in enumerate("23456789TJQKA")}
_S = {c: i for i, c in enumerate("cdhs")}


def cid(s: str) -> int:
    return _R[s[0]] * 4 + _S[s[1]]


# =============================================================================
# Non-terminal raises
# =============================================================================
class TestNonTerminalRaises:
    def test_terminal_utility_on_root_raises(self) -> None:
        with pytest.raises(ValueError, match="non-terminal"):
            _root_preflop().terminal_utility()


# =============================================================================
# FOLD resolution
# =============================================================================
class TestFoldResolution:
    def test_sb_fold_preflop_p0_wins_sb_blind(self) -> None:
        """SB folds preflop without raising. P0 (BB) wins SB's 1-chip
        blind. terminal_utility = +1."""
        s = _root_preflop().next_state(HUNLAction.FOLD)
        assert s.is_terminal
        assert s.terminal_utility() == pytest.approx(+1.0)

    def test_bb_fold_after_sb_raise_p0_loses_bb_blind(self) -> None:
        """SB raises preflop, BB folds. P0 (BB) loses 2-chip blind.
        Uncalled excess (SB's raise above 2) returned to SB."""
        s = _root_preflop()
        s = s.next_state(HUNLAction.BET, bet_size=10)   # SB raise to 10
        s = s.next_state(HUNLAction.FOLD)               # BB fold
        assert s.is_terminal
        assert s.terminal_utility() == pytest.approx(-2.0)

    def test_fold_postflop_returns_correct_utility(self) -> None:
        """SB call preflop, BB check → flop. BB checks, SB bets, BB folds.
        Up to flop start, both contributed 2 each. SB's flop bet returns
        as uncalled."""
        s = _root_preflop()
        s = s.next_state(HUNLAction.CALL)   # SB call
        s = s.next_state(HUNLAction.CALL)   # BB check, → flop
        s = s.next_state(HUNLAction.CALL)   # BB check
        s = s.next_state(HUNLAction.BET, bet_size=10)   # SB bet 10
        s = s.next_state(HUNLAction.FOLD)               # BB fold
        assert s.is_terminal
        # P0 (BB) contributed 2 (preflop). P1 (SB) contributed 2 + 10 = 12.
        # matched = min = 2. P0 (BB) folded → loses 2.
        # P1's 10-chip flop bet is uncalled and returned to SB.
        assert s.terminal_utility() == pytest.approx(-2.0)


# =============================================================================
# Showdown — known fixtures
# =============================================================================
class TestShowdown:
    def _walk_to_showdown(
        self,
        private_cards: tuple[int, int, int, int],
        pending_board: tuple[int, int, int, int, int],
    ) -> HUNLState:
        """Drives a state from root preflop to showdown via check-check
        every round so chip contributions are blind-only (P0=2, P1=2
        after SB completes)."""
        s = _root_preflop(private_cards, pending_board)
        # Preflop: SB call, BB check.
        s = s.next_state(HUNLAction.CALL)
        s = s.next_state(HUNLAction.CALL)
        # Flop, Turn, River: each closes via check-check.
        for _ in range(3):
            s = s.next_state(HUNLAction.CALL)
            s = s.next_state(HUNLAction.CALL)
        assert s.is_terminal
        return s

    def test_p0_wins_showdown_with_pair_vs_high_card(self) -> None:
        """P0: pocket Aces; P1: 7-2 high. Board: K Q T 5 3 (no pair on board).
        P0 wins (pair of aces beats high card 7)."""
        s = self._walk_to_showdown(
            private_cards=(cid("As"), cid("Ah"), cid("7c"), cid("2d")),
            pending_board=(cid("Ks"), cid("Qd"), cid("Tc"), cid("5h"), cid("3d")),
        )
        # matched = 2. P0 wins. utility = +2.
        assert s.terminal_utility() == pytest.approx(+2.0)

    def test_p1_wins_showdown_with_higher_card(self) -> None:
        s = self._walk_to_showdown(
            private_cards=(cid("7c"), cid("2d"), cid("As"), cid("Ah")),
            pending_board=(cid("Ks"), cid("Qd"), cid("Tc"), cid("5h"), cid("3d")),
        )
        assert s.terminal_utility() == pytest.approx(-2.0)

    def test_tie_split_pot_returns_zero(self) -> None:
        """Both players play the board (royal flush on the table)."""
        s = self._walk_to_showdown(
            private_cards=(cid("2c"), cid("3d"), cid("4h"), cid("5s")),
            pending_board=(cid("As"), cid("Ks"), cid("Qs"), cid("Js"), cid("Ts")),
        )
        assert s.terminal_utility() == pytest.approx(0.0)


# =============================================================================
# Uncalled chip return — chip math correctness
# =============================================================================
class TestUncalledChipReturn:
    def test_sb_large_bet_bb_fold_only_blind_changes(self) -> None:
        """SB bets 50 preflop, BB folds. Uncalled 48 returned to SB.
        Net P0 change = -2 (lost only the BB blind)."""
        s = _root_preflop()
        s = s.next_state(HUNLAction.BET, bet_size=50)
        s = s.next_state(HUNLAction.FOLD)
        assert s.is_terminal
        assert s.terminal_utility() == pytest.approx(-2.0)

    def test_bb_raise_river_sb_fold_uncalled_returned(self) -> None:
        """Walk to river (both at 2 contribution), BB shoves on river,
        SB folds. Uncalled portion of BB's shove returned at terminal.
        Net P0 (BB) change = +2 (the SB's matched 2)."""
        s = _root_preflop()
        # Preflop check-check.
        s = s.next_state(HUNLAction.CALL)
        s = s.next_state(HUNLAction.CALL)
        # Flop check-check.
        s = s.next_state(HUNLAction.CALL)
        s = s.next_state(HUNLAction.CALL)
        # Turn check-check.
        s = s.next_state(HUNLAction.CALL)
        s = s.next_state(HUNLAction.CALL)
        # River: BB acts first.
        # BB shoves, SB folds.
        sizes = s.legal_bet_sizes()
        all_in = sizes[-1]
        s = s.next_state(HUNLAction.BET, bet_size=all_in)
        s = s.next_state(HUNLAction.FOLD)
        assert s.is_terminal
        # SB contribution before BB river shove: 2. matched = 2.
        # P0 (BB) wins matched → utility = +2.
        assert s.terminal_utility() == pytest.approx(+2.0)


# =============================================================================
# Helpers — _find_folder and _total_contributions probe
# =============================================================================
class TestHelpers:
    def test_find_folder_returns_none_at_showdown(self) -> None:
        s = TestShowdown()._walk_to_showdown(
            private_cards=(0, 1, 2, 3),
            pending_board=(4, 5, 6, 7, 8),
        )
        assert s._find_folder() is None

    def test_find_folder_identifies_p0(self) -> None:
        s = _root_preflop()
        s = s.next_state(HUNLAction.BET, bet_size=10)
        s = s.next_state(HUNLAction.FOLD)
        assert s._find_folder() == 0

    def test_find_folder_identifies_p1(self) -> None:
        s = _root_preflop().next_state(HUNLAction.FOLD)
        assert s._find_folder() == 1

    def test_total_contributions_at_root(self) -> None:
        s = _root_preflop()
        p0, p1 = s._total_contributions()
        assert (p0, p1) == (BB_BLIND_CHIPS_VALUE, SB_BLIND_CHIPS)

    def test_total_contributions_after_preflop_close(self) -> None:
        s = _root_preflop()
        s = s.next_state(HUNLAction.CALL)
        s = s.next_state(HUNLAction.CALL)
        # Preflop done, both at 2 chips. Round 1 empty.
        p0, p1 = s._total_contributions()
        assert (p0, p1) == (BB_BLIND_CHIPS_VALUE, BB_BLIND_CHIPS_VALUE)
