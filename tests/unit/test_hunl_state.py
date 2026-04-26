"""Unit tests for HUNL data structures (Phase 4 M1.2).

Scope: HUNLAction enum + HUNLState frozen dataclass + post-init
invariants. State **transitions** (legal_actions, next_state) are
M1.3, not covered here.
"""

from __future__ import annotations

import dataclasses

import pytest

from poker_ai.games.hunl_state import (
    BIG_BLIND_CHIPS,
    HISTORY_MAX_LEN,
    HUNLAction,
    HUNLState,
    NUM_BOARD_CARDS_FULL,
    NUM_PRIVATE_CARDS,
    STARTING_STACK_BB,
    STARTING_STACK_CHIPS,
)


# =============================================================================
# Module-level constants
# =============================================================================
class TestModuleConstants:
    def test_betting_history_max_len_is_40(self) -> None:
        """Mentor's padding decision: 10 actions × 4 rounds."""
        assert HISTORY_MAX_LEN == 40

    def test_starting_stack_is_100_bb(self) -> None:
        """Heads-up cash standard (Slumbot / DecisionHoldem benchmark)."""
        assert STARTING_STACK_BB == 100

    def test_chip_granularity_doubled_big_blind(self) -> None:
        """1 BB = 2 chips so 0.5 BB (small blind) = 1 chip exact."""
        assert BIG_BLIND_CHIPS == 2
        assert STARTING_STACK_CHIPS == STARTING_STACK_BB * BIG_BLIND_CHIPS
        assert STARTING_STACK_CHIPS == 200


# =============================================================================
# HUNLAction enum
# =============================================================================
class TestHUNLAction:
    def test_four_members(self) -> None:
        """FOLD, CALL, BET, NULL_PADDING — exactly four."""
        assert len(list(HUNLAction)) == 4

    def test_member_values(self) -> None:
        assert HUNLAction.FOLD.value == 0
        assert HUNLAction.CALL.value == 1
        assert HUNLAction.BET.value == 2
        assert HUNLAction.NULL_PADDING.value == 3


# =============================================================================
# HUNLState — fixture builder
# =============================================================================
def _root_state(
    *,
    private_cards: tuple[int, int, int, int] = (0, 1, 2, 3),
    pending_board: tuple[int, int, int, int, int] = (4, 5, 6, 7, 8),
    board_cards: tuple[int, ...] = (),
    current_round: int = 0,
    current_player: int = 0,
    stack_p0: int = STARTING_STACK_CHIPS,
    stack_p1: int = STARTING_STACK_CHIPS,
    last_bet_size: int = 0,
    pot: int = 0,
    betting_history: tuple[HUNLAction, ...] = tuple(
        [HUNLAction.NULL_PADDING] * HISTORY_MAX_LEN
    ),
    betting_sizes: tuple[int, ...] = tuple([0] * HISTORY_MAX_LEN),
) -> HUNLState:
    """Builds a valid HUNLState with the given overrides."""
    return HUNLState(
        private_cards=private_cards,
        pending_board=pending_board,
        board_cards=board_cards,
        betting_history=betting_history,
        betting_sizes=betting_sizes,
        current_round=current_round,
        current_player=current_player,
        stack_p0=stack_p0,
        stack_p1=stack_p1,
        last_bet_size=last_bet_size,
        pot=pot,
    )


# =============================================================================
# HUNLState — happy-path construction
# =============================================================================
class TestHUNLStateConstruction:
    def test_root_state_constructs(self) -> None:
        s = _root_state()
        assert s.current_round == 0
        assert s.pot == 0
        assert s.stack_p0 == STARTING_STACK_CHIPS
        assert s.stack_p1 == STARTING_STACK_CHIPS

    def test_flop_state_constructs(self) -> None:
        """Round 1 (flop) — 3 board cards revealed."""
        s = _root_state(
            board_cards=(4, 5, 6),
            current_round=1,
            stack_p0=STARTING_STACK_CHIPS - 2,
            stack_p1=STARTING_STACK_CHIPS - 2,
            pot=4,
        )
        assert s.current_round == 1
        assert len(s.board_cards) == 3

    def test_turn_state_constructs(self) -> None:
        s = _root_state(
            board_cards=(4, 5, 6, 7),
            current_round=2,
            stack_p0=STARTING_STACK_CHIPS - 4,
            stack_p1=STARTING_STACK_CHIPS - 4,
            pot=8,
        )
        assert len(s.board_cards) == 4

    def test_river_state_constructs(self) -> None:
        s = _root_state(
            board_cards=(4, 5, 6, 7, 8),
            current_round=3,
            stack_p0=STARTING_STACK_CHIPS - 8,
            stack_p1=STARTING_STACK_CHIPS - 8,
            pot=16,
        )
        assert len(s.board_cards) == 5


# =============================================================================
# HUNLState — frozen + immutable + equality
# =============================================================================
class TestHUNLStateImmutability:
    def test_state_is_frozen(self) -> None:
        s = _root_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.pot = 100  # type: ignore[misc]

    def test_state_uses_slots(self) -> None:
        """slots=True prevents adding arbitrary attributes — the frozen
        dataclass __setattr__ takes the first crack at any assignment
        and raises FrozenInstanceError, but the slot mechanism is also
        verified by checking that no __dict__ attribute is created."""
        s = _root_state()
        # frozen + slots: the dataclass __setattr__ blocks assignment.
        with pytest.raises(
            (AttributeError, TypeError, dataclasses.FrozenInstanceError)
        ):
            s.new_attribute = 42  # type: ignore[attr-defined]
        # slots=True: no __dict__.
        assert not hasattr(s, "__dict__")

    def test_states_with_same_fields_are_equal(self) -> None:
        a = _root_state()
        b = _root_state()
        assert a == b

    def test_states_with_different_fields_are_unequal(self) -> None:
        a = _root_state()
        b = _root_state(pot=4, stack_p0=STARTING_STACK_CHIPS - 2,
                        stack_p1=STARTING_STACK_CHIPS - 2)
        assert a != b

    def test_dataclass_replace_returns_new_instance(self) -> None:
        a = _root_state()
        b = dataclasses.replace(
            a, pot=4,
            stack_p0=STARTING_STACK_CHIPS - 2,
            stack_p1=STARTING_STACK_CHIPS - 2,
        )
        assert a is not b
        assert a.pot == 0
        assert b.pot == 4


# =============================================================================
# HUNLState — invariants raise on violation
# =============================================================================
class TestHUNLStateInvariants:
    def test_private_cards_must_be_4(self) -> None:
        with pytest.raises(ValueError, match="private_cards"):
            _root_state(private_cards=(0, 1, 2))  # type: ignore[arg-type]

    def test_private_cards_distinct(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            _root_state(private_cards=(0, 0, 1, 2))

    def test_private_cards_in_deck_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _root_state(private_cards=(0, 1, 2, 52))

    def test_pending_board_must_be_5(self) -> None:
        with pytest.raises(ValueError, match="pending_board"):
            _root_state(pending_board=(4, 5, 6, 7))  # type: ignore[arg-type]

    def test_pending_board_disjoint_from_privates(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            _root_state(pending_board=(0, 5, 6, 7, 8))   # 0 is in private_cards

    def test_board_cards_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="board_cards length"):
            _root_state(board_cards=(4, 5))   # length 2 not allowed

    def test_board_cards_must_be_prefix_of_pending(self) -> None:
        with pytest.raises(ValueError, match="prefix"):
            _root_state(
                board_cards=(4, 5, 9),   # 9 ≠ pending_board[2]=6
                current_round=1,
            )

    def test_board_cards_inconsistent_with_round(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            # current_round=0 (preflop) requires 0 board cards.
            _root_state(board_cards=(4, 5, 6), current_round=0)

    def test_betting_history_length_must_be_40(self) -> None:
        short_history = tuple([HUNLAction.NULL_PADDING] * 5)
        with pytest.raises(ValueError, match="betting_history"):
            _root_state(betting_history=short_history)

    def test_betting_sizes_length_must_be_40(self) -> None:
        with pytest.raises(ValueError, match="betting_sizes"):
            _root_state(betting_sizes=tuple([0] * 5))

    def test_betting_sizes_non_negative(self) -> None:
        bad = list([0] * HISTORY_MAX_LEN)
        bad[3] = -1
        with pytest.raises(ValueError, match="betting_sizes"):
            _root_state(betting_sizes=tuple(bad))

    def test_current_round_in_range(self) -> None:
        with pytest.raises(ValueError, match="current_round"):
            _root_state(current_round=4)

    def test_current_player_in_range(self) -> None:
        with pytest.raises(ValueError, match="current_player"):
            _root_state(current_player=2)

    def test_stack_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="stack_p0"):
            _root_state(
                stack_p0=-1,
                stack_p1=STARTING_STACK_CHIPS,
                pot=0,
            )

    def test_bankroll_invariant_violation(self) -> None:
        """pot + stack_p0 + stack_p1 must equal 2 * STARTING_STACK_CHIPS."""
        with pytest.raises(ValueError, match="bankroll invariant"):
            _root_state(
                stack_p0=STARTING_STACK_CHIPS,
                stack_p1=STARTING_STACK_CHIPS,
                pot=10,   # total = 410, not 400
            )
