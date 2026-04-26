"""Unit tests for HUNL hand evaluator (Phase 4 M1.1).

Coverage:
- card_id ↔ treys conversion is bijective on 0..51
- evaluate_seven correctness on hand-category fixtures
- compare_hands sign + tie semantics
- 1 000-pair random cross-check: treys ordering matches the naive
  enumerate-21 evaluator
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.hunl_hand_eval import (
    card_id_to_treys,
    cards_to_treys,
    compare_hands,
    cross_check_random_hands,
    evaluate_seven,
    naive_evaluate_seven,
)

# Helper: build card_id from rank/suit char.
_RANK_TO_INT = {c: i for i, c in enumerate("23456789TJQKA")}
_SUIT_TO_INT = {c: i for i, c in enumerate("cdhs")}


def cid(s: str) -> int:
    """e.g. cid('As') = ace of spades = 12*4 + 3 = 51."""
    return _RANK_TO_INT[s[0]] * 4 + _SUIT_TO_INT[s[1]]


# =============================================================================
# Card-id ↔ treys conversion bijectivity
# =============================================================================
class TestCardIdConversion:
    def test_all_52_card_ids_are_distinct(self) -> None:
        treys_ids = [card_id_to_treys(c) for c in range(52)]
        assert len(set(treys_ids)) == 52

    def test_invalid_card_id_raises(self) -> None:
        with pytest.raises(ValueError, match="card_id"):
            card_id_to_treys(-1)
        with pytest.raises(ValueError, match="card_id"):
            card_id_to_treys(52)

    def test_known_mapping_aces(self) -> None:
        """Treys uses string-constructor format like 'As' = ace of spades.
        Our card_id 51 = 12 * 4 + 3 → rank 12 (A), suit 3 (s). The
        roundtrip via card_id_to_treys should equal Card.new('As')."""
        from treys import Card
        assert card_id_to_treys(51) == int(Card.new("As"))   # A♠
        assert card_id_to_treys(0) == int(Card.new("2c"))    # 2♣
        assert card_id_to_treys(8) == int(Card.new("4c"))    # 4♣

    def test_cards_to_treys_iterates(self) -> None:
        out = cards_to_treys([0, 1, 2, 3])
        assert len(out) == 4
        assert all(isinstance(x, int) for x in out)


# =============================================================================
# evaluate_seven — fixtures for each hand category
# =============================================================================
class TestEvaluateSeven:
    def test_input_validation_hole_count(self) -> None:
        with pytest.raises(ValueError, match="hole"):
            evaluate_seven([cid("As")], [cid("Kh"), cid("Qd"), cid("Js"), cid("Tc"), cid("2h")])

    def test_input_validation_board_count(self) -> None:
        with pytest.raises(ValueError, match="board"):
            evaluate_seven([cid("As"), cid("Kh")], [cid("Qd"), cid("Js")])

    def test_duplicate_cards_rejected(self) -> None:
        # As appears twice.
        with pytest.raises(ValueError, match="distinct"):
            evaluate_seven(
                [cid("As"), cid("Kh")],
                [cid("As"), cid("Js"), cid("Tc"), cid("2h"), cid("3d")],
            )

    def test_royal_flush_is_strongest(self) -> None:
        rank = evaluate_seven(
            [cid("As"), cid("Ks")],
            [cid("Qs"), cid("Js"), cid("Ts"), cid("2h"), cid("3d")],
        )
        assert rank == 1   # treys: 1 = royal flush

    def test_pair_weaker_than_straight(self) -> None:
        pair = evaluate_seven(
            [cid("As"), cid("Ah")],
            [cid("Kc"), cid("Qd"), cid("Tc"), cid("2h"), cid("3d")],
        )
        straight = evaluate_seven(
            [cid("As"), cid("Kh")],
            [cid("Qd"), cid("Js"), cid("Tc"), cid("2h"), cid("3d")],
        )
        assert straight < pair  # lower = stronger

    def test_high_card_is_weakest_category(self) -> None:
        hi = evaluate_seven(
            [cid("As"), cid("Kh")],
            [cid("Qd"), cid("Js"), cid("9c"), cid("2h"), cid("3d")],
        )
        # Should be in the high-card range (treys: 6186..7462).
        assert 6186 <= hi <= 7462


# =============================================================================
# compare_hands — sign + tie semantics
# =============================================================================
class TestCompareHands:
    def test_player_a_wins_returns_plus_one(self) -> None:
        # A has royal flush, B has high card.
        sgn = compare_hands(
            hole_a=[cid("As"), cid("Ks")],
            hole_b=[cid("2c"), cid("3d")],
            board=[cid("Qs"), cid("Js"), cid("Ts"), cid("8h"), cid("4d")],
        )
        assert sgn == +1

    def test_player_b_wins_returns_minus_one(self) -> None:
        sgn = compare_hands(
            hole_a=[cid("2c"), cid("3d")],
            hole_b=[cid("As"), cid("Ks")],
            board=[cid("Qs"), cid("Js"), cid("Ts"), cid("8h"), cid("4d")],
        )
        assert sgn == -1

    def test_chop_returns_zero_when_board_plays(self) -> None:
        """Both players play the board (royal flush on the table).
        Best 5 from each player's 7 = the same 5 (royal flush).
        Chop expected."""
        # Royal flush on the board.
        board = [cid("As"), cid("Ks"), cid("Qs"), cid("Js"), cid("Ts")]
        sgn = compare_hands(
            hole_a=[cid("2c"), cid("3d")],
            hole_b=[cid("4h"), cid("5d")],
            board=board,
        )
        assert sgn == 0


# =============================================================================
# 1 000-pair random cross-check vs naive enumerate-21
# =============================================================================
class TestRandomCrossCheck:
    def test_treys_vs_naive_orderings_match_1000_pairs(self) -> None:
        """The mentor-required cross-check: treys's ordering of 7-card
        sets must match a from-scratch enumeration on 1 000 random
        hand pairs. Mismatch implies a bug in either the treys wrapper
        or the naive scorer."""
        rng = np.random.default_rng(seed=42)
        matches, total = cross_check_random_hands(1000, rng)
        # Allow zero mismatches — both evaluators are deterministic and
        # should agree on every comparable pair.
        assert matches == total, (
            f"treys vs naive mismatch on {total - matches}/{total} pairs"
        )

    def test_naive_evaluator_runs_independently(self) -> None:
        """Sanity: naive evaluator returns finite tuples for any
        legal 7-card input."""
        rng = np.random.default_rng(seed=42)
        for _ in range(30):
            deck = rng.permutation(52)
            hole = deck[0:2].tolist()
            board = deck[2:7].tolist()
            score = naive_evaluate_seven(hole, board)
            assert isinstance(score, tuple)
            assert len(score) >= 1
