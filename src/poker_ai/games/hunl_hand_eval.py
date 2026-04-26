"""HUNL 7-card best-5 hand evaluator (Phase 4 M1.1).

Wraps the ``treys`` library's evaluator with a card-id ↔ treys-id
adapter so that downstream HUNL code can use plain integer card ids
``0..51`` (rank × 4 + suit, our internal encoding) without ever
touching treys's Card-object API directly.

Design notes:

- **Card id convention** (HUNL-internal): ``card_id = rank * 4 +
  suit``, with ``rank ∈ 0..12`` (2..A) and ``suit ∈ 0..3`` (clubs,
  diamonds, hearts, spades, in alphabetical order). This matches the
  Phase 2 LeducPoker pattern of dense integer card ids; we just
  extend the range from 0..5 (Leduc) to 0..51 (HUNL).
- **Treys's convention**: 32-bit packed ints with rank in `0xF0000000`
  and suit in `0x00FFF000` (prime encoding). We adapt via the
  ``treys.Card.new("As")``-style string constructor, which is the
  documented stable interface.
- **Lower rank = stronger hand** (treys convention; royal flush = 1,
  high-card 7-2 ≈ 7 462). Our wrapper preserves this convention so
  comparisons in :func:`compare_hands` are direct integer compares.

This module exposes two functions:

- :func:`evaluate_seven` — given (2 hole, 5 board) cards, returns
  the treys rank (1 = strongest, 7 462 = weakest).
- :func:`compare_hands` — given two (2 hole, 5 board) sets, returns
  ``+1`` if hand A wins, ``-1`` if B wins, ``0`` on tie.

A naive cross-check evaluator (enumerate all C(7,5)=21 sub-hands and
score each via standard rules) is provided in :func:`naive_evaluate_seven`
purely for testing; it is not used on the MCCFR hot path.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from treys import Card, Evaluator  # type: ignore[import-untyped]

# Single shared evaluator — treys's Evaluator is stateless wrt the
# input but allocates lookup tables on construction. Reuse the table.
_EVALUATOR = Evaluator()

# Rank chars in treys order (2..A). Suit chars in treys lowercase.
_RANK_CHARS: tuple[str, ...] = (
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
)
_SUIT_CHARS: tuple[str, ...] = ("c", "d", "h", "s")


def card_id_to_treys(card_id: int) -> int:
    """Maps our HUNL card_id ∈ 0..51 to a treys-packed Card int.

    ``card_id = rank * 4 + suit`` with ``rank ∈ 0..12`` (2..A,
    treys's order) and ``suit ∈ 0..3`` (c/d/h/s).
    """
    if not 0 <= card_id < 52:
        raise ValueError(f"card_id must be in 0..51, got {card_id}")
    rank = card_id // 4
    suit = card_id % 4
    return int(Card.new(_RANK_CHARS[rank] + _SUIT_CHARS[suit]))


def cards_to_treys(card_ids: Sequence[int]) -> list[int]:
    """Vectorises :func:`card_id_to_treys` over an iterable."""
    return [card_id_to_treys(c) for c in card_ids]


def evaluate_seven(hole: Sequence[int], board: Sequence[int]) -> int:
    """Returns the treys rank of the best 5-card hand from (hole, board).

    Lower is stronger: 1 = royal flush, 7 462 = 7-2 high. This is the
    canonical treys convention and the wrapper preserves it.
    """
    if len(hole) != 2:
        raise ValueError(f"hole must have 2 cards, got {len(hole)}")
    if len(board) != 5:
        raise ValueError(f"board must have 5 cards, got {len(board)}")
    all_ids = list(hole) + list(board)
    if len(set(all_ids)) != 7:
        raise ValueError(
            f"7 cards must be distinct, got {len(set(all_ids))} unique"
        )
    h = cards_to_treys(hole)
    b = cards_to_treys(board)
    return int(_EVALUATOR.evaluate(b, h))


def compare_hands(
    hole_a: Sequence[int],
    hole_b: Sequence[int],
    board: Sequence[int],
) -> int:
    """+1 if A wins, -1 if B wins, 0 on tie.

    Both players see the same 5-card board; evaluator picks the best
    5 from each player's (2 hole + 5 board) = 7-card set independently.
    """
    rank_a = evaluate_seven(hole_a, board)
    rank_b = evaluate_seven(hole_b, board)
    # Lower rank = stronger.
    if rank_a < rank_b:
        return +1
    if rank_a > rank_b:
        return -1
    return 0


# =============================================================================
# Naive cross-check evaluator — testing only, not on hot path
# =============================================================================
def naive_evaluate_seven(
    hole: Sequence[int], board: Sequence[int]
) -> tuple[int, ...]:
    """Independent re-evaluation: enumerate all C(7,5)=21 sub-hands,
    score each via :func:`_score_five_card_hand`, return the best.

    "Best" = the smallest rank under our naive scoring; the absolute
    rank values do NOT match treys (different lookup table), but the
    **ordering** of any two 7-card sets is invariant. Cross-check
    tests verify ``sign(naive_a - naive_b) == sign(treys_a - treys_b)``
    on random 7-card pairs.
    """
    if len(hole) != 2:
        raise ValueError(f"hole must have 2 cards, got {len(hole)}")
    if len(board) != 5:
        raise ValueError(f"board must have 5 cards, got {len(board)}")
    all7 = list(hole) + list(board)
    if len(set(all7)) != 7:
        raise ValueError(
            f"7 cards must be distinct, got {len(set(all7))} unique"
        )
    best = None
    for combo in combinations(all7, 5):
        s = _score_five_card_hand(combo)
        if best is None or s < best:
            best = s
    assert best is not None
    return best


def _score_five_card_hand(cards: Sequence[int]) -> tuple[int, ...]:
    """Returns a tuple comparable with normal int comparison (lower =
    stronger), ordered by hand category then by rank tiebreakers.

    Categories (low-int = strong):
        0 straight flush
        1 four of a kind
        2 full house
        3 flush
        4 straight
        5 three of a kind
        6 two pair
        7 one pair
        8 high card

    Ties within a category are broken by ranks descending (we negate
    for "lower=stronger" ordering).
    """
    ranks = sorted([c // 4 for c in cards], reverse=True)  # 0..12, A=12
    suits = [c % 4 for c in cards]

    is_flush = len(set(suits)) == 1
    rank_set = sorted(set(ranks), reverse=True)
    is_straight = (
        len(rank_set) == 5
        and (rank_set[0] - rank_set[4] == 4)
    )
    # Wheel A-2-3-4-5 — A treated as low.
    if not is_straight and set(rank_set) == {12, 0, 1, 2, 3}:
        is_straight = True
        # Reorder so A is low: ranks for tiebreak become [3, 2, 1, 0, -1].
        ranks = [3, 2, 1, 0, -1]
        rank_set = [3, 2, 1, 0, -1]

    counts = sorted(
        ((ranks.count(r), r) for r in set(ranks)),
        key=lambda t: (-t[0], -t[1]),
    )
    pattern = tuple(c for c, _ in counts)

    # Negate ranks for comparison so that "lower tuple = stronger".
    rank_break = tuple(-r for r, _ in [(rank, _) for _, rank in counts])

    if is_straight and is_flush:
        return (0, -rank_set[0])
    if pattern == (4, 1):
        return (1,) + rank_break
    if pattern == (3, 2):
        return (2,) + rank_break
    if is_flush:
        return (3,) + tuple(-r for r in ranks)
    if is_straight:
        return (4, -rank_set[0])
    if pattern == (3, 1, 1):
        return (5,) + rank_break
    if pattern == (2, 2, 1):
        return (6,) + rank_break
    if pattern == (2, 1, 1, 1):
        return (7,) + rank_break
    return (8,) + tuple(-r for r in ranks)


# =============================================================================
# Cross-check helper — used by tests to validate treys vs naive ordering
# =============================================================================
def cross_check_random_hands(
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Runs ``n_pairs`` independent random 7-card-pair comparisons.

    For each, computes the comparison sign via both treys and the naive
    scorer. Returns ``(matches, total)``: pairs where the orderings
    agreed, and the total number of pairs.

    A perfectly correct treys wrapper has ``matches == total`` modulo
    a small number of exact-tie cases where both rankers must agree on
    the tie itself (treys's rank == naive's tuple equal); the test asserts
    the percent match.
    """
    matches = 0
    for _ in range(n_pairs):
        deck = rng.permutation(52)
        # P1 hole, P2 hole, board
        hole_a = deck[0:2].tolist()
        hole_b = deck[2:4].tolist()
        board = deck[4:9].tolist()
        sign_treys = compare_hands(hole_a, hole_b, board)
        rank_a_naive = naive_evaluate_seven(hole_a, board)
        rank_b_naive = naive_evaluate_seven(hole_b, board)
        if rank_a_naive < rank_b_naive:
            sign_naive = +1
        elif rank_a_naive > rank_b_naive:
            sign_naive = -1
        else:
            sign_naive = 0
        if sign_treys == sign_naive:
            matches += 1
    return matches, n_pairs
