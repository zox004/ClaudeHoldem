"""HUNL state representation — Phase 4 M1.2 (data structure only).

This module defines the immutable data layer for HUNL:

- :class:`HUNLAction` — IntEnum for raw HUNL actions (FOLD / CALL /
  BET / NULL_PADDING). The ``BET`` size travels in a parallel array
  on :class:`HUNLState` (``round_bet_sizes``); ``NULL_PADDING`` is
  reserved for encode-time flattening padding (M1.5), not used inside
  per-round history tuples.
- :class:`HUNLState` — frozen dataclass holding the full HUNL infoset
  (cards, board, per-round betting history, pot, stacks, current
  player).

State **transitions** (legal_actions, next_state, round closure) are
intentionally NOT in this file — those land in M1.3. The split
mirrors mentor's request: M1.2 is pure data + invariants, M1.3 is
behaviour. Tests below the implementation verify only field shapes,
immutability, equality, and the post-init invariants.

Betting-history representation (M1.2.1 self-audit #16, 2026-04-26):
the original M1.2 design used a flat 40-slot tuple with NULL_PADDING
trailers. M1.3 round-closure logic — which is the next step — needs
to know how many actions were in the *current* round, and a flat
representation makes that derivation awkward (an auxiliary
round-start-indices field is the cheap fix; an explicit per-round
tuple of tuples is the cleanest fix). This module ships the cleaner
**per-round** representation, mirroring Phase 2 LeducState's
``round_history: tuple[tuple[LeducAction, ...], tuple[LeducAction,
...]]`` extended to four HUNL rounds. Encode-time flattening +
padding to 40 slots happens in M1.5's ``encode``; the mentor's
padding decision is preserved at the encoding boundary, not inside
the state.

Constants:

- ``HISTORY_MAX_LEN = 40`` — flattened betting-history length used at
  encode time (mentor padding decision: 10 actions × 4 rounds).
- ``BIG_BLIND_CHIPS = 2`` — internal chip granularity (1 BB = 2 chips
  so 0.5 BB = 1 chip exact, no fractional arithmetic on the hot path).
- ``STARTING_STACK_BB = 100`` — Slumbot / DecisionHoldem benchmark
  match (mentor sign-off Q on stack depth).
- ``STARTING_STACK_CHIPS = STARTING_STACK_BB * BIG_BLIND_CHIPS = 200``.

The bankroll invariant ``pot + stack_p0 + stack_p1 == 400`` is enforced
in :meth:`HUNLState.__post_init__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class HUNLAction(IntEnum):
    """Raw HUNL action label.

    BET carries a chip-size payload that lives on :class:`HUNLState`'s
    ``betting_sizes`` array; the action enum itself is just the
    category. NULL_PADDING marks unused slots in the fixed-length
    ``betting_history`` so encode-time logic can skip them without
    needing a separate length counter.
    """
    FOLD = 0
    CALL = 1
    BET = 2
    NULL_PADDING = 3


HISTORY_MAX_LEN: Final[int] = 40
BIG_BLIND_CHIPS: Final[int] = 2
STARTING_STACK_BB: Final[int] = 100
STARTING_STACK_CHIPS: Final[int] = STARTING_STACK_BB * BIG_BLIND_CHIPS

NUM_PRIVATE_CARDS_PER_PLAYER: Final[int] = 2
NUM_PRIVATE_CARDS: Final[int] = 4   # 2 × 2 players
NUM_BOARD_CARDS_FULL: Final[int] = 5
NUM_DECK_CARDS: Final[int] = 52


@dataclass(frozen=True, slots=True)
class HUNLState:
    """Immutable HUNL game state.

    Card-id convention is the project-wide ``rank * 4 + suit`` packing;
    see :mod:`poker_ai.games.hunl_hand_eval` for the rank/suit chars.

    Fields:
        private_cards: ``(p0_h1, p0_h2, p1_h1, p1_h2)`` — both players'
            hole cards. Acting-player perspective is applied at
            ``infoset_key`` computation (M1.3), not here.
        pending_board: full 5-card board, kept private until the
            corresponding round transition reveals it. Mirrors Phase 2
            LeducState's ``_pending_board``.
        board_cards: cards revealed so far. Length 0 / 3 / 4 / 5 for
            preflop / flop / turn / river respectively. Always a prefix
            of ``pending_board`` (the deal's pre-determined layout).
        round_history: 4-tuple of variable-length tuples — one per
            round (preflop, flop, turn, river). The ``current_round``
            tuple holds actions taken so far in the current round;
            future rounds are empty tuples. NULL_PADDING never appears
            here (it is for encode-time flattening only).
        round_bet_sizes: chip sizes parallel to ``round_history``;
            non-BET slots carry 0. Same 4-tuple shape.
        current_round: 0=preflop, 1=flop, 2=turn, 3=river.
        current_player: 0 or 1 — the player to act now. Stored
            explicitly because heads-up postflop reverses act-order
            relative to preflop (BB acts first postflop), and computing
            it from ``round_history`` alone would require a stateful
            walk that respects the SB-first-preflop / BB-first-postflop
            convention.
        stack_p0, stack_p1: chips remaining behind pot, in the
            BIG_BLIND_CHIPS=2 internal granularity.
        last_bet_size: chip size of the most recent raise increment
            in the current round (0 if no raise yet this round). Used
            by M1.3's ``legal_actions`` to enforce the no-limit
            min-raise rule (a raise increment must be ≥ the previous
            raise increment within the same round). Note: this is the
            **increment**, not the total bet; an all-in shorter than
            the min-raise increment does not "reopen" the betting.
        pot: chips already committed to the middle this hand.

    Bankroll invariant (heads-up, no side pots): ``pot + stack_p0 +
    stack_p1 == 2 × STARTING_STACK_CHIPS = 400``. Any all-in resolution
    that returns excess to the all-in player happens at
    ``terminal_utility`` time (M1.4), so during a non-terminal hand the
    invariant always holds.
    """

    private_cards: tuple[int, int, int, int]
    pending_board: tuple[int, int, int, int, int]
    board_cards: tuple[int, ...]
    round_history: tuple[
        tuple[HUNLAction, ...],
        tuple[HUNLAction, ...],
        tuple[HUNLAction, ...],
        tuple[HUNLAction, ...],
    ]
    round_bet_sizes: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    current_round: int
    current_player: int
    stack_p0: int
    stack_p1: int
    last_bet_size: int
    pot: int

    # --------------------------------------------------------- validation

    def __post_init__(self) -> None:
        # --- private_cards: 4 distinct ints in 0..51
        if len(self.private_cards) != NUM_PRIVATE_CARDS:
            raise ValueError(
                f"private_cards must have {NUM_PRIVATE_CARDS} entries, "
                f"got {len(self.private_cards)}"
            )
        for c in self.private_cards:
            if not 0 <= c < NUM_DECK_CARDS:
                raise ValueError(
                    f"private_cards entry {c} out of range 0..{NUM_DECK_CARDS-1}"
                )
        if len(set(self.private_cards)) != NUM_PRIVATE_CARDS:
            raise ValueError("private_cards entries must be distinct")

        # --- pending_board: 5 distinct ints in 0..51, disjoint from privates
        if len(self.pending_board) != NUM_BOARD_CARDS_FULL:
            raise ValueError(
                f"pending_board must have {NUM_BOARD_CARDS_FULL} entries, "
                f"got {len(self.pending_board)}"
            )
        for c in self.pending_board:
            if not 0 <= c < NUM_DECK_CARDS:
                raise ValueError(
                    f"pending_board entry {c} out of range 0..{NUM_DECK_CARDS-1}"
                )
        if len(set(self.pending_board)) != NUM_BOARD_CARDS_FULL:
            raise ValueError("pending_board entries must be distinct")
        if set(self.pending_board) & set(self.private_cards):
            raise ValueError(
                "pending_board and private_cards must be disjoint"
            )

        # --- board_cards: prefix of pending_board, length ∈ {0, 3, 4, 5}
        if len(self.board_cards) not in (0, 3, 4, 5):
            raise ValueError(
                f"board_cards length must be one of {{0, 3, 4, 5}}, "
                f"got {len(self.board_cards)}"
            )
        if tuple(self.board_cards) != self.pending_board[: len(self.board_cards)]:
            raise ValueError(
                "board_cards must be a prefix of pending_board"
            )

        # --- round_history / round_bet_sizes: per-round structure
        if len(self.round_history) != 4:
            raise ValueError(
                f"round_history must have 4 round-tuples, "
                f"got {len(self.round_history)}"
            )
        if len(self.round_bet_sizes) != 4:
            raise ValueError(
                f"round_bet_sizes must have 4 round-tuples, "
                f"got {len(self.round_bet_sizes)}"
            )
        for r_idx in range(4):
            actions = self.round_history[r_idx]
            sizes = self.round_bet_sizes[r_idx]
            if len(actions) != len(sizes):
                raise ValueError(
                    f"round_history[{r_idx}] length {len(actions)} ≠ "
                    f"round_bet_sizes[{r_idx}] length {len(sizes)}"
                )
            for a in actions:
                if not isinstance(a, HUNLAction):
                    raise ValueError(
                        f"round_history[{r_idx}] entries must be HUNLAction, "
                        f"got {type(a)}"
                    )
                if a == HUNLAction.NULL_PADDING:
                    raise ValueError(
                        "NULL_PADDING is reserved for encode-time flattening "
                        "and must not appear inside round_history"
                    )
            for sz in sizes:
                if sz < 0:
                    raise ValueError(
                        f"round_bet_sizes[{r_idx}] entries must be ≥ 0, "
                        f"got {sz}"
                    )
        # Future rounds (> current_round) must be empty — actions cannot
        # exist before the round opens.
        for r_idx in range(self.current_round + 1, 4):
            if self.round_history[r_idx]:
                raise ValueError(
                    f"round_history[{r_idx}] (future round) must be empty "
                    f"while current_round={self.current_round}"
                )

        # --- current_round, current_player ranges
        if self.current_round not in (0, 1, 2, 3):
            raise ValueError(
                f"current_round must be 0..3, got {self.current_round}"
            )
        if self.current_player not in (0, 1):
            raise ValueError(
                f"current_player must be 0 or 1, got {self.current_player}"
            )

        # --- chip non-negativity + bankroll invariant
        if self.stack_p0 < 0:
            raise ValueError(f"stack_p0 must be ≥ 0, got {self.stack_p0}")
        if self.stack_p1 < 0:
            raise ValueError(f"stack_p1 must be ≥ 0, got {self.stack_p1}")
        if self.last_bet_size < 0:
            raise ValueError(
                f"last_bet_size must be ≥ 0, got {self.last_bet_size}"
            )
        if self.pot < 0:
            raise ValueError(f"pot must be ≥ 0, got {self.pot}")
        total_bankroll = self.pot + self.stack_p0 + self.stack_p1
        expected = 2 * STARTING_STACK_CHIPS
        if total_bankroll != expected:
            raise ValueError(
                f"bankroll invariant: pot + stack_p0 + stack_p1 must equal "
                f"{expected} (=2×{STARTING_STACK_CHIPS}), got {total_bankroll}"
            )

        # --- board_cards length consistent with current_round
        # preflop: 0 cards, flop: 3, turn: 4, river: 5
        expected_board_lens = {0: 0, 1: 3, 2: 4, 3: 5}
        if len(self.board_cards) != expected_board_lens[self.current_round]:
            raise ValueError(
                f"board_cards length {len(self.board_cards)} inconsistent "
                f"with current_round {self.current_round} (expected "
                f"{expected_board_lens[self.current_round]})"
            )
