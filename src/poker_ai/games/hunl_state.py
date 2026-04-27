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
  Slumbot adapter (M4.1) multiplies by 50 to match the Slumbot HTTP
  API's BB=100 absolute chip convention.
- ``STARTING_STACK_BB = 200`` — Slumbot / ACPC Doyle's Game standard
  (M4.0 reconfigure, mentor #9 self-correction: prior 100 BB design
  was based on incorrect "Slumbot uses 100 BB" fact statement;
  Slumbot 2019 / ACPC 2017+ all use 200 BB).
- ``STARTING_STACK_CHIPS = STARTING_STACK_BB * BIG_BLIND_CHIPS = 400``.

The bankroll invariant ``pot + stack_p0 + stack_p1 == 800`` is
enforced in :meth:`HUNLState.__post_init__` (= 2 × STARTING_STACK_CHIPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from poker_ai.games.hunl_hand_eval import compare_hands


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
STARTING_STACK_BB: Final[int] = 200
STARTING_STACK_CHIPS: Final[int] = STARTING_STACK_BB * BIG_BLIND_CHIPS   # 400

NUM_PRIVATE_CARDS_PER_PLAYER: Final[int] = 2
NUM_PRIVATE_CARDS: Final[int] = 4   # 2 × 2 players
NUM_BOARD_CARDS_FULL: Final[int] = 5
NUM_DECK_CARDS: Final[int] = 52

# Heads-up blind sizes in chips (with BIG_BLIND_CHIPS=2 granularity).
# Convention (mentor 2026-04-26): player 0 = BB (acts first postflop),
# player 1 = SB (acts first preflop).
SB_BLIND_CHIPS: Final[int] = 1   # 0.5 BB
BB_BLIND_CHIPS_VALUE: Final[int] = 2   # 1 BB; alias for clarity at use sites
NUM_ROUNDS: Final[int] = 4
# Board cards revealed after each round transition: round 0 → 3, 1 → 4, 2 → 5.
_BOARD_LEN_AT_ROUND: Final[dict[int, int]] = {0: 0, 1: 3, 2: 4, 3: 5}


def first_actor_for_round(round_idx: int) -> int:
    """Heads-up act-order rule (mentor 2026-04-26):
    preflop SB (player 1) acts first; postflop BB (player 0) acts first."""
    if round_idx == 0:
        return 1   # SB
    return 0   # BB


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
        last_raise_increment: chip size of the most recent raise increment
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
    last_raise_increment: int
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
        if self.last_raise_increment < 0:
            raise ValueError(
                f"last_raise_increment must be ≥ 0, got {self.last_raise_increment}"
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

    # ============================================================== helpers
    # Internal computations used by both legal_actions/next_state and the
    # is_terminal property. Pure functions of the immutable state.

    def _round_contributions(self, round_idx: int) -> tuple[int, int]:
        """Returns ``(p0_chip_contrib, p1_chip_contrib)`` to the round
        identified by ``round_idx``, including blinds where applicable.

        Walks the round's actions linearly, replaying chip commitments
        in act-order. Used by :meth:`legal_actions` to compute
        ``to_call`` and by :meth:`next_state` to detect round closure.
        """
        if round_idx == 0:
            p0, p1 = BB_BLIND_CHIPS_VALUE, SB_BLIND_CHIPS
        else:
            p0, p1 = 0, 0
        matched = max(p0, p1)
        actor = first_actor_for_round(round_idx)
        for action, size in zip(
            self.round_history[round_idx],
            self.round_bet_sizes[round_idx],
            strict=True,
        ):
            if action == HUNLAction.FOLD:
                # Fold ends the round; remaining contributions stay frozen.
                break
            if action == HUNLAction.CALL:
                # Actor matches the current matched amount.
                if actor == 0:
                    p0 = matched
                else:
                    p1 = matched
            elif action == HUNLAction.BET:
                # ``size`` records the actor's *total round contribution
                # after this BET*. matched updates to that.
                if actor == 0:
                    p0 = size
                else:
                    p1 = size
                matched = size
            actor = 1 - actor
        return p0, p1

    def _is_round_closed(self, round_idx: int) -> bool:
        """A round is closed when both players have acted and bets are
        matched, OR a player folded, OR an all-in is matched/called.

        Heads-up specifics:
        - Preflop: BB has the option to raise after SB completes; round
          closes only after BB acts (CALL=check, BET=raise) on a matched
          pot. SB CALL alone does NOT close preflop because BB has not
          yet acted in the betting round.
        - Postflop: round closes when both have acted and matched.
        - All-in: a CALL of an all-in BET closes the round (no further
          betting possible).
        """
        actions = self.round_history[round_idx]
        if not actions:
            return False
        if actions[-1] == HUNLAction.FOLD:
            return True
        # Bets matched check.
        p0, p1 = self._round_contributions(round_idx)
        if p0 != p1:
            return False
        # Both players must have acted in this round (no one is sitting on
        # an unanswered bet). Heads-up: at least 2 actions in the round.
        if len(actions) < 2:
            return False
        # Preflop limp + BB option special case: SB calls then BB option
        # is required. So preflop needs ≥ 2 actions for the round to
        # close (SB action + BB action). The check above covers it.
        # All-in detection: if the last BET drove the actor's stack to
        # 0 and the opponent CALL'd, that CALL closes the round.
        return True

    @property
    def is_terminal(self) -> bool:
        """True if any player folded, or river closed normally."""
        for r_idx in range(NUM_ROUNDS):
            actions = self.round_history[r_idx]
            if actions and actions[-1] == HUNLAction.FOLD:
                return True
        # River close.
        if self.current_round == NUM_ROUNDS - 1 and self._is_round_closed(
            NUM_ROUNDS - 1
        ):
            return True
        return False

    # ========================================================= legal action
    # Returns the set of HUNLActions the current player may play. Bet sizes
    # are returned separately by :meth:`legal_bet_sizes`. Mentor's two-layer
    # contract (option-a-for-raw + option-c-for-abstracted): raw HUNLState
    # exposes both methods independently.

    def legal_actions(self) -> tuple[HUNLAction, ...]:
        """Tuple of FOLD/CALL/BET subsetted by this state's legality.

        Heads-up no-side-pot:
        - FOLD legal only when there is something to call (``to_call > 0``)
        - CALL always legal at non-terminal nodes (matches opponent's
          bet, or "checks" with 0 chips when matched)
        - BET legal when the actor still has chips AND the opponent
          still has chips (no one to raise to once opponent is all-in)

        Terminal states return ``()``.
        """
        if self.is_terminal:
            return ()
        actor = self.current_player
        actor_stack = self.stack_p0 if actor == 0 else self.stack_p1
        opp_stack = self.stack_p1 if actor == 0 else self.stack_p0
        p0, p1 = self._round_contributions(self.current_round)
        actor_contrib = p0 if actor == 0 else p1
        matched = max(p0, p1)
        to_call = matched - actor_contrib  # always ≥ 0

        legal: list[HUNLAction] = []
        # FOLD: only when chips are at risk (otherwise check via CALL).
        if to_call > 0:
            legal.append(HUNLAction.FOLD)
        # CALL: always legal at non-terminal. Note: an all-in CALL is
        # allowed even if actor_stack < to_call — partial-call is legal
        # in heads-up (the uncalled portion is returned at terminal).
        legal.append(HUNLAction.CALL)
        # BET: actor has chips AND opponent has chips (heads-up no side pot).
        # An all-in BET that exceeds opponent's available chips is allowed
        # at the action-legality level; the excess is returned at terminal.
        if actor_stack > to_call and opp_stack > 0:
            legal.append(HUNLAction.BET)
        return tuple(legal)

    def legal_action_mask(self) -> np.ndarray:
        """Bool mask shape ``(3,)`` aligned with ``HUNLAction`` integer
        values FOLD=0 / CALL=1 / BET=2. NULL_PADDING (3) is encode-time
        only and never a legal action."""
        mask = np.zeros(3, dtype=bool)
        for a in self.legal_actions():
            mask[int(a)] = True
        return mask

    def legal_bet_sizes(self) -> tuple[int, ...]:
        """Returns valid total-round-contribution sizes for a BET action.

        Empty tuple if BET is not in :meth:`legal_actions`. Otherwise
        a contiguous integer range from min-raise total up to all-in
        cap (= actor's full stack contributed this round).

        The discrete grid here is at single-chip granularity (raw HUNL).
        :class:`AbstractedHUNLGame` (M2) will narrow this to the 6-size
        fold/call/0.5p/1p/2p/all-in subset.
        """
        if HUNLAction.BET not in self.legal_actions():
            return ()
        actor = self.current_player
        actor_stack = self.stack_p0 if actor == 0 else self.stack_p1
        opp_stack = self.stack_p1 if actor == 0 else self.stack_p0
        p0, p1 = self._round_contributions(self.current_round)
        actor_contrib = p0 if actor == 0 else p1
        opp_contrib = p1 if actor == 0 else p0
        matched = max(p0, p1)

        # Min-raise rule: new total ≥ matched + last_raise_increment.
        min_raise_total = matched + self.last_raise_increment
        # Player can also go all-in below min_raise_total — all-ins are
        # always allowed.
        actor_max = actor_contrib + actor_stack   # actor's all-in cap
        # Heads-up effective cap: no use raising past opponent's match
        # capacity. Cap at min(actor_max, opp_contrib + opp_stack).
        opp_max = opp_contrib + opp_stack
        max_useful = min(actor_max, opp_max)

        # Lower bound: min_raise_total OR actor_max (if all-in is below
        # min_raise_total). Whichever is smaller AND > matched.
        lo = min(min_raise_total, actor_max)
        if lo <= matched:
            # Must raise above current matched (else it isn't a BET).
            # If actor's all-in lands at or below matched, BET is not
            # legal — but legal_actions would have caught that. Safety:
            return ()
        hi = max_useful
        if hi < lo:
            return ()
        return tuple(range(lo, hi + 1))

    # ========================================================= transition
    # Applies an action and returns a new HUNLState, advancing rounds and
    # revealing board cards as needed. The current_player flips after each
    # action within a round; round transitions reset current_player to the
    # round's first actor.

    def next_state(
        self, action: HUNLAction, bet_size: int = 0
    ) -> "HUNLState":
        """Applies ``action`` (with optional ``bet_size`` for BET) and
        returns the resulting :class:`HUNLState`.

        Validates the action is in :meth:`legal_actions` and (for BET)
        that ``bet_size`` is in :meth:`legal_bet_sizes`. Updates pot,
        stacks, the relevant round_history / round_bet_sizes tuple,
        ``last_raise_increment``, ``current_player``, and (on round
        closure) advances ``current_round`` + reveals board cards.
        """
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(
                f"action {action!r} not in legal_actions {legal!r}"
            )
        if action == HUNLAction.BET:
            valid_sizes = self.legal_bet_sizes()
            if bet_size not in valid_sizes:
                raise ValueError(
                    f"bet_size {bet_size} not in legal_bet_sizes "
                    f"(min/max = {valid_sizes[0] if valid_sizes else None}/"
                    f"{valid_sizes[-1] if valid_sizes else None})"
                )
        elif bet_size != 0:
            raise ValueError(
                f"bet_size must be 0 for non-BET action {action!r}, "
                f"got {bet_size}"
            )

        actor = self.current_player
        actor_stack = self.stack_p0 if actor == 0 else self.stack_p1
        p0, p1 = self._round_contributions(self.current_round)
        actor_contrib = p0 if actor == 0 else p1
        matched = max(p0, p1)
        to_call = matched - actor_contrib

        # Determine chips committed this action and new last_raise_increment.
        if action == HUNLAction.FOLD:
            chips_added = 0
            new_last_raise = self.last_raise_increment
            stored_size = 0
        elif action == HUNLAction.CALL:
            # Cap the call by the actor's stack (partial-call all-in).
            chips_added = min(to_call, actor_stack)
            new_last_raise = self.last_raise_increment
            stored_size = 0
        else:  # BET
            chips_added = bet_size - actor_contrib
            increment = bet_size - matched
            # Min-raise rule: only BETs ≥ min_raise_total reset
            # last_raise_increment. All-in shorts do NOT reopen betting.
            min_raise_total = matched + self.last_raise_increment
            if bet_size >= min_raise_total:
                new_last_raise = increment
            else:
                # All-in for less than min-raise: keep prior increment.
                new_last_raise = self.last_raise_increment
            stored_size = bet_size

        # Apply chip movement.
        new_stack_p0 = self.stack_p0 - (chips_added if actor == 0 else 0)
        new_stack_p1 = self.stack_p1 - (chips_added if actor == 1 else 0)
        new_pot = self.pot + chips_added

        # Append to current round's history/sizes.
        new_round_history = list(self.round_history)
        new_round_history[self.current_round] = (
            self.round_history[self.current_round] + (action,)
        )
        new_round_bet_sizes = list(self.round_bet_sizes)
        new_round_bet_sizes[self.current_round] = (
            self.round_bet_sizes[self.current_round] + (stored_size,)
        )

        # Determine round closure on the *new* history+sizes. Walk
        # the just-updated round to compute new (p0, p1) contributions.
        new_p0, new_p1 = self._replay_round_contributions(
            self.current_round,
            tuple(new_round_history)[self.current_round],
            tuple(new_round_bet_sizes)[self.current_round],
        )
        if action == HUNLAction.FOLD:
            round_closed = True
        else:
            # Round closes when both players have acted AND either bets
            # are matched OR an all-in has been called (one stack = 0).
            # The all-in case handles partial-call all-ins where the
            # caller's chips ran out before fully matching: round closes,
            # uncalled excess returned at terminal_utility (M1.4).
            bets_matched = (new_p0 == new_p1)
            both_acted = len(new_round_history[self.current_round]) >= 2
            either_all_in = new_stack_p0 == 0 or new_stack_p1 == 0
            round_closed = both_acted and (bets_matched or either_all_in)
        next_round = self.current_round
        next_board = self.board_cards
        next_player = 1 - actor
        next_last_raise = new_last_raise
        if round_closed and action != HUNLAction.FOLD:
            # Advance round + reveal board.
            if self.current_round < NUM_ROUNDS - 1:
                next_round = self.current_round + 1
                next_board = self.pending_board[
                    : _BOARD_LEN_AT_ROUND[next_round]
                ]
                next_player = first_actor_for_round(next_round)
                # Reset last_raise_increment for the new round to 1 BB
                # (min-bet rule for the first action in a new round).
                next_last_raise = BB_BLIND_CHIPS_VALUE
            # Else: river just closed, terminal. Leave fields as-is —
            # is_terminal will return True and no further actions occur.

        return HUNLState(
            private_cards=self.private_cards,
            pending_board=self.pending_board,
            board_cards=next_board,
            round_history=tuple(new_round_history),  # type: ignore[arg-type]
            round_bet_sizes=tuple(new_round_bet_sizes),  # type: ignore[arg-type]
            current_round=next_round,
            current_player=next_player,
            stack_p0=new_stack_p0,
            stack_p1=new_stack_p1,
            last_raise_increment=next_last_raise,
            pot=new_pot,
        )

    # ============================================================== M1.4
    # Terminal utility: hand-resolution at hands that have already
    # reached :prop:`is_terminal`. Mirrors Phase 2 LeducPoker's static
    # method API; ``HUNLGame.terminal_utility(state)`` (M1.5) will
    # delegate to this.

    def _total_contributions(self) -> tuple[int, int]:
        """Total chips each player committed across all rounds.

        Round 0 always contributes blinds (P0=BB=2, P1=SB=1) even if
        the round has no actions yet (which can only happen at the
        non-terminal root). Round r ≥ 1 contributes only if it has
        actions.
        """
        total_p0 = 0
        total_p1 = 0
        for r in range(NUM_ROUNDS):
            if r > 0 and not self.round_history[r]:
                continue
            cp0, cp1 = self._round_contributions(r)
            total_p0 += cp0
            total_p1 += cp1
        return total_p0, total_p1

    def _find_folder(self) -> int | None:
        """Returns the player who folded (0 or 1), or ``None`` if no
        fold occurred (river-close terminal)."""
        for r in range(NUM_ROUNDS):
            actions = self.round_history[r]
            if actions and actions[-1] == HUNLAction.FOLD:
                # Walk to determine who folded.
                actor = first_actor_for_round(r)
                for _ in actions[:-1]:
                    actor = 1 - actor
                return actor
        return None

    def terminal_utility(self) -> float:
        """Player 0's chip-net change at terminal (positive ⇒ P0 won
        chips). Heads-up zero-sum: P1's utility is the negation.

        Three resolution paths:

        1. **Fold**: the non-folding player wins the matched pot. The
           folder loses ``matched`` chips; the winner gains ``matched``.
           Any uncalled excess (the winner's contribution beyond
           ``matched``) is conceptually returned to the winner — the
           formula collapses to a single ``±matched`` because both
           players' chip changes are exactly ``matched`` in opposite
           directions.

        2. **Showdown win**: hand_eval picks the winner via
           :func:`compare_hands` on the river board + each player's
           hole cards; winner gains ``matched``, loser loses ``matched``.

        3. **Tie (split pot)**: equal hand strength → 0 chip net
           change for either player. Pot is split, both contribute
           and recover ``matched``.

        Raises ``ValueError`` if called on a non-terminal state.
        """
        if not self.is_terminal:
            raise ValueError(
                "terminal_utility called on non-terminal state"
            )

        total_p0, total_p1 = self._total_contributions()
        matched = min(total_p0, total_p1)

        folder = self._find_folder()
        if folder is not None:
            return -float(matched) if folder == 0 else +float(matched)

        # Showdown — board must be the full 5-card river by now since
        # is_terminal returned True via the river-close path.
        sign = compare_hands(
            list(self.private_cards[0:2]),
            list(self.private_cards[2:4]),
            list(self.board_cards),
        )
        if sign == +1:
            return +float(matched)
        if sign == -1:
            return -float(matched)
        return 0.0   # tie split

    @staticmethod
    def _replay_round_contributions(
        round_idx: int,
        actions: tuple[HUNLAction, ...],
        sizes: tuple[int, ...],
    ) -> tuple[int, int]:
        """Pure replay of a round's actions to (p0_contrib, p1_contrib).
        Used by :meth:`next_state` to walk the just-updated round."""
        if round_idx == 0:
            p0, p1 = BB_BLIND_CHIPS_VALUE, SB_BLIND_CHIPS
        else:
            p0, p1 = 0, 0
        matched = max(p0, p1)
        actor = first_actor_for_round(round_idx)
        for a, size in zip(actions, sizes, strict=True):
            if a == HUNLAction.FOLD:
                break
            if a == HUNLAction.CALL:
                if actor == 0:
                    p0 = matched
                else:
                    p1 = matched
            elif a == HUNLAction.BET:
                if actor == 0:
                    p0 = size
                else:
                    p1 = size
                matched = size
            actor = 1 - actor
        return p0, p1
