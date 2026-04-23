"""Leduc Hold'em Poker Engine.

Game parameters (Southey 2005 / Neller & Lanctot 2013 §5 / OpenSpiel standard):
- Deck: 6 cards (3 ranks × 2 suits; suit is value-irrelevant)
- Ante: 1 chip per player (starting pot = 2)
- Round 1 bet/raise size: 2 chips
- Round 2 bet/raise size: 4 chips
- Max raises per round: 2 (first bet + 1 raise)
- Raise semantics: call + bet_size (e.g. round 1 raise commits 4 chips)

Terminal utility (P1 perspective):
- Fold: folder loses own committed chips, non-folder wins opponent's commit.
- Showdown: pair (private rank = board rank) beats non-pair. Otherwise higher
  rank wins. Same private rank (no pair either side) = tie.

Commitment trace anchors (regression lock-in for pot accounting):
- Round 1 ``rrf`` (P1 bet, P2 raise, P1 fold): P1 commits 3 (ante+bet),
  P2 commits 5 (ante+call+raise). P1 folds → utility = -3.
- ``cc.rrf`` (round 1 cc, round 2 bet-raise-fold, P1 folder): P1 commits 5
  (ante + round-2 bet 4), P2 commits 9 (ante + round-2 call+raise = 8).
  P1 folds → utility = -5.

Card encoding: ``card_id`` ∈ 0..5; ``rank = card_id // 2`` (0=J, 1=Q, 2=K);
``suit = card_id % 2``.

Infoset key: ``"<own_rank>|<round1_chars>[.<board_rank><round2_chars>]"`` using
``'f'/'c'/'r'`` for actions and ``'J'/'Q'/'K'`` for ranks. Perfect recall is
guaranteed by including the full action history of both rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import permutations

import numpy as np

# --- Rule constants (no magic numbers in game logic) -------------------------

_RANK_CHARS: tuple[str, str, str] = ("J", "Q", "K")
_BET_SIZES: tuple[int, int] = (2, 4)   # bet/raise size by round_idx (0, 1)
_ANTE: int = 1
_MAX_RAISES_PER_ROUND: int = 2
_ACTION_CHARS: dict[int, str] = {0: "f", 1: "c", 2: "r"}


# --- Action enum -------------------------------------------------------------


class LeducAction(IntEnum):
    """Three unified actions for Leduc (OpenSpiel / Brown 2019 convention).

    Interpretation is context-dependent:
    - ``CALL`` = "check" when no prior bet; "call" when facing a bet.
    - ``RAISE`` = "bet" when opening the round; "raise" when facing a bet.
    - ``FOLD`` = forfeit. Illegal when no prior bet exists in the round.
    """

    FOLD = 0
    CALL = 1
    RAISE = 2

    def __str__(self) -> str:
        return _ACTION_CHARS[int(self)]


# --- Pure helpers: round closure + per-round commitment ---------------------


def _is_round_closed(round_actions: tuple[LeducAction, ...]) -> bool:
    """True if ``round_actions`` closes the round via a CALL that follows at
    least one prior action.

    Closing sequences: ``cc``, ``rc``, ``crc``, ``rrc``, ``crrc``. A fold does
    NOT "close" a round here — it terminates the game directly and is handled
    separately by :attr:`LeducState.is_terminal`.
    """
    if len(round_actions) < 2:
        return False
    return round_actions[-1] == LeducAction.CALL


def _round_commits(
    round_actions: tuple[LeducAction, ...],
    bet_size: int,
) -> tuple[int, int]:
    """Chip commitment per player in a single round (excluding ante).

    Raise semantics: a ``RAISE`` increases the round's stake by ``bet_size``;
    the raiser's total round commit becomes the new stake. ``CALL`` after a
    raise matches the current stake. ``CALL`` with no prior raise is a free
    check (commits 0). ``FOLD`` contributes nothing further.
    """
    commits = [0, 0]
    stake = 0
    for i, action in enumerate(round_actions):
        player = i % 2
        if action == LeducAction.FOLD:
            continue
        if action == LeducAction.CALL:
            commits[player] = stake
        elif action == LeducAction.RAISE:
            stake += bet_size
            commits[player] = stake
    return commits[0], commits[1]


# --- Immutable state --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeducState:
    """Immutable Leduc game state.

    ``board_card`` is exposed as ``None`` during round 1 and as the dealt
    card once round 1 closes normally (non-fold). ``_pending_board`` carries
    the deal's board through round 1 so that :meth:`next_state` can promote
    it at round 1's closure without any external chance input — all chance
    is absorbed into :meth:`LeducPoker.all_deals`.
    """

    private_cards: tuple[int, int]
    board_card: int | None
    round_history: tuple[
        tuple[LeducAction, ...],
        tuple[LeducAction, ...],
    ]
    _pending_board: int

    # ---------- derived properties -----------------------------------------

    @property
    def round_idx(self) -> int:
        """0 during round 1; 1 after round 1 closes and the board is revealed."""
        return 0 if self.board_card is None else 1

    @property
    def bets_this_round(self) -> int:
        """Number of ``RAISE`` actions so far in the current round (0..2)."""
        return sum(
            1
            for a in self.round_history[self.round_idx]
            if a == LeducAction.RAISE
        )

    @property
    def current_player(self) -> int:
        """Player to act. Both rounds start with P1 (index 0) acting first."""
        return len(self.round_history[self.round_idx]) % 2

    @property
    def is_terminal(self) -> bool:
        """True if any round ended with a fold, or round 2 closed normally."""
        for round_actions in self.round_history:
            if round_actions and round_actions[-1] == LeducAction.FOLD:
                return True
        if self.round_idx == 1 and _is_round_closed(self.round_history[1]):
            return True
        return False

    @property
    def infoset_key(self) -> str:
        """Perfect-recall infoset key ``<own_rank>|<round1>[.<board><round2>]``."""
        own_card = self.private_cards[self.current_player]
        own_rank_char = _RANK_CHARS[own_card // 2]
        round1_str = "".join(str(a) for a in self.round_history[0])
        if self.round_idx == 0:
            return f"{own_rank_char}|{round1_str}"
        assert self.board_card is not None, "round 2 state must have board revealed"
        board_char = _RANK_CHARS[self.board_card // 2]
        round2_str = "".join(str(a) for a in self.round_history[1])
        return f"{own_rank_char}|{round1_str}.{board_char}{round2_str}"

    # ---------- action API --------------------------------------------------

    def legal_actions(self) -> tuple[LeducAction, ...]:
        """Tuple of legal actions given current bet state. Empty at terminal."""
        if self.is_terminal:
            return ()
        n_bets = self.bets_this_round
        if n_bets == 0:
            return (LeducAction.CALL, LeducAction.RAISE)
        if n_bets == 1:
            return (LeducAction.FOLD, LeducAction.CALL, LeducAction.RAISE)
        # 2-bet cap reached → RAISE illegal
        return (LeducAction.FOLD, LeducAction.CALL)

    def legal_action_mask(self) -> np.ndarray:
        """Bool mask of shape ``(LeducPoker.NUM_ACTIONS,)`` aligned with action indices."""
        mask = np.zeros(LeducPoker.NUM_ACTIONS, dtype=bool)
        for a in self.legal_actions():
            mask[int(a)] = True
        return mask

    def next_state(self, action: LeducAction) -> LeducState:
        """Return a new state with ``action`` appended to the current round.

        Promotes ``_pending_board`` to ``board_card`` when round 1 closes
        via a non-fold action — the next state is then in round 2.
        """
        cur_idx = self.round_idx
        appended = self.round_history[cur_idx] + (action,)
        if cur_idx == 0:
            new_rhist: tuple[
                tuple[LeducAction, ...],
                tuple[LeducAction, ...],
            ] = (appended, self.round_history[1])
        else:
            new_rhist = (self.round_history[0], appended)

        new_board = self.board_card
        if (
            cur_idx == 0
            and action != LeducAction.FOLD
            and _is_round_closed(new_rhist[0])
        ):
            new_board = self._pending_board

        return LeducState(
            private_cards=self.private_cards,
            board_card=new_board,
            round_history=new_rhist,
            _pending_board=self._pending_board,
        )


# --- Game factory + terminal utility ----------------------------------------


class LeducPoker:
    """Leduc Hold'em game API: static factories and terminal utility dispatch."""

    NUM_ACTIONS: int = 3
    ENCODING_DIM: int = 13  # Phase 3 Day 1 — see encode() layout

    @staticmethod
    def all_deals() -> tuple[tuple[int, int, int], ...]:
        """All 120 ordered ``(p1_card, p2_card, board_card)`` triples.

        Each of the 6 × 5 × 4 permutations of three distinct ``card_id``s
        in ``0..5``. Uniform chance: each deal has probability ``1/120``.
        This is the ``reach_opp`` initial value consumed by
        :meth:`VanillaCFR.train` and :func:`eval.exploitability.best_response_value`.
        """
        return tuple(permutations(range(6), 3))

    @staticmethod
    def state_from_deal(deal: tuple[int, int, int]) -> LeducState:
        """Build the round-1 root state from a ``(P1, P2, board)`` deal triple.

        Board is stored internally (``_pending_board``) but not revealed
        (``board_card = None``) until round 1 closes.
        """
        p1_card, p2_card, board_card = deal
        return LeducState(
            private_cards=(p1_card, p2_card),
            board_card=None,
            round_history=((), ()),
            _pending_board=board_card,
        )

    @staticmethod
    def terminal_utility(state: LeducState) -> float:
        """P1-perspective chip utility at a terminal state.

        Static dispatch over fold / showdown branches (intentionally
        hardcoded rather than dynamic — matches Kuhn engine pattern, avoids
        pot-accounting bugs from re-derivation at each call site).
        """
        round1 = state.round_history[0]
        round2 = state.round_history[1]
        p1_r1, p2_r1 = _round_commits(round1, bet_size=_BET_SIZES[0])
        p1_r2, p2_r2 = _round_commits(round2, bet_size=_BET_SIZES[1])
        p1_total = _ANTE + p1_r1 + p1_r2
        p2_total = _ANTE + p2_r1 + p2_r2

        # Fold branches take precedence (any fold ends the game immediately).
        if round1 and round1[-1] == LeducAction.FOLD:
            folder = (len(round1) - 1) % 2
            return -float(p1_total) if folder == 0 else float(p2_total)
        if round2 and round2[-1] == LeducAction.FOLD:
            folder = (len(round2) - 1) % 2
            return -float(p1_total) if folder == 0 else float(p2_total)

        # Showdown: both players must have matched the final stake.
        assert p1_total == p2_total, (
            f"showdown with unequal commit: P1={p1_total}, P2={p2_total}"
        )
        assert state.board_card is not None, "showdown requires board revealed"

        p1_rank = state.private_cards[0] // 2
        p2_rank = state.private_cards[1] // 2
        board_rank = state.board_card // 2

        p1_pair = p1_rank == board_rank
        p2_pair = p2_rank == board_rank

        if p1_pair and not p2_pair:
            return float(p2_total)
        if p2_pair and not p1_pair:
            return -float(p1_total)
        # Both-pair is impossible (each rank has only 2 copies; if P1 and P2
        # share a rank, the board cannot also be that rank). Fall through to
        # straight rank comparison.
        if p1_rank > p2_rank:
            return float(p2_total)
        if p2_rank > p1_rank:
            return -float(p1_total)
        return 0.0

    @staticmethod
    def encode(state: LeducState) -> np.ndarray:
        """Acting-player-perspective encoding (shape ``(13,)``, float32).

        Layout::

            [0-2]  hole rank one-hot (J/Q/K)          — acting player's card
            [3-5]  board rank one-hot (J/Q/K)         — all-zero while round 1
            [6]    1.0 if board not revealed (round 1)
            [7-8]  round one-hot (round 1 / round 2)
            [9]    len(round_history[0]) / 4.0
            [10]   raise_count(round_history[0]) / 2.0
            [11]   len(round_history[1]) / 4.0
            [12]   raise_count(round_history[1]) / 2.0

        Uniqueness for all 288 reachable infosets: own_card (3) × round (2) ×
        round-1 shape × board-rank × round-2 shape separate them. Verified by
        DFS test.
        """
        out = np.zeros(LeducPoker.ENCODING_DIM, dtype=np.float32)
        hole = state.private_cards[state.current_player]
        out[hole // 2] = 1.0
        if state.board_card is None:
            out[6] = 1.0
        else:
            out[3 + state.board_card // 2] = 1.0
        out[7 + state.round_idx] = 1.0
        r1 = state.round_history[0]
        r2 = state.round_history[1]
        out[9] = len(r1) / 4.0
        out[10] = sum(1 for a in r1 if a == LeducAction.RAISE) / 2.0
        out[11] = len(r2) / 4.0
        out[12] = sum(1 for a in r2 if a == LeducAction.RAISE) / 2.0
        return out
