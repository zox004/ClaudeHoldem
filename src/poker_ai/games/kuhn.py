"""Kuhn Poker game engine.

Kuhn Poker is a 2-player zero-sum imperfect-information toy poker with a
3-card deck {J, Q, K}, 1 card dealt per player, 1-chip ante, at most one
bet/raise. It is the smallest nontrivial extensive-form game with private
information and is the canonical vehicle for validating a CFR implementation
(Neller & Lanctot 2013, Section 4).

Terminal scoring (P1 perspective; u2 = -u1 by zero-sum)
=======================================================

+------------------+------------------------------------+-----+------------------------+-----------------+
| Terminal history | Meaning                            | Pot | P1 utility             | Card-dependent? |
+==================+====================================+=====+========================+=================+
| pp               | both check → showdown              |  2  | +1 if c1>c2 else -1    |       yes       |
| bp               | P1 bet, P2 fold                    |  2  | +1 (fixed)             |       no        |
| bb               | both bet → showdown                |  4  | +2 if c1>c2 else -2    |       yes       |
| pbp              | P1 pass, P2 bet, P1 fold           |  2  | -1 (fixed)             |       no        |
| pbb              | P1 pass, P2 bet, P1 call → sdown   |  4  | +2 if c1>c2 else -2    |       yes       |
+------------------+------------------------------------+-----+------------------------+-----------------+

Convention notes:
    - Fold: the folding player loses all chips they have committed so far.
    - Showdown: higher rank (J=0 < Q=1 < K=2) wins the entire pot.
    - Utility = net chips gained (a called bet is recovered from the pot and
      cancels out in net accounting; hence bp/pbp are ±1, not ±2).

Design choices
==============

1.  No chance-node abstraction. A CFR trainer iterates over ``all_deals()``
    externally: the six permutations of (P1 card, P2 card) are weighted
    uniformly. When Leduc's mid-game board card is added in Phase 2, a
    ``ChanceNode`` protocol will be introduced there.

2.  ``KuhnAction`` is an ``IntEnum`` for interoperability with numpy action
    indexing and neural-network action heads in Phase 3+. String rendering
    ("p"/"b") is only used for infoset-key formatting.

3.  State is a ``@dataclass(frozen=True, slots=True)``. Immutability makes
    CFR's functional recursion safe by construction; ``slots=True`` trims
    per-instance memory cost for the ~10⁴ states a Kuhn traversal creates.

4.  Infoset key is rendered as ``"<rank_char>|<history_str>"`` (e.g. ``"K|pb"``)
    using ``"JQK"[card]`` at the boundary only — internal card identifiers
    remain ``int``. This matches Neller & Lanctot's paper notation for easy
    eyeball-debugging against their published Nash strategy tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

_RANK_CHARS = "JQK"


class KuhnAction(IntEnum):
    PASS = 0
    BET = 1

    def __str__(self) -> str:
        return "p" if self is KuhnAction.PASS else "b"


@dataclass(frozen=True, slots=True)
class KuhnState:
    deal: tuple[int, int]
    history: tuple[KuhnAction, ...]

    @property
    def is_terminal(self) -> bool:
        h = self.history
        if len(h) < 2:
            return False
        if len(h) == 2:
            # Of the four length-2 histories, only (PASS, BET) continues; the
            # other three (pp, bp, bb) end the game.
            return h != (KuhnAction.PASS, KuhnAction.BET)
        # Length 3 is only reachable after "pb"; both pbp and pbb end the game.
        return True

    @property
    def current_player(self) -> int:
        # For Kuhn's 4 non-terminal histories, turn order is strict alternation.
        return len(self.history) % 2

    @property
    def infoset_key(self) -> str:
        own_card = self.deal[self.current_player]
        history_str = "".join(str(a) for a in self.history)
        return f"{_RANK_CHARS[own_card]}|{history_str}"

    def legal_actions(self) -> tuple[KuhnAction, ...]:
        return (KuhnAction.PASS, KuhnAction.BET)

    def next_state(self, action: KuhnAction) -> "KuhnState":
        return KuhnState(deal=self.deal, history=self.history + (action,))


class KuhnPoker:
    NUM_PLAYERS = 2
    NUM_CARDS = 3
    NUM_ACTIONS: int = 2   # PASS, BET — Phase 2 GameProtocol conformance

    # Six deals of 2 distinct cards from {J, Q, K}. Fixed order for determinism.
    _DEALS: tuple[tuple[int, int], ...] = (
        (0, 1), (0, 2),
        (1, 0), (1, 2),
        (2, 0), (2, 1),
    )

    @staticmethod
    def all_deals() -> tuple[tuple[int, int], ...]:
        return KuhnPoker._DEALS

    @staticmethod
    def state_from_deal(deal: tuple[int, int]) -> KuhnState:
        return KuhnState(deal=deal, history=())

    @staticmethod
    def terminal_utility(state: KuhnState) -> float:
        """Return Player 1's utility at a terminal state (zero-sum: u2 = -u1).

        Hardcoded dispatch over the five terminal histories — see the scoring
        table in the module docstring.
        """
        if not state.is_terminal:
            raise ValueError(
                f"terminal_utility called on non-terminal state: {state!r}"
            )

        c1, c2 = state.deal
        p1_wins_showdown = c1 > c2
        history = state.history

        if history == (KuhnAction.PASS, KuhnAction.PASS):
            return 1.0 if p1_wins_showdown else -1.0
        if history == (KuhnAction.BET, KuhnAction.PASS):
            return 1.0  # P2 folded
        if history == (KuhnAction.BET, KuhnAction.BET):
            return 2.0 if p1_wins_showdown else -2.0
        if history == (KuhnAction.PASS, KuhnAction.BET, KuhnAction.PASS):
            return -1.0  # P1 folded
        if history == (KuhnAction.PASS, KuhnAction.BET, KuhnAction.BET):
            return 2.0 if p1_wins_showdown else -2.0

        raise AssertionError(f"unreachable terminal history: {history!r}")
