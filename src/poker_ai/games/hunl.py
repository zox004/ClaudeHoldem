"""HUNL game factory — Phase 4 M1.5 (GameProtocol implementation).

Wires :class:`HUNLState` (M1.2–M1.4) onto :class:`GameProtocol` so
that Phase 2 algorithms (Vanilla CFR for small games, MCCFR for
scalable games) can drop in without code changes. The factory:

- exposes ``NUM_ACTIONS`` (3 — FOLD/CALL/BET) and ``ENCODING_DIM``
  (102, see :func:`encode` for layout) as the GameProtocol contract.
- provides ``sample_deal`` for Monte-Carlo chance sampling (the
  intended path; HUNL has ~10^14 deals).
- raises NotImplementedError from ``all_deals`` (the GameProtocol
  scaling decision from M0 — finite games like Kuhn / Leduc keep
  ``all_deals`` for Vanilla CFR; HUNL skips it because Vanilla CFR
  is impractical here regardless).
- delegates ``terminal_utility`` to :meth:`HUNLState.terminal_utility`.
- implements ``encode`` with a compact rank/suit layout suitable for
  the M2 MCCFR-tabular path. **M2 reconsideration**: if neural-net
  components are reintroduced (Phase 5+), replace this compact
  encoding with one-hot per the design-spec note (Q4).

Encoding layout (102 dimensions, all float32):
- [0..4)   acting player's 2 hole cards: 2 × (rank/12, suit/3)
- [4..14)  5 board cards: 5 × (rank/12, suit/3); unrevealed = -1.0
- [14..18) current_round one-hot (preflop / flop / turn / river)
- [18..22) scalars: pot/400, stack_p0/200, stack_p1/200,
           last_raise_increment/200
- [22..102) betting history flat: 40 slots × (action_id/3, size/200);
            NULL_PADDING fills trailing slots
"""

from __future__ import annotations

from typing import Any

import numpy as np

from poker_ai.games.hunl_state import (
    BB_BLIND_CHIPS_VALUE,
    HUNLAction,
    HUNLState,
    SB_BLIND_CHIPS,
    STARTING_STACK_CHIPS,
)

_DECK_SIZE: int = 52
_DEAL_SIZE: int = 9     # 2 P0 hole + 2 P1 hole + 5 board (Q3 sign-off)
_ENCODING_DIM: int = 102

# Encoding offsets (kept module-private; tests cross-check via the
# helper ``_encode_offset``).
_OFFSET_HOLE: int = 0
_OFFSET_BOARD: int = 4
_OFFSET_ROUND: int = 14
_OFFSET_SCALARS: int = 18
_OFFSET_HISTORY: int = 22
_HISTORY_FLAT_LEN: int = 40

_NORM_RANK: float = 12.0
_NORM_SUIT: float = 3.0
_NORM_POT: float = 2.0 * float(STARTING_STACK_CHIPS)   # 2 × stack (bankroll cap)
_NORM_STACK: float = float(STARTING_STACK_CHIPS)
_NORM_LAST_RAISE: float = float(STARTING_STACK_CHIPS)
_NORM_ACTION_ID: float = 3.0   # max HUNLAction value (NULL_PADDING=3)
_NORM_BET_SIZE: float = float(STARTING_STACK_CHIPS)


def _encode_card(arr: np.ndarray, offset: int, card_id: int) -> None:
    """Writes (rank/12, suit/3) into ``arr[offset:offset+2]``.
    Negative card_id (sentinel for unrevealed) writes (-1, -1)."""
    if card_id < 0:
        arr[offset] = -1.0
        arr[offset + 1] = -1.0
        return
    rank = card_id // 4
    suit = card_id % 4
    arr[offset] = rank / _NORM_RANK
    arr[offset + 1] = suit / _NORM_SUIT


class HUNLGame:
    """HUNL Hold'em factory.

    Stack depth, blind sizes, and other invariants live as module
    constants in :mod:`poker_ai.games.hunl_state`. ``HUNLGame`` is a
    thin GameProtocol facade.

    The class is intentionally stateless — methods are static and a
    fresh state is built for each ``state_from_deal`` call.
    """

    NUM_ACTIONS: int = 3   # FOLD, CALL, BET (NULL_PADDING is encode-only)
    ENCODING_DIM: int = _ENCODING_DIM

    @staticmethod
    def all_deals() -> tuple[Any, ...]:
        """Raises ``NotImplementedError`` — HUNL has ~10^14 ordered deals;
        enumeration is impractical. Use :meth:`sample_deal` instead.
        Phase 4 M0 contract: large-deal games skip ``all_deals``."""
        raise NotImplementedError(
            "HUNL has ~10^14 deals; use sample_deal(rng) (Phase 4 M0 contract)"
        )

    @staticmethod
    def sample_deal(rng: np.random.Generator) -> tuple[int, ...]:
        """Uniform sample of a 9-card deal without replacement.

        Layout (Q3 sign-off):
            ``(p0_h1, p0_h2, p1_h1, p1_h2, b1, b2, b3, b4, b5)``
        """
        return tuple(
            int(c) for c in rng.choice(_DECK_SIZE, size=_DEAL_SIZE, replace=False)
        )

    @staticmethod
    def state_from_deal(deal: tuple[int, ...]) -> HUNLState:
        """Builds the root preflop state from a 9-flat deal tuple.

        Validates length 9, distinct entries, and 0..51 range. Posts
        SB and BB blinds: P0 (BB) starts with 198 chips behind +
        2 in pot; P1 (SB) starts with 199 chips behind + 1 in pot.
        Pot=3 (1 SB + 2 BB), current_player=1 (SB acts first preflop).
        """
        if len(deal) != _DEAL_SIZE:
            raise ValueError(
                f"deal must have {_DEAL_SIZE} entries (P0 hole 2 + P1 hole 2 "
                f"+ board 5), got {len(deal)}"
            )
        for c in deal:
            if not 0 <= c < _DECK_SIZE:
                raise ValueError(
                    f"deal entry {c} out of range 0..{_DECK_SIZE - 1}"
                )
        if len(set(deal)) != _DEAL_SIZE:
            raise ValueError(
                f"deal must contain {_DEAL_SIZE} distinct cards, "
                f"got {len(set(deal))} unique"
            )
        return HUNLState(
            private_cards=(deal[0], deal[1], deal[2], deal[3]),
            pending_board=(deal[4], deal[5], deal[6], deal[7], deal[8]),
            board_cards=(),
            round_history=((), (), (), ()),
            round_bet_sizes=((), (), (), ()),
            current_round=0,
            current_player=1,   # SB acts first preflop
            stack_p0=STARTING_STACK_CHIPS - BB_BLIND_CHIPS_VALUE,
            stack_p1=STARTING_STACK_CHIPS - SB_BLIND_CHIPS,
            last_raise_increment=BB_BLIND_CHIPS_VALUE,
            pot=BB_BLIND_CHIPS_VALUE + SB_BLIND_CHIPS,
        )

    @staticmethod
    def terminal_utility(state: HUNLState) -> float:
        """Delegates to :meth:`HUNLState.terminal_utility`. P0 perspective,
        zero-sum (P1 utility = negation)."""
        return state.terminal_utility()

    @staticmethod
    def encode(state: HUNLState) -> np.ndarray:
        """Compact rank/suit encoding suitable for M2 MCCFR-tabular use.

        Returns a ``float32`` array of shape ``(ENCODING_DIM,) = (102,)``.
        See module docstring for the layout. M2 reconsideration: replace
        with one-hot if neural components return (Phase 5+).
        """
        arr = np.zeros(_ENCODING_DIM, dtype=np.float32)

        # --- acting-player hole cards (2 cards × 2 dim = 4 dim) ---
        actor = state.current_player
        hole_offset_in_state = actor * 2
        actor_hole = state.private_cards[hole_offset_in_state : hole_offset_in_state + 2]
        for i, c in enumerate(actor_hole):
            _encode_card(arr, _OFFSET_HOLE + i * 2, c)

        # --- board cards (5 cards × 2 dim = 10 dim) ---
        for i in range(5):
            if i < len(state.board_cards):
                _encode_card(arr, _OFFSET_BOARD + i * 2, state.board_cards[i])
            else:
                _encode_card(arr, _OFFSET_BOARD + i * 2, -1)

        # --- current round one-hot (4 dim) ---
        arr[_OFFSET_ROUND + state.current_round] = 1.0

        # --- scalars (4 dim) ---
        arr[_OFFSET_SCALARS + 0] = state.pot / _NORM_POT
        arr[_OFFSET_SCALARS + 1] = state.stack_p0 / _NORM_STACK
        arr[_OFFSET_SCALARS + 2] = state.stack_p1 / _NORM_STACK
        arr[_OFFSET_SCALARS + 3] = state.last_raise_increment / _NORM_LAST_RAISE

        # --- betting history flat 40 slots × 2 dim = 80 dim ---
        flat_actions: list[HUNLAction] = []
        flat_sizes: list[int] = []
        for round_actions, round_sizes in zip(
            state.round_history, state.round_bet_sizes, strict=True
        ):
            flat_actions.extend(round_actions)
            flat_sizes.extend(round_sizes)
        # Pad to history length with NULL_PADDING + size 0.
        while len(flat_actions) < _HISTORY_FLAT_LEN:
            flat_actions.append(HUNLAction.NULL_PADDING)
            flat_sizes.append(0)
        for i in range(_HISTORY_FLAT_LEN):
            arr[_OFFSET_HISTORY + i * 2] = (
                int(flat_actions[i]) / _NORM_ACTION_ID
            )
            arr[_OFFSET_HISTORY + i * 2 + 1] = (
                flat_sizes[i] / _NORM_BET_SIZE
            )

        return arr
