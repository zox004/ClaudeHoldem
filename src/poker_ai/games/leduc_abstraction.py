"""Leduc Hold'em card abstraction wrapper — Phase 4 Step 2 path validation.

Wraps :class:`LeducPoker` to alias multiple ranks onto the same infoset
bucket. CFR/MCCFR algorithms that key strategy by ``state.infoset_key``
will then share strategy across grouped ranks, exactly the lossy
abstraction approach Pluribus uses for HUNL.

Two design notes (Phase 4 Step 2 design clarification):

1. Leduc's raw ``infoset_key`` already collapses suit (``own_card // 2``
   maps the 6-card deck onto the 3 visible ranks J/Q/K). A "3-bucket"
   abstraction on Leduc therefore reduces to the identity map — useful as
   a wrapper-sanity smoke but not a real information-loss test. The real
   test on Leduc is **2-bucket** (e.g. {J, Q} → "low", {K} → "high"),
   which loses ~37 % of the rank entropy (1.58 bits → 1.0 bit).

2. The wrapper does NOT mutate the underlying game tree, terminal
   utility, encoding, legal actions, or chance distribution — only the
   ``infoset_key`` returned by :class:`AbstractedLeducState`. This keeps
   the actual exploitability evaluation HONEST: best-response is still
   computed against the raw game tree, while the strategy under test
   uses the abstracted keys.

Example:
    >>> game = AbstractedLeducPoker(n_buckets=2)
    >>> deals = game.all_deals()
    >>> state = game.state_from_deal(deals[0])
    >>> state.infoset_key
    'L|'        # bucket 0 (J or Q) at the root, no betting yet

Reference algorithms: :class:`MCCFRExternalSampling`, :class:`VanillaCFR`,
:class:`CFRPlus` all consume :class:`GameProtocol` so the wrapper is
plug-compatible without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from poker_ai.games.leduc import LeducPoker, LeducState

_BUCKET_CHARS_2: tuple[str, str] = ("L", "H")
_BUCKET_CHARS_3: tuple[str, str, str] = ("J", "Q", "K")


class CardAbstractor:
    """Maps a raw Leduc rank index (0=J, 1=Q, 2=K) onto a bucket index.

    Bucket assignments are deterministic — same rank always maps to the
    same bucket — so infoset_key strings are stable within a run and
    across runs of the same configuration.

    Supported configurations:

    - ``n_buckets=3``: identity. J → 0, Q → 1, K → 2. The wrapper's
      infoset_key still differs from raw Leduc's (uses bucket characters
      "J"/"Q"/"K" by coincidence, but via a different code path), so a
      3-bucket run smoke-tests the wrapper without losing information.

    - ``n_buckets=2``: E[HS] grouping {J, Q} → 0 ("L"), {K} → 1 ("H"). On
      Leduc this is the simplest meaningful abstraction; (J, Q) share a
      bucket because they are the two weaker ranks. Round-2 hand-strength
      ordering still carries through because the board character is left
      raw — only the player's private rank is bucketed.
    """

    def __init__(self, n_buckets: int) -> None:
        if n_buckets not in (2, 3):
            raise ValueError(
                f"n_buckets must be 2 or 3 for Leduc abstraction, got {n_buckets}"
            )
        self.n_buckets = n_buckets

    def bucket(self, rank: int) -> int:
        """Maps rank ∈ {0, 1, 2} → bucket ∈ {0, ..., n_buckets-1}."""
        if rank not in (0, 1, 2):
            raise ValueError(f"rank must be in {{0, 1, 2}}, got {rank}")
        if self.n_buckets == 3:
            return rank
        # n_buckets == 2: {J=0, Q=1} → 0, {K=2} → 1.
        return 0 if rank < 2 else 1

    def bucket_char(self, rank: int) -> str:
        """Stable single-character label for the bucket — used in keys."""
        b = self.bucket(rank)
        if self.n_buckets == 3:
            return _BUCKET_CHARS_3[b]
        return _BUCKET_CHARS_2[b]


@dataclass(frozen=True, slots=True)
class AbstractedLeducState:
    """Wraps a :class:`LeducState` and overrides ``infoset_key``.

    All other StateProtocol members delegate to the underlying state, so
    the game tree, legality, and chance distribution are untouched.
    """

    _raw: LeducState
    _abstractor: CardAbstractor

    @property
    def is_terminal(self) -> bool:
        return self._raw.is_terminal

    @property
    def current_player(self) -> int:
        return self._raw.current_player

    @property
    def infoset_key(self) -> str:
        """Mirrors :meth:`LeducState.infoset_key` but with bucketed
        characters for the player's private rank.

        The board character stays raw (J/Q/K) because abstraction here is
        applied to the player's private hand only — Pluribus's E[HS²]
        bucketing typically conditions on the board, but for a Leduc
        sanity test the simplest "private-only" abstraction is enough to
        register an information-loss vs no-loss comparison.
        """
        own_card = self._raw.private_cards[self._raw.current_player]
        own_rank = own_card // 2
        own_char = self._abstractor.bucket_char(own_rank)
        round1_str = "".join(str(a) for a in self._raw.round_history[0])
        if self._raw.round_idx == 0:
            return f"{own_char}|{round1_str}"
        assert self._raw.board_card is not None
        # Board character left raw — see docstring rationale.
        from poker_ai.games.leduc import _RANK_CHARS
        board_char = _RANK_CHARS[self._raw.board_card // 2]
        round2_str = "".join(str(a) for a in self._raw.round_history[1])
        return f"{own_char}|{round1_str}.{board_char}{round2_str}"

    def legal_actions(self) -> tuple[IntEnum, ...]:
        return self._raw.legal_actions()

    def legal_action_mask(self) -> np.ndarray:
        return self._raw.legal_action_mask()

    def next_state(self, action: IntEnum) -> AbstractedLeducState:
        # The protocol-level signature uses IntEnum, but the underlying
        # LeducState only accepts LeducAction. Both are IntEnum subtypes
        # with identical numeric values for the actions Leduc supports.
        from poker_ai.games.leduc import LeducAction
        leduc_action = LeducAction(int(action))
        return AbstractedLeducState(
            _raw=self._raw.next_state(leduc_action),
            _abstractor=self._abstractor,
        )


class AbstractedLeducPoker:
    """GameProtocol-compatible Leduc with private-rank abstraction.

    Delegates ``all_deals``, ``state_from_deal``, ``terminal_utility``,
    and ``encode`` to a wrapped :class:`LeducPoker`. Only state's
    ``infoset_key`` differs from the raw game.
    """

    NUM_ACTIONS: int = LeducPoker.NUM_ACTIONS
    ENCODING_DIM: int = LeducPoker.ENCODING_DIM

    def __init__(self, n_buckets: int = 2) -> None:
        self.abstractor = CardAbstractor(n_buckets)
        self._raw = LeducPoker()

    def all_deals(self) -> tuple[tuple[int, int, int], ...]:
        return self._raw.all_deals()

    def sample_deal(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """Delegates to the raw Leduc sampler — abstraction does not
        affect chance distribution."""
        return self._raw.sample_deal(rng)

    def state_from_deal(self, deal: tuple[int, int, int]) -> AbstractedLeducState:
        return AbstractedLeducState(
            _raw=self._raw.state_from_deal(deal),
            _abstractor=self.abstractor,
        )

    def terminal_utility(self, state: Any) -> float:
        # Accept both AbstractedLeducState and raw LeducState — strategy
        # algorithms always pass the wrapped variant, but tests may pass
        # raw states.
        raw = state._raw if isinstance(state, AbstractedLeducState) else state
        return self._raw.terminal_utility(raw)

    def encode(self, state: Any) -> np.ndarray:
        raw = state._raw if isinstance(state, AbstractedLeducState) else state
        return self._raw.encode(raw)

    def n_infosets(self) -> int:
        """Counts unique abstracted infoset keys by traversing the raw
        game tree once. Useful for ablation reports — Phase 2 raw Leduc
        has 288 infosets; 2-bucket abstraction collapses to fewer."""
        seen: set[str] = set()
        deals = self._raw.all_deals()

        def _walk(state: LeducState) -> None:
            if state.is_terminal:
                return
            ab_state = AbstractedLeducState(_raw=state, _abstractor=self.abstractor)
            seen.add(ab_state.infoset_key)
            for action in state.legal_actions():
                _walk(state.next_state(action))

        for deal in deals:
            _walk(self._raw.state_from_deal(deal))
        return len(seen)
