"""Structural typing for game engines usable by CFR / BR algorithms.

Uses :class:`typing.Protocol` (runtime-checkable) rather than ABC so that
concrete games (Kuhn, Leduc) satisfy these via duck typing alone — no
inheritance required. This keeps the Phase 2 refactor minimal: existing
Kuhn and Leduc classes need no changes beyond a trivial ``NUM_ACTIONS``
constant (Kuhn) that was missing before.

Over-specification guardrails (intentionally EXCLUDED from the Protocols
to avoid locking in premature abstractions):

- ``legal_action_mask``: Leduc-only. Kuhn has all actions legal at every
  decision, so exposing an explicit mask on every state would be noise.
- ``private_cards`` / ``cards``: name differs (Kuhn uses ``cards``, Leduc
  uses ``private_cards``). Don't bake either name into the structural type.
- ``round_idx`` / ``board_card`` / ``bets_this_round``: Leduc-only.
- ``history`` / ``round_history``: shape differs between games.

If Phase 3 NLHE requires a richer contract (e.g. action abstraction info,
card bucketing handles), reconsider promoting to ABC at that point.

Reference: :pep:`544` Protocol structural subtyping.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateProtocol(Protocol):
    """Structural type for a game state usable by CFR / BR algorithms."""

    @property
    def is_terminal(self) -> bool: ...

    @property
    def current_player(self) -> int: ...

    @property
    def infoset_key(self) -> str: ...

    def legal_actions(self) -> tuple[IntEnum, ...]: ...

    def legal_action_mask(self) -> np.ndarray: ...

    def next_state(self, action: IntEnum) -> StateProtocol: ...


@runtime_checkable
class GameProtocol(Protocol):
    """Structural type for a 2-player imperfect-information game factory.

    Methods are declared with instance-method signatures, but a conforming
    implementation may use ``@staticmethod`` (both Kuhn and Leduc do so).
    Runtime ``isinstance`` checks only verify attribute presence.

    ``ENCODING_DIM`` + ``encode`` were added in Phase 3 Day 1 to bridge
    tabular CFR and neural function-approximating Deep CFR: the advantage
    and strategy networks consume a fixed-dim numeric vector per infoset,
    which :meth:`encode` produces from any non-terminal state.

    ``sample_deal`` was added in Phase 4 Step 3 (M0) to support games
    where ``all_deals()`` is impractical to enumerate (HUNL has ~10^14
    deals). Algorithms that scale via Monte Carlo sampling (External-
    Sampling MCCFR, Deep CFR's outer loop) should use ``sample_deal``;
    enumeration-based algorithms (Vanilla CFR, exact best-response /
    exploitability) keep using ``all_deals``. Finite-deal games (Kuhn,
    Leduc, AbstractedLeducPoker) implement both — ``sample_deal`` simply
    picks uniformly from ``all_deals``. Large-deal games (HUNL) raise
    ``NotImplementedError`` from ``all_deals`` and provide a direct
    sampler in ``sample_deal``.
    """

    NUM_ACTIONS: int
    ENCODING_DIM: int

    def all_deals(self) -> tuple[Any, ...]: ...

    def sample_deal(self, rng: np.random.Generator) -> Any: ...

    def state_from_deal(self, deal: Any) -> StateProtocol: ...

    def terminal_utility(self, state: StateProtocol) -> float: ...

    def encode(self, state: StateProtocol) -> np.ndarray: ...
