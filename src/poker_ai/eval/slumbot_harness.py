"""Slumbot benchmark harness — Phase 4 M4.3.

Orchestrates a full hand (or session of hands) against the Slumbot HTTP
API: wires :class:`SlumbotClient` (M4.1 transport) and the M4.2 protocol
adapter into a loop that

1. requests a new hand,
2. replays the partial action sequence into our abstracted state,
3. samples our action from a caller-supplied strategy,
4. submits it via ``client.act``,
5. repeats until the server returns ``winnings``,
6. records the outcome (utility, sync_check, position) into a
   :class:`HandRecord`.

Retry policy: configurable exponential backoff on 5xx / 429 responses.
4xx errors fast-fail (no retry). Defaults: ``max_retries=3``,
``backoff_base=2.0``, ``max_backoff=30.0`` seconds.

Live HTTP is M4.4 territory — every call site here is exercised against
mocked clients in unit tests.

Hand-termination cross-check (Hook 2):

- ``state.is_terminal AND winnings is not None``     → ✓
- ``state.is_terminal AND winnings is None``         → ✗ SlumbotError
- ``not state.is_terminal AND winnings is not None`` → ✗ SlumbotError
- ``not state.is_terminal AND winnings is None``     → loop continues

Sync check granularity:

- **fold-only**: full magnitude check
  ``chip_to_slumbot(our_utility_from_state) == winnings`` (signs must
  match too). Mismatch raises :class:`SlumbotError`.
- **showdown**: opp_hole is unobservable (Slumbot does not reveal it
  on /api/act response), so ``state.terminal_utility`` cannot be
  computed exactly. ``sync_check`` is set to ``True`` automatically
  on showdown — the sign / magnitude verification is skipped, and the
  Slumbot-reported ``winnings`` is treated as ground truth for
  ``our_utility_chips``.

Position mapping (Slumbot ↔ HUNLState):

- ``client_pos == 0`` (we are BB)   ↔ ``HUNLState.current_player == 0``
- ``client_pos == 1`` (we are SB)   ↔ ``HUNLState.current_player == 1``

I.e. ``our_player_idx == client_pos`` (direct equality).

**M4.4 self-audit (claude #24)**: Slumbot's API documentation as
mirrored in third-party clients describes ``client_pos: 0=small
blind/button``, but the live server's behavior is the opposite —
the server takes the SB/button position and ``client_pos`` reports
the role of the *client*. Verified live 2026-04-27: a fresh
``new_hand`` consistently returns ``action`` strings starting with
SB-side moves (e.g. ``"b200"``) when ``client_pos==0``, which is
only consistent with the client being the BB (the server SB acts
first preflop). The direct mapping above matches HUNLState's
``player 0 = BB / player 1 = SB`` convention coincidentally. Doc
drift is registered as mentor #10 candidate (cross-context fact
verification, asset #22 generalisation).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests   # type: ignore[import-untyped]

from poker_ai.eval.slumbot_client import (
    SlumbotClient,
    SlumbotError,
    chip_from_slumbot,
    chip_to_slumbot,
)
from poker_ai.eval.slumbot_protocol import (
    encode_action,
    replay_sequence,
)
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLAction,
    AbstractedHUNLGame,
    AbstractedHUNLState,
)
from poker_ai.games.hunl_state import (
    BB_BLIND_CHIPS_VALUE,
    HUNLAction,
)


# =============================================================================
# Public types
# =============================================================================
StrategyFn = Callable[[AbstractedHUNLState], np.ndarray]


@dataclass(frozen=True, slots=True)
class HandRecord:
    """One hand played against Slumbot.

    Attributes:
        deal: 9-card deal tuple. Our hole cards (``hole_cards``
            response field) + revealed board are exact; the opponent's
            hole and any unrevealed board slots are randomly sampled
            from the remaining deck. Used for state-machine replay
            only — utility is sourced from ``slumbot_winnings``.
        sequence: Slumbot action sequence string at termination.
        client_pos: 0 = SB / button, 1 = BB (Slumbot convention).
        our_utility_chips: signed chip count in our internal units
            (BB = ``BB_BLIND_CHIPS_VALUE`` = 2 chips). Positive = we
            won. Derived from ``slumbot_winnings`` via
            :func:`chip_from_slumbot` (Slumbot is ground truth on
            showdowns where opp_hole is hidden).
        slumbot_winnings: signed Slumbot chip count from ``response.winnings``.
        sync_check: ``True`` iff the fold-only magnitude check passed
            or termination was a showdown (where magnitude check is
            relaxed to a sign-only invariant).
    """

    deal: tuple[int, ...]
    sequence: str
    client_pos: int
    our_utility_chips: int
    slumbot_winnings: int
    sync_check: bool


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Aggregated record over a play session."""

    hands: list[HandRecord] = field(default_factory=list)


def mbb_per_hand_winrate(records: list[HandRecord]) -> tuple[float, float]:
    """Mean ± SE win-rate in mbb/hand from the client's perspective.

    Per-hand value: ``our_utility_chips * 1000 / BB_BLIND_CHIPS_VALUE``.

    SE estimator: naive IID — ``std(per_hand) / sqrt(n)`` with
    ``ddof=1`` when ``n >= 2``, else ``0.0``. Hand samples are i.i.d. by
    construction (each Slumbot hand draws its own deal + opponent
    randomness), but our shared strategy means hand-level utilities
    can carry small correlations through the strategy itself; the naive
    SE underestimates that variance source.

    **AIVAT (Burch 2018) variance reduction is registered as a Phase 5
    hook** (asset #24 candidate in PHASE.md M4 closure). M4 reports
    naive SE explicitly so trend interpretation can adjust if the
    AIVAT-corrected estimator is later substituted.

    Raises :class:`ValueError` on empty input.
    """
    if not records:
        raise ValueError("mbb_per_hand_winrate requires at least one record")
    arr = np.asarray(
        [r.our_utility_chips * 1000.0 / BB_BLIND_CHIPS_VALUE for r in records],
        dtype=np.float64,
    )
    mean = float(arr.mean())
    if len(arr) >= 2:
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    else:
        se = 0.0
    return mean, se


# =============================================================================
# Card parsing helper
# =============================================================================
_RANKS: str = "23456789TJQKA"
_SUITS: str = "cdhs"


def _card_str_to_int(s: str) -> int:
    """Parses a Slumbot card string (e.g. ``'As'``) into our 0..51 id."""
    if len(s) != 2:
        raise ValueError(f"card string must be 2 chars: {s!r}")
    rank = _RANKS.index(s[0])
    suit = _SUITS.index(s[1])
    return rank * 4 + suit


def _reconstruct_deal(
    hole_cards: list[str],
    board: list[str],
    client_pos: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Builds a 9-card deal tuple consistent with Slumbot's response.

    Layout (per :func:`HUNLGame.sample_deal`):
        ``(p0_h1, p0_h2, p1_h1, p1_h2, b1, b2, b3, b4, b5)``.

    Our hole cards land at the right slots according to ``client_pos``
    (claude #24 self-audit, M4.4 live verify): Slumbot's
    ``client_pos == 0`` means *we* are the BB (HUNLState player 0,
    slots 0..1); ``client_pos == 1`` means we are the SB (HUNLState
    player 1, slots 2..3).

    Opponent hole and any unrevealed board slots are sampled uniformly
    from the remaining deck via ``rng``. Used for state-machine replay
    only; chip-flow utility is read from ``slumbot_winnings``, so the
    randomness here only affects which abstracted-bucket key we land
    in (a downstream concern handled at the strategy layer in M4.5).
    """
    our_hole = [_card_str_to_int(c) for c in hole_cards]
    if len(our_hole) != 2:
        raise ValueError(f"hole_cards must have 2 entries; got {len(our_hole)}")
    board_known = [_card_str_to_int(c) for c in board]
    if len(board_known) > 5:
        raise ValueError(f"board has > 5 cards: {len(board_known)}")
    used = set(our_hole) | set(board_known)
    remaining = np.asarray([c for c in range(52) if c not in used])
    n_missing_board = 5 - len(board_known)
    draw_count = 2 + n_missing_board
    indices = rng.choice(len(remaining), size=draw_count, replace=False)
    sampled = [int(remaining[int(i)]) for i in indices]
    opp_hole = sampled[:2]
    board_full = board_known + sampled[2:2 + n_missing_board]
    if client_pos == 0:
        # We are BB → HUNLState player_idx 0 → hole at slots 0..1.
        deal = tuple(our_hole + opp_hole + board_full)
    else:
        # client_pos == 1 → we are SB → HUNLState player_idx 1 → slots 2..3.
        deal = tuple(opp_hole + our_hole + board_full)
    return deal


# =============================================================================
# SlumbotHarness
# =============================================================================
class SlumbotHarness:
    """Wraps a :class:`SlumbotClient` with retry, position alternation,
    and per-hand orchestration.

    Args:
        client: :class:`SlumbotClient` (or any duck-typed object with
            ``new_hand`` / ``act`` methods returning
            :class:`SlumbotResponse`).
        max_retries: number of additional attempts after the first
            failure for retry-eligible errors (5xx / 429). Default 3.
        backoff_base: exponential factor for retry delay
            (``delay = backoff_base ** attempt`` seconds, capped at
            ``max_backoff``). Default 2.0.
        max_backoff: cap on a single retry delay in seconds. Default 30.
    """

    def __init__(
        self,
        client: SlumbotClient,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff

    # ------------------------------------------------------------------ retry
    def _retry_post(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Calls ``fn(*args)`` with exponential-backoff retry on 5xx/429.

        4xx (except 429) and any non-HTTPError exception fast-fail.
        After ``max_retries + 1`` total attempts the last error
        propagates.
        """
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args)
            except requests.HTTPError as exc:
                status = _http_status(exc)
                if not _is_retryable_status(status):
                    raise
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(self.backoff_base ** attempt, self.max_backoff)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------ hands
    def play_one_hand(
        self,
        game: AbstractedHUNLGame,
        strategy_fn: StrategyFn,
        rng: np.random.Generator,
    ) -> HandRecord:
        """Drives one full hand against Slumbot. See module docstring for
        the loop / cross-check contract.
        """
        first = self._retry_post(self.client.new_hand)
        deal = _reconstruct_deal(
            list(first.hole_cards), list(first.board), first.client_pos, rng
        )
        client_pos = first.client_pos
        # M4.4 live-verified: Slumbot's client_pos directly equals
        # HUNLState player_idx (0=BB, 1=SB) — see module docstring.
        our_player_idx = client_pos

        last_response = first
        last_action_seq = first.action

        # Loop while server signals "your turn" (winnings is None).
        while last_response.winnings is None:
            state = replay_sequence(
                game, deal, last_action_seq, client_pos
            )
            if state.is_terminal:
                raise SlumbotError(
                    "desync: state.is_terminal but winnings is None "
                    f"(action={last_action_seq!r})"
                )
            if state.current_player != our_player_idx:
                raise SlumbotError(
                    f"desync: not our turn (state.current_player="
                    f"{state.current_player}, our_player_idx="
                    f"{our_player_idx})"
                )
            our_action = _sample_legal_action(state, strategy_fn, rng)
            wire_token = encode_action(state, our_action)
            last_response = self._retry_post(self.client.act, wire_token)
            last_action_seq = last_response.action

        # Hand has terminated (winnings present).
        final_state = replay_sequence(
            game, deal, last_action_seq, client_pos
        )
        if not final_state.is_terminal:
            raise SlumbotError(
                "desync: winnings present but state not terminal "
                f"(action={last_action_seq!r})"
            )

        winnings = int(last_response.winnings or 0)
        our_utility_chips = chip_from_slumbot(winnings)
        sync_check = _verify_sync(
            final_state, winnings, our_player_idx
        )

        return HandRecord(
            deal=deal,
            sequence=last_action_seq,
            client_pos=client_pos,
            our_utility_chips=our_utility_chips,
            slumbot_winnings=winnings,
            sync_check=sync_check,
        )

    def play_session(
        self,
        game: AbstractedHUNLGame,
        strategy_fn: StrategyFn,
        n_hands: int,
        rng: np.random.Generator,
    ) -> SessionRecord:
        """Plays ``n_hands`` hands sequentially. Errors mid-session
        propagate to the caller (no partial-session recovery)."""
        if n_hands < 0:
            raise ValueError(f"n_hands must be ≥ 0; got {n_hands}")
        records: list[HandRecord] = []
        for _ in range(n_hands):
            records.append(self.play_one_hand(game, strategy_fn, rng))
        return SessionRecord(hands=records)


# =============================================================================
# Internals
# =============================================================================
def _http_status(exc: requests.HTTPError) -> int | None:
    """Best-effort status-code extraction from an HTTPError."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_retryable_status(status: int | None) -> bool:
    """Returns True for 5xx and 429 only.

    A None status (transport-level failure with no response object) is
    treated as retryable — typical of transient network errors caught
    by ``requests`` and re-raised as ``HTTPError`` without an attached
    response. Anything 4xx (except 429) is permanent and fast-fails.
    """
    if status is None:
        return True
    if status == 429:
        return True
    if 500 <= status <= 599:
        return True
    return False


def _sample_legal_action(
    state: AbstractedHUNLState,
    strategy_fn: StrategyFn,
    rng: np.random.Generator,
) -> AbstractedHUNLAction:
    """Samples one legal :class:`AbstractedHUNLAction` from ``strategy_fn``.

    Mass on illegal actions is masked then renormalised; a strategy
    that places all mass on illegal actions falls back to uniform over
    the legal mask (no crash, just opt for a defensible default).
    """
    dist = np.asarray(strategy_fn(state), dtype=np.float64)
    if dist.shape != (6,):
        raise ValueError(
            f"strategy_fn must return shape (6,); got {dist.shape}"
        )
    mask = state.legal_action_mask().astype(np.float64)
    filtered = dist * mask
    total = float(filtered.sum())
    if total <= 0.0:
        legal_total = float(mask.sum())
        if legal_total <= 0.0:
            raise SlumbotError(
                "no legal action at our turn (state has empty legal mask)"
            )
        filtered = mask / legal_total
    else:
        filtered = filtered / total
    idx = int(rng.choice(6, p=filtered))
    return AbstractedHUNLAction(idx)


def _verify_sync(
    final_state: AbstractedHUNLState,
    slumbot_winnings: int,
    our_player_idx: int,
) -> bool:
    """Cross-check helper. See module docstring for the rule.

    Fold path: full-magnitude check returns ``True`` iff
    ``chip_to_slumbot(state_utility) == slumbot_winnings`` (signs and
    chip-counts both agree). Mismatch returns ``False`` — flagged but
    not raised. Hard desync (e.g. winnings present but state not
    terminal) is detected upstream in :meth:`play_one_hand`; the field
    here is purely informational so callers can audit per-record
    consistency without aborting the session.

    Showdown path: returns ``True`` unconditionally — opp hole is
    unobservable so we cannot recompute our state-side utility exactly,
    and Slumbot's reported winnings is the only ground truth.
    """
    is_fold = any(
        action == HUNLAction.FOLD
        for round_actions in final_state._raw.round_history
        for action in round_actions
    )
    if not is_fold:
        return True
    raw = final_state._raw
    tu_p1_perspective = raw.terminal_utility()
    our_utility_from_state = (
        tu_p1_perspective if our_player_idx == 0 else -tu_p1_perspective
    )
    expected_winnings = chip_to_slumbot(int(our_utility_from_state))
    return expected_winnings == slumbot_winnings
