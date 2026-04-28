"""Slumbot action protocol adapter — Phase 4 M4.2.

Bridges :class:`AbstractedHUNLState`'s 6-way abstracted action grid with
Slumbot's wire format on top of the M4.1 transport layer:

    Slumbot wire:    f  c  k  b<int>             tokens, ``/`` between streets
    Our domain:      AbstractedHUNLAction         FOLD / CALL / BET_HALF / BET_POT
                                                  / BET_DOUBLE / BET_ALLIN

Two directions are supported:

- :func:`encode_action`     — our chosen abstracted action → Slumbot token
- :func:`ingest_opponent_token` — Slumbot opponent token → state advance
- :func:`replay_sequence`   — full Slumbot action history → final state

Translation choice (M4.2 spec lock, claude push-back accepted):

  **(i) nearest-bucket** with **larger-size tie-break**.

We deliberately reject (ii) Schnizlein 2009 probabilistic state translation
for M4.2 because it would require restructuring infoset_key / strategy
lookup (~150 LoC scope creep) and verifying the transfer of a paper-level
algorithm to our history-encoding pattern. (ii) is registered as a Phase
5 hook (see asset #24 candidate, M4 closure).

Tie-break = larger size: measurement conservatism. Under-estimating
opponent strength biases our own win-rate report downward (Slumbot side
overstated); over-estimating biases honestly. M4 is a measurement
exercise, so we err toward over-estimation.

Bet-to semantics: Slumbot ``b500`` = bet TO 500 chips total street.
``HUNLState.legal_bet_sizes()`` returns "total round contribution" by the
same definition (M3.2 spec sign-off), so dispatch is direct after the
×50 chip-granularity bridge from M4.1.

Audit hook coverage (M4.2 spec):

- Hook 1 — Re-raise chain + non-grid sizes: handled by
  :func:`split_action_sequence` parsing arbitrary chains and by
  :func:`nearest_abstracted_bet_size` snapping non-grid raw sizes onto
  our 6-action grid via floor-divide-by-50 then nearest bucket.
- Hook 2 — All-in clamp: BET_ALLIN's ``compute_size`` already returns
  ``state.legal_bet_sizes()[-1]`` (heads-up effective stack); the encode
  side multiplies by 50 to match Slumbot's full-stack ``b<full>``.
- Hook 3 — Empty round / street transition: :func:`split_action_sequence`
  pads to 4 streets even on ``b20000c///`` style sequences;
  :func:`replay_sequence` early-returns when the state goes terminal so
  trailing slashes after an all-in resolve cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import numpy as np

from poker_ai.eval.probabilistic_dispatch import bucket_weights, sample_bucket
from poker_ai.eval.slumbot_client import chip_from_slumbot, chip_to_slumbot
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLAction,
    AbstractedHUNLGame,
    AbstractedHUNLState,
    _bet_mask,
    compute_size,
)
from poker_ai.games.hunl_state import HUNLAction


_BET_PATTERN: Final[re.Pattern[str]] = re.compile(r"^b(\d+)$")


# =============================================================================
# Action token parsing
# =============================================================================
@dataclass(frozen=True, slots=True)
class SlumbotActionToken:
    """Parsed single Slumbot action token.

    ``kind`` ∈ {"f", "c", "k", "b"}; ``bet_to_slumbot_chips`` is the
    ``X`` from a ``b<X>`` raise-to amount and zero for non-bet tokens.
    """

    kind: str
    bet_to_slumbot_chips: int


def parse_action_token(token: str) -> SlumbotActionToken:
    """Parses a single Slumbot action token.

    Accepts ``f`` (fold), ``c`` (call), ``k`` (check), and ``b<int>``
    (raise to total street contribution of <int> Slumbot chips).
    Anything else raises :class:`ValueError`.
    """
    if token in ("f", "c", "k"):
        return SlumbotActionToken(kind=token, bet_to_slumbot_chips=0)
    match = _BET_PATTERN.match(token)
    if match is None:
        raise ValueError(f"unknown Slumbot action token: {token!r}")
    return SlumbotActionToken(
        kind="b", bet_to_slumbot_chips=int(match.group(1))
    )


def split_action_sequence(seq: str) -> list[list[str]]:
    """Splits a Slumbot action sequence into per-street token lists.

    Streets are separated by ``/``; each street is tokenised into
    individual moves (single-char ``f``/``c``/``k`` plus ``b<digits>``
    spans). Always returns exactly 4 inner lists — empty lists pad
    shorter sequences and trailing empty streets after an all-in
    resolve (e.g. ``b20000c///``).

    Raises :class:`ValueError` on unknown characters within a street.
    """
    if seq == "":
        return [[], [], [], []]
    streets = seq.split("/")
    out: list[list[str]] = []
    for street_str in streets:
        tokens: list[str] = []
        i = 0
        while i < len(street_str):
            char = street_str[i]
            if char == "b":
                j = i + 1
                while j < len(street_str) and street_str[j].isdigit():
                    j += 1
                if j == i + 1:
                    raise ValueError(
                        f"bare 'b' in street {street_str!r} (no digits)"
                    )
                tokens.append(street_str[i:j])
                i = j
            elif char in ("f", "c", "k"):
                tokens.append(char)
                i += 1
            else:
                raise ValueError(
                    f"unknown character {char!r} in street {street_str!r}"
                )
        out.append(tokens)
    while len(out) < 4:
        out.append([])
    return out


# =============================================================================
# Nearest-bucket translation (i)
# =============================================================================
def nearest_abstracted_bet_size(
    raw_chip_size: int,
    legal_abstracted_sizes: list[int],
) -> int:
    """Returns the legal abstracted bet size closest to ``raw_chip_size``.

    Tie-break: when two legal sizes are equidistant, the **larger** one
    is selected (measurement-conservatism choice; see module docstring).
    Below the minimum legal size: snaps to the lowest. Above the maximum:
    snaps to the highest.

    All sizes are in our internal chip units (BB=2 chips); convert from
    Slumbot chips via :func:`poker_ai.eval.slumbot_client.chip_from_slumbot`
    before calling.

    Raises :class:`ValueError` on empty ``legal_abstracted_sizes``.
    """
    if not legal_abstracted_sizes:
        raise ValueError("legal_abstracted_sizes must not be empty")
    sorted_legal = sorted(set(legal_abstracted_sizes))
    if raw_chip_size <= sorted_legal[0]:
        return sorted_legal[0]
    if raw_chip_size >= sorted_legal[-1]:
        return sorted_legal[-1]
    best = sorted_legal[0]
    best_dist = abs(raw_chip_size - best)
    for size in sorted_legal[1:]:
        dist = abs(raw_chip_size - size)
        if dist < best_dist or (dist == best_dist and size > best):
            best = size
            best_dist = dist
    return best


# =============================================================================
# Action emit (our abstracted → Slumbot token)
# =============================================================================
def encode_action(
    state: AbstractedHUNLState,
    action: AbstractedHUNLAction,
) -> str:
    """Emits the Slumbot wire token for the chosen abstracted action.

    Mapping:
      - FOLD → ``"f"``
      - CALL → ``"c"`` if there is a chip amount to call, else ``"k"``
      - BET_HALF / BET_POT / BET_DOUBLE / BET_ALLIN → ``"b<X>"`` where
        ``X = chip_to_slumbot(compute_size(action, state._raw))``

    The action must be legal at ``state``; otherwise :class:`ValueError`.
    """
    if action not in state.legal_actions():
        raise ValueError(
            f"action {action!r} is not legal at this state "
            f"(legal={list(state.legal_actions())})"
        )
    if action == AbstractedHUNLAction.FOLD:
        return "f"
    if action == AbstractedHUNLAction.CALL:
        raw = state._raw
        p0_contrib, p1_contrib = raw._round_contributions(raw.current_round)
        actor = raw.current_player
        actor_contrib = p0_contrib if actor == 0 else p1_contrib
        opp_contrib = p1_contrib if actor == 0 else p0_contrib
        to_call = opp_contrib - actor_contrib
        return "c" if to_call > 0 else "k"
    raw_chip_size = compute_size(action, state._raw)
    return f"b{chip_to_slumbot(raw_chip_size)}"


# =============================================================================
# Action ingest (Slumbot token → state advance)
# =============================================================================
def ingest_opponent_token(
    state: AbstractedHUNLState,
    token: SlumbotActionToken,
    *,
    dispatch_mode: str = "deterministic",
    rng: np.random.Generator | None = None,
) -> AbstractedHUNLState:
    """Advances the state given a Slumbot wire token.

    For ``b<X>``: converts X / 50 to our chips, finds legal abstracted
    BET sizes via :func:`compute_size` over each legal BET enum, then
    dispatches:

    * ``dispatch_mode="deterministic"`` (M4.2 default) —
      :func:`nearest_abstracted_bet_size` snaps to the closest legal
      bucket (larger-size tie-break).
    * ``dispatch_mode="probabilistic"`` (M4.5.2 Schnizlein 2009) —
      :func:`bucket_weights` builds a 2-bucket distribution from the
      paper Eq. 5/6 geometric similarity, then :func:`sample_bucket`
      picks one bucket using ``rng``. ``rng`` is required in this
      mode and should be seeded per hand for paper §3.2 internal-
      consistency invariant.

    For ``f`` / ``c`` / ``k``: direct enum mapping (``k`` collapses to
    CALL since our state machine encodes "check" as a no-money CALL).

    The function is whose-turn-agnostic — it applies the action to
    whichever player is currently to act. ``replay_sequence`` uses
    that property to walk a full Slumbot history regardless of side.
    """
    if token.kind == "f":
        return state.next_state(AbstractedHUNLAction.FOLD)
    if token.kind in ("c", "k"):
        return state.next_state(AbstractedHUNLAction.CALL)
    if token.kind != "b":
        raise ValueError(f"unknown token kind {token.kind!r}")

    raw = state._raw
    bet_mask = _bet_mask(raw)
    candidates: dict[int, AbstractedHUNLAction] = {}
    for bet_action, is_legal in bet_mask.items():
        if not is_legal:
            continue
        candidates[compute_size(bet_action, raw)] = bet_action
    if not candidates:
        raise ValueError(
            "Slumbot 'b' token received but no abstracted BET legal here"
        )
    our_chip = chip_from_slumbot(token.bet_to_slumbot_chips)
    if dispatch_mode == "deterministic":
        chosen_size = nearest_abstracted_bet_size(
            our_chip, list(candidates.keys())
        )
    elif dispatch_mode == "probabilistic":
        if rng is None:
            raise ValueError(
                "dispatch_mode='probabilistic' requires rng "
                "(paper §3.2 internal-consistency: seed per hand)"
            )
        weights = bucket_weights(
            raw_chip=our_chip, legal_sizes=list(candidates.keys())
        )
        chosen_size = sample_bucket(weights, rng)
    else:
        raise ValueError(
            f"dispatch_mode must be 'deterministic' or 'probabilistic'; "
            f"got {dispatch_mode!r}"
        )
    return state.next_state(candidates[chosen_size])


# =============================================================================
# Full sequence replay
# =============================================================================
def replay_sequence(
    game: AbstractedHUNLGame,
    deal: tuple[int, ...],
    sequence: str,
    client_pos: int,
    *,
    dispatch_mode: str = "deterministic",
    rng: np.random.Generator | None = None,
) -> AbstractedHUNLState:
    """Replays a full Slumbot action sequence onto a fresh state.

    Args:
        game: AbstractedHUNLGame whose ``state_from_deal`` produces the
            initial state.
        deal: 9-card deal tuple (P0 hole, P1 hole, board) per
            :meth:`HUNLGame.sample_deal`.
        sequence: Slumbot action history string, e.g. ``"cb500c/kk/kk/kk"``.
        client_pos: which side we (the client) sit on per Slumbot
            convention (0 = SB / button, 1 = BB). Currently informational
            only — token dispatch is whose-turn-driven and identical
            regardless of side. Reserved for future per-side bookkeeping
            (M4.3+ when stats are aggregated per-side).

    Returns the resulting :class:`AbstractedHUNLState` (terminal if the
    sequence ends a hand). Trailing slashes after a terminal token are
    safely ignored (the loop short-circuits on ``state.is_terminal``).
    """
    del client_pos   # M4.2: dispatch identical regardless; reserved for M4.3
    state = game.state_from_deal(deal)
    streets = split_action_sequence(sequence)
    for tokens in streets:
        for token_str in tokens:
            if state.is_terminal:
                return state
            parsed = parse_action_token(token_str)
            state = ingest_opponent_token(
                state, parsed, dispatch_mode=dispatch_mode, rng=rng,
            )
    # All-in roll-out: when both stacks reach 0 mid-hand, Slumbot encodes
    # the resulting "no-action" remaining streets as trailing slashes
    # (e.g. ``b20000c///``). Our state machine still expects the rounds
    # to close explicitly; auto-advance with CALLs (which collapse to
    # checks under matched-contributions=0) until terminal.
    while (
        not state.is_terminal
        and state._raw.stack_p0 == 0
        and state._raw.stack_p1 == 0
    ):
        state = state.next_state(AbstractedHUNLAction.CALL)
    return state


# =============================================================================
# Re-export from sibling module for tests / callers
# =============================================================================
__all__ = [
    "SlumbotActionToken",
    "encode_action",
    "ingest_opponent_token",
    "nearest_abstracted_bet_size",
    "parse_action_token",
    "replay_sequence",
    "split_action_sequence",
    # convenient access to opaque imports
    "HUNLAction",
]
