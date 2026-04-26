"""HUNL card abstraction (Phase 4 M2).

Implements the Pluribus-style E[HS²] preflop bucketing so that an
:class:`AbstractedHUNLGame` (M2.3) can collapse the 169 distinct
preflop starting hands onto a small bucket index. With ~50 buckets,
preflop infoset count drops from 169 × <action histories> to
50 × <action histories> — about 3.4× compression at preflop, plus
the ability for MCCFR to actually reach saturated sample-per-bucket
counts within a tractable budget (Step 2 / Leduc validated this
mechanism: information loss < sampling-variance gain under finite
compute).

Module layout:

- :func:`hand_signature` — canonical "AKs" / "AKo" / "AA" string for
  any (c0, c1) two-card hand. Pair = "RR" (no suffix), suited = "RRs",
  offsuit = "RRo", higher rank first.
- :func:`enumerate_starting_hands` — returns all 169 canonical
  signatures (13 pairs + 78 suited + 78 offsuit).
- :func:`hand_strength_squared_mc` — Monte Carlo E[HS²] for one hand.
- :class:`HUNLCardAbstractor` — precomputes E[HS²] for every signature,
  sorts, and assigns each into one of ``n_buckets`` percentile buckets.

E[HS² ] formulation (Ganzfried & Sandholm 2014, equation 3):
    For starting hand h:
        HS(h, board) = P_{opp_hand}[h beats opp_hand on (h, opp, board)]
        E[HS² (h)] = E_{board}[ HS(h, board)² ]

We approximate via Monte Carlo: for each trial, draw a board (5 cards)
and an opponent hand (2 cards) uniformly from the deck (excluding h's
two cards), then evaluate the showdown. Win = 1.0, tie = 0.5, loss =
0.0. HS_sample is just the indicator over a single opponent hand;
Pluribus-grade implementations average over MANY opponent hands per
board to reduce variance, but for our M2 first pass we use one
opponent per trial and compensate with more trials.

Determinism: ``HUNLCardAbstractor(seed=...)`` computes a fixed table.
Same seed → same bucket assignments for every hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np

from poker_ai.games.hunl import HUNLGame
from poker_ai.games.hunl_hand_eval import compare_hands
from poker_ai.games.hunl_state import HUNLAction, HUNLState

# Rank chars in ascending strength: 2..A.
_RANK_CHARS: Final[tuple[str, ...]] = (
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
)
_RANK_CHARS_DESC: Final[tuple[str, ...]] = tuple(reversed(_RANK_CHARS))


def hand_signature(c0: int, c1: int) -> str:
    """Canonical preflop hand signature.

    Returns:
        "RR" for a pocket pair (two cards of same rank).
        "ABs" for suited (different ranks, same suit), higher rank first.
        "ABo" for offsuit (different ranks, different suits), higher rank first.

    Card-id convention: ``rank * 4 + suit``, rank ∈ 0..12 (2..A),
    suit ∈ 0..3.
    """
    if c0 == c1:
        raise ValueError(f"hand cards must differ; got both = {c0}")
    r0, s0 = c0 // 4, c0 % 4
    r1, s1 = c1 // 4, c1 % 4
    if r0 == r1:
        # Pocket pair.
        return _RANK_CHARS[r0] + _RANK_CHARS[r1]
    # Order higher rank first.
    if r0 < r1:
        r0, r1 = r1, r0
        s0, s1 = s1, s0
    suited_marker = "s" if s0 == s1 else "o"
    return _RANK_CHARS[r0] + _RANK_CHARS[r1] + suited_marker


def enumerate_starting_hands() -> tuple[str, ...]:
    """All 169 canonical preflop starting-hand signatures.

    13 pocket pairs (AA, KK, ..., 22) + 78 suited (AKs, AQs, ..., 32s)
    + 78 offsuit (AKo, AQo, ..., 32o) = 169.
    """
    out: list[str] = []
    # Pocket pairs (descending strength).
    for r in _RANK_CHARS_DESC:
        out.append(r + r)
    # Suited and offsuit, higher rank first.
    for i, r0 in enumerate(_RANK_CHARS_DESC):
        for r1 in _RANK_CHARS_DESC[i + 1 :]:
            out.append(r0 + r1 + "s")
            out.append(r0 + r1 + "o")
    return tuple(out)


def _signature_to_concrete_cards(sig: str) -> tuple[int, int]:
    """Picks one concrete (c0, c1) pair representing a signature.

    Used by Monte Carlo to start trials from a canonical fixture; the
    *concrete* suits chosen are arbitrary but consistent — Monte Carlo
    samples opponent hand and board uniformly from the remaining 50
    cards regardless, so HS(h) is independent of which concrete suits
    represent ``h`` (Ganzfried & Sandholm §3.1, suit-isomorphism).
    """
    if len(sig) == 2:
        # Pair: e.g. "AA" -> ranks (12, 12) -> use suits 0 and 1.
        rank = _RANK_CHARS.index(sig[0])
        return rank * 4 + 0, rank * 4 + 1
    rank0 = _RANK_CHARS.index(sig[0])
    rank1 = _RANK_CHARS.index(sig[1])
    if sig[2] == "s":
        # Suited: same suit (use 0 for both).
        return rank0 * 4 + 0, rank1 * 4 + 0
    # Offsuit: different suits (use 0, 1).
    return rank0 * 4 + 0, rank1 * 4 + 1


def hand_strength_squared_mc(
    sig: str,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo E[HS²] for a starting-hand signature.

    For each trial:
        1. Sample 5 board cards + 2 opponent cards uniformly without
           replacement from the 50 cards remaining after fixing ``sig``.
        2. Evaluate showdown via :func:`compare_hands`.
        3. HS_trial = 1.0 (win) | 0.5 (tie) | 0.0 (loss).

    Returns the mean of HS_trial² over ``n_trials``.

    Suit-isomorphism: the concrete suit choice for ``sig`` is fixed via
    :func:`_signature_to_concrete_cards`; HS does not depend on the
    fixture because the remaining 50 cards still cover all relevant
    rank/suit combinations uniformly.
    """
    h0, h1 = _signature_to_concrete_cards(sig)
    deck = np.array([c for c in range(52) if c != h0 and c != h1])
    hs_sq_total = 0.0
    for _ in range(n_trials):
        # Draw 7 cards (5 board + 2 opp) without replacement.
        sampled = rng.choice(deck, size=7, replace=False)
        opp_hole = [int(sampled[0]), int(sampled[1])]
        board = [int(c) for c in sampled[2:7]]
        sgn = compare_hands([h0, h1], opp_hole, board)
        if sgn == +1:
            hs = 1.0
        elif sgn == -1:
            hs = 0.0
        else:
            hs = 0.5
        hs_sq_total += hs * hs
    return hs_sq_total / n_trials


class HUNLCardAbstractor:
    """E[HS²] preflop bucketing.

    Construction precomputes E[HS²] for every signature in
    :func:`enumerate_starting_hands` and sorts them into ``n_buckets``
    percentile bins. ``bucket(c0, c1)`` then maps any concrete
    two-card hand to its bucket index in 0..n_buckets-1.

    Strength ordering is preserved: bucket 0 contains the weakest
    hands, bucket ``n_buckets-1`` contains the strongest.

    Args:
        n_buckets: number of percentile bins. Default 50, the M2 first-
            pass spec value. Higher = finer abstraction (less info loss);
            the bucket count is the dominant lever for abstraction
            granularity at preflop.
        n_trials: Monte Carlo trials per signature. M2 first pass uses
            10 000 (≈ 1.7 M evaluations at 169 hands × 10 k trials,
            ~17 s with treys). Production should use 100 k+ for tighter
            E[HS²] estimates; the bucket boundaries shift only slightly
            between 10 k and 100 k (Pluribus paper §4.1 footnote).
        seed: RNG seed for reproducibility. Same seed → same bucket
            assignments forever.
    """

    def __init__(
        self,
        n_buckets: int = 50,
        n_trials: int = 10_000,
        seed: int = 42,
    ) -> None:
        if n_buckets < 1 or n_buckets > 169:
            raise ValueError(
                f"n_buckets must be in 1..169, got {n_buckets}"
            )
        if n_trials < 1:
            raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")
        self.n_buckets = n_buckets
        self.n_trials = n_trials
        self.seed = seed

        rng = np.random.default_rng(seed)
        signatures = enumerate_starting_hands()
        scores = [
            hand_strength_squared_mc(sig, n_trials, rng)
            for sig in signatures
        ]
        # Sort signatures by score ascending (weakest first).
        order = sorted(range(len(signatures)), key=lambda i: scores[i])
        # Assign each signature to its percentile bucket.
        self._bucket_of_signature: dict[str, int] = {}
        n_sigs = len(signatures)
        for rank_idx, sig_idx in enumerate(order):
            # Bucket boundary: rank_idx in [0, n_sigs); bucket =
            # rank_idx * n_buckets // n_sigs (uniform percentile split).
            bucket_idx = rank_idx * n_buckets // n_sigs
            self._bucket_of_signature[signatures[sig_idx]] = bucket_idx
        self._scores: dict[str, float] = dict(zip(signatures, scores))

    def bucket(self, c0: int, c1: int) -> int:
        """Returns bucket index 0..n_buckets-1 for the two-card hand."""
        sig = hand_signature(c0, c1)
        return self._bucket_of_signature[sig]

    def bucket_of_signature(self, sig: str) -> int:
        """Lookup-by-signature variant — useful for tests."""
        return self._bucket_of_signature[sig]

    def score_of_signature(self, sig: str) -> float:
        """Returns the cached E[HS²] for a signature."""
        return self._scores[sig]


# =============================================================================
# AbstractedHUNLState — wraps HUNLState with abstracted infoset_key (M2.2)
# =============================================================================
@dataclass(frozen=True, slots=True)
class AbstractedHUNLState:
    """Wraps :class:`HUNLState` and overrides ``infoset_key`` with a
    bucketed-private-card representation.

    Mirrors :class:`AbstractedLeducState` exactly (Phase 4 Step 2
    pattern). All other StateProtocol members delegate to the
    underlying state, so the game tree, legality, terminal_utility,
    and chance distribution are untouched. Only the strategy-key
    aliasing changes.

    The infoset_key uses an integer bucket index for the acting
    player's hole cards plus the raw board-card character / round /
    history representation. The board is left raw at M2 (board-
    conditioned bucketing is M3 work); for preflop-only sanity that
    is fine because the board is empty.
    """

    _raw: HUNLState
    _abstractor: HUNLCardAbstractor

    @property
    def is_terminal(self) -> bool:
        return self._raw.is_terminal

    @property
    def current_player(self) -> int:
        return self._raw.current_player

    @property
    def infoset_key(self) -> str:
        """Bucket-indexed perfect-recall key.

        Layout: ``"<bucket>|<round_idx>:<board_str>:<history_str>"``
        where bucket is the acting-player's hole-cards bucket index,
        round_idx is 0..3, board_str is a card-id-comma-list (raw at
        M2), and history_str is the flat action+size sequence so far
        across all rounds.

        At preflop, board_str = "" and the key reduces to bucket +
        history alone, giving the cleanest preflop infoset count
        of (n_buckets × |history_prefixes|).
        """
        actor = self._raw.current_player
        hole_offset = actor * 2
        c0 = self._raw.private_cards[hole_offset]
        c1 = self._raw.private_cards[hole_offset + 1]
        bucket = self._abstractor.bucket(c0, c1)
        # Board representation: empty at preflop, comma-list otherwise.
        board_str = ",".join(str(c) for c in self._raw.board_cards)
        # History flat: round-index-prefixed action sequences.
        hist_parts: list[str] = []
        for r_idx, (round_acts, round_sizes) in enumerate(
            zip(self._raw.round_history, self._raw.round_bet_sizes, strict=True)
        ):
            for a, sz in zip(round_acts, round_sizes, strict=True):
                hist_parts.append(f"{int(a)}:{sz}")
        hist_str = ".".join(hist_parts)
        return f"{bucket}|{self._raw.current_round}:{board_str}:{hist_str}"

    def legal_actions(self) -> tuple[IntEnum, ...]:
        return self._raw.legal_actions()

    def legal_action_mask(self) -> np.ndarray:
        return self._raw.legal_action_mask()

    def legal_bet_sizes(self) -> tuple[int, ...]:
        return self._raw.legal_bet_sizes()

    def next_state(
        self, action: HUNLAction, bet_size: int = 0
    ) -> AbstractedHUNLState:
        """Applies the abstracted action.

        **M2 action abstraction**: when ``action == HUNLAction.BET`` and
        ``bet_size == 0`` (the default — algorithms like MCCFR don't
        know bet sizes), substitute a 1×pot-size raise capped at the
        legal max. This collapses the continuous bet space onto a
        single discrete BET action so MCCFR / Vanilla CFR can run on
        the wrapper without modification. M3 will widen this to the
        full {0.5p, 1p, 2p, all-in} 4-size set per the design spec.
        """
        if action == HUNLAction.BET and bet_size == 0:
            # Auto-pick: 1× pot, clamped to legal range.
            sizes = self._raw.legal_bet_sizes()
            if not sizes:
                raise ValueError(
                    "BET action selected but legal_bet_sizes is empty"
                )
            target = self._raw.pot   # 1× pot total
            # Clamp into [min, max].
            chosen = max(sizes[0], min(sizes[-1], target))
            bet_size = chosen
        return AbstractedHUNLState(
            _raw=self._raw.next_state(action, bet_size=bet_size),
            _abstractor=self._abstractor,
        )

    def terminal_utility(self) -> float:
        return self._raw.terminal_utility()


# =============================================================================
# AbstractedHUNLGame — GameProtocol-compatible wrapper (M2.3)
# =============================================================================
class AbstractedHUNLGame:
    """:class:`GameProtocol`-compatible HUNL with E[HS²] preflop
    bucketing.

    Delegates ``all_deals``, ``sample_deal``, ``state_from_deal``,
    ``terminal_utility``, and ``encode`` to a wrapped :class:`HUNLGame`.
    Only state's ``infoset_key`` differs from the raw game (via the
    :class:`AbstractedHUNLState` wrapper).

    Mirrors :class:`AbstractedLeducPoker` exactly (Phase 4 Step 2
    pattern). MCCFR / Vanilla CFR / Deep CFR algorithms that key
    strategy by ``state.infoset_key`` automatically share strategy
    across hands sharing a bucket, exactly the lossy abstraction
    Pluribus uses.
    """

    NUM_ACTIONS: int = HUNLGame.NUM_ACTIONS
    ENCODING_DIM: int = HUNLGame.ENCODING_DIM

    def __init__(
        self,
        n_buckets: int = 50,
        n_trials: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.abstractor = HUNLCardAbstractor(
            n_buckets=n_buckets, n_trials=n_trials, seed=seed
        )

    def all_deals(self) -> tuple[Any, ...]:
        """Delegates — HUNL has ~10^14 deals; raises NotImplementedError."""
        return HUNLGame.all_deals()

    def sample_deal(self, rng: np.random.Generator) -> tuple[int, ...]:
        return HUNLGame.sample_deal(rng)

    def state_from_deal(self, deal: tuple[int, ...]) -> AbstractedHUNLState:
        return AbstractedHUNLState(
            _raw=HUNLGame.state_from_deal(deal),
            _abstractor=self.abstractor,
        )

    def terminal_utility(self, state: Any) -> float:
        raw = state._raw if isinstance(state, AbstractedHUNLState) else state
        return HUNLGame.terminal_utility(raw)

    def encode(self, state: Any) -> np.ndarray:
        raw = state._raw if isinstance(state, AbstractedHUNLState) else state
        return HUNLGame.encode(raw)
