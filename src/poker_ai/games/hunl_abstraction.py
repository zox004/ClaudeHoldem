"""HUNL card abstraction (Phase 4 M2 + M3.1).

Implements the Pluribus-style E[HS²] bucketing for both preflop hole
cards (M2) and postflop public boards (M3.1). With ~50 buckets per
round, preflop infoset count drops from 169 × <action histories> to
50 × <action histories>, and postflop nodes (~10^7 raw signatures
per round) collapse to ~50 × <history prefix> at each of flop / turn /
river. Step 2 / Leduc validated the mechanism: information loss <
sampling-variance gain under finite compute.

Module layout:

- :func:`hand_signature` — canonical "AKs" / "AKo" / "AA" string for
  any (c0, c1) two-card hand. Pair = "RR" (no suffix), suited = "RRs",
  offsuit = "RRo", higher rank first.
- :func:`enumerate_starting_hands` — returns all 169 canonical
  signatures (13 pairs + 78 suited + 78 offsuit).
- :func:`hand_strength_squared_mc` — Monte Carlo E[HS²] for one preflop
  hand (M2).
- :func:`hand_strength_squared_postflop_mc` — Monte Carlo E[HS²] for a
  given (hole, board) postflop pair (M3.1).
- :class:`HUNLCardAbstractor` — precomputes E[HS²] for every preflop
  signature, sorts, and assigns each into one of ``n_buckets``
  percentile buckets (M2).
- :class:`PostflopBoardAbstractor` — round-aware (flop/turn/river)
  E[HS²] percentile bucketing with a lazy per-call cache and hit-rate
  reporting (M3.1).

E[HS² ] formulation (Ganzfried & Sandholm 2014, equation 3):
    For starting hand h on a public board b:
        HS(h, b) = P_{opp,future board}[h beats opp on (h, opp, b ∪ future)]
        E[HS² (h)] = E_{b, future}[ HS(h, b ∪ future)² ]

We approximate via Monte Carlo. Preflop: draw a board (5 cards) and
an opponent hand (2 cards) uniformly. Postflop: fix the public board
prefix (3/4/5 cards), then draw the remaining cards (turn/river that
have not yet appeared) plus opponent hand uniformly. Win = 1.0, tie =
0.5, loss = 0.0; HS_sample is the indicator over a single opponent
hand; we compensate via larger n_trials. Pluribus-grade implementations
average over many opponent hands per board to reduce variance —
deferred to Phase 5.

M3.1 audit candidate (PHASE.md self-audit #22 hook): scalar percentile
binning is a *deviation* from the Pluribus-standard distribution-aware
K-means with EMD. We keep scalar percentiles for M3 because (a) the
Step 2 Leduc result already validates the mechanism on scalar features,
(b) Phase 4 timeline favours pattern-transfer over re-design, (c) the
upgrade to distribution-aware buckets is a well-defined Phase 5 hook.

Determinism: every abstractor seeds its RNG; same seed reproduces
thresholds and bucket assignments exactly.

M2 → M3 strategy compatibility (intentional break)
--------------------------------------------------
:class:`AbstractedHUNLState`'s infoset_key under M3 substitutes a
postflop bucket index for the raw card-id comma list used in M2.
**Strategies trained under M2 cannot be transferred onto M3 trainers**
because the keys no longer line up. This is intentional — the M3 keys
are what allow MCCFR to actually share strategy across boards within a
bucket. The break is documented and tested
(``test_m2_compat_break_is_visible``); fresh strategies start from M3.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
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
# Postflop board bucketing (M3.1)
# =============================================================================
# Round indices match HUNLState.current_round: 0=preflop, 1=flop, 2=turn,
# 3=river. The postflop abstractor is responsible only for rounds 1..3.
_POSTFLOP_ROUNDS: Final[tuple[int, ...]] = (1, 2, 3)
_BOARD_LEN_BY_ROUND: Final[dict[int, int]] = {1: 3, 2: 4, 3: 5}


def hand_strength_squared_postflop_mc(
    hole: tuple[int, int],
    board: tuple[int, ...],
    mc_trials: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo E[HS²] for a fixed (hole, board) postflop pair.

    Round is inferred from ``len(board)`` (3=flop, 4=turn, 5=river).
    For each trial:
        1. Draw the missing board cards (turn/river that haven't fallen
           yet) plus 2 opponent hole cards uniformly from the deck
           remaining after fixing ``hole + board``.
        2. Evaluate showdown via :func:`compare_hands`.
        3. HS_trial = 1.0 (win) | 0.5 (tie) | 0.0 (loss).

    Returns the mean of HS_trial² over ``mc_trials``.

    The function does not validate inputs (callers — typically
    :class:`PostflopBoardAbstractor` — already validate). Negative or
    out-of-range card ids are passed through to ``rng.choice``'s deck
    construction, which would silently produce garbage; valid inputs
    are the caller's contract.
    """
    used = set(hole) | set(board)
    deck = np.array([c for c in range(52) if c not in used])
    board_len = len(board)
    # Cards still to be revealed: 5 - board_len board cards + 2 opp hole.
    sample_size = (5 - board_len) + 2
    hs_sq_total = 0.0
    h0, h1 = hole[0], hole[1]
    board_list = list(board)
    for _ in range(mc_trials):
        sampled = rng.choice(deck, size=sample_size, replace=False)
        # Layout: first (5 - board_len) entries fill the future board,
        # last 2 are the opponent's hole.
        future_board_n = 5 - board_len
        full_board = board_list + [int(c) for c in sampled[:future_board_n]]
        opp_hole = [int(sampled[future_board_n]),
                    int(sampled[future_board_n + 1])]
        sgn = compare_hands([h0, h1], opp_hole, full_board)
        hs = 1.0 if sgn == 1 else (0.5 if sgn == 0 else 0.0)
        hs_sq_total += hs * hs
    return hs_sq_total / mc_trials


class PostflopBoardAbstractor:
    """Round-aware postflop E[HS²] percentile bucketing (M3.1).

    Buckets every (acting-player hole, public board, round_idx ∈
    {1, 2, 3}) into one of ``n_buckets`` strength bins. Same E[HS²]
    scalar-percentile pattern as :class:`HUNLCardAbstractor`, but
    conditioned on the public board, with separate thresholds per
    round.

    The bucket assignment for any concrete (hole, board) is computed
    on first request and cached; repeated lookups (typical inside
    MCCFR) are dict-O(1). The cache is keyed on a canonical signature
    that ignores the concrete ordering of cards but currently keeps
    suit identity (i.e. partial canonicalisation only — full suit-
    isomorphism is deferred to Phase 5 because the marginal cache-hit
    gain is small and the implementation cost is non-trivial).

    Args:
        n_buckets: number of percentile bins per round. Default 50,
            matching the M2 first pass. Higher = finer abstraction
            (less info loss) at the cost of more bucket cells to fill.
        mc_trials: Monte Carlo trials per E[HS²] evaluation. Default
            300; production should consider 1 000+. Memory cost is
            independent of mc_trials.
        threshold_sample_size: number of random (hole, board) pairs
            drawn per round to derive the percentile thresholds.
            Default 200 — small enough that test fixtures construct
            in seconds; production runs (M3.4 baseline measurement)
            should use 5 000+ via explicit override.
        seed: deterministic RNG seed for both the threshold-derivation
            sampling and the per-call MC evaluations.
    """

    def __init__(
        self,
        n_buckets: int = 50,
        mc_trials: int = 300,
        threshold_sample_size: int = 200,
        seed: int = 42,
    ) -> None:
        if n_buckets < 1:
            raise ValueError(f"n_buckets must be ≥ 1, got {n_buckets}")
        if mc_trials < 1:
            raise ValueError(f"mc_trials must be ≥ 1, got {mc_trials}")
        if threshold_sample_size < n_buckets:
            raise ValueError(
                f"threshold_sample_size must be ≥ n_buckets ({n_buckets}), "
                f"got {threshold_sample_size}"
            )
        self.n_buckets = n_buckets
        self.mc_trials = mc_trials
        self.threshold_sample_size = threshold_sample_size
        self.seed = seed

        # Two RNGs: one for threshold derivation (consumed during init),
        # one for online lookup MC evaluations. Decoupling keeps the
        # threshold table independent of how many lookups happen later.
        rng_thresh = np.random.default_rng(seed)
        self._rng_query = np.random.default_rng(seed + 0xC0FFEE)

        self.thresholds: dict[int, list[float]] = {}
        for round_idx in _POSTFLOP_ROUNDS:
            board_len = _BOARD_LEN_BY_ROUND[round_idx]
            scores: list[float] = []
            for _ in range(threshold_sample_size):
                draw = rng_thresh.choice(52, size=2 + board_len, replace=False)
                hole = (int(draw[0]), int(draw[1]))
                board = tuple(int(c) for c in draw[2 : 2 + board_len])
                scores.append(
                    hand_strength_squared_postflop_mc(
                        hole, board, mc_trials, rng_thresh
                    )
                )
            scores.sort()
            # n_buckets - 1 internal cut points at uniform percentile.
            cuts = [
                scores[(i + 1) * threshold_sample_size // n_buckets - 1]
                for i in range(n_buckets - 1)
            ]
            self.thresholds[round_idx] = cuts

        self._cache: dict[tuple[str, tuple[int, ...], int], int] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def bucket(
        self,
        hole: tuple[int, int],
        board: tuple[int, ...],
        round_idx: int,
    ) -> int:
        """Returns a bucket index in ``[0, n_buckets)`` for the given
        postflop (hole, board) pair.

        Validates inputs strictly; raises :class:`ValueError` on any
        violation. Lazy: each (hole_canonical, board_sorted, round_idx)
        signature triggers one E[HS²] evaluation on first access and is
        cached thereafter.
        """
        if round_idx not in _POSTFLOP_ROUNDS:
            raise ValueError(
                f"round_idx must be in {_POSTFLOP_ROUNDS} "
                f"(preflop=0 is out of scope for PostflopBoardAbstractor); "
                f"got {round_idx}"
            )
        if len(hole) != 2:
            raise ValueError(f"hole must have 2 cards, got {len(hole)}")
        expected_board_len = _BOARD_LEN_BY_ROUND[round_idx]
        if len(board) != expected_board_len:
            raise ValueError(
                f"board for round_idx={round_idx} must have "
                f"{expected_board_len} cards, got {len(board)}"
            )
        all_cards = list(hole) + list(board)
        for c in all_cards:
            if not 0 <= c < 52:
                raise ValueError(f"card id {c} out of range 0..51")
        if len(set(all_cards)) != len(all_cards):
            raise ValueError(
                f"hole + board must have distinct cards; got {all_cards}"
            )

        hole_sig = hand_signature(hole[0], hole[1])
        board_sorted = tuple(sorted(board))
        key = (hole_sig, board_sorted, round_idx)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        score = hand_strength_squared_postflop_mc(
            hole, board, self.mc_trials, self._rng_query
        )
        cuts = self.thresholds[round_idx]
        # bisect_left places score into [0, n_buckets - 1] (inclusive);
        # ties to the lower side. Strength order is preserved (lower
        # score → lower bucket).
        bucket_idx = bisect.bisect_left(cuts, score)
        if bucket_idx >= self.n_buckets:
            bucket_idx = self.n_buckets - 1
        self._cache[key] = bucket_idx
        return bucket_idx

    def cache_stats(self) -> dict[str, int | float]:
        """Returns ``{hits, misses, total, hit_rate}`` for the lazy
        cache. ``hit_rate`` is ``hits / total`` or 0.0 if ``total ==
        0``. Used by M3.4 baseline run reporting (PHASE.md note: a
        hit-rate < 0.5 indicates the lazy strategy is not paying off
        and signals an architecture re-think for M5)."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
        }


# =============================================================================
# AbstractedHUNLState — wraps HUNLState with abstracted infoset_key (M2.2)
# =============================================================================
@dataclass(frozen=True, slots=True)
class AbstractedHUNLState:
    """Wraps :class:`HUNLState` and overrides ``infoset_key`` with a
    bucketed private-card AND public-board representation.

    Mirrors :class:`AbstractedLeducState` exactly (Phase 4 Step 2
    pattern). All other StateProtocol members delegate to the
    underlying state, so the game tree, legality, terminal_utility,
    and chance distribution are untouched. Only the strategy-key
    aliasing changes.

    M2 layout (preflop only): ``"<hole_bucket>|0::<history>"`` (board
    segment empty).

    M3.1 layout (postflop, when ``_postflop_abstractor`` is set):
    ``"<hole_bucket>|<round>:<board_bucket>:<history>"``. The
    ``board_bucket`` is a single integer produced by
    :class:`PostflopBoardAbstractor`; the M2 raw card-id comma list
    is gone (intentional compatibility break — see module docstring).

    Backward-compatibility: ``_postflop_abstractor`` defaults to
    ``None``. When None and the state is postflop, the wrapper falls
    back to the M2 raw board-string layout. The standard
    :class:`AbstractedHUNLGame` factory always provides the postflop
    abstractor; the None branch exists only for tests / direct
    construction patterns from M2.
    """

    _raw: HUNLState
    _abstractor: HUNLCardAbstractor
    _postflop_abstractor: PostflopBoardAbstractor | None = field(default=None)

    @property
    def is_terminal(self) -> bool:
        return self._raw.is_terminal

    @property
    def current_player(self) -> int:
        return self._raw.current_player

    @property
    def infoset_key(self) -> str:
        """Bucket-indexed perfect-recall key.

        Layout: ``"<hole_bucket>|<round_idx>:<board_segment>:<history>"``
        where ``hole_bucket`` is the acting-player's hole-cards bucket
        index, ``round_idx`` is 0..3, ``board_segment`` is empty at
        preflop and a single bucket index at flop / turn / river (M3.1),
        and ``history`` is the flat action+size sequence across rounds.

        Strength order is preserved across buckets (lower bucket =
        weaker hand) for both preflop and postflop, so MCCFR's strategy
        sharing is meaningful.
        """
        actor = self._raw.current_player
        hole_offset = actor * 2
        c0 = self._raw.private_cards[hole_offset]
        c1 = self._raw.private_cards[hole_offset + 1]
        hole_bucket = self._abstractor.bucket(c0, c1)
        board_str = self._board_segment(c0, c1)
        # History flat: round-index-prefixed action sequences.
        hist_parts: list[str] = []
        for round_acts, round_sizes in zip(
            self._raw.round_history, self._raw.round_bet_sizes, strict=True
        ):
            for a, sz in zip(round_acts, round_sizes, strict=True):
                hist_parts.append(f"{int(a)}:{sz}")
        hist_str = ".".join(hist_parts)
        return (
            f"{hole_bucket}|{self._raw.current_round}:"
            f"{board_str}:{hist_str}"
        )

    def _board_segment(self, c0: int, c1: int) -> str:
        """Postflop board segment (M3.1) or empty string (preflop).

        Falls back to the M2 raw card-id comma list when no postflop
        abstractor is wired in — only used by direct test fixtures
        that bypass :class:`AbstractedHUNLGame`.
        """
        if self._raw.current_round == 0 or not self._raw.board_cards:
            return ""
        if self._postflop_abstractor is None:
            return ",".join(str(c) for c in self._raw.board_cards)
        return str(
            self._postflop_abstractor.bucket(
                (c0, c1),
                self._raw.board_cards,
                self._raw.current_round,
            )
        )

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
            _postflop_abstractor=self._postflop_abstractor,
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
        postflop_mc_trials: int | None = None,
        postflop_threshold_sample_size: int | None = None,
    ) -> None:
        """Constructs both the preflop abstractor (M2) and the postflop
        abstractor (M3.1).

        ``postflop_mc_trials`` and ``postflop_threshold_sample_size``
        default to test-friendly small values derived from
        ``n_trials`` / ``n_buckets``. Production runs (M3.4 baseline)
        should pass explicit larger values; default precompute is a
        few seconds for the test suite.
        """
        self.abstractor = HUNLCardAbstractor(
            n_buckets=n_buckets, n_trials=n_trials, seed=seed
        )
        # Default scaling: postflop MC ~ preflop MC / 4 (postflop hand
        # strength has lower variance because the board is fixed and
        # only turn/river plus opp hole are sampled). Threshold sample
        # size ~ max(4 × n_buckets, 200) — enough to derive percentile
        # cuts without dominating fixture-construction time.
        mc = (
            postflop_mc_trials
            if postflop_mc_trials is not None
            else max(n_trials // 4, 50)
        )
        sample = (
            postflop_threshold_sample_size
            if postflop_threshold_sample_size is not None
            else max(4 * n_buckets, 200)
        )
        self.postflop_abstractor = PostflopBoardAbstractor(
            n_buckets=n_buckets,
            mc_trials=mc,
            threshold_sample_size=sample,
            # Decorrelate from preflop seed to avoid spurious
            # cross-round correlation from a shared RNG state.
            seed=seed + 0xBEEF,
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
            _postflop_abstractor=self.postflop_abstractor,
        )

    def terminal_utility(self, state: Any) -> float:
        raw = state._raw if isinstance(state, AbstractedHUNLState) else state
        return HUNLGame.terminal_utility(raw)

    def encode(self, state: Any) -> np.ndarray:
        raw = state._raw if isinstance(state, AbstractedHUNLState) else state
        return HUNLGame.encode(raw)
