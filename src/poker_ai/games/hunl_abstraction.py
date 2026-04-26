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

from typing import Final

import numpy as np

from poker_ai.games.hunl_hand_eval import compare_hands

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
