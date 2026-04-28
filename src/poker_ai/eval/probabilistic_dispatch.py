"""Probabilistic state translation (Schnizlein 2009 §3.2-§4).

Implements the **soft translation** geometric similarity from
Schnizlein, Bowling, Szafron 2009 — "Probabilistic State Translation
in Extensive Games with Large Action Sets" (IJCAI 2009).

Paper Eq. 5 / 6 (with ``b_1 < b < b_2``)::

    S(h, a, a_1) = (b_1/b - b_1/b_2) / (1 - b_1/b_2)
    S(h, a, a_2) = (b/b_2 - b_1/b_2) / (1 - b_1/b_2)

Boundary cases:

* ``b == b_1`` → ``S_1 = 1, S_2 = 0`` (paper §3.2 last paragraph)
* ``b == b_2`` → ``S_1 = 0, S_2 = 1``
* ``b == sqrt(b_1 * b_2)`` → ``S_1 = S_2 = 1/3`` (Eq. 4 hard-translation
  flip point; normalised weights are 0.5 each)

The "non-exponential" sampling form (§3.2 final paragraph) is used:
within one hand, a single bucket is sampled per opponent action with
a deterministic ``rng`` so the same hand always replays to the same
abstract history.

This module is the M4.5.2 swap-in for the M4.2 ``nearest_abstracted_bet_size``
hard-translation dispatch (which is preserved as the deterministic
A-arm of the dispatch_mode flag for the A/B mid-pilot).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# =============================================================================
# Eq. 5 / 6 — geometric similarity
# =============================================================================
def soft_similarity(*, b: int, b1: int, b2: int) -> tuple[float, float]:
    """Returns the unnormalised paper-Eq.-5/6 similarity pair
    ``(S_1, S_2)`` for a real bet ``b`` between abstract bets
    ``b_1 < b_2``.

    Edge cases ``b == b_1`` and ``b == b_2`` are handled explicitly
    so the formula's removable singularities never reach the divide:

    * ``b == b_1`` → ``(1.0, 0.0)``
    * ``b == b_2`` → ``(0.0, 1.0)``

    Raises:
        ValueError: if ``b1`` / ``b`` / ``b2`` are not all positive,
            if ``b1 >= b2``, or if ``b`` falls outside ``[b1, b2]``.
    """
    if b1 <= 0 or b <= 0 or b2 <= 0:
        raise ValueError(
            f"sizes must be positive; got b1={b1}, b={b}, b2={b2}"
        )
    if b1 >= b2:
        raise ValueError(f"b1 < b2 required; got b1={b1}, b2={b2}")
    if b < b1 or b > b2:
        raise ValueError(
            f"b1 < b < b2 required (boundaries inclusive); "
            f"got b1={b1}, b={b}, b2={b2}"
        )
    if b == b1:
        return 1.0, 0.0
    if b == b2:
        return 0.0, 1.0
    ratio = b1 / b2
    denom = 1.0 - ratio
    s1 = (b1 / b - ratio) / denom
    s2 = (b / b2 - ratio) / denom
    return s1, s2


# =============================================================================
# bucket_weights — over a sorted legal-bucket list
# =============================================================================
def bucket_weights(
    *, raw_chip: int, legal_sizes: list[int]
) -> dict[int, float]:
    """Returns ``{bucket_size: probability_weight}`` summing to 1.0
    using paper §3.2 + §4 soft translation.

    The two adjacent legal buckets straddling ``raw_chip`` get
    non-zero weight from :func:`soft_similarity` (normalised to sum
    to 1). All other buckets get weight 0 and are *omitted* from the
    returned dict (caller can iterate ``.items()`` without filtering).

    Edge cases:

    * ``raw_chip <= min(legal_sizes)`` → ``{min: 1.0}`` (deterministic
      single-bucket fallback; matches M4.2 ``nearest_abstracted_bet_size``
      below-min snap behaviour for arm comparability).
    * ``raw_chip >= max(legal_sizes)`` → ``{max: 1.0}``.
    * ``raw_chip`` exactly equals a legal size → ``{that_size: 1.0}``.

    Args:
        raw_chip: the real-game bet size in our internal chip units
            (BB=2 chips; convert from Slumbot via ``chip_from_slumbot``
            before calling).
        legal_sizes: legal abstract bet sizes; need not be sorted or
            deduplicated (we sort+dedup internally).

    Raises:
        ValueError: on empty ``legal_sizes``.
    """
    if not legal_sizes:
        raise ValueError("legal_sizes must not be empty")
    sorted_legal = sorted(set(legal_sizes))

    if raw_chip <= sorted_legal[0]:
        return {sorted_legal[0]: 1.0}
    if raw_chip >= sorted_legal[-1]:
        return {sorted_legal[-1]: 1.0}

    # raw_chip strictly inside [min, max]. Find the adjacent pair
    # (b1, b2) with b1 <= raw_chip <= b2.
    for i in range(len(sorted_legal) - 1):
        b1 = sorted_legal[i]
        b2 = sorted_legal[i + 1]
        if b1 <= raw_chip <= b2:
            if raw_chip == b1:
                return {b1: 1.0}
            if raw_chip == b2:
                return {b2: 1.0}
            s1, s2 = soft_similarity(b=raw_chip, b1=b1, b2=b2)
            total = s1 + s2
            return {b1: s1 / total, b2: s2 / total}

    # Should be unreachable given the boundary checks above.
    raise AssertionError(
        f"bucket_weights internal invariant: raw_chip={raw_chip} "
        f"did not fall in any adjacent pair of {sorted_legal}"
    )


# =============================================================================
# sample_bucket — non-exponential method (paper §3.2)
# =============================================================================
def sample_bucket(
    weights: Mapping[int, float], rng: np.random.Generator
) -> int:
    """Samples a single bucket key from ``weights`` using ``rng``.
    Singleton dicts (deterministic fallbacks from
    :func:`bucket_weights`) short-circuit without consuming randomness
    so the rng stream stays aligned with the per-hand seed contract.

    Args:
        weights: ``{bucket_size: probability}`` summing to 1.0 (no
            re-normalisation done here).
        rng: a :class:`numpy.random.Generator` (caller seeds it per
            hand for paper §3.2 internal-consistency invariant).
    """
    if len(weights) == 1:
        return next(iter(weights.keys()))
    keys = list(weights.keys())
    probs = np.asarray([weights[k] for k in keys], dtype=np.float64)
    idx = int(rng.choice(len(keys), p=probs))
    return int(keys[idx])
