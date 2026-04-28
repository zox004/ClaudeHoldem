"""Phase 4 M4.5.2 — Probabilistic state translation (Schnizlein 2009 §3.2-§4).

Locks in the **paper Eq. 5/6** geometric similarity exactly:

    S(h, a, a_1) = (b_1/b - b_1/b_2) / (1 - b_1/b_2)
    S(h, a, a_2) = (b/b_2 - b_1/b_2) / (1 - b_1/b_2)

with ``b_1 < b < b_2``. The unit tests cover:

* boundary cases (``b == b_1`` → ``(1, 0)``, ``b == b_2`` → ``(0, 1)``);
* the geometric-mean midpoint ``b == sqrt(b_1 * b_2)`` (where Eq. 4
  hard-translation flips between the two adjacent buckets);
* sum-to-one of the normalised weights;
* ``bucket_weights`` edge cases (``raw`` below min / above max /
  equal to a legal size → deterministic single-bucket fallback);
* internal consistency: same ``rng`` seed → same sampled bucket.

Source: Schnizlein, Bowling, Szafron 2009 — "Probabilistic State
Translation in Extensive Games with Large Action Sets" (IJCAI),
Section 3.2 + Section 4 (Eq. 5, 6). PDF cached locally.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from poker_ai.eval.probabilistic_dispatch import (
    bucket_weights,
    sample_bucket,
    soft_similarity,
)


# ---------------------------------------------------------------------------
# soft_similarity — paper Eq. 5/6
# ---------------------------------------------------------------------------
def test_soft_similarity_b_equals_b1_returns_1_0() -> None:
    """``b == b_1`` → ``S_1 = 1, S_2 = 0`` (paper invariant, §3.2 last
    paragraph: "if the original history being translated is a legal
    history in the abstract game, then soft translation will return
    this history with weight 1").
    """
    s1, s2 = soft_similarity(b=100, b1=100, b2=400)
    assert s1 == pytest.approx(1.0)
    assert s2 == pytest.approx(0.0)


def test_soft_similarity_b_equals_b2_returns_0_1() -> None:
    s1, s2 = soft_similarity(b=400, b1=100, b2=400)
    assert s1 == pytest.approx(0.0)
    assert s2 == pytest.approx(1.0)


def test_soft_similarity_geometric_midpoint_yields_one_third_each() -> None:
    """At ``b = sqrt(b_1 * b_2)`` (the hard-translation flip point),
    Eq. 5/6 give ``S_1 = S_2 = 1/3`` (proven on paper). Normalised
    weights are 0.5 each. This is the most discriminating numeric
    check that distinguishes our implementation from a naive linear
    interpolation.
    """
    b1, b2 = 100, 400
    b = int(math.sqrt(b1 * b2))   # 200
    s1, s2 = soft_similarity(b=b, b1=b1, b2=b2)
    assert s1 == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert s2 == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_soft_similarity_intermediate_value() -> None:
    """One concrete numeric: b1=100, b2=400, b=300.
    S_1 = (100/300 - 100/400) / (1 - 100/400)
        = (0.333... - 0.25) / 0.75
        = 0.0833.../0.75 = 0.1111...
    S_2 = (300/400 - 100/400) / 0.75
        = 0.5/0.75 = 0.666...
    """
    s1, s2 = soft_similarity(b=300, b1=100, b2=400)
    expected_s1 = (100 / 300 - 100 / 400) / (1 - 100 / 400)
    expected_s2 = (300 / 400 - 100 / 400) / (1 - 100 / 400)
    assert s1 == pytest.approx(expected_s1, rel=1e-9)
    assert s2 == pytest.approx(expected_s2, rel=1e-9)


def test_soft_similarity_rejects_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="b1 < b < b2"):
        soft_similarity(b=100, b1=200, b2=400)
    with pytest.raises(ValueError, match="b1 < b < b2"):
        soft_similarity(b=500, b1=100, b2=400)
    with pytest.raises(ValueError, match="b1 < b2"):
        soft_similarity(b=100, b1=400, b2=200)


def test_soft_similarity_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="positive"):
        soft_similarity(b=100, b1=0, b2=400)
    with pytest.raises(ValueError, match="positive"):
        soft_similarity(b=-1, b1=100, b2=400)


# ---------------------------------------------------------------------------
# bucket_weights — over a sorted legal-bucket list
# ---------------------------------------------------------------------------
def test_bucket_weights_normalised_pair_sums_to_one() -> None:
    """Output weights MUST sum to 1.0 — strategy lookup will weight
    per-bucket policies by these and over-weight by ε leaks into
    out-of-distribution behaviour at action selection time.
    """
    weights = bucket_weights(raw_chip=300, legal_sizes=[100, 200, 400, 800])
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)


def test_bucket_weights_picks_two_adjacent_buckets() -> None:
    """raw=300 between b1=200 and b2=400 → exactly those 2 buckets,
    others 0 (paper §4 second-to-last paragraph: "all other actions
    are given weight 0").
    """
    weights = bucket_weights(raw_chip=300, legal_sizes=[100, 200, 400, 800])
    assert set(weights.keys()) == {200, 400}


def test_bucket_weights_below_min_snaps_to_min() -> None:
    """Below-range raw → single-bucket deterministic fallback (matches
    nearest-bucket M4.2 contract at the boundary).
    """
    weights = bucket_weights(raw_chip=50, legal_sizes=[100, 200, 400])
    assert weights == {100: 1.0}


def test_bucket_weights_above_max_snaps_to_max() -> None:
    weights = bucket_weights(raw_chip=1000, legal_sizes=[100, 200, 400])
    assert weights == {400: 1.0}


def test_bucket_weights_exact_match_returns_singleton() -> None:
    """raw exactly equals a legal bucket → that bucket only, weight 1."""
    weights = bucket_weights(raw_chip=200, legal_sizes=[100, 200, 400])
    assert weights == {200: 1.0}


def test_bucket_weights_geometric_midpoint_is_50_50() -> None:
    """The strongest end-to-end check: at the geometric midpoint of
    two adjacent buckets, the *normalised* weights are 0.5/0.5
    (because the unnormalised similarities are both 1/3, see
    test_soft_similarity_geometric_midpoint_yields_one_third_each).
    """
    weights = bucket_weights(raw_chip=200, legal_sizes=[100, 400])
    # 200 is exactly sqrt(100*400). 200 also happens to equal one of
    # the legals only if 200 is in the list — here it isn't.
    assert weights[100] == pytest.approx(0.5)
    assert weights[400] == pytest.approx(0.5)


def test_bucket_weights_rejects_empty_legal() -> None:
    with pytest.raises(ValueError, match="legal_sizes"):
        bucket_weights(raw_chip=100, legal_sizes=[])


def test_bucket_weights_dedups_and_sorts_input() -> None:
    """Caller robustness — duplicate / unsorted inputs must yield
    the same result as a clean sorted list.
    """
    a = bucket_weights(raw_chip=300, legal_sizes=[400, 100, 200, 800])
    b = bucket_weights(raw_chip=300, legal_sizes=[800, 200, 200, 400, 100])
    assert a == b


# ---------------------------------------------------------------------------
# sample_bucket — paper §3.2 "non-exponential method"
# ---------------------------------------------------------------------------
def test_sample_bucket_deterministic_singleton_returns_only_key() -> None:
    """Single-key dict (raw == legal, or below-min / above-max) ⇒ no
    randomness consumed.
    """
    rng = np.random.default_rng(0)
    chosen = sample_bucket({200: 1.0}, rng)
    assert chosen == 200


def test_sample_bucket_same_seed_same_result() -> None:
    """Internal-consistency invariant from paper §3.2 final paragraph:
    "By assigning an ID to every game we play, we can seed our
    sampling process with a hash of the ID to ensure that, within
    one game, we will always return the same history given the same
    input." Our test uses raw seed but the contract is the same.
    """
    weights = {100: 0.5, 400: 0.5}
    a = sample_bucket(weights, np.random.default_rng(42))
    b = sample_bucket(weights, np.random.default_rng(42))
    assert a == b


def test_sample_bucket_distribution_matches_weights_in_expectation() -> None:
    """Statistical sanity: 10000 draws from {100: 0.3, 400: 0.7}
    should land within 3 sigma of the binomial expectation.
    """
    weights = {100: 0.3, 400: 0.7}
    rng = np.random.default_rng(0)
    n = 10_000
    counts = {100: 0, 400: 0}
    for _ in range(n):
        counts[sample_bucket(weights, rng)] += 1
    # binomial std for n=10000, p=0.3 is sqrt(10000*0.3*0.7) ≈ 45.8
    assert abs(counts[100] - 3000) < 3 * 46
