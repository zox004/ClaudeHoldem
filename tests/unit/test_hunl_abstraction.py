"""Unit tests for HUNL card abstraction (Phase 4 M2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.hunl_abstraction import (
    HUNLCardAbstractor,
    enumerate_starting_hands,
    hand_signature,
    hand_strength_squared_mc,
)


# Card-id helper for fixtures.
_R = {c: i for i, c in enumerate("23456789TJQKA")}
_S = {c: i for i, c in enumerate("cdhs")}


def cid(s: str) -> int:
    return _R[s[0]] * 4 + _S[s[1]]


# =============================================================================
# hand_signature — canonical form
# =============================================================================
class TestHandSignature:
    def test_pocket_pair_aa(self) -> None:
        assert hand_signature(cid("As"), cid("Ah")) == "AA"
        assert hand_signature(cid("Ah"), cid("As")) == "AA"   # symmetric

    def test_pocket_pair_22(self) -> None:
        assert hand_signature(cid("2c"), cid("2d")) == "22"

    def test_suited(self) -> None:
        assert hand_signature(cid("As"), cid("Ks")) == "AKs"
        # Order independent of arg order.
        assert hand_signature(cid("Ks"), cid("As")) == "AKs"

    def test_offsuit(self) -> None:
        assert hand_signature(cid("As"), cid("Kh")) == "AKo"

    def test_higher_rank_first(self) -> None:
        assert hand_signature(cid("7c"), cid("2d")) == "72o"

    def test_duplicate_card_raises(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            hand_signature(0, 0)


# =============================================================================
# enumerate_starting_hands — 169 canonical signatures
# =============================================================================
class TestEnumerateStartingHands:
    def test_count_169(self) -> None:
        """13 pairs + 78 suited + 78 offsuit = 169."""
        assert len(enumerate_starting_hands()) == 169

    def test_all_distinct(self) -> None:
        sigs = enumerate_starting_hands()
        assert len(set(sigs)) == 169

    def test_contains_pairs(self) -> None:
        sigs = enumerate_starting_hands()
        for r in "23456789TJQKA":
            assert (r + r) in sigs

    def test_contains_extreme_offsuits_and_suiteds(self) -> None:
        sigs = enumerate_starting_hands()
        assert "AKs" in sigs
        assert "AKo" in sigs
        assert "32s" in sigs
        assert "32o" in sigs

    def test_no_invalid_signatures(self) -> None:
        sigs = enumerate_starting_hands()
        for sig in sigs:
            if len(sig) == 2:
                assert sig[0] == sig[1]   # pair
            else:
                assert len(sig) == 3
                assert sig[2] in ("s", "o")
                assert sig[0] != sig[1]


# =============================================================================
# Monte Carlo E[HS²]
# =============================================================================
class TestHandStrengthSquaredMC:
    def test_aa_stronger_than_72o(self) -> None:
        """Sanity: AA's E[HS²] should exceed 72o's by a wide margin."""
        rng = np.random.default_rng(seed=42)
        aa = hand_strength_squared_mc("AA", 500, rng)
        rng = np.random.default_rng(seed=42)   # reset so MC trials match
        seventytwoo = hand_strength_squared_mc("72o", 500, rng)
        assert aa > seventytwoo

    def test_aa_above_05(self) -> None:
        """AA wins more than 50% of the time → HS > 0.5 → E[HS²] > 0.25."""
        rng = np.random.default_rng(seed=42)
        score = hand_strength_squared_mc("AA", 1000, rng)
        assert score > 0.5

    def test_seed_reproducibility(self) -> None:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        score_a = hand_strength_squared_mc("AKs", 500, rng_a)
        score_b = hand_strength_squared_mc("AKs", 500, rng_b)
        assert score_a == score_b


# =============================================================================
# HUNLCardAbstractor — bucket assignment
# =============================================================================
class TestHUNLCardAbstractor:
    @pytest.fixture(scope="class")
    def abstractor(self) -> HUNLCardAbstractor:
        # Small n_trials for fast tests.
        return HUNLCardAbstractor(n_buckets=50, n_trials=500, seed=42)

    def test_invalid_n_buckets_raises(self) -> None:
        with pytest.raises(ValueError, match="n_buckets"):
            HUNLCardAbstractor(n_buckets=0)
        with pytest.raises(ValueError, match="n_buckets"):
            HUNLCardAbstractor(n_buckets=170)

    def test_invalid_n_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            HUNLCardAbstractor(n_buckets=50, n_trials=0)

    def test_aa_in_top_bucket(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        """AA should land in the highest bucket (n_buckets - 1)."""
        assert abstractor.bucket_of_signature("AA") == abstractor.n_buckets - 1

    def test_32o_in_bottom_bucket(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        """32o should land in the lowest bucket (0)."""
        assert abstractor.bucket_of_signature("32o") == 0

    def test_bucket_method_matches_signature_lookup(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        """``bucket(c0, c1)`` and ``bucket_of_signature(sig)`` agree."""
        c0, c1 = cid("As"), cid("Ah")
        assert abstractor.bucket(c0, c1) == abstractor.bucket_of_signature("AA")

    def test_bucket_indices_in_range(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        sigs = enumerate_starting_hands()
        for sig in sigs:
            b = abstractor.bucket_of_signature(sig)
            assert 0 <= b < abstractor.n_buckets

    def test_buckets_cover_all_169_hands(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        sigs = enumerate_starting_hands()
        assigned = set(
            abstractor.bucket_of_signature(s) for s in sigs
        )
        # With 169 hands × 50 buckets, every bucket should have hands.
        assert len(assigned) == abstractor.n_buckets

    def test_seed_reproducibility(self) -> None:
        ab_a = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=42)
        ab_b = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=42)
        for sig in enumerate_starting_hands():
            assert ab_a.bucket_of_signature(sig) == ab_b.bucket_of_signature(sig)

    def test_different_seeds_can_differ(self) -> None:
        """Different seeds shouldn't collapse to identical bucket maps —
        Monte Carlo noise should shuffle at least the boundary hands.

        At low n_trials (300 here), the MC estimate is noisy enough
        that many mid-strength hands shift buckets between seeds. The
        assertion here is just "some difference exists"; extreme hands
        (AA, 32o) should still anchor consistent extreme buckets — see
        :meth:`test_extreme_hands_stable_across_seeds`."""
        ab_a = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=42)
        ab_b = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=43)
        sigs = enumerate_starting_hands()
        diffs = sum(
            1 for s in sigs
            if ab_a.bucket_of_signature(s) != ab_b.bucket_of_signature(s)
        )
        assert diffs > 0   # at least one difference

    def test_extreme_hands_stable_across_seeds(self) -> None:
        """AA must stay in the top bucket and 32o in the bottom bucket
        regardless of seed — these are the most decisively-strong /
        weak preflop hands and are not boundary cases."""
        ab_a = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=42)
        ab_b = HUNLCardAbstractor(n_buckets=50, n_trials=300, seed=43)
        assert ab_a.bucket_of_signature("AA") == ab_a.n_buckets - 1
        assert ab_b.bucket_of_signature("AA") == ab_b.n_buckets - 1
        assert ab_a.bucket_of_signature("32o") == 0
        assert ab_b.bucket_of_signature("32o") == 0

    def test_smaller_buckets_compress(self) -> None:
        """n_buckets=10 means each bucket holds ~17 hands; fewer
        distinct bucket indices used."""
        ab10 = HUNLCardAbstractor(n_buckets=10, n_trials=300, seed=42)
        sigs = enumerate_starting_hands()
        used = set(ab10.bucket_of_signature(s) for s in sigs)
        assert len(used) == 10

    def test_score_lookup_returns_finite(
        self, abstractor: HUNLCardAbstractor
    ) -> None:
        sigs = enumerate_starting_hands()
        for sig in sigs:
            score = abstractor.score_of_signature(sig)
            assert 0.0 <= score <= 1.0
