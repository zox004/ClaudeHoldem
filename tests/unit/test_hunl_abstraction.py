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


# =============================================================================
# AbstractedHUNLState — wraps HUNLState with bucketed infoset_key (M2.2)
# =============================================================================
class TestAbstractedHUNLState:
    @pytest.fixture(scope="class")
    def game(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import AbstractedHUNLGame
        return AbstractedHUNLGame(n_buckets=50, n_trials=300, seed=42)

    def test_state_protocol_compliance(self, game) -> None:  # type: ignore[no-untyped-def]
        from poker_ai.games.protocol import StateProtocol
        rng = np.random.default_rng(42)
        deal = game.sample_deal(rng)
        state = game.state_from_deal(deal)
        assert isinstance(state, StateProtocol)

    def test_infoset_key_is_string(self, game) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(42)
        deal = game.sample_deal(rng)
        state = game.state_from_deal(deal)
        assert isinstance(state.infoset_key, str)

    def test_root_key_has_bucket_prefix(self, game) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(42)
        deal = game.sample_deal(rng)
        state = game.state_from_deal(deal)
        # Format: "<bucket>|<round>:<board>:<history>"
        prefix = state.infoset_key.split("|")[0]
        assert prefix.isdigit()
        assert 0 <= int(prefix) < game.abstractor.n_buckets

    def test_acting_player_perspective_bucket(self, game) -> None:  # type: ignore[no-untyped-def]
        """At root, current_player=1 (SB), so the bucket should be P1's
        hole-card bucket. After SB CALL, current_player=0 (BB), so the
        bucket should switch to P0's bucket."""
        from poker_ai.games.hunl_abstraction import AbstractedHUNLAction
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)   # P0 hole=(0,1), P1 hole=(2,3)
        state = game.state_from_deal(deal)
        # P1 acts first preflop.
        p1_bucket = game.abstractor.bucket(deal[2], deal[3])
        assert state.infoset_key.split("|")[0] == str(p1_bucket)
        # After SB CALL, P0 to act.
        state2 = state.next_state(AbstractedHUNLAction.CALL)
        p0_bucket = game.abstractor.bucket(deal[0], deal[1])
        assert state2.infoset_key.split("|")[0] == str(p0_bucket)

    def test_same_bucket_collapses_keys(self, game) -> None:  # type: ignore[no-untyped-def]
        """Two deals where the acting player's bucket is the same must
        produce identical root infoset_keys (proves abstraction is
        actually aliasing)."""
        # Find two different concrete hands with the same bucket.
        ab = game.abstractor
        # Sample some hands and look for collisions.
        rng = np.random.default_rng(0)
        bucket_to_hands: dict[int, list[tuple[int, int]]] = {}
        for _ in range(200):
            cards = rng.choice(52, size=2, replace=False)
            c0, c1 = int(cards[0]), int(cards[1])
            b = ab.bucket(c0, c1)
            bucket_to_hands.setdefault(b, []).append((c0, c1))
        # Find a bucket with ≥ 2 different hands.
        for b, hands in bucket_to_hands.items():
            if len(hands) >= 2:
                (h0_a, h1_a), (h0_b, h1_b) = hands[0], hands[1]
                if (h0_a, h1_a) == (h0_b, h1_b):
                    continue
                # Build deals: P1 (SB) holds the test hand.
                deal_a = (
                    20, 21, h0_a, h1_a,
                    *[c for c in range(52) if c not in (20, 21, h0_a, h1_a)][:5],
                )
                deal_b = (
                    20, 21, h0_b, h1_b,
                    *[c for c in range(52) if c not in (20, 21, h0_b, h1_b)][:5],
                )
                # Skip if deals overlap card-wise.
                try:
                    s_a = game.state_from_deal(deal_a)
                    s_b = game.state_from_deal(deal_b)
                except ValueError:
                    continue
                assert s_a.infoset_key == s_b.infoset_key, (
                    f"bucket {b}: hands {hands[0]} vs {hands[1]} should "
                    f"produce same root key"
                )
                return   # one example is sufficient
        pytest.skip("no two distinct hands sharing a bucket within sample")


# =============================================================================
# AbstractedHUNLGame — GameProtocol (M2.3)
# =============================================================================
class TestAbstractedHUNLGame:
    @pytest.fixture(scope="class")
    def game(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import AbstractedHUNLGame
        return AbstractedHUNLGame(n_buckets=50, n_trials=300, seed=42)

    def test_game_protocol_compliance(self, game) -> None:  # type: ignore[no-untyped-def]
        from poker_ai.games.protocol import GameProtocol
        assert isinstance(game, GameProtocol)
        # M3.2: NUM_ACTIONS lifted from 3 (raw HUNLGame) to 6
        # (AbstractedHUNLAction grid).
        assert game.NUM_ACTIONS == 6
        assert game.ENCODING_DIM == 102

    def test_sample_deal_delegates(self, game) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(42)
        deal = game.sample_deal(rng)
        assert len(deal) == 9
        assert len(set(deal)) == 9

    def test_all_deals_raises(self, game) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(NotImplementedError):
            game.all_deals()

    def test_state_from_deal_returns_abstracted(self, game) -> None:  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import AbstractedHUNLState
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        assert isinstance(state, AbstractedHUNLState)

    def test_terminal_utility_unchanged(self, game) -> None:  # type: ignore[no-untyped-def]
        """Wrapping doesn't change terminal_utility — abstraction only
        aliases strategy keys, not chip math."""
        from poker_ai.games.hunl_abstraction import AbstractedHUNLAction
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        terminal = state.next_state(AbstractedHUNLAction.FOLD)
        # SB folds → P0 wins SB blind.
        assert game.terminal_utility(terminal) == pytest.approx(+1.0)

    def test_encode_unchanged(self, game) -> None:  # type: ignore[no-untyped-def]
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        ab_state = game.state_from_deal(deal)
        from poker_ai.games.hunl import HUNLGame
        raw_state = HUNLGame.state_from_deal(deal)
        ab_enc = game.encode(ab_state)
        raw_enc = HUNLGame.encode(raw_state)
        assert np.array_equal(ab_enc, raw_enc)
