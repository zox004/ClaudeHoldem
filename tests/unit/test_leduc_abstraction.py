"""Unit tests for Leduc abstraction wrapper (Phase 4 Step 2)."""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.leduc import LeducAction, LeducPoker
from poker_ai.games.leduc_abstraction import (
    AbstractedLeducPoker,
    AbstractedLeducState,
    CardAbstractor,
)
from poker_ai.games.protocol import GameProtocol, StateProtocol


# =============================================================================
# CardAbstractor — bucket assignment correctness
# =============================================================================
class TestCardAbstractor:
    def test_invalid_n_buckets_raises(self) -> None:
        with pytest.raises(ValueError, match="n_buckets"):
            CardAbstractor(n_buckets=1)
        with pytest.raises(ValueError, match="n_buckets"):
            CardAbstractor(n_buckets=4)

    def test_invalid_rank_raises(self) -> None:
        ab = CardAbstractor(n_buckets=2)
        with pytest.raises(ValueError, match="rank"):
            ab.bucket(-1)
        with pytest.raises(ValueError, match="rank"):
            ab.bucket(3)

    def test_n_buckets_3_is_identity(self) -> None:
        ab = CardAbstractor(n_buckets=3)
        assert ab.bucket(0) == 0
        assert ab.bucket(1) == 1
        assert ab.bucket(2) == 2

    def test_n_buckets_2_collapses_low_pair(self) -> None:
        """{J=0, Q=1} → bucket 0 ('L'); {K=2} → bucket 1 ('H')."""
        ab = CardAbstractor(n_buckets=2)
        assert ab.bucket(0) == 0  # J
        assert ab.bucket(1) == 0  # Q
        assert ab.bucket(2) == 1  # K

    def test_bucket_char_n_3(self) -> None:
        ab = CardAbstractor(n_buckets=3)
        assert ab.bucket_char(0) == "J"
        assert ab.bucket_char(1) == "Q"
        assert ab.bucket_char(2) == "K"

    def test_bucket_char_n_2(self) -> None:
        ab = CardAbstractor(n_buckets=2)
        assert ab.bucket_char(0) == "L"
        assert ab.bucket_char(1) == "L"
        assert ab.bucket_char(2) == "H"

    def test_deterministic(self) -> None:
        """Same rank → same bucket on every call."""
        ab = CardAbstractor(n_buckets=2)
        for _ in range(50):
            for r in (0, 1, 2):
                assert ab.bucket(r) == ab.bucket(r)
                assert ab.bucket_char(r) == ab.bucket_char(r)


# =============================================================================
# AbstractedLeducState — StateProtocol compliance + infoset_key correctness
# =============================================================================
class TestAbstractedLeducState:
    def _root_state(
        self, n_buckets: int = 2
    ) -> tuple[AbstractedLeducPoker, AbstractedLeducState]:
        game = AbstractedLeducPoker(n_buckets=n_buckets)
        state = game.state_from_deal(game.all_deals()[0])
        return game, state

    def test_state_protocol_compliance(self) -> None:
        _, state = self._root_state()
        assert isinstance(state, StateProtocol)

    def test_forwards_is_terminal(self) -> None:
        _, state = self._root_state()
        assert state.is_terminal is False

    def test_forwards_current_player(self) -> None:
        _, state = self._root_state()
        assert state.current_player == 0

    def test_forwards_legal_actions(self) -> None:
        _, state = self._root_state()
        legal = state.legal_actions()
        # Both rounds open with CALL + RAISE legal (no FOLD on first action).
        assert LeducAction.CALL in legal
        assert LeducAction.RAISE in legal

    def test_forwards_legal_action_mask(self) -> None:
        _, state = self._root_state()
        mask = state.legal_action_mask()
        assert mask.shape == (LeducPoker.NUM_ACTIONS,)
        assert mask.dtype == np.bool_

    def test_next_state_returns_abstracted(self) -> None:
        _, state = self._root_state()
        nxt = state.next_state(LeducAction.CALL)
        assert isinstance(nxt, AbstractedLeducState)

    def test_n_buckets_3_infoset_key_matches_raw(self) -> None:
        """3-bucket is the identity map — every abstracted infoset_key
        must equal the raw LeducState's key."""
        game = AbstractedLeducPoker(n_buckets=3)
        raw_game = LeducPoker()
        for deal in game.all_deals()[:30]:
            ab_state = game.state_from_deal(deal)
            raw_state = raw_game.state_from_deal(deal)
            assert ab_state.infoset_key == raw_state.infoset_key

    def test_n_buckets_2_uses_LH_chars_for_private_card(self) -> None:
        """2-bucket key starts with 'L' (J or Q hand) or 'H' (K hand)."""
        game = AbstractedLeducPoker(n_buckets=2)
        for deal in game.all_deals():
            state = game.state_from_deal(deal)
            assert state.infoset_key[0] in ("L", "H")

    def test_n_buckets_2_collapses_J_and_Q_keys(self) -> None:
        """Different deals where current player holds J vs Q must produce
        the same bucket character at the root."""
        game = AbstractedLeducPoker(n_buckets=2)
        # Find a deal where P1 holds J (card 0 or 1) and one where holds Q.
        # all_deals returns (p1_card, p2_card, board) tuples.
        deal_with_J = next(d for d in game.all_deals() if d[0] in (0, 1))
        # Among deals where p1 holds J, find one where p1 also has card 0
        # (rank 0). That's J. Now find a deal where p1 holds Q (rank 1).
        deal_with_Q = next(d for d in game.all_deals() if d[0] in (2, 3))
        s_j = game.state_from_deal(deal_with_J)
        s_q = game.state_from_deal(deal_with_Q)
        # Both should start with 'L' at the root (P1 acts first).
        assert s_j.infoset_key.startswith("L")
        assert s_q.infoset_key.startswith("L")
        # And they should be equal (both same bucket, same empty history).
        assert s_j.infoset_key == s_q.infoset_key

    def test_board_char_remains_raw_in_round_2(self) -> None:
        """Board character is left raw (J/Q/K) — only the player's
        private rank is bucketed."""
        game = AbstractedLeducPoker(n_buckets=2)
        # Step into round 2: P1 CALL, P2 CALL.
        deal = (0, 2, 4)  # J, K, K (deck)  — board card 4 is rank 2 = K
        state = game.state_from_deal(deal)
        s1 = state.next_state(LeducAction.CALL)
        s2 = s1.next_state(LeducAction.CALL)
        # Now we're in round 2.
        assert "K" in s2.infoset_key  # board char K should appear


# =============================================================================
# AbstractedLeducPoker — GameProtocol compliance + infoset count
# =============================================================================
class TestAbstractedLeducPoker:
    def test_default_n_buckets_is_2(self) -> None:
        game = AbstractedLeducPoker()
        assert game.abstractor.n_buckets == 2

    def test_game_protocol_compliance(self) -> None:
        game = AbstractedLeducPoker(n_buckets=2)
        assert isinstance(game, GameProtocol)
        assert game.NUM_ACTIONS == LeducPoker.NUM_ACTIONS
        assert game.ENCODING_DIM == LeducPoker.ENCODING_DIM

    def test_all_deals_matches_raw(self) -> None:
        game = AbstractedLeducPoker(n_buckets=2)
        raw = LeducPoker()
        assert game.all_deals() == raw.all_deals()

    def test_terminal_utility_unchanged(self) -> None:
        """Wrapping must not alter zero-sum utility."""
        game = AbstractedLeducPoker(n_buckets=2)
        # Terminal via FOLD on the first action. P1 holds J, P2 holds K.
        state = game.state_from_deal((0, 4, 5))
        # P1 FOLD → P2 wins ante (1 chip).
        terminal = state.next_state(LeducAction.FOLD)
        assert terminal.is_terminal
        u = game.terminal_utility(terminal)
        # Raw Leduc sign convention: P1 fold loses 1 chip → -1.0.
        assert u == pytest.approx(-1.0)

    def test_encode_unchanged(self) -> None:
        game = AbstractedLeducPoker(n_buckets=2)
        state = game.state_from_deal(game.all_deals()[0])
        raw = LeducPoker().state_from_deal(LeducPoker().all_deals()[0])
        assert np.array_equal(game.encode(state), LeducPoker().encode(raw))

    def test_n_infosets_n_buckets_3_matches_raw(self) -> None:
        """3-bucket abstraction must preserve infoset count (288)."""
        game = AbstractedLeducPoker(n_buckets=3)
        assert game.n_infosets() == 288

    def test_n_infosets_n_buckets_2_reduces(self) -> None:
        """2-bucket abstraction must reduce infoset count below 288."""
        game = AbstractedLeducPoker(n_buckets=2)
        n = game.n_infosets()
        assert n < 288
        assert n > 0

    def test_n_infosets_n_buckets_2_exact_count(self) -> None:
        """Regression on the smoke-measured value."""
        game = AbstractedLeducPoker(n_buckets=2)
        assert game.n_infosets() == 192

    def test_sample_deal_delegates_to_raw(self) -> None:
        """Phase 4 Step 3 (M0) — abstracted game's sample_deal delegates
        to the raw Leduc sampler (abstraction does not affect chance)."""
        game = AbstractedLeducPoker(n_buckets=2)
        rng = np.random.default_rng(42)
        deal = game.sample_deal(rng)
        assert deal in game.all_deals()


# =============================================================================
# Round-trip — full traversal works end-to-end on the wrapper
# =============================================================================
class TestAbstractedTraversal:
    def test_traversal_to_terminal_works(self) -> None:
        game = AbstractedLeducPoker(n_buckets=2)
        state = game.state_from_deal((0, 4, 5))
        # Drive to terminal: CALL, CALL, CALL, CALL (round 1 closes,
        # round 2 opens, then round 2 closes by both calling).
        state = state.next_state(LeducAction.CALL)
        state = state.next_state(LeducAction.CALL)
        # Round 2 begins with P1 to act.
        state = state.next_state(LeducAction.CALL)
        state = state.next_state(LeducAction.CALL)
        assert state.is_terminal
        # Both players invested 1 ante; round 1 had 0 raises so each side
        # is in for 1 chip. Round 2 also 0 raises, no extra chips. So at
        # showdown the pot is 2 chips. Winner takes net +1.
        u = game.terminal_utility(state)
        assert abs(u) == pytest.approx(1.0)
