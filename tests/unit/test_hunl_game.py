"""Unit tests for HUNLGame factory + encode (Phase 4 M1.5)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from poker_ai.games.hunl import HUNLGame
from poker_ai.games.hunl_state import (
    BB_BLIND_CHIPS_VALUE,
    HUNLAction,
    HUNLState,
    SB_BLIND_CHIPS,
    STARTING_STACK_CHIPS,
)
from poker_ai.games.protocol import GameProtocol


# =============================================================================
# Class constants + GameProtocol compliance
# =============================================================================
class TestHUNLGameCompliance:
    def test_num_actions_is_3(self) -> None:
        """FOLD, CALL, BET — NULL_PADDING is encode-only."""
        assert HUNLGame.NUM_ACTIONS == 3

    def test_encoding_dim_is_102(self) -> None:
        """Compact rank/suit M1 encoding total: hole 4 + board 10 +
        round 4 + scalars 4 + history 80 = 102."""
        assert HUNLGame.ENCODING_DIM == 102

    def test_isinstance_game_protocol(self) -> None:
        game = HUNLGame()
        assert isinstance(game, GameProtocol)

    def test_all_deals_raises(self) -> None:
        """Phase 4 M0 contract: HUNL skips all_deals (impractical to
        enumerate ~10^14 deals)."""
        with pytest.raises(NotImplementedError, match="sample_deal"):
            HUNLGame.all_deals()


# =============================================================================
# sample_deal
# =============================================================================
class TestSampleDeal:
    def test_returns_9_distinct_card_ids_in_range(self) -> None:
        rng = np.random.default_rng(42)
        deal = HUNLGame.sample_deal(rng)
        assert len(deal) == 9
        for c in deal:
            assert isinstance(c, int)
            assert 0 <= c < 52
        assert len(set(deal)) == 9

    def test_seed_reproducibility(self) -> None:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        seq_a = [HUNLGame.sample_deal(rng_a) for _ in range(10)]
        seq_b = [HUNLGame.sample_deal(rng_b) for _ in range(10)]
        assert seq_a == seq_b

    def test_uniform_card_appearance_over_10k_samples(self) -> None:
        """Each of 52 cards should appear in ≈ 9/52 = 17.3 % of deals.
        Over 10 k samples, each card appears in ~1730 deals (±15 %)."""
        rng = np.random.default_rng(42)
        cnt = Counter()
        N = 10_000
        for _ in range(N):
            deal = HUNLGame.sample_deal(rng)
            cnt.update(deal)
        # Expected per-card count: 10 000 × 9 / 52 ≈ 1731.
        for card_id in range(52):
            assert 1450 < cnt[card_id] < 2000, (
                f"card {card_id} count {cnt[card_id]} outside expected band"
            )


# =============================================================================
# state_from_deal
# =============================================================================
class TestStateFromDeal:
    def test_root_state_has_blinds_posted(self) -> None:
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = HUNLGame.state_from_deal(deal)
        assert state.current_round == 0
        assert state.current_player == 1   # SB
        assert state.pot == BB_BLIND_CHIPS_VALUE + SB_BLIND_CHIPS
        assert state.stack_p0 == STARTING_STACK_CHIPS - BB_BLIND_CHIPS_VALUE
        assert state.stack_p1 == STARTING_STACK_CHIPS - SB_BLIND_CHIPS
        assert state.board_cards == ()
        assert state.last_raise_increment == BB_BLIND_CHIPS_VALUE

    def test_private_and_board_layout(self) -> None:
        """9-flat layout: hole P0 = deal[0:2], hole P1 = deal[2:4],
        pending_board = deal[4:9]."""
        deal = (10, 11, 20, 21, 30, 31, 40, 41, 50)
        state = HUNLGame.state_from_deal(deal)
        assert state.private_cards == (10, 11, 20, 21)
        assert state.pending_board == (30, 31, 40, 41, 50)

    def test_invalid_deal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="9 entries"):
            HUNLGame.state_from_deal((0, 1, 2))

    def test_duplicate_cards_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            HUNLGame.state_from_deal((0, 0, 1, 2, 3, 4, 5, 6, 7))

    def test_out_of_range_card_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 52))


# =============================================================================
# terminal_utility — delegation to HUNLState
# =============================================================================
class TestTerminalUtilityDelegation:
    def test_delegates_to_state_terminal_utility(self) -> None:
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = HUNLGame.state_from_deal(deal)
        terminal = state.next_state(HUNLAction.FOLD)
        # SB folds → P0 wins SB blind → utility +1.
        assert HUNLGame.terminal_utility(terminal) == pytest.approx(+1.0)

    def test_delegates_raises_on_non_terminal(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        with pytest.raises(ValueError, match="non-terminal"):
            HUNLGame.terminal_utility(state)


# =============================================================================
# encode — shape, dtype, layout sections
# =============================================================================
class TestEncodeShapeAndDtype:
    def test_encode_shape_is_102(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        assert arr.shape == (102,)

    def test_encode_dtype_float32(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        assert arr.dtype == np.float32


class TestEncodeCardsActingPerspective:
    def test_root_preflop_acting_is_p1_sees_p1_hole(self) -> None:
        """deal layout = (P0_h1, P0_h2, P1_h1, P1_h2, ...) and SB acts
        first preflop, so encode[0..4) should match P1's hole, NOT P0's."""
        deal = (0, 4, 8, 12, 16, 20, 24, 28, 32)
        state = HUNLGame.state_from_deal(deal)
        # P1 hole = (8, 12). Rank/suit pairs:
        #   8 = rank 2 / suit 0 → (2/12, 0/3)
        #   12 = rank 3 / suit 0 → (3/12, 0/3)
        arr = HUNLGame.encode(state)
        assert arr[0] == pytest.approx(2 / 12.0)
        assert arr[1] == pytest.approx(0 / 3.0)
        assert arr[2] == pytest.approx(3 / 12.0)
        assert arr[3] == pytest.approx(0 / 3.0)

    def test_after_sb_action_acting_is_p0_sees_p0_hole(self) -> None:
        deal = (0, 4, 8, 12, 16, 20, 24, 28, 32)
        state = HUNLGame.state_from_deal(deal).next_state(HUNLAction.CALL)
        # Now BB (P0) to act; encode hole = P0's = (0, 4).
        # 0 = rank 0, suit 0; 4 = rank 1, suit 0.
        arr = HUNLGame.encode(state)
        assert arr[0] == pytest.approx(0 / 12.0)
        assert arr[1] == pytest.approx(0 / 3.0)
        assert arr[2] == pytest.approx(1 / 12.0)
        assert arr[3] == pytest.approx(0 / 3.0)


class TestEncodeBoard:
    def test_root_preflop_board_all_unrevealed_minus_one(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        # Board section [4..14) all -1.0 (sentinel for unrevealed).
        for i in range(4, 14):
            assert arr[i] == -1.0

    def test_flop_reveals_three_cards(self) -> None:
        state = HUNLGame.state_from_deal((0, 4, 8, 12, 16, 20, 24, 28, 32))
        # SB call, BB check → flop reveals 3 cards (16, 20, 24).
        state = state.next_state(HUNLAction.CALL)
        state = state.next_state(HUNLAction.CALL)
        arr = HUNLGame.encode(state)
        # Card 16 = rank 4, suit 0 → (4/12, 0/3).
        assert arr[4] == pytest.approx(4 / 12.0)
        assert arr[5] == pytest.approx(0 / 3.0)
        # Card 20 = rank 5, suit 0; card 24 = rank 6, suit 0.
        assert arr[6] == pytest.approx(5 / 12.0)
        assert arr[8] == pytest.approx(6 / 12.0)
        # Slots 4 and 5 (cards 4 and 5 of the board, indices 10..14) still -1.
        for i in range(10, 14):
            assert arr[i] == -1.0


class TestEncodeRoundOnehot:
    def test_preflop_round_idx_0(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        # Round one-hot section [14..18).
        assert arr[14] == 1.0   # preflop
        assert arr[15] == 0.0
        assert arr[16] == 0.0
        assert arr[17] == 0.0

    def test_flop_round_idx_1(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        state = state.next_state(HUNLAction.CALL)
        state = state.next_state(HUNLAction.CALL)
        arr = HUNLGame.encode(state)
        assert arr[14] == 0.0
        assert arr[15] == 1.0   # flop
        assert arr[16] == 0.0
        assert arr[17] == 0.0


class TestEncodeScalarsNormalization:
    def test_root_preflop_scalars(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        # pot = 3 / 400 = 0.0075
        assert arr[18] == pytest.approx(3 / 400.0)
        # stack_p0 = 198 / 200 = 0.99
        assert arr[19] == pytest.approx(198 / 200.0)
        # stack_p1 = 199 / 200 = 0.995
        assert arr[20] == pytest.approx(199 / 200.0)
        # last_raise = 2 / 200 = 0.01
        assert arr[21] == pytest.approx(2 / 200.0)

    def test_after_sb_raise_scalars_update(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        state = state.next_state(HUNLAction.BET, bet_size=10)
        arr = HUNLGame.encode(state)
        # pot = 12 (SB 10 + BB 2). stack_p1 = 200-10 = 190.
        assert arr[18] == pytest.approx(12 / 400.0)
        assert arr[20] == pytest.approx(190 / 200.0)


class TestEncodeBettingHistory:
    def test_root_preflop_history_is_all_null_padding(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        # NULL_PADDING action_id = 3, normalized = 1.0; size = 0/200 = 0.
        for i in range(40):
            assert arr[22 + i * 2] == pytest.approx(3 / 3.0)
            assert arr[22 + i * 2 + 1] == 0.0

    def test_first_action_recorded_then_null_padding(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        state = state.next_state(HUNLAction.CALL)
        arr = HUNLGame.encode(state)
        # Slot 0 = CALL (action_id 1), size 0.
        assert arr[22 + 0 * 2] == pytest.approx(1 / 3.0)
        assert arr[22 + 0 * 2 + 1] == 0.0
        # Slot 1 onward = NULL_PADDING.
        for i in range(1, 40):
            assert arr[22 + i * 2] == pytest.approx(3 / 3.0)
            assert arr[22 + i * 2 + 1] == 0.0

    def test_bet_action_records_size(self) -> None:
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        state = state.next_state(HUNLAction.BET, bet_size=10)
        arr = HUNLGame.encode(state)
        # Slot 0 = BET (id 2), size 10/200 = 0.05.
        assert arr[22 + 0] == pytest.approx(2 / 3.0)
        assert arr[22 + 1] == pytest.approx(10 / 200.0)


class TestEncodeBoundsSanity:
    def test_all_normalized_values_in_minus_1_to_1(self) -> None:
        """No element of the encoding should fall outside [-1, +1] under
        valid input. Sentinel -1 (unrevealed cards) is the only negative
        value; all others are 0..1."""
        state = HUNLGame.state_from_deal((0, 1, 2, 3, 4, 5, 6, 7, 8))
        arr = HUNLGame.encode(state)
        assert (arr >= -1.0).all()
        assert (arr <= 1.0).all()
