"""Unit tests for Leduc Hold'em game engine.

Target module (NOT YET IMPLEMENTED — these tests must fail with
``ModuleNotFoundError`` on collection; Phase 2 Week 1 Day 1 RED state):
    src/poker_ai/games/leduc.py

Target API:
    class LeducAction(IntEnum):
        FOLD = 0
        CALL = 1
        RAISE = 2
        def __str__(self) -> str: ...   # "f" | "c" | "r"

    @dataclass(frozen=True, slots=True)
    class LeducState:
        private_cards: tuple[int, int]     # (P1 card_id, P2 card_id), id in 0..5
        board_card: int | None             # None in round 1, in 0..5 after flop
        round_history: tuple[
            tuple[LeducAction, ...],       # round 1 actions
            tuple[LeducAction, ...],       # round 2 actions (empty if round 1)
        ]
        # Derived:
        @property
        def round_idx(self) -> int: ...          # 0 or 1
        @property
        def bets_this_round(self) -> int: ...    # RAISE count in current round
        @property
        def current_player(self) -> int: ...     # P1 leads both rounds
        @property
        def is_terminal(self) -> bool: ...
        @property
        def infoset_key(self) -> str: ...
        def legal_actions(self) -> tuple[LeducAction, ...]: ...
        def legal_action_mask(self) -> np.ndarray: ...   # shape (3,), dtype bool
        def next_state(self, action: LeducAction) -> "LeducState": ...

    class LeducPoker:
        NUM_ACTIONS = 3
        @staticmethod
        def all_deals() -> tuple[tuple[int, int, int], ...]:
            # 120 (P1, P2, board) triples with distinct card_ids in 0..5.
        @staticmethod
        def state_from_deal(deal: tuple[int, int, int]) -> LeducState: ...
        @staticmethod
        def terminal_utility(state: LeducState) -> float: ...

Card encoding convention (rank-only semantics, suit only for uniqueness):
    card_id in 0..5, rank = card_id // 2 (0=J, 1=Q, 2=K), suit = card_id % 2.

Reference: Neller & Lanctot 2013, Section 5 (Leduc Hold'em tree and infoset
counting); Southey et al. 2005 (Leduc rules).
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pytest

from poker_ai.games.leduc import LeducAction, LeducPoker, LeducState

# -----------------------------------------------------------------------------
# Convenience aliases (match target API's 0=FOLD, 1=CALL, 2=RAISE).
# -----------------------------------------------------------------------------
FOLD, CALL, RAISE = LeducAction.FOLD, LeducAction.CALL, LeducAction.RAISE

# Card IDs 0..5; rank = id // 2; ranks 0=J, 1=Q, 2=K.
J0, J1 = 0, 1  # two Jacks
Q0, Q1 = 2, 3  # two Queens
K0, K1 = 4, 5  # two Kings
ALL_CARD_IDS = (J0, J1, Q0, Q1, K0, K1)


def _deal(
    p1: int, p2: int, board: int
) -> tuple[int, int, int]:
    return (p1, p2, board)


def _root_with_board(
    p1: int, p2: int, board: int
) -> LeducState:
    """Helper: state_from_deal returns the round-1 root; board_card is None."""
    return LeducPoker.state_from_deal(_deal(p1, p2, board))


# -----------------------------------------------------------------------------
# LeducAction: IntEnum behaviour
# -----------------------------------------------------------------------------
class TestLeducAction:
    def test_int_values(self) -> None:
        """FOLD=0, CALL=1, RAISE=2 (network action head / numpy indexing)."""
        assert int(LeducAction.FOLD) == 0
        assert int(LeducAction.CALL) == 1
        assert int(LeducAction.RAISE) == 2

    def test_str_is_single_character(self) -> None:
        """str(action)은 infoset key용 단문자."""
        assert str(LeducAction.FOLD) == "f"
        assert str(LeducAction.CALL) == "c"
        assert str(LeducAction.RAISE) == "r"

    def test_members_count(self) -> None:
        """Leduc은 fold/call/raise 세 액션만 존재."""
        assert len(LeducAction) == 3


# -----------------------------------------------------------------------------
# LeducPoker.all_deals()
# -----------------------------------------------------------------------------
class TestAllDeals:
    def test_exactly_120_deals(self) -> None:
        """6*5*4 = 120 permutations of 3 distinct card_ids from 6."""
        deals = LeducPoker.all_deals()
        assert len(deals) == 120

    def test_all_deals_are_triples_of_card_ids(self) -> None:
        """각 deal은 길이 3, 각 원소 ∈ 0..5."""
        for deal in LeducPoker.all_deals():
            assert len(deal) == 3
            for cid in deal:
                assert isinstance(cid, int)
                assert 0 <= cid <= 5

    def test_all_deals_have_distinct_card_ids(self) -> None:
        """각 deal의 세 card_id는 서로 다름 (중복 드로우 방지)."""
        for deal in LeducPoker.all_deals():
            assert len(set(deal)) == 3

    def test_deals_are_deterministic(self) -> None:
        """두 번 호출해도 동일 순서 (dict/set 순회 순서 의존 금지)."""
        assert LeducPoker.all_deals() == LeducPoker.all_deals()

    def test_deals_are_unique(self) -> None:
        """120개 deal 사이에 중복 없음."""
        deals = LeducPoker.all_deals()
        assert len(set(deals)) == 120

    @pytest.mark.parametrize(
        "pair_category,expected_count",
        [
            ("pair_private", 24),      # P1.rank == P2.rank
            ("no_pair_private", 96),   # P1.rank != P2.rank
        ],
    )
    def test_pair_private_deal_counts(
        self, pair_category: str, expected_count: int
    ) -> None:
        """Private pair deals are 3 ranks * 2 (P1 suit-pick) * 1 (P2 suit) * 4 (board) = 24.
        Non-pair private deals = 120 - 24 = 96.
        """
        deals = LeducPoker.all_deals()
        pair_deals = [d for d in deals if (d[0] // 2) == (d[1] // 2)]
        if pair_category == "pair_private":
            assert len(pair_deals) == expected_count
        else:
            assert len(deals) - len(pair_deals) == expected_count

    def test_chance_probability_matches_cfr_convention(self) -> None:
        """The 1/n chance probability must match the convention used by
        VanillaCFR.train() when initializing reach_opp.

        Leduc: reach_opp = 1.0 / 120 (chance over all deals).
        This test locks in the 'uniform chance' contract that BR and CFR rely on.
        """
        deals = LeducPoker.all_deals()
        assert len(deals) == 120
        assert len(set(deals)) == 120  # No duplicates
        expected_chance_prob = 1.0 / len(deals)
        assert expected_chance_prob == pytest.approx(1.0 / 120.0)


# -----------------------------------------------------------------------------
# LeducState: frozen dataclass with __slots__
# -----------------------------------------------------------------------------
class TestLeducStateFrozen:
    def test_cannot_mutate_attribute(self) -> None:
        """frozen=True → 속성 재할당 시 FrozenInstanceError."""
        import dataclasses

        state = _root_with_board(J0, Q0, K0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.board_card = 0  # type: ignore[misc]

    def test_has_slots_no_dict(self) -> None:
        """slots=True → 인스턴스에 __dict__ 없음."""
        state = _root_with_board(J0, Q0, K0)
        assert not hasattr(state, "__dict__")


# -----------------------------------------------------------------------------
# LeducState derived properties
# -----------------------------------------------------------------------------
class TestLeducStateProperties:
    def test_initial_root_state(self) -> None:
        """Root: round 0, 0 bets, P1 to act, not terminal, board hidden."""
        s = _root_with_board(J0, Q0, K0)
        assert s.round_idx == 0
        assert s.bets_this_round == 0
        assert s.current_player == 0
        assert s.is_terminal is False
        assert s.board_card is None

    def test_after_check(self) -> None:
        """P1 check → P2 to act, still 0 bets, not terminal."""
        s = _root_with_board(J0, Q0, K0).next_state(CALL)
        assert s.round_idx == 0
        assert s.bets_this_round == 0
        assert s.current_player == 1
        assert s.is_terminal is False

    def test_after_bet(self) -> None:
        """P1 bet → P2 to act, 1 bet on table."""
        s = _root_with_board(J0, Q0, K0).next_state(RAISE)
        assert s.round_idx == 0
        assert s.bets_this_round == 1
        assert s.current_player == 1
        assert s.is_terminal is False

    def test_after_bet_raise(self) -> None:
        """P1 bet, P2 raise → 2-bet cap reached (in round)."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(RAISE)
            .next_state(RAISE)
        )
        assert s.bets_this_round == 2
        assert s.current_player == 0
        assert s.is_terminal is False

    def test_fold_is_terminal(self) -> None:
        """Any fold immediately terminates the game."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(RAISE)
            .next_state(FOLD)
        )
        assert s.is_terminal is True

    def test_check_check_advances_to_round_2(self) -> None:
        """cc in round 1 → round 2 starts; board revealed; P1 leads."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL)
            .next_state(CALL)
        )
        assert s.is_terminal is False
        assert s.round_idx == 1
        assert s.board_card is not None
        assert s.bets_this_round == 0
        assert s.current_player == 0  # P1 leads round 2 as well
        assert s.round_history[1] == ()  # round-2 history starts empty

    def test_round2_check_check_is_terminal(self) -> None:
        """cc.cc → both rounds closed via check → showdown terminal."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)    # round 1 cc
            .next_state(CALL).next_state(CALL)    # round 2 cc
        )
        assert s.is_terminal is True

    def test_round2_bet_call_is_terminal(self) -> None:
        """Round 2 bet-call closes the game at showdown."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)    # round 1 cc
            .next_state(RAISE).next_state(CALL)   # round 2 bet-call
        )
        assert s.is_terminal is True

    def test_bet_raise_call_closes_round(self) -> None:
        """rrc in round 1 → round 1 ends, round 2 starts."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(RAISE).next_state(RAISE).next_state(CALL)
        )
        assert s.round_idx == 1
        assert s.is_terminal is False
        assert s.board_card is not None


# -----------------------------------------------------------------------------
# legal_actions
# -----------------------------------------------------------------------------
class TestLegalActions:
    def test_first_to_act_no_bet_has_check_or_bet(self) -> None:
        """Bet 없이 먼저 수를 둘 때: FOLD 불가 — (CALL, RAISE)."""
        s = _root_with_board(J0, Q0, K0)
        assert s.legal_actions() == (CALL, RAISE)

    def test_facing_bet_under_cap_has_all_three(self) -> None:
        """1 bet 직면 + 아직 cap 안 참: (FOLD, CALL, RAISE) 모두 합법."""
        s = _root_with_board(J0, Q0, K0).next_state(RAISE)
        assert s.legal_actions() == (FOLD, CALL, RAISE)

    def test_at_two_bet_cap_raise_is_illegal(self) -> None:
        """2-bet cap 도달 시 RAISE 금지: (FOLD, CALL) 만 합법."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(RAISE)
            .next_state(RAISE)
        )
        assert s.legal_actions() == (FOLD, CALL)

    def test_round2_first_to_act_no_bet(self) -> None:
        """Round 2 open: bet이 carry-over 되지 않음 → (CALL, RAISE)."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)
        )
        assert s.legal_actions() == (CALL, RAISE)

    def test_round2_at_cap(self) -> None:
        """Round 2에서도 2-bet cap 동일 규칙."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)       # round 1 cc
            .next_state(RAISE).next_state(RAISE)     # round 2 bet-raise
        )
        assert s.legal_actions() == (FOLD, CALL)


# -----------------------------------------------------------------------------
# legal_action_mask
# -----------------------------------------------------------------------------
class TestLegalActionMask:
    def test_mask_shape_and_dtype(self) -> None:
        """mask는 shape (3,), dtype bool ndarray."""
        s = _root_with_board(J0, Q0, K0)
        mask = s.legal_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3,)
        assert mask.dtype == np.bool_

    @pytest.mark.parametrize(
        "build,expected",
        [
            # (state builder, expected mask as tuple of bools for FOLD, CALL, RAISE)
            (lambda: _root_with_board(J0, Q0, K0),
             (False, True, True)),
            (lambda: _root_with_board(J0, Q0, K0).next_state(RAISE),
             (True, True, True)),
            (lambda: _root_with_board(J0, Q0, K0).next_state(RAISE).next_state(RAISE),
             (True, True, False)),
        ],
        ids=["open_no_bet", "facing_bet_under_cap", "at_two_bet_cap"],
    )
    def test_mask_matches_legal_actions(self, build, expected) -> None:
        """legal_actions()의 index 집합과 mask의 True 위치가 정확히 일치."""
        state = build()
        mask = state.legal_action_mask()
        legal = state.legal_actions()
        expected_arr = np.array(expected, dtype=np.bool_)
        np.testing.assert_array_equal(mask, expected_arr)
        # Cross-check consistency with tuple:
        indices_from_mask = tuple(int(i) for i, v in enumerate(mask) if v)
        indices_from_legal = tuple(int(a) for a in legal)
        assert indices_from_mask == indices_from_legal


# -----------------------------------------------------------------------------
# infoset_key
# -----------------------------------------------------------------------------
class TestInfosetKey:
    def test_root_p1_jack(self) -> None:
        """Root, P1=J (card_id 0 or 1) → 'J|'."""
        for p1_jack in (J0, J1):
            s = _root_with_board(p1_jack, Q0, K0)
            assert s.infoset_key == "J|"

    def test_root_p1_queen(self) -> None:
        for p1_q in (Q0, Q1):
            s = _root_with_board(p1_q, K0, J0)
            assert s.infoset_key == "Q|"

    def test_root_p1_king(self) -> None:
        for p1_k in (K0, K1):
            s = _root_with_board(p1_k, J0, Q0)
            assert s.infoset_key == "K|"

    def test_after_p1_check_p2_view(self) -> None:
        """P1 check → P2 to move; key shows P2's rank + 'c'."""
        s = _root_with_board(J0, Q0, K0).next_state(CALL)
        # P2 card id = Q0 → rank 'Q'
        assert s.infoset_key == "Q|c"

    def test_after_p1_bet_p2_view(self) -> None:
        s = _root_with_board(J0, K0, Q0).next_state(RAISE)
        # P2 card id = K0 → rank 'K'
        assert s.infoset_key == "K|r"

    def test_round2_start_key_shape(self) -> None:
        """cc.<board_rank> — round 2 opening 시 board rank가 key에 등장."""
        # board_card = Q0 → rank Q. P1 perspective after cc.
        s = (
            _root_with_board(J0, K0, Q0)
            .next_state(CALL).next_state(CALL)
        )
        assert s.infoset_key == "J|cc.Q"

    def test_round2_after_check_p2_view(self) -> None:
        """cc → round 2 P1 check → P2 view with P2's rank + board + c."""
        s = (
            _root_with_board(J0, K0, Q0)
            .next_state(CALL).next_state(CALL)    # round 1 cc
            .next_state(CALL)                     # round 2 P1 check
        )
        # P2 card K0 → rank 'K'; board Q → 'Q'; round2 hist 'c'
        assert s.infoset_key == "K|cc.Qc"

    def test_round2_after_bet_raise_p1_view(self) -> None:
        """crrc in round 2 path requires round 1 to also close; test separately.

        Here we check round 1 'cr' then P1 view — still in round 1, so no board char.
        """
        s = _root_with_board(Q0, K0, J0).next_state(CALL).next_state(RAISE)
        # After 'cr', P1 to move (len=2 even); P1's rank Q + history 'cr'
        assert s.infoset_key == "Q|cr"

    def test_round2_infoset_with_bet_raise(self) -> None:
        """Round-2 P1 after crrc path: 'K|crc.Jr' example from spec."""
        # Need P1 view in round 2 after sequence that ends with 'Jr' chars.
        # Round 1 'crc' closes via check-bet-call. Round 2 starts, board=J.
        # In round 2 P1 bets ('r'); then it's P2's turn — P1 view must be BEFORE
        # round 2 starts raising or right when P1 is to act. For 'K|crc.Jr' the
        # current player must be P2 (since 'r' just happened and next is P2).
        s = (
            _root_with_board(Q0, K0, J0)  # board card id J0 → rank 'J'
            .next_state(CALL).next_state(RAISE).next_state(CALL)  # round 1 crc
            .next_state(RAISE)                                      # round 2 P1 bet
        )
        # Now P2 to move; P2 rank K; round2 hist 'r'; board J.
        assert s.infoset_key == "K|crc.Jr"

    def test_infoset_key_is_string(self) -> None:
        """CLAUDE.md 규약: infoset key는 결정론적 str."""
        s = _root_with_board(J0, Q0, K0)
        assert isinstance(s.infoset_key, str)

    def test_infoset_key_determinism(self) -> None:
        """동일 state를 여러 번 조회해도 동일 key (dict 순회 의존 금지)."""
        s = (
            _root_with_board(J0, K0, Q0)
            .next_state(CALL).next_state(CALL)
        )
        keys = [s.infoset_key for _ in range(5)]
        assert len(set(keys)) == 1


# -----------------------------------------------------------------------------
# Terminal utility (P1 perspective; zero-sum so u2 = -u1)
# -----------------------------------------------------------------------------
#
# Pot accounting conventions:
#   - Ante: each player posts 1 at start → pot starts at 2.
#   - Round 1: bet size = 2; max 2 raises/round means up to 2 chips + 2 chips.
#   - Round 2: bet size = 4.
#   - On fold: folder loses chips committed so far; winner's utility equals
#     the opponent's total contribution beyond their own ante refund.
#   - On showdown: higher rank wins entire pot; utility = opponent's
#     contribution (net chips, since each side gets their own chips back
#     before distribution).
#
# We only test a small fixed sanity table here to keep tests focused; full
# pot-tree enumeration lives in integration/regression tests later.
# -----------------------------------------------------------------------------
class TestTerminalUtility:
    def test_round1_fold_after_bet_returns_ante(self) -> None:
        """P1 bet, P2 fold in round 1: P1 wins P2's ante (=+1)."""
        s = _root_with_board(J0, Q0, K0).next_state(RAISE).next_state(FOLD)
        assert s.is_terminal
        assert LeducPoker.terminal_utility(s) == pytest.approx(1.0)

    def test_round1_p1_fold_after_p2_raise_negative(self) -> None:
        """P1 check, P2 bet, P1 fold: P1 loses 1 (own ante)."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(RAISE).next_state(FOLD)
        )
        assert s.is_terminal
        assert LeducPoker.terminal_utility(s) == pytest.approx(-1.0)

    def test_showdown_cc_cc_higher_rank_wins(self) -> None:
        """cc.cc, P1 rank K vs P2 rank J: P1 wins ante only (pot 2)."""
        # p1 = K0 (rank 2), p2 = J0 (rank 0), board = Q0 (rank 1, no pair either)
        s = (
            _root_with_board(K0, J0, Q0)
            .next_state(CALL).next_state(CALL)
            .next_state(CALL).next_state(CALL)
        )
        assert s.is_terminal
        u = LeducPoker.terminal_utility(s)
        assert u > 0
        assert u == pytest.approx(1.0)

    def test_showdown_pair_beats_higher_rank(self) -> None:
        """Board pairs with P2 (Q,Q); P1 holds K (no pair). Pair wins."""
        # p1 = K0 (rank 2), p2 = Q0 (rank 1), board = Q1 (rank 1 → pair with P2)
        s = (
            _root_with_board(K0, Q0, Q1)
            .next_state(CALL).next_state(CALL)
            .next_state(CALL).next_state(CALL)
        )
        assert s.is_terminal
        u = LeducPoker.terminal_utility(s)
        assert u < 0  # P1 loses to P2's pair

    def test_showdown_tie_same_private_rank_no_pair(self) -> None:
        """Impossible by rules (distinct card_ids) — but same rank via different suits possible.

        With p1=J0 (J), p2=J1 (J), board=K0 (K, no pair for either): tie → 0.
        """
        s = (
            _root_with_board(J0, J1, K0)
            .next_state(CALL).next_state(CALL)
            .next_state(CALL).next_state(CALL)
        )
        assert s.is_terminal
        assert LeducPoker.terminal_utility(s) == pytest.approx(0.0)

    def test_p2_fold_round2_after_raise(self) -> None:
        """cc then round-2 bet by P1, P2 fold: P1 wins P2's committed chips."""
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)    # round 1 cc
            .next_state(RAISE).next_state(FOLD)   # round 2 P1 bet, P2 fold
        )
        assert s.is_terminal
        # P2 committed only the ante (1) before folding → P1 wins 1.
        assert LeducPoker.terminal_utility(s) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "deal",
        [
            (J0, Q0, K0),  # cross-check 1: P1=J, P2=Q, board=K
            (K0, J0, Q1),  # cross-check 2: P1=K, P2=J, board=Q (다른 rank 조합)
        ],
        ids=["JQ_K", "KJ_Q"],
    )
    def test_round1_fold_after_cap_loses_bet(self, deal) -> None:
        """Round 1 rrf: P1 bet, P2 raise, P1 fold. P1 loses ante+bet = 3 chips.

        기존 ±1 fold 테스트는 single bet 후 fold만 다룸. 이 테스트가 "cap 도달
        후 fold" = 큰 pot commit 후 fold 패턴을 처음 커버.
        """
        s = (
            LeducPoker.state_from_deal(deal)
            .next_state(RAISE)  # P1 bet(2)
            .next_state(RAISE)  # P2 raise(2+2=4 committed)
            .next_state(FOLD)   # P1 fold
        )
        assert s.is_terminal
        # P1 committed: ante 1 + bet 2 = 3 → utility = -3 (P1 loses committed)
        assert LeducPoker.terminal_utility(s) == pytest.approx(-3.0)

    def test_round2_fold_after_raise_large_pot(self) -> None:
        """cc.rrf: round 1 cc, round 2 bet-raise-fold. P1 loses 5 chips.

        round 1 (bet=2)와 round 2 (bet=4) 사이즈 차이가 pot 누적에 올바르게
        반영되는지 검증.
        """
        s = (
            _root_with_board(J0, Q0, K0)
            .next_state(CALL).next_state(CALL)    # round 1 cc
            .next_state(RAISE)                     # round 2 P1 bet(4)
            .next_state(RAISE)                     # round 2 P2 raise (cap)
            .next_state(FOLD)                      # round 2 P1 fold
        )
        assert s.is_terminal
        # P1 committed: 1 ante + 4 round-2 bet = 5 → utility = -5
        assert LeducPoker.terminal_utility(s) == pytest.approx(-5.0)

    def test_showdown_p1_pair_wins_over_higher_opponent(self) -> None:
        """P1=Q, P2=K, board=Q → P1 has pair Q, P2 high K no pair. P1 wins.

        대칭 보강: 기존에 'P2 pair win'만 있고 'P1 pair win' 케이스가 없었음.
        """
        # p1 = Q0 (rank 1), p2 = K0 (rank 2), board = Q1 (rank 1 → pair with P1)
        s = (
            _root_with_board(Q0, K0, Q1)
            .next_state(CALL).next_state(CALL)  # round 1 cc
            .next_state(CALL).next_state(CALL)  # round 2 cc
        )
        assert s.is_terminal
        u = LeducPoker.terminal_utility(s)
        assert u > 0, f"P1 pair should win, got utility={u}"
        # Pot = antes only (both checked all the way) → P1 gains P2's ante = +1.0
        assert u == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "p2_card",
        [Q0, Q1, K0, K1],  # 4개 (J0/J1 제외: P1이 J0 사용)
        ids=["p2_Q0", "p2_Q1", "p2_K0", "p2_K1"],
    )
    def test_fold_utility_invariant_across_opponent_cards(self, p2_card: int) -> None:
        """Fold 결과 utility는 상대 private card와 무관해야 함 (포커 규칙 불변식).

        P1 bet → P2 fold 시나리오. P2 카드가 Q/K 어느 것이든 P1 utility = +1.
        """
        # P1 = J0 고정. P2 카드만 parametrize. Board는 remaining pool에서 결정.
        used = {J0, p2_card}
        board = next(c for c in range(6) if c not in used)
        s = (
            LeducPoker.state_from_deal((J0, p2_card, board))
            .next_state(RAISE)   # P1 bet(2)
            .next_state(FOLD)    # P2 fold
        )
        assert s.is_terminal
        # P2 committed: ante 1 only → P1 gains +1 regardless of P2's card
        assert LeducPoker.terminal_utility(s) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Cross-check: next_state immutability (like Kuhn's TestNextStateImmutability).
# -----------------------------------------------------------------------------
class TestNextStateImmutability:
    def test_original_round_history_not_mutated(self) -> None:
        original = _root_with_board(J0, Q0, K0)
        _ = original.next_state(CALL)
        assert original.round_history == ((), ())

    def test_next_state_returns_new_instance(self) -> None:
        s0 = _root_with_board(J0, Q0, K0)
        s1 = s0.next_state(CALL)
        assert s1 is not s0

    def test_private_cards_preserved_across_transitions(self) -> None:
        s0 = _root_with_board(K0, J0, Q0)
        s1 = s0.next_state(RAISE)
        s2 = s1.next_state(CALL)
        assert s1.private_cards == (K0, J0)
        assert s2.private_cards == (K0, J0)
