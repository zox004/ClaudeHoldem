"""Unit tests for Kuhn Poker game engine.

Target module (NOT YET IMPLEMENTED — these tests must fail with ModuleNotFoundError):
    src/poker_ai/games/kuhn.py

Target API:
    class KuhnAction(IntEnum):
        PASS = 0
        BET = 1
        def __str__(self) -> str: ...   # "p" | "b"

    @dataclass(frozen=True, slots=True)
    class KuhnState:
        deal: tuple[int, int]                 # (P1 card, P2 card) — 0=J, 1=Q, 2=K
        history: tuple[KuhnAction, ...]       # length 0..3
        # Methods / properties:
        @property
        def is_terminal(self) -> bool: ...
        @property
        def current_player(self) -> int: ...  # 0 or 1; undefined/raises if terminal
        @property
        def infoset_key(self) -> str: ...     # e.g. "J|pb" — deterministic string
        def legal_actions(self) -> tuple[KuhnAction, ...]: ...
        def next_state(self, action: KuhnAction) -> "KuhnState": ...

    class KuhnPoker:
        NUM_PLAYERS = 2
        NUM_CARDS = 3
        @staticmethod
        def all_deals() -> tuple[tuple[int, int], ...]: ...   # 6 permutations
        @staticmethod
        def state_from_deal(deal: tuple[int, int]) -> KuhnState: ...
        @staticmethod
        def terminal_utility(state: KuhnState) -> float: ...  # P1 perspective

Reference: Neller & Lanctot 2013, Section 4 (Kuhn poker tree and Nash family).
"""

from __future__ import annotations

from itertools import product

import pytest

from poker_ai.games.kuhn import KuhnAction, KuhnPoker, KuhnState

# Card codes (match target API's 0=J, 1=Q, 2=K).
JACK, QUEEN, KING = 0, 1, 2
PASS, BET = KuhnAction.PASS, KuhnAction.BET


# -----------------------------------------------------------------------------
# KuhnAction: IntEnum behaviour
# -----------------------------------------------------------------------------
class TestKuhnAction:
    def test_int_values(self) -> None:
        """PASS=0, BET=1 (numpy 인덱싱 및 네트워크 action head와 호환)."""
        assert int(KuhnAction.PASS) == 0
        assert int(KuhnAction.BET) == 1

    def test_str_is_single_character(self) -> None:
        """str(action)은 infoset key에 들어가는 단문자."""
        assert str(KuhnAction.PASS) == "p"
        assert str(KuhnAction.BET) == "b"

    def test_members_count(self) -> None:
        """Kuhn은 pass/bet 두 액션만 존재."""
        assert len(KuhnAction) == 2


# -----------------------------------------------------------------------------
# KuhnPoker.all_deals()
# -----------------------------------------------------------------------------
class TestAllDeals:
    def test_exactly_six_unique_deals(self) -> None:
        """3!/1! = 6 permutations of 2 distinct cards from {J,Q,K}."""
        deals = KuhnPoker.all_deals()
        assert len(deals) == 6
        assert len(set(deals)) == 6

    def test_every_deal_uses_two_distinct_cards(self) -> None:
        """각 딜은 서로 다른 두 카드로 구성."""
        for deal in KuhnPoker.all_deals():
            assert len(deal) == 2
            assert deal[0] != deal[1]
            assert set(deal).issubset({JACK, QUEEN, KING})

    def test_deals_are_deterministic(self) -> None:
        """두 번 호출해도 동일한 순서로 반환 (dict 순회 순서 의존 금지)."""
        assert KuhnPoker.all_deals() == KuhnPoker.all_deals()


# -----------------------------------------------------------------------------
# current_player
# -----------------------------------------------------------------------------
class TestCurrentPlayer:
    @pytest.mark.parametrize(
        "history,expected_player",
        [
            ((), 0),                       # P1 opens
            ((PASS,), 1),                  # P2 responds
            ((BET,), 1),                   # P2 responds
            ((PASS, BET), 0),              # P1 responds to pb
        ],
    )
    def test_non_terminal_player(
        self, history: tuple[KuhnAction, ...], expected_player: int
    ) -> None:
        """비-terminal 노드에서 수 둘 플레이어 규약."""
        state = KuhnState(deal=(JACK, QUEEN), history=history)
        assert state.current_player == expected_player


# -----------------------------------------------------------------------------
# legal_actions
# -----------------------------------------------------------------------------
class TestLegalActions:
    @pytest.mark.parametrize(
        "history",
        [(), (PASS,), (BET,), (PASS, BET)],
    )
    def test_both_actions_always_legal_in_non_terminal(
        self, history: tuple[KuhnAction, ...]
    ) -> None:
        """Kuhn에서는 모든 비-terminal 노드에서 PASS·BET 두 액션 모두 합법."""
        state = KuhnState(deal=(JACK, QUEEN), history=history)
        assert state.legal_actions() == (KuhnAction.PASS, KuhnAction.BET)


# -----------------------------------------------------------------------------
# is_terminal classification
# -----------------------------------------------------------------------------
class TestIsTerminal:
    @pytest.mark.parametrize(
        "history,terminal",
        [
            ((), False),
            ((PASS,), False),
            ((BET,), False),
            ((PASS, BET), False),
            ((PASS, PASS), True),          # both check → showdown
            ((BET, PASS), True),           # P2 fold
            ((BET, BET), True),            # both bet → showdown
            ((PASS, BET, PASS), True),     # P1 fold after pb
            ((PASS, BET, BET), True),      # call after pb → showdown
        ],
    )
    def test_terminal_classification(
        self, history: tuple[KuhnAction, ...], terminal: bool
    ) -> None:
        state = KuhnState(deal=(JACK, QUEEN), history=history)
        assert state.is_terminal is terminal


# -----------------------------------------------------------------------------
# infoset_key: determinism + player-correct private info
# -----------------------------------------------------------------------------
class TestInfosetKey:
    @pytest.mark.parametrize(
        "deal,history,expected",
        [
            # P1-to-move nodes encode P1's card
            ((JACK, QUEEN), (), "J|"),
            ((QUEEN, JACK), (), "Q|"),
            ((KING, QUEEN), (), "K|"),
            ((QUEEN, KING), (PASS, BET), "Q|pb"),
            # P2-to-move nodes encode P2's card
            ((JACK, QUEEN), (PASS,), "Q|p"),
            ((JACK, KING), (BET,), "K|b"),
        ],
    )
    def test_infoset_key_format(
        self,
        deal: tuple[int, int],
        history: tuple[KuhnAction, ...],
        expected: str,
    ) -> None:
        """infoset_key = '<own_rank>|<history_str>' 고정 포맷."""
        state = KuhnState(deal=deal, history=history)
        assert state.infoset_key == expected

    def test_same_own_card_same_history_same_key(self) -> None:
        """상대 카드가 달라도 내 카드·히스토리가 같으면 같은 infoset."""
        s1 = KuhnState(deal=(JACK, QUEEN), history=())  # P1 sees J
        s2 = KuhnState(deal=(JACK, KING), history=())   # P1 sees J
        assert s1.infoset_key == s2.infoset_key

    def test_different_player_different_key(self) -> None:
        """빈 히스토리(P1) vs PASS 뒤(P2)는 서로 다른 infoset."""
        s1 = KuhnState(deal=(JACK, QUEEN), history=())
        s2 = KuhnState(deal=(JACK, QUEEN), history=(PASS,))
        assert s1.infoset_key != s2.infoset_key

    def test_key_is_string(self) -> None:
        """CLAUDE.md 규약: infoset key는 결정론적 '문자열'."""
        state = KuhnState(deal=(QUEEN, KING), history=(PASS, BET))
        assert isinstance(state.infoset_key, str)


def test_infoset_key_exact_format() -> None:
    """포맷 규약: '<rank_char>|<history_str>', rank ∈ {J, Q, K}.

    state_from_deal → next_state 경계를 통해 perspective가 P1 → P2로
    넘어갈 때 own card가 올바르게 switch되는지 동시 검증.
    """
    # 빈 history, P1 관점, own card = K (index 2)
    state = KuhnPoker.state_from_deal((2, 0))  # P1=K, P2=J
    assert state.infoset_key == "K|"  # 포맷: "rank_char|history_str"

    # P1이 bet → P2 차례. own card는 P2의 J.
    next_s = state.next_state(KuhnAction.BET)
    assert next_s.infoset_key == "J|b"


# -----------------------------------------------------------------------------
# next_state: frozen dataclass, no mutation
# -----------------------------------------------------------------------------
class TestNextStateImmutability:
    def test_original_history_not_mutated(self) -> None:
        original = KuhnState(deal=(JACK, QUEEN), history=())
        _ = original.next_state(PASS)
        assert original.history == ()

    def test_extends_history_by_one_action(self) -> None:
        s0 = KuhnState(deal=(JACK, QUEEN), history=())
        s1 = s0.next_state(PASS)
        assert s1.history == (PASS,)
        s2 = s1.next_state(BET)
        assert s2.history == (PASS, BET)

    def test_deal_preserved_across_transitions(self) -> None:
        s0 = KuhnState(deal=(KING, JACK), history=())
        s1 = s0.next_state(BET)
        s2 = s1.next_state(PASS)
        assert s1.deal == (KING, JACK)
        assert s2.deal == (KING, JACK)

    def test_returned_state_is_new_instance(self) -> None:
        """frozen dataclass → 새 객체 반환."""
        s0 = KuhnState(deal=(JACK, QUEEN), history=())
        s1 = s0.next_state(PASS)
        assert s1 is not s0


# -----------------------------------------------------------------------------
# KuhnPoker.state_from_deal
# -----------------------------------------------------------------------------
class TestStateFromDeal:
    def test_state_from_deal_returns_root(self) -> None:
        """딜로부터 만들어진 상태는 빈 히스토리 + P1 차례."""
        s = KuhnPoker.state_from_deal((JACK, QUEEN))
        assert s.deal == (JACK, QUEEN)
        assert s.history == ()
        assert not s.is_terminal
        assert s.current_player == 0


# -----------------------------------------------------------------------------
# KuhnPoker.terminal_utility — exhaustive 5 terminals × 6 deals = 30 cases.
# -----------------------------------------------------------------------------
#
# Standard Kuhn scoring (ante = 1 per player, bet/raise = 1):
#   "pp"  — both check → showdown for pot 2.        u1 = +1 if c1>c2 else -1
#   "bp"  — P1 bet, P2 fold → P1 wins ante.         u1 = +1
#   "bb"  — both bet → showdown for pot 4.          u1 = +2 if c1>c2 else -2
#   "pbp" — P1 pass, P2 bet, P1 fold → P1 -1 ante.  u1 = -1
#   "pbb" — P1 pass, P2 bet, P1 call → showdown 4.  u1 = +2 if c1>c2 else -2
#
# Reference: Neller & Lanctot 2013, Section 4.1.
# -----------------------------------------------------------------------------

_DEALS = [
    (JACK, QUEEN), (QUEEN, JACK),
    (JACK, KING), (KING, JACK),
    (QUEEN, KING), (KING, QUEEN),
]


def _showdown_u1(deal: tuple[int, int], pot_win: int) -> int:
    """Showdown helper: +pot_win if P1's card is higher, else -pot_win."""
    return pot_win if deal[0] > deal[1] else -pot_win


_TERMINAL_CASES: list[tuple[tuple[int, int], tuple[KuhnAction, ...], int]] = []
for d in _DEALS:
    # pp: showdown pot 2 → ±1
    _TERMINAL_CASES.append((d, (PASS, PASS), _showdown_u1(d, 1)))
    # bp: P2 fold → P1 always +1
    _TERMINAL_CASES.append((d, (BET, PASS), +1))
    # bb: showdown pot 4 → ±2
    _TERMINAL_CASES.append((d, (BET, BET), _showdown_u1(d, 2)))
    # pbp: P1 fold → P1 always -1
    _TERMINAL_CASES.append((d, (PASS, BET, PASS), -1))
    # pbb: showdown pot 4 → ±2
    _TERMINAL_CASES.append((d, (PASS, BET, BET), _showdown_u1(d, 2)))


def _case_id(case: tuple[tuple[int, int], tuple[KuhnAction, ...], int]) -> str:
    deal, hist, _ = case
    ranks = "JQK"
    hist_str = "".join(str(a) for a in hist) or "root"
    return f"{ranks[deal[0]]}{ranks[deal[1]]}-{hist_str}"


@pytest.mark.parametrize(
    "deal,history,expected_u1",
    _TERMINAL_CASES,
    ids=[_case_id(c) for c in _TERMINAL_CASES],
)
def test_terminal_utility_exhaustive(
    deal: tuple[int, int],
    history: tuple[KuhnAction, ...],
    expected_u1: int,
) -> None:
    """30개 terminal (5 histories × 6 deals)에 대해 P1 관점 utility 정확성."""
    state = KuhnState(deal=deal, history=history)
    assert state.is_terminal
    assert KuhnPoker.terminal_utility(state) == expected_u1


class TestTerminalUtilityZeroSum:
    """Zero-sum invariant: utility는 P1 관점만 저장. P2는 -u1."""

    @pytest.mark.parametrize(
        "deal,history",
        [(d, h) for d, h, _ in _TERMINAL_CASES],
    )
    def test_utility_is_in_expected_range(
        self, deal: tuple[int, int], history: tuple[KuhnAction, ...]
    ) -> None:
        """Kuhn에서 가능한 utility 절대값은 {1, 2}."""
        u = KuhnPoker.terminal_utility(KuhnState(deal=deal, history=history))
        assert abs(u) in (1, 2)


# -----------------------------------------------------------------------------
# Cross-check: reaching every non-terminal via next_state must not flip is_terminal
# classification compared to constructing KuhnState directly.
# -----------------------------------------------------------------------------
class TestNextStateClassificationConsistency:
    @pytest.mark.parametrize(
        "history",
        [
            (),
            (PASS,), (BET,),
            (PASS, PASS), (PASS, BET), (BET, PASS), (BET, BET),
            (PASS, BET, PASS), (PASS, BET, BET),
        ],
    )
    def test_sequential_next_state_matches_direct_construction(
        self, history: tuple[KuhnAction, ...]
    ) -> None:
        """next_state 연쇄로 만든 상태 == KuhnState 직접 생성 (history 동일)."""
        direct = KuhnState(deal=(QUEEN, KING), history=history)
        walker = KuhnState(deal=(QUEEN, KING), history=())
        for action in history:
            walker = walker.next_state(action)
        assert walker.history == direct.history
        assert walker.is_terminal == direct.is_terminal
        assert walker == direct  # frozen dataclass equality
