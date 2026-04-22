"""Regression test: Perfect recall invariant on Leduc Hold'em infosets.

CLAUDE.md invariant: "Perfect recall 가정 — 동일 플레이어의 과거 관측은
infoset에 모두 포함". For Leduc, this expands into four concrete properties:

1.  **Exactly 288 infosets exist** (Neller & Lanctot 2013, §5 Table 2).
    The count is obtained by full DFS over the game tree, collecting
    infoset_key from every non-terminal state.
2.  **Opponent's card must not leak into infoset_key.** Swapping the
    opponent's private card at any non-terminal must not change the
    to-move player's infoset_key.
3.  **Own card and public history must be recoverable.** Different own
    cards or different histories must yield different keys.
4.  **Board card timing.** The board-rank char appears in the key iff
    we are in round 2 (i.e., the state has transitioned past the "cc",
    "rc", "crc", "rrc", or "crrc" round-1 closure).

Target module (NOT YET IMPLEMENTED — RED for Phase 2 Week 1 Day 1):
    src/poker_ai/games/leduc.py

Reference: Neller & Lanctot 2013, Section 5; Southey et al. 2005.
"""

from __future__ import annotations

import pytest

from poker_ai.games.leduc import LeducAction, LeducPoker, LeducState


# -----------------------------------------------------------------------------
# DFS helper: enumerate all non-terminal states reachable from a given deal.
# -----------------------------------------------------------------------------
def _enumerate_non_terminal(state: LeducState) -> list[LeducState]:
    """DFS over the game tree rooted at ``state``; return every non-terminal node."""
    out: list[LeducState] = []
    stack: list[LeducState] = [state]
    while stack:
        cur = stack.pop()
        if cur.is_terminal:
            continue
        out.append(cur)
        for action in cur.legal_actions():
            stack.append(cur.next_state(action))
    return out


# -----------------------------------------------------------------------------
# Infoset count regression
# -----------------------------------------------------------------------------
class TestInfosetCount:
    def test_exactly_288_infosets_exist(self) -> None:
        """3*4 (round-1 per-player) + 3*3*12 (round-2 per-player) = 288 infosets.

        Expected 288 per Neller & Lanctot 2013 Table 2 (Leduc full tree).
        If this fails, debug by printing the observed count and a sample of keys.
        """
        keys: set[str] = set()
        for deal in LeducPoker.all_deals():
            root = LeducPoker.state_from_deal(deal)
            for state in _enumerate_non_terminal(root):
                keys.add(state.infoset_key)
        assert len(keys) == 288, (
            f"expected 288 Leduc infosets, got {len(keys)}. "
            f"First 10 keys (sorted): {sorted(keys)[:10]}"
        )


# -----------------------------------------------------------------------------
# Opponent card privacy
# -----------------------------------------------------------------------------
class TestOpponentCardPrivacy:
    @pytest.mark.parametrize(
        "history_round1",
        [
            (),
            (LeducAction.CALL,),
            (LeducAction.RAISE,),
            (LeducAction.CALL, LeducAction.RAISE),
        ],
        ids=["root", "after_c", "after_r", "after_cr"],
    )
    def test_round1_opponent_swap_does_not_change_key(
        self, history_round1: tuple[LeducAction, ...]
    ) -> None:
        """Round 1: swapping opponent's card id (same or different rank) must
        leave the to-move player's infoset_key invariant.
        """
        # Fix own card = J0 (rank J), walk to history_round1 where it's our turn.
        # Determine whose perspective is to move: P1 if len even, P2 if odd.
        to_move = len(history_round1) % 2
        # Pick two different opponent card IDs (possibly same rank, possibly not).
        opp_cards = [2, 3, 4, 5]  # Q0, Q1, K0, K1 — all != J0
        # Pair them up by "any two different".
        pairs = [
            (opp_cards[i], opp_cards[j])
            for i in range(len(opp_cards))
            for j in range(i + 1, len(opp_cards))
        ]
        # Board card must be distinct from both private cards; pick one free id.
        for opp_a, opp_b in pairs:
            # board must differ from own (0=J0) and from opp_a and opp_b
            used = {0, opp_a, opp_b}
            board = next(c for c in range(6) if c not in used)
            if to_move == 0:
                deal_a = (0, opp_a, board)
                deal_b = (0, opp_b, board)
            else:
                deal_a = (opp_a, 0, board)
                deal_b = (opp_b, 0, board)
            state_a = LeducPoker.state_from_deal(deal_a)
            state_b = LeducPoker.state_from_deal(deal_b)
            for a in history_round1:
                state_a = state_a.next_state(a)
                state_b = state_b.next_state(a)
            assert state_a.infoset_key == state_b.infoset_key, (
                f"round-1 opponent card leak: history={history_round1}, "
                f"{state_a.infoset_key!r} vs {state_b.infoset_key!r}"
            )

    def test_round2_opponent_swap_does_not_change_key(self) -> None:
        """Round 2 after cc: same own card, same board, same history, different
        opponent card → identical infoset_key.
        """
        # own P1 = J0; board = Q0 (rank Q); opponents differ.
        for opp_a, opp_b in [(4, 5), (2, 4), (3, 5)]:
            deal_a = (0, opp_a, 2)
            deal_b = (0, opp_b, 2)
            s_a = LeducPoker.state_from_deal(deal_a)
            s_b = LeducPoker.state_from_deal(deal_b)
            # round 1 cc, round 2 open (P1 to move again).
            for a in (LeducAction.CALL, LeducAction.CALL):
                s_a = s_a.next_state(a)
                s_b = s_b.next_state(a)
            assert s_a.infoset_key == s_b.infoset_key, (
                f"round-2 opponent card leak: {s_a.infoset_key!r} vs {s_b.infoset_key!r}"
            )


# -----------------------------------------------------------------------------
# Own-card and public-history encoding
# -----------------------------------------------------------------------------
class TestInfosetEncoding:
    def test_different_own_card_different_key(self) -> None:
        """At the root (P1 to move), three distinct own ranks → three distinct keys."""
        # Use distinct (own, opp, board) triples; just need P1's rank to vary.
        root_j = LeducPoker.state_from_deal((0, 2, 4))  # P1 J
        root_q = LeducPoker.state_from_deal((2, 0, 4))  # P1 Q
        root_k = LeducPoker.state_from_deal((4, 0, 2))  # P1 K
        keys = {root_j.infoset_key, root_q.infoset_key, root_k.infoset_key}
        assert len(keys) == 3, f"own-card collapse: {keys}"

    def test_different_history_different_key(self) -> None:
        """Same (own=P1 J) but different round-1 histories → different keys."""
        base = LeducPoker.state_from_deal((0, 2, 4))  # P1 J
        # Non-terminal P1-to-move histories in round 1: () and (C, R) (since 'cr' → P1 turn).
        root_key = base.infoset_key
        after_cr = base.next_state(LeducAction.CALL).next_state(LeducAction.RAISE).infoset_key
        assert root_key != after_cr, (
            f"history collapse: root={root_key!r}, after_cr={after_cr!r}"
        )

    def test_round1_vs_round2_same_own_same_history_string_differ(self) -> None:
        """Round 1 'cc' vs root '' differ; and round-2 key contains board char.

        Even if round-1 and round-2 happened to share the same visible player
        history string, the '.' + board char separates round 2 from round 1.
        """
        base = LeducPoker.state_from_deal((0, 2, 4))  # P1 J, P2 Q, board K
        # After cc: now in round 2, P1 to move, board revealed as K.
        r2 = base.next_state(LeducAction.CALL).next_state(LeducAction.CALL)
        # Root key has no '.'; r2 key must have '.'.
        assert "." not in base.infoset_key
        assert "." in r2.infoset_key


# -----------------------------------------------------------------------------
# Board-card timing in key
# -----------------------------------------------------------------------------
class TestBoardCardTiming:
    def test_round1_keys_never_contain_dot(self) -> None:
        """No infoset_key from any round-1 non-terminal state contains '.'."""
        for deal in LeducPoker.all_deals():
            root = LeducPoker.state_from_deal(deal)
            for state in _enumerate_non_terminal(root):
                if state.round_idx == 0:
                    assert "." not in state.infoset_key, (
                        f"round-1 key contains '.': {state.infoset_key!r}"
                    )

    def test_round2_keys_always_contain_dot_and_board_rank(self) -> None:
        """Every round-2 non-terminal key contains '.' followed by J|Q|K."""
        for deal in LeducPoker.all_deals():
            root = LeducPoker.state_from_deal(deal)
            for state in _enumerate_non_terminal(root):
                if state.round_idx == 1:
                    key = state.infoset_key
                    assert "." in key, f"round-2 key missing '.': {key!r}"
                    dot_idx = key.index(".")
                    # Character immediately after '.' must be a rank char.
                    assert dot_idx + 1 < len(key), (
                        f"round-2 key truncated after '.': {key!r}"
                    )
                    assert key[dot_idx + 1] in "JQK", (
                        f"round-2 key board rank char invalid: {key!r}"
                    )


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
def test_infoset_key_determinism() -> None:
    """Repeated computation of the same state's infoset_key is stable.

    Guards against accidental reliance on dict/set iteration order during key
    construction (e.g., joining actions via an unordered container).
    """
    state = (
        LeducPoker.state_from_deal((0, 2, 4))  # P1 J, P2 Q, board K
        .next_state(LeducAction.CALL)
        .next_state(LeducAction.RAISE)
    )
    keys = [state.infoset_key for _ in range(5)]
    assert len(set(keys)) == 1
