"""Unit tests for game state encoding (Phase 3 Day 1 — Deep CFR infrastructure).

Target API (NOT YET IMPLEMENTED — these tests must fail with ``AttributeError``
on access to ``game.encode`` / ``game.ENCODING_DIM``; Phase 3 Day 1 RED state):

    class KuhnPoker:
        NUM_ACTIONS: int = 2
        ENCODING_DIM: int = 6
        @staticmethod
        def encode(state: KuhnState) -> np.ndarray: ...   # shape (6,), float32

    class LeducPoker:
        NUM_ACTIONS: int = 3
        ENCODING_DIM: int = 13
        @staticmethod
        def encode(state: LeducState) -> np.ndarray: ...  # shape (13,), float32

Encoding layouts
----------------
**Kuhn (6-dim float32):**
    [0-2]  own card one-hot (J/Q/K)
    [3]    1.0 if len(history) >= 1 else 0.0
    [4]    1.0 if history[0] == BET else 0.0
    [5]    1.0 if len(history) >= 2 and history[1] == BET else 0.0

**Leduc (13-dim float32):**
    [0-2]  hole rank one-hot (J/Q/K)
    [3-5]  board rank one-hot (J/Q/K, all-zero while round 1)
    [6]    1.0 if board not revealed (round 1) else 0.0
    [7-8]  round one-hot (round 1 / round 2)
    [9]    len(round_history[0]) / 4.0
    [10]   raise_count(round_history[0]) / 2.0
    [11]   len(round_history[1]) / 4.0
    [12]   raise_count(round_history[1]) / 2.0

Why network-friendly encoding (not ``infoset_key`` strings):
    Deep CFR replaces tabular regret lookup with a neural network that consumes
    a fixed-dim numeric vector. ``infoset_key`` is perfect-recall-unique but
    non-vectorisable. This ``encode`` API is the bridge. Design is intentionally
    minimal (no card-bucketing, no action-abstraction) — Kuhn/Leduc are small
    enough to encode faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.kuhn import KuhnAction, KuhnPoker, KuhnState
from poker_ai.games.leduc import LeducAction, LeducPoker, LeducState

# Kuhn action aliases
PASS, BET = KuhnAction.PASS, KuhnAction.BET

# Leduc action aliases
FOLD, CALL, RAISE = LeducAction.FOLD, LeducAction.CALL, LeducAction.RAISE


# =============================================================================
# Kuhn encoding
# =============================================================================
class TestKuhnEncoding:
    """6-dim float32 encoding for Kuhn states."""

    def test_shape_is_6(self) -> None:
        """encode() returns a 1D array of exactly 6 elements."""
        state = KuhnPoker.state_from_deal((0, 1))
        vec = KuhnPoker.encode(state)
        assert vec.shape == (6,)

    def test_dtype_is_float32(self) -> None:
        """Encoding dtype is float32 (network-friendly)."""
        state = KuhnPoker.state_from_deal((0, 1))
        vec = KuhnPoker.encode(state)
        assert vec.dtype == np.float32

    def test_encoding_dim_constant_is_6(self) -> None:
        """ENCODING_DIM class attribute matches runtime shape."""
        assert KuhnPoker.ENCODING_DIM == 6

    def test_deterministic_for_same_state(self) -> None:
        """Same state → byte-identical encoding across calls."""
        state = KuhnPoker.state_from_deal((1, 2))
        v1 = KuhnPoker.encode(state)
        v2 = KuhnPoker.encode(state)
        np.testing.assert_array_equal(v1, v2)

    def test_p1_root_has_card_onehot_only(self) -> None:
        """At root with P1=J (card=0), deal=(J,Q): own card=J → [1,0,0,0,0,0]."""
        state = KuhnPoker.state_from_deal((0, 1))   # P1=J to act
        vec = KuhnPoker.encode(state)
        expected = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_p2_after_pass_has_hist_bit(self) -> None:
        """After (PASS,), acting player = P2. deal=(J,Q), P2 own card=Q (idx 1).

        Layout: [0,1,0, 1, 0, 0]  — Q one-hot + hist_len>=1 bit.
        """
        root = KuhnPoker.state_from_deal((0, 1))
        state = root.next_state(PASS)
        vec = KuhnPoker.encode(state)
        expected = np.array([0, 1, 0, 1, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_p2_after_bet_encodes_bet_bit(self) -> None:
        """After (BET,), acting = P2=Q. [4] (history[0]==BET) is set."""
        root = KuhnPoker.state_from_deal((0, 1))
        state = root.next_state(BET)
        vec = KuhnPoker.encode(state)
        expected = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_p1_after_pass_bet_encodes_second_action(self) -> None:
        """After (PASS, BET), acting = P1=J. [5] (history[1]==BET) set, [4]=0."""
        root = KuhnPoker.state_from_deal((0, 1))
        state = root.next_state(PASS).next_state(BET)
        vec = KuhnPoker.encode(state)
        expected = np.array([1, 0, 0, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_all_12_infosets_yield_distinct_encodings(self) -> None:
        """All 12 reachable Kuhn infosets produce pairwise distinct encodings.

        12 = 3 cards × 4 non-terminal histories (``""``, ``"p"``, ``"b"``, ``"pb"``).
        If any two collide, the advantage net cannot distinguish them.
        """
        # Collect one representative state per infoset by DFS.
        seen: dict[str, np.ndarray] = {}

        def dfs(state: KuhnState) -> None:
            if state.is_terminal:
                return
            key = state.infoset_key
            if key not in seen:
                seen[key] = KuhnPoker.encode(state)
            for a in state.legal_actions():
                dfs(state.next_state(a))

        for deal in KuhnPoker.all_deals():
            dfs(KuhnPoker.state_from_deal(deal))

        assert len(seen) == 12, f"expected 12 infosets, got {len(seen)}"
        # Pairwise distinctness via tuple-set.
        as_tuples = {tuple(v.tolist()) for v in seen.values()}
        assert len(as_tuples) == 12, (
            f"collision detected: {len(seen)} infosets → {len(as_tuples)} encodings"
        )


# =============================================================================
# Leduc encoding
# =============================================================================
class TestLeducEncoding:
    """13-dim float32 encoding for Leduc states."""

    def test_shape_is_13(self) -> None:
        state = LeducPoker.state_from_deal((0, 2, 4))
        vec = LeducPoker.encode(state)
        assert vec.shape == (13,)

    def test_dtype_is_float32(self) -> None:
        state = LeducPoker.state_from_deal((0, 2, 4))
        vec = LeducPoker.encode(state)
        assert vec.dtype == np.float32

    def test_encoding_dim_constant_is_13(self) -> None:
        assert LeducPoker.ENCODING_DIM == 13

    def test_deterministic_for_same_state(self) -> None:
        state = LeducPoker.state_from_deal((0, 2, 4))
        v1 = LeducPoker.encode(state)
        v2 = LeducPoker.encode(state)
        np.testing.assert_array_equal(v1, v2)

    def test_root_state_board_not_revealed(self) -> None:
        """Root of round 1: board hidden, [6]=1.0, [7]=1.0 (round 1), [9..12]=0."""
        state = LeducPoker.state_from_deal((0, 2, 4))  # P1=J, P2=Q, board=K
        vec = LeducPoker.encode(state)

        # [0-2] P1 hole = J (rank 0) → [1,0,0]
        assert vec[0] == 1.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0
        # [3-5] board hidden → all 0
        assert vec[3] == 0.0
        assert vec[4] == 0.0
        assert vec[5] == 0.0
        # [6] board-not-revealed flag
        assert vec[6] == 1.0
        # [7-8] round one-hot: round 1
        assert vec[7] == 1.0
        assert vec[8] == 0.0
        # [9-12] no history yet
        assert vec[9] == 0.0
        assert vec[10] == 0.0
        assert vec[11] == 0.0
        assert vec[12] == 0.0

    def test_round2_state_reveals_board(self) -> None:
        """After round 1 ``cc`` closes + one round 2 action, board is in [3-5],
        [6]=0, [8]=1, round 2 history length encoded in [11]."""
        # deal: P1=J, P2=Q, board=K (id 4, rank 2)
        root = LeducPoker.state_from_deal((0, 2, 4))
        # Round 1: P1 check, P2 check → round 1 closes, board revealed.
        after_cc = root.next_state(CALL).next_state(CALL)
        # Round 2: P1 checks (one action).
        state = after_cc.next_state(CALL)
        # Acting player = P2 = Q (rank 1 → [0,1,0]).
        vec = LeducPoker.encode(state)

        # [0-2] acting player (P2) hole = Q
        assert vec[0] == 0.0
        assert vec[1] == 1.0
        assert vec[2] == 0.0
        # [3-5] board = K (rank 2) → [0,0,1]
        assert vec[3] == 0.0
        assert vec[4] == 0.0
        assert vec[5] == 1.0
        # [6] board revealed → flag is 0
        assert vec[6] == 0.0
        # [7-8] round one-hot: round 2
        assert vec[7] == 0.0
        assert vec[8] == 1.0
        # [11] round 2 length = 1, normalised by 4.0 → 0.25
        assert vec[11] == pytest.approx(0.25)

    def test_raise_count_encoded(self) -> None:
        """Round 1 history (RAISE, RAISE): len=2 → [9]=0.5, raise_count=2 → [10]=1.0."""
        root = LeducPoker.state_from_deal((0, 2, 4))
        state = root.next_state(RAISE).next_state(RAISE)  # P1 bets, P2 raises
        vec = LeducPoker.encode(state)
        assert vec[9] == pytest.approx(2.0 / 4.0)
        assert vec[10] == pytest.approx(2.0 / 2.0)

    def test_all_288_infosets_yield_distinct_encodings(self) -> None:
        """All 288 reachable Leduc infosets produce pairwise distinct encodings.

        288 is the canonical Leduc infoset count (Southey 2005 / Neller 2013 §5).
        Collision would mean the advantage net cannot separate two distinct
        decision points — catastrophic for Deep CFR.
        """
        seen: dict[str, np.ndarray] = {}

        def dfs(state: LeducState) -> None:
            if state.is_terminal:
                return
            key = state.infoset_key
            if key not in seen:
                seen[key] = LeducPoker.encode(state)
            for a in state.legal_actions():
                dfs(state.next_state(a))

        for deal in LeducPoker.all_deals():
            dfs(LeducPoker.state_from_deal(deal))

        assert len(seen) == 288, f"expected 288 infosets, got {len(seen)}"
        as_tuples = {tuple(v.tolist()) for v in seen.values()}
        assert len(as_tuples) == 288, (
            f"collision detected: {len(seen)} infosets → {len(as_tuples)} encodings"
        )


# =============================================================================
# Protocol conformance
# =============================================================================
class TestProtocolConformance:
    """Both games expose ``ENCODING_DIM`` and ``encode`` — Deep CFR relies on
    these via the GameProtocol extension.

    Direct attribute access (no ``hasattr``) so that missing attributes
    surface as ``AttributeError`` for a clean RED signal in the Day 1 run.
    """

    def test_kuhn_has_encoding_dim_attr(self) -> None:
        dim = KuhnPoker.ENCODING_DIM
        assert isinstance(dim, int) and dim > 0

    def test_leduc_has_encoding_dim_attr(self) -> None:
        dim = LeducPoker.ENCODING_DIM
        assert isinstance(dim, int) and dim > 0

    def test_kuhn_has_encode_method(self) -> None:
        method = KuhnPoker.encode
        assert callable(method)

    def test_leduc_has_encode_method(self) -> None:
        method = LeducPoker.encode
        assert callable(method)
