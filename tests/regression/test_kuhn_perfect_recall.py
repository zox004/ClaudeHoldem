"""Regression test: Perfect recall invariant on Kuhn Poker infosets.

CLAUDE.md invariant: "Perfect recall 가정 — 동일 플레이어의 과거 관측은 infoset에 모두 포함".
For Kuhn, this decomposes into three concrete properties:

1.  **Exactly 12 infosets exist** (3 own cards × 4 non-terminal histories).
2.  **Opponent's card must not leak into infoset_key.** Two states that differ
    only in the opponent's private card must yield the same infoset_key.
3.  **Own card and full public history must be recoverable from infoset_key.**
    Distinct own cards or distinct histories must yield distinct keys.

Reference: Neller & Lanctot 2013, Section 4.
"""

from __future__ import annotations

from itertools import product

import pytest

from poker_ai.games.kuhn import KuhnAction, KuhnPoker, KuhnState

ALL_CARDS = (0, 1, 2)  # J, Q, K

# The four non-terminal histories in Kuhn's game tree.
NON_TERMINAL_HISTORIES: tuple[tuple[KuhnAction, ...], ...] = (
    (),
    (KuhnAction.PASS,),
    (KuhnAction.BET,),
    (KuhnAction.PASS, KuhnAction.BET),
)


def test_exactly_twelve_infosets_exist() -> None:
    """3 own cards × 4 non-terminal histories = 12 distinct infoset keys."""
    keys: set[str] = set()
    for deal in KuhnPoker.all_deals():
        for history in NON_TERMINAL_HISTORIES:
            keys.add(KuhnState(deal=deal, history=history).infoset_key)
    assert len(keys) == 12, f"expected 12 infosets, got {len(keys)}: {sorted(keys)}"


def test_opponent_card_does_not_leak_into_infoset_key() -> None:
    """Swapping the opponent's card cannot change the current player's infoset.

    This is the private-info invariant. If it fails, CFR would treat the same
    observation history as two different decision points depending on the
    opponent's hidden card — information the agent can't observe.
    """
    for history in NON_TERMINAL_HISTORIES:
        to_move = len(history) % 2  # P1 if len even, P2 if odd
        for own_card, opp_a, opp_b in product(ALL_CARDS, ALL_CARDS, ALL_CARDS):
            if opp_a == own_card or opp_b == own_card or opp_a == opp_b:
                continue
            if to_move == 0:
                state_a = KuhnState(deal=(own_card, opp_a), history=history)
                state_b = KuhnState(deal=(own_card, opp_b), history=history)
            else:
                state_a = KuhnState(deal=(opp_a, own_card), history=history)
                state_b = KuhnState(deal=(opp_b, own_card), history=history)
            assert state_a.infoset_key == state_b.infoset_key, (
                f"opponent card leaked: history={history}, own={own_card}, "
                f"{state_a.infoset_key!r} vs {state_b.infoset_key!r}"
            )


def test_own_card_is_encoded_in_infoset_key() -> None:
    """At a fixed non-terminal node, different own cards give 3 distinct keys."""
    for history in NON_TERMINAL_HISTORIES:
        to_move = len(history) % 2
        keys: set[str] = set()
        for own in ALL_CARDS:
            opp = (own + 1) % 3  # any non-equal opponent card
            if to_move == 0:
                keys.add(KuhnState(deal=(own, opp), history=history).infoset_key)
            else:
                keys.add(KuhnState(deal=(opp, own), history=history).infoset_key)
        assert len(keys) == 3, (
            f"history {history}: expected 3 own-card infosets, got {len(keys)}: {keys}"
        )


def test_public_history_is_encoded_in_infoset_key() -> None:
    """Same own card but different public histories must give different keys."""
    # P1's non-terminal decision points: "" and "pb"
    for own in ALL_CARDS:
        opp = (own + 1) % 3
        empty_key = KuhnState(deal=(own, opp), history=()).infoset_key
        pb_key = KuhnState(
            deal=(own, opp), history=(KuhnAction.PASS, KuhnAction.BET)
        ).infoset_key
        assert empty_key != pb_key, (
            f"P1 own={own}: '' and 'pb' collapsed to same key {empty_key!r}"
        )

    # P2's non-terminal decision points: "p" and "b"
    for own in ALL_CARDS:
        opp = (own + 1) % 3
        p_key = KuhnState(deal=(opp, own), history=(KuhnAction.PASS,)).infoset_key
        b_key = KuhnState(deal=(opp, own), history=(KuhnAction.BET,)).infoset_key
        assert p_key != b_key, (
            f"P2 own={own}: 'p' and 'b' collapsed to same key {p_key!r}"
        )


def test_infoset_key_determinism() -> None:
    """Repeated computation of the same state's infoset_key is stable.

    Guards against accidental reliance on dict iteration order or set ordering.
    """
    state = KuhnState(deal=(1, 2), history=(KuhnAction.PASS, KuhnAction.BET))
    keys = [state.infoset_key for _ in range(5)]
    assert len(set(keys)) == 1
