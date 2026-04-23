"""Unit tests for GameProtocol / StateProtocol structural typing.

Target module (NOT YET IMPLEMENTED — these tests must fail with
ModuleNotFoundError at import time):
    src/poker_ai/games/protocol.py

Target API (runtime_checkable Protocols):
    class StateProtocol(Protocol):
        is_terminal: bool (property)
        current_player: int (property)
        infoset_key: str (property)
        def legal_actions(self) -> tuple[IntEnum, ...]: ...
        def next_state(self, action: IntEnum) -> StateProtocol: ...

    class GameProtocol(Protocol):
        NUM_ACTIONS: int
        def all_deals(self) -> tuple[Any, ...]: ...
        def state_from_deal(self, deal: Any) -> StateProtocol: ...
        def terminal_utility(self, state: StateProtocol) -> float: ...

The Protocols intentionally do NOT include concrete-specific members such as
``legal_action_mask`` (Leduc-only) or ``private_cards`` / ``cards`` (name
differs between games) — those would over-specify the structural type and
break Kuhn conformance.

Both KuhnPoker and LeducPoker are expected to conform structurally; neither
needs to inherit from the Protocol classes.
"""

from __future__ import annotations

from enum import IntEnum

import pytest


# -----------------------------------------------------------------------------
# TestProtocolImports — module-level importability of the new protocol module
# -----------------------------------------------------------------------------
class TestProtocolImports:
    def test_game_protocol_importable(self) -> None:
        """GameProtocol symbol is exported from poker_ai.games.protocol."""
        from poker_ai.games.protocol import GameProtocol  # noqa: F401

    def test_state_protocol_importable(self) -> None:
        """StateProtocol symbol is exported from poker_ai.games.protocol."""
        from poker_ai.games.protocol import StateProtocol  # noqa: F401


# -----------------------------------------------------------------------------
# Helpers: reach a terminal state for each game via a simple fold path.
# -----------------------------------------------------------------------------
def _kuhn_terminal_from_first_deal():
    """Kuhn: root → PASS → PASS terminates (pp showdown)."""
    from poker_ai.games.kuhn import KuhnAction, KuhnPoker

    game = KuhnPoker()
    deal = game.all_deals()[0]
    state = game.state_from_deal(deal)
    state = state.next_state(KuhnAction.PASS)
    state = state.next_state(KuhnAction.PASS)
    return game, state


def _leduc_terminal_from_first_deal():
    """Leduc: root → RAISE → FOLD terminates (P2 folds round 1)."""
    from poker_ai.games.leduc import LeducAction, LeducPoker

    game = LeducPoker()
    deal = game.all_deals()[0]
    state = game.state_from_deal(deal)
    state = state.next_state(LeducAction.RAISE)
    state = state.next_state(LeducAction.FOLD)
    return game, state


# -----------------------------------------------------------------------------
# TestGameProtocolConformance — runtime isinstance() checks on Kuhn + Leduc
# -----------------------------------------------------------------------------
def _make_kuhn():
    from poker_ai.games.kuhn import KuhnPoker

    return KuhnPoker()


def _make_leduc():
    from poker_ai.games.leduc import LeducPoker

    return LeducPoker()


@pytest.mark.parametrize(
    "game_factory",
    [_make_kuhn, _make_leduc],
    ids=["kuhn", "leduc"],
)
class TestGameProtocolConformance:
    def test_instance_satisfies_game_protocol(self, game_factory) -> None:
        """Instances of KuhnPoker / LeducPoker structurally satisfy GameProtocol."""
        from poker_ai.games.protocol import GameProtocol

        game = game_factory()
        assert isinstance(game, GameProtocol), (
            f"{type(game).__name__} does not satisfy GameProtocol (missing members?)"
        )

    def test_has_num_actions_int(self, game_factory) -> None:
        """GameProtocol requires ``NUM_ACTIONS: int`` (> 0)."""
        game = game_factory()
        assert isinstance(game.NUM_ACTIONS, int)
        assert game.NUM_ACTIONS > 0

    def test_all_deals_returns_nonempty_tuple(self, game_factory) -> None:
        """``all_deals()`` returns a non-empty tuple of deal objects."""
        game = game_factory()
        deals = game.all_deals()
        assert isinstance(deals, tuple)
        assert len(deals) > 0

    def test_state_from_deal_returns_state_protocol(self, game_factory) -> None:
        """``state_from_deal(deal)`` returns an object satisfying StateProtocol."""
        from poker_ai.games.protocol import StateProtocol

        game = game_factory()
        deal = game.all_deals()[0]
        state = game.state_from_deal(deal)
        assert isinstance(state, StateProtocol), (
            f"{type(state).__name__} does not satisfy StateProtocol"
        )

    def test_terminal_utility_is_callable_on_terminal(self, game_factory) -> None:
        """Terminal utility returns a finite float from a reachable terminal."""
        import math

        game = game_factory()
        # Build a terminal state through the appropriate per-game fold/call path.
        if type(game).__name__ == "KuhnPoker":
            _, terminal = _kuhn_terminal_from_first_deal()
        else:
            _, terminal = _leduc_terminal_from_first_deal()

        assert terminal.is_terminal, "test helper must reach a terminal state"
        u = game.terminal_utility(terminal)
        assert isinstance(u, float)
        assert math.isfinite(u)


# -----------------------------------------------------------------------------
# TestStateProtocolConformance — runtime isinstance() checks on root states
# -----------------------------------------------------------------------------
def _kuhn_root_state():
    from poker_ai.games.kuhn import KuhnPoker

    game = KuhnPoker()
    return game.state_from_deal(game.all_deals()[0])


def _leduc_root_state():
    from poker_ai.games.leduc import LeducPoker

    game = LeducPoker()
    return game.state_from_deal(game.all_deals()[0])


@pytest.mark.parametrize(
    "state_factory",
    [_kuhn_root_state, _leduc_root_state],
    ids=["kuhn", "leduc"],
)
class TestStateProtocolConformance:
    def test_root_state_satisfies_protocol(self, state_factory) -> None:
        from poker_ai.games.protocol import StateProtocol

        state = state_factory()
        assert isinstance(state, StateProtocol), (
            f"{type(state).__name__} root state fails StateProtocol conformance"
        )

    def test_is_terminal_is_bool(self, state_factory) -> None:
        state = state_factory()
        assert isinstance(state.is_terminal, bool)

    def test_current_player_is_int_in_0_1(self, state_factory) -> None:
        state = state_factory()
        cp = state.current_player
        assert isinstance(cp, int)
        assert cp in (0, 1), f"current_player={cp} out of range"

    def test_infoset_key_is_str(self, state_factory) -> None:
        state = state_factory()
        assert isinstance(state.infoset_key, str)
        assert len(state.infoset_key) > 0

    def test_legal_actions_is_tuple_of_intenums(self, state_factory) -> None:
        state = state_factory()
        legal = state.legal_actions()
        assert isinstance(legal, tuple)
        assert len(legal) > 0, "root state should have at least one legal action"
        for a in legal:
            assert isinstance(a, IntEnum), (
                f"legal action {a!r} (type {type(a).__name__}) is not an IntEnum"
            )

    def test_next_state_returns_state_protocol(self, state_factory) -> None:
        from poker_ai.games.protocol import StateProtocol

        state = state_factory()
        first_legal = state.legal_actions()[0]
        new_state = state.next_state(first_legal)
        assert isinstance(new_state, StateProtocol), (
            f"next_state returned {type(new_state).__name__}, "
            "which fails StateProtocol conformance"
        )


# -----------------------------------------------------------------------------
# TestProtocolIsNotOverSpecified — Protocol must stay structurally minimal.
#
# If these members appeared in StateProtocol, KuhnState (which has no
# legal_action_mask and uses `deal` not `private_cards`) would no longer
# conform, defeating the refactor.
# -----------------------------------------------------------------------------
class TestProtocolIsNotOverSpecified:
    def test_protocol_does_not_require_legal_action_mask(self) -> None:
        """``legal_action_mask`` is Leduc-specific — it must NOT be on StateProtocol."""
        from poker_ai.games.protocol import StateProtocol

        # Protocols expose their declared attrs via __annotations__ or as
        # class-level attributes. We check both the class namespace and
        # its annotations to be robust against either declaration style.
        protocol_attrs = set(vars(StateProtocol).keys()) | set(
            getattr(StateProtocol, "__annotations__", {}).keys()
        )
        assert "legal_action_mask" not in protocol_attrs, (
            "StateProtocol must not require legal_action_mask (Kuhn would break)"
        )

    def test_protocol_does_not_require_private_cards(self) -> None:
        """Private-card attr name differs (Kuhn=`deal`, Leduc=`private_cards`);
        neither name should be on StateProtocol."""
        from poker_ai.games.protocol import StateProtocol

        protocol_attrs = set(vars(StateProtocol).keys()) | set(
            getattr(StateProtocol, "__annotations__", {}).keys()
        )
        assert "private_cards" not in protocol_attrs, (
            "StateProtocol must not require private_cards"
        )
        assert "deal" not in protocol_attrs, (
            "StateProtocol must not require deal"
        )
