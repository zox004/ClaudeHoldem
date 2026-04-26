"""FAILING tests for Phase 4 M3.2 — AbstractedHUNLAction 6-size action grid.

These tests are written *before* the implementation per the project's TDD
rule (CLAUDE.md §1). They will fail with ImportError until M3.2 lands the
:class:`AbstractedHUNLAction` enum, lifts ``AbstractedHUNLGame.NUM_ACTIONS``
from 3 to 6, and rewires :meth:`AbstractedHUNLState.legal_actions` /
:meth:`legal_action_mask` / :meth:`next_state` onto the abstracted enum.

M3.2 design summary (mentor-agreed, see PHASE.md):

- New enum ``AbstractedHUNLAction`` (IntEnum) with values:
  FOLD=0, CALL=1, BET_HALF=2, BET_POT=3, BET_DOUBLE=4, BET_ALLIN=5.
  Values 0/1 align with raw ``HUNLAction.FOLD``/``CALL`` for cheap
  dispatch; values 2..5 are abstracted-only.
- ``AbstractedHUNLGame.NUM_ACTIONS = 6``. Raw ``HUNLGame.NUM_ACTIONS``
  remains 3 — the wrapper expands the action space without touching
  the underlying engine.
- Bet-size formulas (``compute_size``):
    * BET_HALF   → ``int(0.5 * state.pot)``
    * BET_POT    → ``state.pot``
    * BET_DOUBLE → ``2 * state.pot``
    * BET_ALLIN  → ``state.legal_bet_sizes()[-1]`` (heads-up effective cap)
- Canonical-collision rule: when two BET_* enums map to the same chip
  size, only the one with the smaller enum value (smaller IntEnum
  ordinal) is legal; the duplicate is masked off. This guarantees the
  policy distribution puts all probability on the canonical action and
  prevents silent strategy fragmentation across equivalent bets.
- ``AbstractedHUNLState.legal_action_mask()`` shape ``(6,)``.
- ``next_state(action: AbstractedHUNLAction)`` — bet_size kwarg is
  removed; the enum determines the size. Compatibility break with
  M2's ``next_state(HUNLAction, bet_size=int)`` signature.
- ``encode`` is unchanged (102-dim) — abstraction operates only on the
  action interface, not the observation.
- ``infoset_key`` history segments still record raw ``HUNLAction.BET=2``
  + the actually committed chip size, so abstracted enum values never
  appear in keys.

Total: 24 tests across 10 categories (A..J), all expected RED.
"""

from __future__ import annotations

import numpy as np
import pytest

# All M3.2 imports — currently undefined in the source. Tests fail at
# collection with ImportError, which is the intended RED signal.
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLAction,
    AbstractedHUNLGame,
    AbstractedHUNLState,
    HUNLCardAbstractor,
    PostflopBoardAbstractor,
    compute_size,
)
from poker_ai.games.hunl_state import HUNLAction, HUNLState


# Card-id helper for fixtures.
_R = {c: i for i, c in enumerate("23456789TJQKA")}
_S = {c: i for i, c in enumerate("cdhs")}


def cid(s: str) -> int:
    return _R[s[0]] * 4 + _S[s[1]]


# -----------------------------------------------------------------------------
# Fabrication helpers — direct HUNLState construction with custom chip layouts
# so collision regimes can be reached deterministically without driving the
# game tree through many actions.
# -----------------------------------------------------------------------------
def _make_flop_state(
    pot: int,
    stack_p0: int,
    stack_p1: int,
    *,
    last_raise_increment: int = 2,
    current_player: int = 0,
    private_cards: tuple[int, int, int, int] = (0, 1, 2, 3),
    pending_board: tuple[int, int, int, int, int] = (4, 5, 6, 7, 8),
) -> HUNLState:
    """Builds a synthetic flop-round state with the requested chip layout.

    Bankroll invariant ``pot + stack_p0 + stack_p1 == 400`` is enforced by
    :class:`HUNLState`; callers must satisfy it.
    """
    return HUNLState(
        private_cards=private_cards,
        pending_board=pending_board,
        board_cards=pending_board[:3],
        round_history=((), (), (), ()),
        round_bet_sizes=((), (), (), ()),
        current_round=1,
        current_player=current_player,
        stack_p0=stack_p0,
        stack_p1=stack_p1,
        last_raise_increment=last_raise_increment,
        pot=pot,
    )


@pytest.fixture(scope="module")
def game() -> AbstractedHUNLGame:
    """Cheap (n_buckets=10, small MC) abstracted game used across tests."""
    return AbstractedHUNLGame(
        n_buckets=10,
        n_trials=100,
        seed=42,
        postflop_mc_trials=50,
        postflop_threshold_sample_size=40,
    )


def _wrap(raw: HUNLState, game: AbstractedHUNLGame) -> AbstractedHUNLState:
    """Wraps a fabricated raw HUNLState so we can call abstracted-API methods."""
    return AbstractedHUNLState(
        _raw=raw,
        _abstractor=game.abstractor,
        _postflop_abstractor=game.postflop_abstractor,
    )


# =============================================================================
# A. Enum definition (2 tests)
# =============================================================================
class TestAbstractedHUNLActionEnum:
    def test_six_values_with_correct_names_and_order(self) -> None:
        """Enum has exactly six members (FOLD/CALL/BET_HALF/BET_POT/
        BET_DOUBLE/BET_ALLIN) at integer values 0..5 in that order."""
        expected = {
            "FOLD": 0,
            "CALL": 1,
            "BET_HALF": 2,
            "BET_POT": 3,
            "BET_DOUBLE": 4,
            "BET_ALLIN": 5,
        }
        members = {m.name: int(m) for m in AbstractedHUNLAction}
        assert members == expected

    def test_is_intenum_subclass(self) -> None:
        """``AbstractedHUNLAction`` must be IntEnum so it satisfies
        ``StateProtocol.legal_actions() -> tuple[IntEnum, ...]`` and
        cleanly indexes into ``np.ndarray`` masks via ``int(action)``."""
        from enum import IntEnum

        for member in AbstractedHUNLAction:
            assert isinstance(member, IntEnum)


# =============================================================================
# B. NUM_ACTIONS / ENCODING_DIM (2 tests)
# =============================================================================
class TestGameClassConstants:
    def test_num_actions_is_six(self) -> None:
        """NUM_ACTIONS must lift from 3 (raw) to 6 (abstracted) in M3.2."""
        assert AbstractedHUNLGame.NUM_ACTIONS == 6

    def test_encoding_dim_unchanged_at_102(self) -> None:
        """ENCODING_DIM is unchanged — encode() still emits 102 features."""
        assert AbstractedHUNLGame.ENCODING_DIM == 102


# =============================================================================
# C. Deep-stack default legal_actions / mask (4 tests)
# =============================================================================
class TestDeepStackLegalActions:
    """At a 'deep' flop state (pot=20, both stacks=190), every BET_* size
    is distinct and within ``legal_bet_sizes()`` so all four BET enums
    plus CALL must be legal. FOLD is illegal because to_call=0 (matched
    pot at flop start)."""

    @pytest.fixture
    def state(self, game: AbstractedHUNLGame) -> AbstractedHUNLState:
        return _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)

    def test_legal_actions_is_subset_of_six(
        self, state: AbstractedHUNLState
    ) -> None:
        """legal_actions() returns a tuple of AbstractedHUNLAction members
        (StateProtocol-compatible) drawn from the 6-element universe."""
        legal = state.legal_actions()
        assert isinstance(legal, tuple)
        for a in legal:
            assert isinstance(a, AbstractedHUNLAction)
        assert set(legal).issubset(set(AbstractedHUNLAction))

    def test_legal_action_mask_shape_is_six(
        self, state: AbstractedHUNLState
    ) -> None:
        """Mask shape is (6,) — one slot per AbstractedHUNLAction."""
        mask = state.legal_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (6,)

    def test_deep_state_all_four_bets_plus_call_legal(
        self, state: AbstractedHUNLState
    ) -> None:
        """pot=20, stacks=190/190 → BET_HALF=10, BET_POT=20, BET_DOUBLE=40,
        BET_ALLIN=190 all distinct & ∈ [lo=2, hi=190]. FOLD illegal (no
        chips to call). All four BET_* enums and CALL must be legal."""
        mask = state.legal_action_mask()
        assert bool(mask[int(AbstractedHUNLAction.FOLD)]) is False
        assert bool(mask[int(AbstractedHUNLAction.CALL)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_HALF)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_POT)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_DOUBLE)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_ALLIN)]) is True

    def test_mask_is_binary(self, state: AbstractedHUNLState) -> None:
        """Mask values must be 0/1 only (bool or int dtype either fine).
        Defends against off-by-one float-mask bugs that have bitten
        previous infoset/legal-mask wiring (see PHASE.md self-audit log)."""
        mask = state.legal_action_mask()
        # Accept bool dtype or any int/float dtype with values in {0, 1}.
        unique = set(int(v) for v in mask.tolist())
        assert unique.issubset({0, 1})


# =============================================================================
# D. compute_size formulas (3 tests)
# =============================================================================
class TestComputeSize:
    """``compute_size(action, state)`` must implement the four bet-size
    formulas exactly (no rounding-mode drift)."""

    @pytest.fixture
    def raw(self) -> HUNLState:
        return _make_flop_state(pot=20, stack_p0=190, stack_p1=190)

    def test_bet_half_is_int_half_pot(self, raw: HUNLState) -> None:
        """BET_HALF size = ``int(0.5 * pot)``. ``int(0.5 * 20) == 10``."""
        assert compute_size(AbstractedHUNLAction.BET_HALF, raw) == 10

    def test_bet_pot_is_pot(self, raw: HUNLState) -> None:
        """BET_POT size = ``state.pot`` exactly."""
        assert compute_size(AbstractedHUNLAction.BET_POT, raw) == 20

    def test_bet_double_is_two_pot(self, raw: HUNLState) -> None:
        """BET_DOUBLE size = ``2 * state.pot``."""
        assert compute_size(AbstractedHUNLAction.BET_DOUBLE, raw) == 40


# =============================================================================
# E. next_state dispatch (4 tests)
# =============================================================================
class TestNextStateDispatch:
    """``next_state`` translates the abstracted enum into a raw
    ``HUNLAction`` (+ bet_size for BET_*) on the underlying engine."""

    def test_next_state_bet_pot_records_raw_bet_with_pot_size(
        self, game: AbstractedHUNLGame
    ) -> None:
        """next_state(BET_POT) → raw round_history gets HUNLAction.BET +
        round_bet_sizes records the chip size returned by compute_size.

        Subtlety: ``round_bet_sizes`` records *total round contribution
        after the BET*, which equals ``actor_contrib_before + chips_added``
        where ``chips_added = bet_size - actor_contrib_before``. At the
        start of a postflop round actor_contrib_before == 0, so the stored
        size equals ``compute_size(BET_POT, state) == state.pot`` exactly.
        """
        state = _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)
        nxt = state.next_state(AbstractedHUNLAction.BET_POT)
        round_actions = nxt._raw.round_history[1]
        round_sizes = nxt._raw.round_bet_sizes[1]
        assert round_actions[-1] == HUNLAction.BET
        assert round_sizes[-1] == 20

    def test_next_state_bet_half_records_raw_bet_with_half_pot(
        self, game: AbstractedHUNLGame
    ) -> None:
        """next_state(BET_HALF) → raw BET + size = int(0.5 * pot) = 10."""
        state = _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)
        nxt = state.next_state(AbstractedHUNLAction.BET_HALF)
        assert nxt._raw.round_history[1][-1] == HUNLAction.BET
        assert nxt._raw.round_bet_sizes[1][-1] == 10

    def test_next_state_fold_dispatches_to_raw_fold(
        self, game: AbstractedHUNLGame
    ) -> None:
        """next_state(FOLD) must dispatch HUNLAction.FOLD onto the raw
        engine. Use a state where FOLD is legal (preflop SB facing BB =
        root state)."""
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        # Root: SB to act, to_call=1 → FOLD legal.
        nxt = state.next_state(AbstractedHUNLAction.FOLD)
        assert nxt._raw.round_history[0][-1] == HUNLAction.FOLD

    def test_next_state_call_dispatches_to_raw_call(
        self, game: AbstractedHUNLGame
    ) -> None:
        """next_state(CALL) → raw HUNLAction.CALL recorded in the
        relevant round_history slot."""
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        nxt = state.next_state(AbstractedHUNLAction.CALL)
        assert nxt._raw.round_history[0][-1] == HUNLAction.CALL


# =============================================================================
# F. Illegal action raises (2 tests)
# =============================================================================
class TestIllegalActionRaises:
    def test_bet_when_raw_bet_illegal_raises(
        self, game: AbstractedHUNLGame
    ) -> None:
        """When opponent is all-in (opp_stack == 0), raw BET is illegal,
        so every BET_* enum must be illegal too. next_state(BET_POT) must
        raise ValueError."""
        # Bankroll 400: pot=200, stack_p0=200, stack_p1=0 (P1 all-in).
        # Actor=P0; opp_stack=0 → raw legal_actions excludes BET.
        raw = _make_flop_state(
            pot=200, stack_p0=200, stack_p1=0, current_player=0
        )
        state = _wrap(raw, game)
        with pytest.raises(ValueError):
            state.next_state(AbstractedHUNLAction.BET_POT)

    def test_collision_suppressed_action_raises(
        self, game: AbstractedHUNLGame
    ) -> None:
        """When BET_POT == BET_ALLIN size, the canonical-collision rule
        suppresses BET_ALLIN (larger enum value loses). Calling
        next_state(BET_ALLIN) on such a state must raise ValueError —
        even though *some* BET enum at that exact size is legal."""
        # pot=100, stack_p0=100, stack_p1=200. Actor=P0.
        # actor_max=0+100=100, opp_max=0+200=200, hi=100. BET_POT=100,
        # BET_ALLIN=hi=100 → collision; BET_POT wins (smaller enum value).
        raw = _make_flop_state(
            pot=100, stack_p0=100, stack_p1=200, current_player=0
        )
        state = _wrap(raw, game)
        with pytest.raises(ValueError):
            state.next_state(AbstractedHUNLAction.BET_ALLIN)


# =============================================================================
# G. Short-stack collision tests (4 tests) — mentor-mandated silent-bug guard
# =============================================================================
class TestShortStackCollisions:
    """Each case constructs a chip layout that pins exactly one collision
    or out-of-range condition. These regimes are easy to miss in
    end-to-end tests because they only trigger at specific pot/stack
    ratios — hence the explicit fabrication."""

    def test_bet_half_below_min_when_actor_stack_tiny(
        self, game: AbstractedHUNLGame
    ) -> None:
        """pot=100, actor stack=10 (P0). compute_size(BET_HALF)=50 but
        actor_max=10 → BET_HALF > hi, mask[BET_HALF]=0. BET_POT=100>10,
        BET_DOUBLE=200>10 → both illegal too. Only BET_ALLIN (size=10)
        is legal among the bets."""
        # Bankroll: 100 + 10 + 290 = 400.
        raw = _make_flop_state(
            pot=100, stack_p0=10, stack_p1=290, current_player=0
        )
        state = _wrap(raw, game)
        mask = state.legal_action_mask()
        assert bool(mask[int(AbstractedHUNLAction.BET_HALF)]) is False
        assert bool(mask[int(AbstractedHUNLAction.BET_POT)]) is False
        assert bool(mask[int(AbstractedHUNLAction.BET_DOUBLE)]) is False
        assert bool(mask[int(AbstractedHUNLAction.BET_ALLIN)]) is True

    def test_bet_pot_collision_with_allin_keeps_smaller_index(
        self, game: AbstractedHUNLGame
    ) -> None:
        """pot=100, stack_p0=100 (actor), stack_p1=200. actor_max=100,
        opp_max=200, hi=100. BET_POT=100==hi=BET_ALLIN. Smaller enum
        value (BET_POT=3) wins → mask[BET_POT]=1, mask[BET_ALLIN]=0."""
        raw = _make_flop_state(
            pot=100, stack_p0=100, stack_p1=200, current_player=0
        )
        state = _wrap(raw, game)
        mask = state.legal_action_mask()
        assert bool(mask[int(AbstractedHUNLAction.BET_POT)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_ALLIN)]) is False

    def test_bet_double_collision_with_allin_keeps_smaller_index(
        self, game: AbstractedHUNLGame
    ) -> None:
        """pot=50, stack_p0=100 (actor), stack_p1=250. actor_max=100,
        opp_max=250, hi=100. BET_DOUBLE=2*50=100==hi=BET_ALLIN. Smaller
        enum (BET_DOUBLE=4) wins → mask[BET_DOUBLE]=1, mask[BET_ALLIN]=0.
        BET_HALF=25 and BET_POT=50 still distinct & legal."""
        raw = _make_flop_state(
            pot=50, stack_p0=100, stack_p1=250, current_player=0
        )
        state = _wrap(raw, game)
        mask = state.legal_action_mask()
        assert bool(mask[int(AbstractedHUNLAction.BET_HALF)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_POT)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_DOUBLE)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_ALLIN)]) is False

    def test_deep_stack_all_four_bets_distinct_and_legal(
        self, game: AbstractedHUNLGame
    ) -> None:
        """pot=20, stacks=190/190 → BET_HALF=10, BET_POT=20, BET_DOUBLE=40,
        BET_ALLIN=190. All distinct, all in legal range. None collide;
        none short. Mirror of the deep-stack default test but explicit
        about the no-collision invariant."""
        raw = _make_flop_state(pot=20, stack_p0=190, stack_p1=190)
        state = _wrap(raw, game)
        mask = state.legal_action_mask()
        assert bool(mask[int(AbstractedHUNLAction.BET_HALF)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_POT)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_DOUBLE)]) is True
        assert bool(mask[int(AbstractedHUNLAction.BET_ALLIN)]) is True


# =============================================================================
# H. encode() unchanged (1 test)
# =============================================================================
class TestEncodeUnchanged:
    def test_encode_shape_and_history_records_raw_action_id(
        self, game: AbstractedHUNLGame
    ) -> None:
        """encode shape stays at (102,). After next_state(BET_POT), the
        history slot in encode must contain raw HUNLAction.BET=2 (not
        AbstractedHUNLAction.BET_POT=3); abstracted enum values must
        never appear in encoded observations."""
        state = _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)
        nxt = state.next_state(AbstractedHUNLAction.BET_POT)
        enc = game.encode(nxt)
        assert enc.shape == (102,)
        # action_id field at history slot index = (preflop empty + flop[0]).
        # Flat history begins at offset 22; first real action is the
        # one we just took (no preflop actions in this fabricated state).
        # Slot layout: [action_id/3, size/200] per slot.
        # Recover action_id ∈ {0=FOLD, 1=CALL, 2=BET, 3=NULL_PADDING}.
        first_slot_action_id_field = float(enc[22]) * 3.0
        action_id = int(round(first_slot_action_id_field))
        assert action_id in {0, 1, 2}, (
            f"history slot must hold raw HUNLAction value (0..2); abstracted "
            f"enum values 3..5 must not appear, got {action_id}"
        )
        assert action_id == int(HUNLAction.BET)


# =============================================================================
# I. infoset_key history consistency (1 test)
# =============================================================================
class TestInfosetKeyHistoryConsistency:
    def test_history_segment_uses_raw_action_id_and_real_size(
        self, game: AbstractedHUNLGame
    ) -> None:
        """next_state(BET_POT) → infoset_key history segment contains
        ``"2:<chip_size>"`` where 2 = int(HUNLAction.BET) and chip_size
        is the actually-committed total round contribution. Abstracted
        enum value 3 (BET_POT) must not appear in the key."""
        state = _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)
        nxt = state.next_state(AbstractedHUNLAction.BET_POT)
        key = nxt.infoset_key
        # History format per M2: "<action_id>:<size>" joined by '.', and
        # round-flat. The fabricated state has no preflop actions, so the
        # first (and only) history entry is the BET we just placed.
        # Acceptance: the history substring contains "2:20" exactly.
        assert "2:20" in key, (
            f"infoset_key history must record raw HUNLAction.BET=2 + the "
            f"committed chip size 20; got {key!r}"
        )
        # Negative: the abstracted enum integer (3 for BET_POT) must NOT
        # appear as an action_id in the history. We cannot just check
        # "3:" globally because the preflop-bucket prefix or board bucket
        # could legitimately contain the digit 3; restrict to the history
        # tail (everything after the third ':' in the key).
        # Key layout: "<bucket>|<round>:<board_bucket>:<history>".
        history_tail = key.split(":", 2)[-1]
        assert "3:" not in history_tail, (
            f"abstracted enum value 3 must not appear as an action_id in "
            f"infoset_key history; got history={history_tail!r}"
        )


# =============================================================================
# J. AbstractedHUNLState.legal_bet_sizes() removed (1 test)
# =============================================================================
class TestLegalBetSizesRemoved:
    def test_abstracted_state_no_longer_exposes_legal_bet_sizes(
        self, game: AbstractedHUNLGame
    ) -> None:
        """M3.2 contract: bet sizes are determined by the enum, so the
        wrapper-level ``legal_bet_sizes`` method (M2 leftover) is gone.
        Either the attribute is missing entirely (preferred) OR calling
        it raises NotImplementedError (acceptable transitional form).

        Raw HUNLState still carries ``legal_bet_sizes()`` for internal
        use by ``compute_size`` / mask logic — this test only asserts
        the *wrapper* surface is cleaned up."""
        state = _wrap(_make_flop_state(pot=20, stack_p0=190, stack_p1=190), game)
        if hasattr(state, "legal_bet_sizes"):
            with pytest.raises(NotImplementedError):
                state.legal_bet_sizes()
        else:
            # Preferred: method removed entirely.
            assert not hasattr(state, "legal_bet_sizes")
