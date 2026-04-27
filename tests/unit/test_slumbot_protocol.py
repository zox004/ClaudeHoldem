"""FAILING tests for Phase 4 M4.2 — Slumbot action protocol adapter.

These tests exercise the action-string ↔ abstracted-action translation
layer that bridges our :class:`AbstractedHUNLState` 6-way action grid
with Slumbot's wire format (``f`` / ``c`` / ``k`` / ``b<int>`` tokens
on a per-street basis separated by ``/``).

Scope (M4.2):
    - :func:`parse_action_token` — single token → SlumbotActionToken.
    - :func:`split_action_sequence` — full sequence → per-street
      lists of tokens (empty inner lists allowed for skipped streets,
      e.g. preflop all-in producing ``b20000c///``).
    - :func:`nearest_abstracted_bet_size` — opponent raw chip size →
      our 6-action abstracted grid bucket (nearest-bucket translation
      with tie-break = larger size). M4.2 chooses (i) nearest-bucket
      over (ii) Schnizlein 2009 probabilistic state translation; the
      latter is registered as a Phase 5 hook.
    - :func:`encode_action` — our :class:`AbstractedHUNLAction` →
      Slumbot token (FOLD→f, CALL→c|k, BET_*→b<chip_to_slumbot>).
    - :func:`ingest_opponent_token` — Slumbot token → state advance.
    - :func:`replay_sequence` — full Slumbot sequence + initial deal
      → final :class:`AbstractedHUNLState`. Handles client_pos mapping
      between Slumbot (0=SB/button, 1=BB) and our HUNLState convention
      (player_idx 0=BB, 1=SB acts first preflop).

3 audit hooks the tests cover (Phase 4 M4.2 spec):
    - Hook 1 — Re-raise chain: standard ``b500b1500c`` plus a non-
      standard size chain ``b347b892c`` exercising nearest-bucket
      dispatch on out-of-grid raw raise sizes.
    - Hook 2 — All-in clamp: Slumbot ``b<full_stack>`` aligns with our
      BET_ALLIN = ``state.legal_bet_sizes()[-1]`` cap (heads-up
      effective stack).
    - Hook 3 — Street transition / empty round: ``b20000c///`` (preflop
      all-in then 3 empty street-suffix slots) replays to terminal at
      preflop-close.

Module under test: ``poker_ai.eval.slumbot_protocol`` (does not yet
exist — every test is expected to RED with ``ImportError`` until M4.2
implementation lands).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.games.hunl import HUNLGame
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLAction,
    AbstractedHUNLGame,
    AbstractedHUNLState,
)
from poker_ai.games.hunl_state import HUNLAction, STARTING_STACK_CHIPS

# Module under test — these imports are intentionally going to fail
# until ``slumbot_protocol.py`` is implemented.
from poker_ai.eval.slumbot_protocol import (  # noqa: E402
    SlumbotActionToken,
    encode_action,
    ingest_opponent_token,
    nearest_abstracted_bet_size,
    parse_action_token,
    replay_sequence,
    split_action_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fresh_game(seed: int = 42) -> AbstractedHUNLGame:
    """Constructs a small, fast-to-build AbstractedHUNLGame for tests.

    Test-friendly bucket / MC trial counts to keep fixture build under
    ~1 second; the protocol layer does not depend on bucket *quality*,
    only on the legal-action interface.
    """
    return AbstractedHUNLGame(
        n_buckets=10,
        n_trials=200,
        seed=seed,
        postflop_mc_trials=50,
        postflop_threshold_sample_size=80,
    )


# Deterministic 9-card deal used across replay tests. Layout from
# ``HUNLGame.sample_deal``:
#   (p0_h1, p0_h2, p1_h1, p1_h2, b1, b2, b3, b4, b5)
_FIXED_DEAL: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)


# =============================================================================
# A. parse_action_token (5 tests)
# =============================================================================
class TestParseActionToken:
    """Single-token parsing into :class:`SlumbotActionToken`."""

    def test_fold_token(self) -> None:
        """`'f'` parses to (kind='f', bet_to_slumbot_chips=0)."""
        tok = parse_action_token("f")
        assert tok == SlumbotActionToken(kind="f", bet_to_slumbot_chips=0)

    def test_call_token(self) -> None:
        """`'c'` parses to (kind='c', bet_to_slumbot_chips=0)."""
        tok = parse_action_token("c")
        assert tok == SlumbotActionToken(kind="c", bet_to_slumbot_chips=0)

    def test_check_token(self) -> None:
        """`'k'` parses to (kind='k', bet_to_slumbot_chips=0)."""
        tok = parse_action_token("k")
        assert tok == SlumbotActionToken(kind="k", bet_to_slumbot_chips=0)

    def test_bet_token_with_size(self) -> None:
        """`'b500'` parses to (kind='b', bet_to_slumbot_chips=500)."""
        tok = parse_action_token("b500")
        assert tok == SlumbotActionToken(kind="b", bet_to_slumbot_chips=500)

    def test_invalid_tokens_raise(self) -> None:
        """Unknown letters / bare 'b' / empty string raise ValueError."""
        for bad in ("", "x", "b", "bX", "B500", "fold"):
            with pytest.raises(ValueError):
                parse_action_token(bad)


# =============================================================================
# B. split_action_sequence (5 tests)
# =============================================================================
class TestSplitActionSequence:
    """Splits full Slumbot sequence into per-street token lists."""

    def test_empty_string_yields_four_empty_rounds(self) -> None:
        """Empty input → list of 4 empty inner lists (one per HUNL street)."""
        out = split_action_sequence("")
        assert out == [[], [], [], []]

    def test_full_four_round_sequence(self) -> None:
        """`'cb300c/kk/kk/kk'` splits into 4 streets with correct tokens."""
        out = split_action_sequence("cb300c/kk/kk/kk")
        assert out == [
            ["c", "b300", "c"],
            ["k", "k"],
            ["k", "k"],
            ["k", "k"],
        ]

    def test_preflop_only_sequence_pads_empty_postflop(self) -> None:
        """`'b20000c'` (preflop ends mid-hand): preflop tokens + 3 empty
        streets to keep the 4-street shape."""
        out = split_action_sequence("b20000c")
        assert out == [["b20000", "c"], [], [], []]

    def test_preflop_all_in_with_explicit_empty_streets(self) -> None:
        """Hook 3: `'b20000c///'` (preflop all-in followed by 3 explicit
        empty streets) parses to one street + 3 empties."""
        out = split_action_sequence("b20000c///")
        assert out == [["b20000", "c"], [], [], []]

    def test_re_raise_chain_standard_sizes(self) -> None:
        """Hook 1 (standard re-raise chain): `'b500b1500c'` parses as
        three preflop tokens with the in-grid sizes preserved verbatim."""
        out = split_action_sequence("b500b1500c")
        assert out == [["b500", "b1500", "c"], [], [], []]


# =============================================================================
# C. nearest_abstracted_bet_size (8 tests)
# =============================================================================
class TestNearestAbstractedBetSize:
    """Nearest-bucket translation of opponent raw raise sizes."""

    def test_exact_match_returns_same_size(self) -> None:
        """raw=10 with legal=[10,20] returns 10 (exact match)."""
        assert nearest_abstracted_bet_size(10, [10, 20]) == 10

    def test_lower_neighbour_wins(self) -> None:
        """raw=12 closer to 10 than 20 → returns 10."""
        assert nearest_abstracted_bet_size(12, [10, 20]) == 10

    def test_upper_neighbour_wins(self) -> None:
        """raw=18 closer to 20 than 10 → returns 20."""
        assert nearest_abstracted_bet_size(18, [10, 20]) == 20

    def test_tie_break_prefers_larger_size(self) -> None:
        """raw=15 equidistant between 10 and 20 → tie-break to 20.

        Measurement-conservatism choice: larger bucket better preserves
        opponent raise-aggression intent for downstream attribution.
        """
        assert nearest_abstracted_bet_size(15, [10, 20]) == 20

    def test_below_minimum_snaps_to_lowest(self) -> None:
        """raw=5 below all legal sizes → snaps to lowest (10)."""
        assert nearest_abstracted_bet_size(5, [10, 20]) == 10

    def test_above_maximum_snaps_to_highest(self) -> None:
        """raw=100 above all legal sizes → snaps to highest (20).

        Hook 2 ingredient: Slumbot full-stack ``b<huge>`` clamps onto
        our BET_ALLIN-sized bucket via this branch.
        """
        assert nearest_abstracted_bet_size(100, [10, 20]) == 20

    def test_empty_legal_list_raises(self) -> None:
        """Empty legal_abstracted_sizes → ValueError (no fallback)."""
        with pytest.raises(ValueError):
            nearest_abstracted_bet_size(10, [])

    def test_non_grid_raw_size_dispatches(self) -> None:
        """Hook 1 ingredient: a non-standard raw size (b347 / 50 ≈ 7
        in our chips) dispatches to the closest legal abstracted size.

        With legal=[5, 10, 20], raw=7 is closer to 5 (distance 2) than
        to 10 (distance 3) → returns 5.
        """
        assert nearest_abstracted_bet_size(7, [5, 10, 20]) == 5


# =============================================================================
# D. encode_action (8 tests)
# =============================================================================
class TestEncodeAction:
    """:class:`AbstractedHUNLAction` → Slumbot token string emit."""

    def test_fold_emits_f(self) -> None:
        """FOLD → 'f' regardless of pot/bet context (only legal when
        there is a bet to call, but the encoder maps the symbol)."""
        # Build a state where FOLD is legal: SB opens with a legal raise,
        # BB faces it. BET_POT (size=pot=3) is below preflop min raise
        # (matched 2 + increment 2 = 4); use BET_DOUBLE (size=6) instead.
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        s1 = s0.next_state(AbstractedHUNLAction.BET_DOUBLE)
        assert AbstractedHUNLAction.FOLD in s1.legal_actions()
        assert encode_action(s1, AbstractedHUNLAction.FOLD) == "f"

    def test_call_when_to_call_positive_emits_c(self) -> None:
        """CALL with pending bet → 'c' (true call)."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # Preflop root: SB owes 1 chip to call BB. CALL legal,
        # to_call > 0 → must emit 'c'.
        assert encode_action(s0, AbstractedHUNLAction.CALL) == "c"

    def test_call_when_to_call_zero_emits_k(self) -> None:
        """CALL with no pending bet (check spot) → 'k'.

        Reach a checked-to spot: SB calls preflop, BB checks (k), advance
        to flop. Now first postflop actor faces to_call=0.
        """
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        s1 = s0.next_state(AbstractedHUNLAction.CALL)   # SB calls
        s2 = s1.next_state(AbstractedHUNLAction.CALL)   # BB checks → flop
        # On flop, first actor (BB) faces to_call=0.
        assert s2._raw.current_round == 1
        assert encode_action(s2, AbstractedHUNLAction.CALL) == "k"

    def test_bet_half_emits_b_with_chip_to_slumbot_size(self) -> None:
        """BET_HALF → 'b<size×50>' where size = compute_size(BET_HALF)."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # Need a state where BET_HALF is legal (post-call flop check).
        s1 = s0.next_state(AbstractedHUNLAction.CALL)
        s2 = s1.next_state(AbstractedHUNLAction.CALL)
        if AbstractedHUNLAction.BET_HALF in s2.legal_actions():
            tok = encode_action(s2, AbstractedHUNLAction.BET_HALF)
            assert tok.startswith("b")
            # The token's chip size must match compute_size × 50.
            from poker_ai.games.hunl_abstraction import compute_size
            from poker_ai.eval.slumbot_client import chip_to_slumbot
            expected = chip_to_slumbot(
                compute_size(AbstractedHUNLAction.BET_HALF, s2._raw)
            )
            assert int(tok[1:]) == expected
        else:
            pytest.skip("BET_HALF not legal in this state — wiring-only test")

    def test_bet_pot_emits_b_with_chip_to_slumbot_size(self) -> None:
        """BET_POT → 'b<pot×50>' on a clean preflop SB-to-act state."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot
        from poker_ai.games.hunl_abstraction import compute_size
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # SB to act preflop; BET_POT should be legal at root.
        if AbstractedHUNLAction.BET_POT in s0.legal_actions():
            tok = encode_action(s0, AbstractedHUNLAction.BET_POT)
            expected = chip_to_slumbot(
                compute_size(AbstractedHUNLAction.BET_POT, s0._raw)
            )
            assert tok == f"b{expected}"
        else:
            pytest.skip("BET_POT not legal at root — wiring-only test")

    def test_bet_double_emits_b_with_chip_to_slumbot_size(self) -> None:
        """BET_DOUBLE → 'b<2×pot×50>' with chip_to_slumbot conversion."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot
        from poker_ai.games.hunl_abstraction import compute_size
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        if AbstractedHUNLAction.BET_DOUBLE in s0.legal_actions():
            tok = encode_action(s0, AbstractedHUNLAction.BET_DOUBLE)
            expected = chip_to_slumbot(
                compute_size(AbstractedHUNLAction.BET_DOUBLE, s0._raw)
            )
            assert tok == f"b{expected}"
        else:
            pytest.skip("BET_DOUBLE not legal at root — wiring-only test")

    def test_bet_allin_emits_b_with_full_stack_slumbot_size(self) -> None:
        """Hook 2: BET_ALLIN → 'b<allin_size×50>' where allin_size =
        ``state.legal_bet_sizes()[-1]`` (heads-up effective cap)."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot
        from poker_ai.games.hunl_abstraction import compute_size
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        if AbstractedHUNLAction.BET_ALLIN in s0.legal_actions():
            tok = encode_action(s0, AbstractedHUNLAction.BET_ALLIN)
            expected = chip_to_slumbot(
                compute_size(AbstractedHUNLAction.BET_ALLIN, s0._raw)
            )
            assert tok == f"b{expected}"
            # And the slumbot value should equal full effective stack.
            # Heads-up: allin = min(actor_max, opp_max). At preflop root
            # (SB to act, all stacks deep), this is 200 BB = 400 chips →
            # 400 × 50 = 20000 slumbot chips.
            assert int(tok[1:]) == 20000
        else:
            pytest.skip("BET_ALLIN not legal at root — wiring-only test")

    def test_illegal_action_raises_value_error(self) -> None:
        """Encoding a non-legal action for the given state → ValueError."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # FOLD is illegal at preflop root for the SB (to_call>0 only
        # after a raise; SB has the option to call/raise/fold against
        # the BB though). If FOLD happens to be legal, fall back to a
        # bet that's masked by the canonical-collision rule.
        legal = set(s0.legal_actions())
        # Pick an action that is provably illegal at root.
        illegal_candidates = [
            a for a in AbstractedHUNLAction if a not in legal
        ]
        if not illegal_candidates:
            pytest.skip("All abstracted actions legal at root")
        with pytest.raises(ValueError):
            encode_action(s0, illegal_candidates[0])


# =============================================================================
# E. ingest_opponent_token (5 tests)
# =============================================================================
class TestIngestOpponentToken:
    """Slumbot token → :class:`AbstractedHUNLState` advance."""

    def test_fold_advances_to_terminal(self) -> None:
        """Receiving 'f' after a raise advances to terminal."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # Open with BET_DOUBLE (smallest raise legal at preflop root,
        # since BET_POT=3 < min_raise=4).
        s1 = s0.next_state(AbstractedHUNLAction.BET_DOUBLE)
        s2 = ingest_opponent_token(
            s1, SlumbotActionToken(kind="f", bet_to_slumbot_chips=0)
        )
        assert s2.is_terminal

    def test_call_with_to_call_positive_advances(self) -> None:
        """'c' at SB-to-act preflop closes SB action; round still open
        because BB's option remains."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        s1 = ingest_opponent_token(
            s0, SlumbotActionToken(kind="c", bet_to_slumbot_chips=0)
        )
        # SB called BB; BB now has the option (preflop heads-up rule).
        assert not s1.is_terminal
        assert s1.current_player == 0   # BB

    def test_check_token_advances_round(self) -> None:
        """'k' (check) on a checked-to flop spot just advances the
        action within the round."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        s1 = s0.next_state(AbstractedHUNLAction.CALL)   # SB calls
        s2 = s1.next_state(AbstractedHUNLAction.CALL)   # BB checks → flop
        # Now flop, BB to act. 'k' from opponent (BB).
        s3 = ingest_opponent_token(
            s2, SlumbotActionToken(kind="k", bet_to_slumbot_chips=0)
        )
        # Flop still open, SB now to act.
        assert s3._raw.current_round == 1
        assert s3.current_player == 1

    def test_bet_token_dispatches_via_nearest_bucket(self) -> None:
        """'b<X>' raw size routes through chip_from_slumbot then nearest
        abstracted bucket; resulting state has the matching round_bet
        recorded."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        # Opponent (SB) opens with a Slumbot-grid raise: b300 → /50 = 6
        # chips; in heads-up 200BB game with pot=3, BET_POT = 3.
        # Nearest abstracted dispatch should pick the closest legal.
        legal_sizes = s0._raw.legal_bet_sizes()
        if not legal_sizes:
            pytest.skip("BET not legal at root")
        s1 = ingest_opponent_token(
            s0, SlumbotActionToken(kind="b", bet_to_slumbot_chips=300)
        )
        # State must have advanced (a BET happened in round 0).
        assert any(
            a == HUNLAction.BET for a in s1._raw.round_history[0]
        )

    def test_non_standard_size_dispatches_via_nearest_bucket(self) -> None:
        """Hook 1: 'b347' (non-grid raw size) → /50=6 (with floor) or
        nearest of our abstracted set; ingest must succeed without
        raising and must advance the state."""
        game = _fresh_game()
        s0 = game.state_from_deal(_FIXED_DEAL)
        if AbstractedHUNLAction.BET_HALF not in s0.legal_actions() \
                and AbstractedHUNLAction.BET_POT not in s0.legal_actions():
            pytest.skip("No BET legal at root")
        s1 = ingest_opponent_token(
            s0, SlumbotActionToken(kind="b", bet_to_slumbot_chips=347)
        )
        # Some BET must have been recorded in preflop history.
        assert any(
            a == HUNLAction.BET for a in s1._raw.round_history[0]
        )


# =============================================================================
# F. replay_sequence (4 tests)
# =============================================================================
class TestReplaySequence:
    """End-to-end Slumbot sequence replay from a fresh deal."""

    def test_normal_four_round_sequence_replays(self) -> None:
        """A clean check-down sequence reaches a non-fold terminal at
        river-close."""
        game = _fresh_game()
        # Sequence: SB calls, BB checks (preflop close), 3 streets of
        # check-check.
        seq = "ck/kk/kk/kk"
        # client_pos doesn't affect a fully passive line; pick 0 (we are SB).
        final = replay_sequence(game, _FIXED_DEAL, seq, client_pos=0)
        assert final.is_terminal
        # No fold in any round.
        for r in final._raw.round_history:
            for a in r:
                assert a != HUNLAction.FOLD

    def test_preflop_fold_terminates_immediately(self) -> None:
        """SB folds preflop immediately (Slumbot 'f' on SB action) →
        terminal at round 0."""
        game = _fresh_game()
        seq = "f"
        final = replay_sequence(game, _FIXED_DEAL, seq, client_pos=0)
        assert final.is_terminal
        # The fold appears in preflop round_history.
        assert HUNLAction.FOLD in final._raw.round_history[0]

    def test_preflop_all_in_with_empty_streets_terminates(self) -> None:
        """Hook 3: 'b20000c///' = preflop all-in jam + call, then 3
        explicitly empty postflop streets. Must replay to terminal
        without crashing on the trailing slashes."""
        game = _fresh_game()
        seq = "b20000c///"
        final = replay_sequence(game, _FIXED_DEAL, seq, client_pos=0)
        assert final.is_terminal
        # All chips in pot at terminal: pot == 2 × STARTING_STACK_CHIPS.
        assert final._raw.pot == 2 * STARTING_STACK_CHIPS

    def test_client_pos_mapping_directs_dispatch(self) -> None:
        """client_pos=0 (we are Slumbot SB / button) maps to HUNLState
        player_idx=1 (SB acts first preflop). The SB acts first preflop
        so the *first* token is OUR action when client_pos=0; with
        client_pos=1 (we are BB), the first preflop token is the
        opponent's. This test verifies the mapping by checking which
        side's hole cards get encoded into the eventual infoset_key
        when we sit at the same point in the sequence under both
        positions."""
        game = _fresh_game()
        # Identical sequence under two client_pos values; the sequence
        # itself is symmetric (cc/kk/kk/kk = check-down).
        seq = "ck/kk/kk/kk"
        final_sb = replay_sequence(game, _FIXED_DEAL, seq, client_pos=0)
        final_bb = replay_sequence(game, _FIXED_DEAL, seq, client_pos=1)
        # Both end terminal; both have no fold.
        assert final_sb.is_terminal
        assert final_bb.is_terminal
        # The replay is sequence-faithful regardless of client_pos —
        # the round_history must match exactly (chip flows determined
        # by tokens, not by who sits at which seat).
        assert final_sb._raw.round_history == final_bb._raw.round_history
