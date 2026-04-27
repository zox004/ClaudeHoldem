"""FAILING tests for Phase 4 M4.3 — Slumbot benchmark harness.

These tests exercise the orchestration layer that wraps the M4.1
:class:`SlumbotClient` transport and the M4.2 action-protocol adapter
into a single end-to-end driver:

- :class:`HandRecord` / :class:`SessionRecord` — frozen+slotted record
  dataclasses for one hand and an aggregated session.
- :func:`mbb_per_hand_winrate` — mean ± SE win-rate (mbb/hand) with
  naive-IID estimator and an explicit Phase 5 / AIVAT hook docstring.
- :class:`SlumbotHarness` — ``play_one_hand`` + ``play_session`` plus
  a configurable exponential-backoff retry policy for 5xx / 429.

Scope (M4.3):
    The tests deliberately mock :class:`SlumbotClient` (no real HTTP).
    Live-network calls are M4.4 pilot territory.

Audit hook coverage (M4.3 spec):
    - Hook 1 — Position alternation: Slumbot rotates ``client_pos`` on
      each new hand; over a 100-hand mock session the harness records
      a roughly 50/50 split.
    - Hook 2 — Winnings sign convention: Slumbot's ``winnings`` is
      "client perspective" (positive = we win) while
      :meth:`HUNLState.terminal_utility` is in P0 perspective. The
      harness must apply the ``client_pos ↔ player_idx`` flip.
        * Fold-only termination: full magnitude check
          (``chip_to_slumbot(our_utility) == slumbot_winnings``).
        * Showdown: opp_hole is unobservable so magnitude check is
          skipped — sign-only sync_check.
    - Hook 3 — Variance estimator declaration: the
      :func:`mbb_per_hand_winrate` docstring explicitly notes that the
      naive-IID SE estimator is a placeholder for the AIVAT (Burch
      2018) reduction registered as a Phase 5 hook.

Module under test: ``poker_ai.eval.slumbot_harness`` (does not yet
exist — every test is expected to RED with ``ImportError`` until M4.3
implementation lands).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests   # type: ignore[import-untyped]

from poker_ai.eval.slumbot_client import (
    SlumbotError,
    SlumbotResponse,
    chip_to_slumbot,
)
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLAction,
    AbstractedHUNLGame,
)
from poker_ai.games.hunl_state import BB_BLIND_CHIPS_VALUE

# Module under test — these imports are intentionally going to fail
# until ``slumbot_harness.py`` is implemented.
from poker_ai.eval.slumbot_harness import (  # noqa: E402
    HandRecord,
    SessionRecord,
    SlumbotHarness,
    StrategyFn,
    mbb_per_hand_winrate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fresh_game(seed: int = 42) -> AbstractedHUNLGame:
    """Test-scaled AbstractedHUNLGame (fast fixture build)."""
    return AbstractedHUNLGame(
        n_buckets=10,
        n_trials=200,
        seed=seed,
        postflop_mc_trials=50,
        postflop_threshold_sample_size=80,
    )


# Deterministic deal layout: (p0_h1, p0_h2, p1_h1, p1_h2, b1..b5).
_FIXED_DEAL: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def _resp(
    *,
    token: str = "tok-1",
    action: str = "",
    client_pos: int = 0,
    hole_cards: list[str] | None = None,
    board: list[str] | None = None,
    winnings: int | None = None,
) -> SlumbotResponse:
    """Builds a :class:`SlumbotResponse` with sensible defaults."""
    return SlumbotResponse(
        token=token,
        action=action,
        client_pos=client_pos,
        hole_cards=hole_cards if hole_cards is not None else ["As", "Kd"],
        board=board if board is not None else [],
        winnings=winnings,
        baseline_winnings=None,
        session_num_hands=None,
        session_total=None,
        session_baseline_total=None,
    )


def _uniform_strategy_fn() -> StrategyFn:
    """Strategy that always returns a uniform 6-vector (the harness
    masks illegal actions before sampling)."""
    def fn(_state: Any) -> np.ndarray:
        return np.full(6, 1.0 / 6.0, dtype=np.float64)
    return fn


def _always_fold_strategy_fn() -> StrategyFn:
    """Strategy that puts all mass on FOLD when legal, else CALL."""
    def fn(_state: Any) -> np.ndarray:
        dist = np.zeros(6, dtype=np.float64)
        dist[int(AbstractedHUNLAction.FOLD)] = 1.0
        return dist
    return fn


def _make_mock_client(
    new_hand_responses: list[SlumbotResponse | Exception] | None = None,
    act_responses: list[SlumbotResponse | Exception] | None = None,
) -> MagicMock:
    """Builds a MagicMock that emulates :class:`SlumbotClient`.

    Each call to ``new_hand`` / ``act`` consumes the next entry from
    the corresponding list. Entries that are :class:`Exception`
    instances are raised; otherwise returned.
    """
    client = MagicMock()
    client._new_hand_iter = iter(new_hand_responses or [])
    client._act_iter = iter(act_responses or [])

    def _new_hand() -> SlumbotResponse:
        item = next(client._new_hand_iter)
        if isinstance(item, Exception):
            raise item
        return item

    def _act(_incr: str) -> SlumbotResponse:
        item = next(client._act_iter)
        if isinstance(item, Exception):
            raise item
        return item

    client.new_hand.side_effect = _new_hand
    client.act.side_effect = _act
    return client


# =============================================================================
# A. HandRecord / SessionRecord dataclass (3 tests)
# =============================================================================
class TestHandSessionDataclass:
    """Frozen + slotted record types for hand- and session-level data."""

    def test_hand_record_frozen_and_six_fields(self) -> None:
        """HandRecord is frozen, slotted, and exposes the documented six fields."""
        rec = HandRecord(
            deal=_FIXED_DEAL,
            sequence="cb500c/kk/kk/kk",
            client_pos=1,
            our_utility_chips=4,
            slumbot_winnings=200,
            sync_check=True,
        )
        assert rec.deal == _FIXED_DEAL
        assert rec.sequence == "cb500c/kk/kk/kk"
        assert rec.client_pos == 1
        assert rec.our_utility_chips == 4
        assert rec.slumbot_winnings == 200
        assert rec.sync_check is True
        with pytest.raises((AttributeError, Exception)):
            rec.client_pos = 1   # type: ignore[misc]
        assert not hasattr(rec, "__dict__")

    def test_session_record_frozen_and_holds_hand_list(self) -> None:
        """SessionRecord exposes ``hands: list[HandRecord]`` and is slotted."""
        h = HandRecord(
            deal=_FIXED_DEAL, sequence="f", client_pos=1,
            our_utility_chips=-1, slumbot_winnings=-50, sync_check=True,
        )
        sess = SessionRecord(hands=[h])
        assert sess.hands == [h]
        assert not hasattr(sess, "__dict__")

    def test_session_record_default_empty_list(self) -> None:
        """SessionRecord() with no args yields an empty hand list."""
        sess = SessionRecord()
        assert sess.hands == []


# =============================================================================
# B. mbb_per_hand_winrate (4 tests)
# =============================================================================
class TestMbbPerHandWinrate:
    """Win-rate aggregation in mbb/hand from client perspective."""

    def test_single_hand_mean_uses_bb_chip_value(self) -> None:
        """One hand at our_utility=2 chips (= 1 BB) → 1000 mbb/hand mean."""
        rec = HandRecord(
            deal=_FIXED_DEAL, sequence="f", client_pos=1,
            our_utility_chips=BB_BLIND_CHIPS_VALUE,
            slumbot_winnings=100, sync_check=True,
        )
        mean, _se = mbb_per_hand_winrate([rec])
        assert mean == pytest.approx(1000.0)

    def test_n_hands_mean_is_average(self) -> None:
        """Mean over n hands == average of per-hand mbb values."""
        recs = [
            HandRecord(
                deal=_FIXED_DEAL, sequence="f", client_pos=1,
                our_utility_chips=2, slumbot_winnings=100, sync_check=True,
            ),
            HandRecord(
                deal=_FIXED_DEAL, sequence="f", client_pos=1,
                our_utility_chips=-2, slumbot_winnings=-100, sync_check=True,
            ),
        ]
        mean, _ = mbb_per_hand_winrate(recs)
        assert mean == pytest.approx(0.0)

    def test_se_naive_iid_formula(self) -> None:
        """SE = std / sqrt(n) under naive-IID assumption."""
        recs = [
            HandRecord(
                deal=_FIXED_DEAL, sequence="f", client_pos=1,
                our_utility_chips=k, slumbot_winnings=k * 50,
                sync_check=True,
            )
            for k in (-2, 0, 2, 4)
        ]
        mean, se = mbb_per_hand_winrate(recs)
        # Per-hand mbb values: {-1000, 0, 1000, 2000}.
        per_hand = np.array([-1000.0, 0.0, 1000.0, 2000.0])
        assert mean == pytest.approx(per_hand.mean())
        # Match either ddof=0 or ddof=1; both are defensible — test
        # checks the value is in that band.
        candidates = (
            per_hand.std(ddof=0) / np.sqrt(len(per_hand)),
            per_hand.std(ddof=1) / np.sqrt(len(per_hand)),
        )
        assert any(se == pytest.approx(c, rel=1e-6) for c in candidates)

    def test_empty_records_raises_value_error(self) -> None:
        """Empty input → ValueError (no degenerate-zero return)."""
        with pytest.raises(ValueError):
            mbb_per_hand_winrate([])


# =============================================================================
# C. Retry policy (5 tests)
# =============================================================================
class TestRetryPolicy:
    """Configurable exponential-backoff retry for 5xx / 429."""

    def test_5xx_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """500 once → backoff sleep → retry succeeds → returns response."""
        monkeypatch.setattr(
            "poker_ai.eval.slumbot_harness.time.sleep", lambda _s: None
        )
        good = _resp(winnings=100, client_pos=1)
        client = _make_mock_client(
            new_hand_responses=[requests.HTTPError("500"), good]
        )
        h = SlumbotHarness(client, max_retries=3, backoff_base=2.0)
        out = h._retry_post(client.new_hand)
        assert out is good
        assert client.new_hand.call_count == 2

    def test_4xx_no_retry_immediate_raise(self) -> None:
        """4xx (e.g. 400 bad request) is fast-fail — no retry."""
        err = requests.HTTPError("400 Bad Request")
        # Mark the error with a synthetic .response.status_code = 400 so
        # the harness can distinguish 4xx from 5xx.
        err.response = MagicMock()
        err.response.status_code = 400
        client = _make_mock_client(new_hand_responses=[err])
        h = SlumbotHarness(client, max_retries=3)
        with pytest.raises(requests.HTTPError):
            h._retry_post(client.new_hand)
        assert client.new_hand.call_count == 1

    def test_429_triggers_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 Too Many Requests is treated like 5xx (retried)."""
        monkeypatch.setattr(
            "poker_ai.eval.slumbot_harness.time.sleep", lambda _s: None
        )
        err = requests.HTTPError("429")
        err.response = MagicMock()
        err.response.status_code = 429
        good = _resp(winnings=0, client_pos=1)
        client = _make_mock_client(new_hand_responses=[err, good])
        h = SlumbotHarness(client, max_retries=3)
        out = h._retry_post(client.new_hand)
        assert out is good
        assert client.new_hand.call_count == 2

    def test_max_retries_exceeded_raises_last_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After ``max_retries`` 5xx responses, the final error propagates."""
        monkeypatch.setattr(
            "poker_ai.eval.slumbot_harness.time.sleep", lambda _s: None
        )
        errors: list[SlumbotResponse | Exception] = [
            requests.HTTPError("500") for _ in range(4)
        ]
        client = _make_mock_client(new_hand_responses=errors)
        h = SlumbotHarness(client, max_retries=2)
        with pytest.raises(requests.HTTPError):
            h._retry_post(client.new_hand)
        # 1 initial attempt + 2 retries = 3 calls total.
        assert client.new_hand.call_count == 3

    def test_max_retries_one_caps_at_two_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configurable: max_retries=1 means one initial + one retry."""
        monkeypatch.setattr(
            "poker_ai.eval.slumbot_harness.time.sleep", lambda _s: None
        )
        client = _make_mock_client(
            new_hand_responses=[
                requests.HTTPError("500"),
                requests.HTTPError("500"),
            ]
        )
        h = SlumbotHarness(client, max_retries=1)
        with pytest.raises(requests.HTTPError):
            h._retry_post(client.new_hand)
        assert client.new_hand.call_count == 2


# =============================================================================
# D. play_one_hand orchestration (5 tests)
# =============================================================================
class TestPlayOneHand:
    """Single-hand orchestration loop: new_hand + act loop + termination."""

    def test_immediate_winnings_preflop_fold(self) -> None:
        """Slumbot folds preflop on first response → HandRecord, sync_check True."""
        # We are BB (client_pos=0); Slumbot (SB) opens with fold.
        # Action sequence "f" terminates immediately.
        winnings_slumbot = 50  # 1 SB stolen, in Slumbot chips
        first = _resp(
            action="f",
            client_pos=0,
            hole_cards=["As", "Ad"],
            winnings=winnings_slumbot,
        )
        client = _make_mock_client(new_hand_responses=[first])
        h = SlumbotHarness(client)
        game = _fresh_game()
        rng = np.random.default_rng(0)
        rec = h.play_one_hand(game, _uniform_strategy_fn(), rng)
        assert isinstance(rec, HandRecord)
        assert rec.sequence == "f"
        assert rec.client_pos == 0
        # client_pos=1 means we are BB; opponent folded → we collect blinds.
        assert rec.slumbot_winnings == winnings_slumbot
        assert rec.sync_check is True

    def test_mid_hand_loop_calls_act(self) -> None:
        """Mid-hand response triggers strategy sample → client.act()."""
        # We are BB (client_pos=0). Slumbot SB raised "b300" preflop.
        # Our turn (BB facing a raise) — FOLD is legal.
        first = _resp(
            action="b300",
            client_pos=0,
            hole_cards=["2s", "3d"],
            winnings=None,
        )
        # We fold. Hand terminates with us losing the BB blind.
        second = _resp(
            action="b300f",
            client_pos=0,
            hole_cards=["2s", "3d"],
            winnings=-100,
        )
        client = _make_mock_client(
            new_hand_responses=[first],
            act_responses=[second],
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(
            game, _always_fold_strategy_fn(), np.random.default_rng(0)
        )
        # We are BB, faced a raise, folded. Mock shows one act call.
        assert client.act.call_count >= 1
        assert isinstance(rec, HandRecord)

    def test_showdown_termination_uses_slumbot_winnings_as_truth(self) -> None:
        """Showdown end: opp_hole unknown → utility from Slumbot, sync sign-only."""
        # We are SB (client_pos=1). Both check down 4 streets (no money in
        # past blinds), Slumbot wins by showdown — winnings = -50 (we lose
        # 1 SB blind to Slumbot's BB).
        first = _resp(
            action="cc/kk/kk/kk",
            client_pos=1,
            hole_cards=["2s", "3d"],
            board=["7h", "Tc", "Js", "Qs", "Kc"],
            winnings=-50,
        )
        client = _make_mock_client(
            new_hand_responses=[first], act_responses=[]
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(
            game, _uniform_strategy_fn(), np.random.default_rng(123)
        )
        # Ground truth: utility derived from slumbot_winnings via
        # chip_from_slumbot, *not* from our own terminal_utility (opp
        # hole unknown). Sign matches.
        assert rec.slumbot_winnings == -50
        assert rec.our_utility_chips < 0
        # sync_check is sign-only on showdown — must be True (signs agree).
        assert rec.sync_check is True

    def test_desync_terminal_state_but_no_winnings_raises(self) -> None:
        """Hook 2 desync: terminal sequence but winnings=None → SlumbotError."""
        bad = _resp(
            action="f",
            client_pos=1,
            hole_cards=["As", "Ad"],
            winnings=None,   # <-- desync: should be set on terminal
        )
        # After we act, Slumbot may continue returning winnings=None;
        # the harness must detect the terminal-state-without-winnings
        # condition.
        client = _make_mock_client(
            new_hand_responses=[bad], act_responses=[]
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        with pytest.raises(SlumbotError):
            h.play_one_hand(
                game, _uniform_strategy_fn(), np.random.default_rng(0)
            )

    def test_multi_step_hand_three_exchanges(self) -> None:
        """Multi-step hand (≥3 client.act calls) terminates with HandRecord."""
        # We are BB (client_pos=0). Slumbot opens with limp ("c"); we
        # check; flop runs; we check; river all check; showdown.
        first = _resp(
            action="c",
            client_pos=0,
            hole_cards=["As", "Ad"],
            board=[],
            winnings=None,
        )
        # After our preflop check ("k"/CALL), flop dealt, our turn again.
        second = _resp(
            action="cc/",
            client_pos=0,
            hole_cards=["As", "Ad"],
            board=["7h", "Tc", "Js"],
            winnings=None,
        )
        # After flop check, turn dealt, our turn again.
        third = _resp(
            action="cc/kk/",
            client_pos=0,
            hole_cards=["As", "Ad"],
            board=["7h", "Tc", "Js", "Qs"],
            winnings=None,
        )
        # After turn check, river dealt, our turn again.
        fourth = _resp(
            action="cc/kk/kk/",
            client_pos=0,
            hole_cards=["As", "Ad"],
            board=["7h", "Tc", "Js", "Qs", "Kc"],
            winnings=None,
        )
        # After river check, showdown — we win as P0 (BB).
        fifth = _resp(
            action="cc/kk/kk/kk",
            client_pos=0,
            hole_cards=["As", "Ad"],
            board=["7h", "Tc", "Js", "Qs", "Kc"],
            winnings=100,
        )
        # Strategy that always picks CALL when legal (collapses to check
        # in matched-contributions states).
        def call_strategy(_s: Any) -> np.ndarray:
            d = np.zeros(6, dtype=np.float64)
            d[int(AbstractedHUNLAction.CALL)] = 1.0
            return d

        client = _make_mock_client(
            new_hand_responses=[first],
            act_responses=[second, third, fourth, fifth],
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(game, call_strategy, np.random.default_rng(0))
        assert isinstance(rec, HandRecord)
        assert client.act.call_count >= 3
        assert rec.slumbot_winnings == 100


# =============================================================================
# E. play_session aggregation (5 tests)
# =============================================================================
class TestPlaySession:
    """Multi-hand session orchestration."""

    def test_n_hands_yields_n_hand_records(self) -> None:
        """play_session(n=10) returns SessionRecord with 10 HandRecords."""
        # Each hand is an immediate preflop fold (Slumbot folds → we win blinds).
        responses = [
            _resp(
                action="f",
                client_pos=i % 2,
                hole_cards=["As", "Ad"],
                winnings=50 if (i % 2) == 1 else 100,
            )
            for i in range(10)
        ]
        client = _make_mock_client(new_hand_responses=responses)
        h = SlumbotHarness(client)
        game = _fresh_game()
        sess = h.play_session(
            game, _uniform_strategy_fn(),
            n_hands=10, rng=np.random.default_rng(0),
        )
        assert isinstance(sess, SessionRecord)
        assert len(sess.hands) == 10

    def test_position_alternation_hook_1(self) -> None:
        """Hook 1: 100 hands with alternating client_pos → ~50/50 split."""
        responses = [
            _resp(
                action="f",
                client_pos=i % 2,   # alternates 0,1,0,1,...
                hole_cards=["As", "Ad"],
                winnings=50 if (i % 2) == 1 else 100,
            )
            for i in range(100)
        ]
        client = _make_mock_client(new_hand_responses=responses)
        h = SlumbotHarness(client)
        game = _fresh_game()
        sess = h.play_session(
            game, _uniform_strategy_fn(),
            n_hands=100, rng=np.random.default_rng(0),
        )
        n_pos0 = sum(1 for r in sess.hands if r.client_pos == 1)
        n_pos1 = sum(1 for r in sess.hands if r.client_pos == 0)
        assert n_pos0 + n_pos1 == 100
        assert abs(n_pos0 - 50) <= 10, f"client_pos=0 count {n_pos0} too far from 50"
        assert abs(n_pos1 - 50) <= 10, f"client_pos=1 count {n_pos1} too far from 50"

    def test_session_winrate_computation_end_to_end(self) -> None:
        """SessionRecord → mbb_per_hand_winrate end-to-end pipeline."""
        responses = [
            _resp(
                action="f", client_pos=1, hole_cards=["As", "Ad"],
                winnings=100,
            )
            for _ in range(5)
        ]
        client = _make_mock_client(new_hand_responses=responses)
        h = SlumbotHarness(client)
        game = _fresh_game()
        sess = h.play_session(
            game, _uniform_strategy_fn(),
            n_hands=5, rng=np.random.default_rng(0),
        )
        mean, _se = mbb_per_hand_winrate(sess.hands)
        # +100 Slumbot chips per hand = +2 our chips = +1000 mbb.
        assert mean == pytest.approx(1000.0)

    def test_n_hands_zero_raises_or_returns_empty(self) -> None:
        """n_hands=0 → ValueError or empty SessionRecord; both acceptable."""
        client = _make_mock_client(new_hand_responses=[])
        h = SlumbotHarness(client)
        game = _fresh_game()
        try:
            sess = h.play_session(
                game, _uniform_strategy_fn(),
                n_hands=0, rng=np.random.default_rng(0),
            )
            assert sess.hands == []
        except ValueError:
            pass   # also acceptable

    def test_mid_session_unrecoverable_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistent network failure mid-session propagates to caller."""
        monkeypatch.setattr(
            "poker_ai.eval.slumbot_harness.time.sleep", lambda _s: None
        )
        good = _resp(action="f", client_pos=1, winnings=100)
        # First two hands succeed, third hand fails permanently.
        responses: list[SlumbotResponse | Exception] = [
            good, good,
            requests.HTTPError("500"),
            requests.HTTPError("500"),
            requests.HTTPError("500"),
            requests.HTTPError("500"),
        ]
        client = _make_mock_client(new_hand_responses=responses)
        h = SlumbotHarness(client, max_retries=1)
        game = _fresh_game()
        with pytest.raises(requests.HTTPError):
            h.play_session(
                game, _uniform_strategy_fn(),
                n_hands=5, rng=np.random.default_rng(0),
            )


# =============================================================================
# F. strategy_fn interface (3 tests)
# =============================================================================
class TestStrategyFnInterface:
    """Contract for ``StrategyFn = Callable[[state], np.ndarray]``."""

    def test_distribution_sums_to_one_legal_action_sampled(self) -> None:
        """Uniform distribution sums to 1; harness samples a legal action."""
        first = _resp(
            action="", client_pos=1, hole_cards=["As", "Ad"],
            winnings=None,
        )
        second = _resp(
            action="f", client_pos=1, hole_cards=["As", "Ad"],
            winnings=-100,
        )
        client = _make_mock_client(
            new_hand_responses=[first], act_responses=[second]
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        # Should not raise on illegal action — masking is harness's job.
        rec = h.play_one_hand(
            game, _uniform_strategy_fn(), np.random.default_rng(0)
        )
        assert isinstance(rec, HandRecord)

    def test_illegal_action_mass_clipped_by_legal_mask(self) -> None:
        """Strategy puts all mass on FOLD; if FOLD is illegal at preflop
        SB-to-act (no bet to call yet), harness must renormalise over
        legal actions instead of crashing."""
        first = _resp(
            action="", client_pos=1, hole_cards=["As", "Ad"],
            winnings=None,
        )
        second = _resp(
            action="cc/kk/kk/kk", client_pos=1,
            hole_cards=["As", "Ad"],
            board=["7h", "Tc", "Js", "Qs", "Kc"],
            winnings=-100,
        )
        client = _make_mock_client(
            new_hand_responses=[first], act_responses=[second]
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        # Strategy that places all mass on FOLD (illegal at preflop SB).
        # Harness must handle gracefully (uniform over legal mask
        # fallback) and return a HandRecord.
        rec = h.play_one_hand(
            game, _always_fold_strategy_fn(), np.random.default_rng(0)
        )
        assert isinstance(rec, HandRecord)

    def test_callable_signature_matches_strategy_fn_alias(self) -> None:
        """Custom strategy_fn callable type-checks under StrategyFn alias."""
        # StrategyFn must be a Callable type alias.
        def my_strategy(_state: Any) -> np.ndarray:
            return np.full(6, 1.0 / 6.0, dtype=np.float64)

        # Must be assignable to a StrategyFn-typed variable.
        fn: StrategyFn = my_strategy
        out = fn(None)
        assert out.shape == (6,)
        assert out.sum() == pytest.approx(1.0)


# =============================================================================
# G. Audit-hook coverage (5 tests)
# =============================================================================
class TestAuditHookCoverage:
    """Spec-locked Hook 1 / Hook 2 / Hook 3 declarations."""

    def test_hook1_position_alternation_distribution(self) -> None:
        """Hook 1: 100-hand mock with alternating client_pos → ~50/50."""
        responses = [
            _resp(
                action="f", client_pos=i % 2,
                hole_cards=["As", "Ad"],
                winnings=50 if (i % 2) == 1 else 100,
            )
            for i in range(100)
        ]
        client = _make_mock_client(new_hand_responses=responses)
        h = SlumbotHarness(client)
        game = _fresh_game()
        sess = h.play_session(
            game, _uniform_strategy_fn(),
            n_hands=100, rng=np.random.default_rng(0),
        )
        n0 = sum(1 for r in sess.hands if r.client_pos == 1)
        # Strict ±10 of 50/50 (perfect alternation in mock should give 50).
        assert 40 <= n0 <= 60

    def test_hook2_winnings_sign_mapping_per_position(self) -> None:
        """Hook 2: utility sign matches Slumbot winnings sign on fold-only."""
        # client_pos=0 (we are SB), Slumbot folds preflop → we win.
        resp_pos0 = _resp(
            action="f", client_pos=1,
            hole_cards=["As", "Ad"], winnings=100,
        )
        client = _make_mock_client(new_hand_responses=[resp_pos0])
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(
            game, _uniform_strategy_fn(), np.random.default_rng(0)
        )
        assert rec.slumbot_winnings > 0
        assert rec.our_utility_chips > 0

        # client_pos=1 (we are BB), Slumbot folds preflop → we win.
        resp_pos1 = _resp(
            action="f", client_pos=0,
            hole_cards=["As", "Ad"], winnings=50,
        )
        client = _make_mock_client(new_hand_responses=[resp_pos1])
        h = SlumbotHarness(client)
        rec = h.play_one_hand(
            game, _uniform_strategy_fn(), np.random.default_rng(0)
        )
        assert rec.slumbot_winnings > 0
        assert rec.our_utility_chips > 0

    def test_hook2_showdown_sign_only_check(self) -> None:
        """Hook 2: showdown termination → sync_check via sign agreement only.

        Magnitude check is skipped because the opponent's hole cards are
        unobservable (no showdown reveal in /api/act response), so our
        terminal_utility cannot be computed exactly.
        """
        # Showdown end: 4 checks per street, board fully revealed.
        # Slumbot reports negative winnings (we lose).
        showdown = _resp(
            action="cc/kk/kk/kk", client_pos=1,
            hole_cards=["2s", "3d"],
            board=["7h", "Tc", "Js", "Qs", "Kc"],
            winnings=-50,
        )
        client = _make_mock_client(new_hand_responses=[showdown])
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(
            game, _uniform_strategy_fn(), np.random.default_rng(0)
        )
        # Sign check: both negative.
        assert rec.slumbot_winnings < 0
        assert rec.our_utility_chips < 0
        # Sync passed under the looser sign-only rule.
        assert rec.sync_check is True

    def test_hook3_variance_estimator_declaration_in_docstring(self) -> None:
        """Hook 3: docstring mentions AIVAT or Phase-5 variance hook."""
        doc = mbb_per_hand_winrate.__doc__ or ""
        lower = doc.lower()
        assert ("aivat" in lower) or ("phase 5" in lower) or ("phase-5" in lower), (
            f"mbb_per_hand_winrate docstring must declare variance "
            f"reduction (AIVAT / Phase 5 hook); got: {doc[:200]!r}"
        )

    def test_winnings_sign_mismatch_flags_sync_check_false(self) -> None:
        """Hook 2: fold-only termination with magnitude mismatch (incl.
        sign mismatch) flags ``sync_check=False`` on the resulting
        HandRecord without raising.

        Hard desyncs (state terminal but winnings None, or winnings
        present but state not terminal) raise :class:`SlumbotError`
        upstream in play_one_hand — those are tested elsewhere
        (test_desync_terminal_state_but_no_winnings_raises). Per-record
        chip-count mismatches between state replay and Slumbot's
        reported winnings are informational only: mocks, partial-pot
        scenarios, and abstracted-action approximation can all produce
        such mismatches without breaking the session.
        """
        first = _resp(
            action="", client_pos=1, hole_cards=["As", "Ad"],
            winnings=None,
        )
        second = _resp(
            action="f", client_pos=1, hole_cards=["As", "Ad"],
            winnings=100,
        )
        client = _make_mock_client(
            new_hand_responses=[first], act_responses=[second]
        )
        h = SlumbotHarness(client)
        game = _fresh_game()
        rec = h.play_one_hand(
            game, _always_fold_strategy_fn(), np.random.default_rng(0)
        )
        assert isinstance(rec, HandRecord)
        # Magnitude check fails (state says we lost SB; Slumbot says +100).
        assert rec.sync_check is False
        # The Slumbot-reported winnings remain the ground truth.
        assert rec.slumbot_winnings == 100


# Tally: A=3, B=4, C=5, D=5, E=5, F=3, G=5 → 30 tests.
