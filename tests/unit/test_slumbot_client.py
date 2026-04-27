"""FAILING tests for Phase 4 M4.1 — SlumbotClient HTTP transport.

These tests exercise *only* the transport layer:

- ``SlumbotResponse`` parsed dataclass (frozen + slots, all 9 fields).
- ``chip_to_slumbot`` / ``chip_from_slumbot`` chip-granularity helpers
  (our BB=2 chips ↔ Slumbot BB=100 chips, ×50 multiplier).
- ``SlumbotClient`` HTTPS POST wrapper for ``/api/new_hand`` and
  ``/api/act``, with token tracking and error propagation.

The action-protocol semantics (abstracted action ↔ Slumbot action
string `b500` etc.) are explicitly out of scope for M4.1; that is M4.2.

All HTTP I/O is mocked via :mod:`unittest.mock`; no real network calls.

Module under test: ``poker_ai.eval.slumbot_client`` (does not yet
exist — every test is expected to RED with ``ImportError`` until
M4.1 implementation lands).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Mock HTTP response helper
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal stand-in for :class:`requests.Response` used in patches."""

    def __init__(self, json_data: dict[str, Any] | None, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data) if json_data is not None else ""

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise requests.JSONDecodeError("no json", "", 0)
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _full_new_hand_payload(
    token: str = "tok-abc",
    action: str = "",
    client_pos: int = 0,
) -> dict[str, Any]:
    """A representative /api/new_hand response body."""
    return {
        "token": token,
        "action": action,
        "client_pos": client_pos,
        "hole_cards": ["As", "Kd"],
        "board": [],
        "winnings": None,
        "baseline_winnings": None,
        "session_num_hands": 0,
        "session_total": 0,
        "session_baseline_total": 0,
    }


# ---------------------------------------------------------------------------
# A. SlumbotResponse dataclass (3 tests)
# ---------------------------------------------------------------------------


class TestSlumbotResponseDataclass:
    def test_is_frozen_and_uses_slots(self) -> None:
        """SlumbotResponse must be frozen (immutable) and slots=True."""
        from poker_ai.eval.slumbot_client import SlumbotResponse

        resp = SlumbotResponse(
            token="t",
            action="",
            client_pos=0,
            hole_cards=["As", "Kd"],
            board=[],
            winnings=None,
            baseline_winnings=None,
            session_num_hands=None,
            session_total=None,
            session_baseline_total=None,
        )
        with pytest.raises((AttributeError, Exception)):
            resp.token = "other"  # type: ignore[misc]
        # slots=True precludes __dict__
        assert not hasattr(resp, "__dict__")

    def test_preserves_all_nine_fields(self) -> None:
        """All 9 documented Slumbot fields must round-trip exactly."""
        from poker_ai.eval.slumbot_client import SlumbotResponse

        resp = SlumbotResponse(
            token="tok-1",
            action="cb300/kk/kk",
            client_pos=1,
            hole_cards=["As", "Kd"],
            board=["7h", "2c", "Js"],
            winnings=150,
            baseline_winnings=-50,
            session_num_hands=42,
            session_total=1234,
            session_baseline_total=-567,
        )
        assert resp.token == "tok-1"
        assert resp.action == "cb300/kk/kk"
        assert resp.client_pos == 1
        assert resp.hole_cards == ["As", "Kd"]
        assert resp.board == ["7h", "2c", "Js"]
        assert resp.winnings == 150
        assert resp.baseline_winnings == -50
        assert resp.session_num_hands == 42
        assert resp.session_total == 1234
        assert resp.session_baseline_total == -567

    def test_optional_fields_accept_none(self) -> None:
        """winnings / baseline_winnings / session_* may be None mid-hand."""
        from poker_ai.eval.slumbot_client import SlumbotResponse

        resp = SlumbotResponse(
            token="t",
            action="",
            client_pos=0,
            hole_cards=["As", "Kd"],
            board=[],
            winnings=None,
            baseline_winnings=None,
            session_num_hands=None,
            session_total=None,
            session_baseline_total=None,
        )
        assert resp.winnings is None
        assert resp.baseline_winnings is None
        assert resp.session_num_hands is None


# ---------------------------------------------------------------------------
# B. chip_to_slumbot / chip_from_slumbot (5 tests)
# ---------------------------------------------------------------------------


class TestChipConversion:
    def test_chip_to_slumbot_bb_equivalent(self) -> None:
        """Our 2 chips (= 1 BB) → Slumbot 100 chips (= 1 BB)."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot

        assert chip_to_slumbot(2) == 100

    def test_chip_to_slumbot_full_stack(self) -> None:
        """Our 400 chips (200 BB stack) → Slumbot 20000 chips."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot

        assert chip_to_slumbot(400) == 20000

    def test_chip_from_slumbot_round_trip(self) -> None:
        """Slumbot 100 chips → our 2 chips (1 BB round-trip)."""
        from poker_ai.eval.slumbot_client import chip_from_slumbot

        assert chip_from_slumbot(100) == 2

    def test_chip_from_slumbot_floor_divides_arbitrary_size(self) -> None:
        """Non-multiple-of-50 Slumbot amounts floor-divide (lossy by design).

        Slumbot's opponent may raise to e.g. 149 chips; we accept the
        loss at this helper level. M4.2 routes such opponent sizes
        through nearest-bucket abstraction translation rather than
        relying on this helper.
        """
        from poker_ai.eval.slumbot_client import chip_from_slumbot

        assert chip_from_slumbot(149) == 2

    def test_chip_to_slumbot_zero(self) -> None:
        """Zero chips maps to zero chips (no offset)."""
        from poker_ai.eval.slumbot_client import chip_to_slumbot

        assert chip_to_slumbot(0) == 0


# ---------------------------------------------------------------------------
# C. SlumbotClient init (3 tests)
# ---------------------------------------------------------------------------


class TestSlumbotClientInit:
    def test_default_host_is_slumbot_com(self) -> None:
        """No-arg construction targets slumbot.com."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        assert client.host == "slumbot.com"

    def test_custom_host_accepted(self) -> None:
        """host kwarg overrides default (for staging / test mirror)."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient(host="staging.slumbot.example")
        assert client.host == "staging.slumbot.example"

    def test_token_property_none_initially(self) -> None:
        """Pre-new_hand the client holds no token."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        assert client.token is None


# ---------------------------------------------------------------------------
# D. new_hand HTTP POST (5 tests)
# ---------------------------------------------------------------------------


class TestNewHand:
    def test_posts_to_slumbot_new_hand_https_url(self) -> None:
        """new_hand() POSTs to https://slumbot.com/api/new_hand."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload()),
        ) as mock_post:
            client.new_hand()
            args, _ = mock_post.call_args
            url = args[0] if args else mock_post.call_args.kwargs.get("url", "")
            assert url == "https://slumbot.com/api/new_hand"

    def test_anonymous_first_call_sends_null_or_empty_token(self) -> None:
        """First new_hand sends body with token=None (anonymous play)."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload()),
        ) as mock_post:
            client.new_hand()
            kwargs = mock_post.call_args.kwargs
            body = kwargs.get("json", {})
            # Either {"token": None} or {} accepted as anonymous-equivalent.
            assert body == {"token": None} or body == {}

    def test_response_parsed_into_dataclass(self) -> None:
        """Server payload is parsed into a SlumbotResponse instance."""
        from poker_ai.eval.slumbot_client import SlumbotClient, SlumbotResponse

        client = SlumbotClient()
        payload = _full_new_hand_payload(token="tok-xyz", client_pos=1)
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(payload),
        ):
            resp = client.new_hand()
        assert isinstance(resp, SlumbotResponse)
        assert resp.token == "tok-xyz"
        assert resp.client_pos == 1
        assert resp.hole_cards == ["As", "Kd"]

    def test_token_property_updated_after_new_hand(self) -> None:
        """After new_hand the client caches the response token."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="tok-first")),
        ):
            client.new_hand()
        assert client.token == "tok-first"

    def test_token_refreshed_when_server_rotates(self) -> None:
        """If a later new_hand returns a different token, client follows."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="tok-1")),
        ):
            client.new_hand()
        assert client.token == "tok-1"

        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="tok-2")),
        ):
            client.new_hand()
        assert client.token == "tok-2"


# ---------------------------------------------------------------------------
# E. act HTTP POST (4 tests)
# ---------------------------------------------------------------------------


class TestAct:
    def test_posts_to_slumbot_act_https_url(self) -> None:
        """act() POSTs to https://slumbot.com/api/act."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        # Prime token via new_hand
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="t1")),
        ):
            client.new_hand()

        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="t1", action="c")),
        ) as mock_post:
            client.act("c")
            args, _ = mock_post.call_args
            url = args[0] if args else mock_post.call_args.kwargs.get("url", "")
            assert url == "https://slumbot.com/api/act"

    def test_act_request_body_carries_token_and_incr(self) -> None:
        """Body == {'token': <current>, 'incr': <input>}."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="tok-A")),
        ):
            client.new_hand()

        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="tok-A", action="b300")),
        ) as mock_post:
            client.act("b300")
            body = mock_post.call_args.kwargs.get("json", {})
            assert body == {"token": "tok-A", "incr": "b300"}

    def test_act_response_parsed_correctly(self) -> None:
        """act() returns a populated SlumbotResponse."""
        from poker_ai.eval.slumbot_client import SlumbotClient, SlumbotResponse

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="t")),
        ):
            client.new_hand()

        payload = _full_new_hand_payload(token="t", action="cb300")
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(payload),
        ):
            resp = client.act("b300")
        assert isinstance(resp, SlumbotResponse)
        assert resp.action == "cb300"

    def test_act_without_prior_new_hand_raises(self) -> None:
        """Calling act() before new_hand() (token=None) is an error."""
        from poker_ai.eval.slumbot_client import SlumbotClient, SlumbotError

        client = SlumbotClient()
        assert client.token is None
        with pytest.raises((ValueError, SlumbotError)):
            client.act("c")


# ---------------------------------------------------------------------------
# F. Error handling (5 tests)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_http_500_propagates_httperror(self) -> None:
        """Server 500 → requests.HTTPError surfaces to caller."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse({"error": "boom"}, status_code=500),
        ):
            with pytest.raises(requests.HTTPError):
                client.new_hand()

    def test_error_msg_field_raises_slumbot_error(self) -> None:
        """200 OK with `error_msg` field → SlumbotError carrying msg."""
        from poker_ai.eval.slumbot_client import SlumbotClient, SlumbotError

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse({"error_msg": "bad incr 'b1'"}),
        ):
            with pytest.raises(SlumbotError) as exc_info:
                client.new_hand()
        assert "bad incr" in str(exc_info.value)

    def test_http_404_propagates_httperror(self) -> None:
        """404 also propagates as HTTPError."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse({"error": "not found"}, status_code=404),
        ):
            with pytest.raises(requests.HTTPError):
                client.new_hand()

    def test_non_json_body_raises_jsondecodeerror(self) -> None:
        """Non-JSON body → requests.JSONDecodeError propagates."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(None),  # .json() raises JSONDecodeError
        ):
            with pytest.raises(requests.JSONDecodeError):
                client.new_hand()

    def test_connection_refused_propagates(self) -> None:
        """requests.ConnectionError raised by transport propagates."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(requests.ConnectionError):
                client.new_hand()


# ---------------------------------------------------------------------------
# G. Token refresh (3 tests)
# ---------------------------------------------------------------------------


class TestTokenRefresh:
    def test_initial_new_hand_sets_token(self) -> None:
        """First new_hand response token X → client.token == X."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="X")),
        ):
            client.new_hand()
        assert client.token == "X"

    def test_act_response_with_rotated_token_updates_client(self) -> None:
        """If act response's token != prior, client picks up the new one."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="X")),
        ):
            client.new_hand()
        assert client.token == "X"

        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="Y", action="c")),
        ):
            client.act("c")
        assert client.token == "Y"

    def test_subsequent_request_uses_latest_token(self) -> None:
        """After rotation, the next request body carries the *new* token."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient()
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="X")),
        ):
            client.new_hand()

        # First act rotates token X → Y
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="Y", action="c")),
        ):
            client.act("c")

        # Second act must use Y, not X.
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload(token="Y", action="cc")),
        ) as mock_post:
            client.act("c")
            body = mock_post.call_args.kwargs.get("json", {})
            assert body.get("token") == "Y"


# ---------------------------------------------------------------------------
# H. URL construction (2 tests)
# ---------------------------------------------------------------------------


class TestURLConstruction:
    def test_scheme_is_always_https(self) -> None:
        """Even custom hosts must be reached via HTTPS (no plaintext)."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient(host="mirror.example.com")
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload()),
        ) as mock_post:
            client.new_hand()
            args, _ = mock_post.call_args
            url = args[0] if args else mock_post.call_args.kwargs.get("url", "")
            assert url.startswith("https://")

    def test_path_joined_with_host(self) -> None:
        """host + '/api/new_hand' joins to full URL with no double-slash."""
        from poker_ai.eval.slumbot_client import SlumbotClient

        client = SlumbotClient(host="slumbot.com")
        with patch(
            "poker_ai.eval.slumbot_client.requests.post",
            return_value=MockResponse(_full_new_hand_payload()),
        ) as mock_post:
            client.new_hand()
            args, _ = mock_post.call_args
            url = args[0] if args else mock_post.call_args.kwargs.get("url", "")
            assert url == "https://slumbot.com/api/new_hand"
            assert "//api" not in url.replace("https://", "")
