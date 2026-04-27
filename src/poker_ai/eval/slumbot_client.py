"""Slumbot HTTP transport — Phase 4 M4.1.

Thin wrapper around ``slumbot.com`` JSON-over-HTTPS API for HUNL
benchmark integration. M4.1 covers transport only:

- ``SlumbotResponse`` parsed dataclass (frozen + slots).
- ``SlumbotClient`` POSTing to ``/api/new_hand`` and ``/api/act`` with
  automatic token tracking and error propagation.
- ``chip_to_slumbot`` / ``chip_from_slumbot`` for the chip-granularity
  delta (our internal BB=2 chips ↔ Slumbot BB=100 chips, ×50).

M4.1 deliberately excludes action-protocol semantics (the
``b500``/``c``/``f``/``k`` strings, street boundaries, opponent-
action ingestion, all-in handling) — those are M4.2's surface.

Verified spec (claude-side, asset #22 cross-context fact verification):

- HTTPS host = ``slumbot.com``.
- Endpoints: ``/api/login`` (out of M4.1 scope), ``/api/new_hand``,
  ``/api/act``.
- ``/api/new_hand`` body: ``{"token": <uuid|None>}``; token is optional
  on the first call (anonymous play supported).
- ``/api/act`` body: ``{"token": <uuid>, "incr": "<action_str>"}``.
- Response keys: ``token``, ``action``, ``client_pos``, ``hole_cards``,
  ``board``, ``winnings``, ``baseline_winnings``, ``session_num_hands``,
  ``session_total``, ``session_baseline_total``.
- Token may rotate per response — client refreshes from each reply.
- HTTP 200 with an ``error_msg`` field is a soft error; non-200 is a
  hard ``HTTPError``.

Error policy:
- HTTP 4xx/5xx ⇒ ``requests.HTTPError`` propagates to the caller.
- HTTP 200 + ``error_msg`` ⇒ :class:`SlumbotError` raised with the msg.
- JSON decode failure ⇒ ``requests.JSONDecodeError`` propagates.
- Network failures (``ConnectionError``, ``Timeout``) propagate.
- No retry — M4.5 production harness layers retry above this client.

Game parameter mismatch (claude-side mentor #9 self-correction):
Slumbot uses BB=100 chips / stack=20000 chips (200 BB Doyle's Game).
Our internal HUNLState uses BB=2 chips / stack=400 chips (also 200 BB
post-M4.0). The ×50 chip multiplier bridges the two views without
changing either state space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import requests   # type: ignore[import-untyped]


DEFAULT_HOST: Final[str] = "slumbot.com"
SLUMBOT_BB_CHIPS: Final[int] = 100
OUR_BB_CHIPS: Final[int] = 2
CHIP_MULTIPLIER: Final[int] = SLUMBOT_BB_CHIPS // OUR_BB_CHIPS   # 50
DEFAULT_TIMEOUT_S: Final[float] = 30.0


# =============================================================================
# Errors
# =============================================================================
class SlumbotError(Exception):
    """Slumbot returned a 200 response with an ``error_msg`` field.

    Distinct from ``requests.HTTPError`` (non-2xx status) — this is the
    server's way of reporting application-level errors (bad action
    string, illegal incr, expired session) on otherwise-successful
    HTTPS responses.
    """


# =============================================================================
# Response dataclass
# =============================================================================
@dataclass(frozen=True, slots=True)
class SlumbotResponse:
    """Parsed Slumbot JSON response from /api/new_hand or /api/act.

    Optional fields (``winnings`` and the ``session_*`` group) are
    ``None`` mid-hand; populated at hand termination and session
    boundaries respectively.

    ``token`` is also optional: M4.4 live verify (claude #24 self-
    audit) — Slumbot omits the ``token`` field on certain ``/api/act``
    responses (e.g. mid-hand replies that don't rotate the session).
    Callers / :meth:`SlumbotClient._post` keep the existing held
    token in that case.
    """

    token: str | None
    action: str
    client_pos: int
    hole_cards: list[str]
    board: list[str]
    winnings: int | None
    baseline_winnings: int | None
    session_num_hands: int | None
    session_total: int | None
    session_baseline_total: int | None


def _parse_response(body: dict[str, Any]) -> SlumbotResponse:
    """Turns the raw JSON dict into a :class:`SlumbotResponse`.

    Raises :class:`SlumbotError` if the body contains an ``error_msg``
    field (Slumbot's soft-error convention on 200 responses).
    """
    if "error_msg" in body:
        raise SlumbotError(str(body["error_msg"]))
    return SlumbotResponse(
        token=body.get("token"),
        action=body.get("action", ""),
        client_pos=int(body.get("client_pos", 0)),
        hole_cards=list(body.get("hole_cards", [])),
        board=list(body.get("board", [])),
        winnings=body.get("winnings"),
        baseline_winnings=body.get("baseline_winnings"),
        session_num_hands=body.get("session_num_hands"),
        session_total=body.get("session_total"),
        session_baseline_total=body.get("session_baseline_total"),
    )


# =============================================================================
# Chip granularity helpers
# =============================================================================
def chip_to_slumbot(our_chips: int) -> int:
    """Converts our internal chips (BB=2) to Slumbot chips (BB=100).

    Multiplies by 50. Always exact for any integer input — our internal
    bet sizes (computed by ``compute_size`` in
    :mod:`poker_ai.games.hunl_abstraction`) are integer chip counts at
    BB=2 granularity, so the upscaled value is always a multiple of 50
    on the Slumbot side.
    """
    return our_chips * CHIP_MULTIPLIER


def chip_from_slumbot(slumbot_chips: int) -> int:
    """Converts Slumbot chips (BB=100) to our internal chips (BB=2).

    Floor-divides by 50. **Lossy by design** when Slumbot returns an
    arbitrary opponent raise size that is not a multiple of 50 (e.g.
    a Slumbot human-player-style raise of ``b347``). M4.2 will route
    those through nearest-abstraction-bucket translation rather than
    using this helper directly; this helper exists only for the round-
    trip case where the value originated on our side.
    """
    return slumbot_chips // CHIP_MULTIPLIER


# =============================================================================
# Client
# =============================================================================
class SlumbotClient:
    """Thin HTTPS POST wrapper for slumbot.com /api/* endpoints.

    Supports anonymous play (no login) — the first ``new_hand`` call
    omits the token. The server returns a session token in the
    response which is then used (and refreshed) on every subsequent
    request.

    Args:
        host: HTTPS host name. Default ``slumbot.com``. The scheme is
            always HTTPS regardless of the host argument.
        timeout: per-request socket timeout in seconds. Default 30.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.host = host
        self.timeout = timeout
        self._token: str | None = None

    @property
    def token(self) -> str | None:
        """Currently held session token; ``None`` until the first
        ``new_hand`` succeeds."""
        return self._token

    def new_hand(self) -> SlumbotResponse:
        """Starts a new hand. Body sends the current token if held;
        omits the token field entirely on the first anonymous call
        (Slumbot rejects ``{"token": null}`` with HTTP 400 "Object
        type exception: Expected string" — verified live in M4.4
        pilot, 2026-04-27, claude self-audit #24).
        """
        body: dict[str, Any] = {}
        if self._token is not None:
            body["token"] = self._token
        return self._post("/api/new_hand", body)

    def act(self, incr: str) -> SlumbotResponse:
        """Sends an action increment string (e.g. ``"c"``, ``"b300"``).

        Raises :class:`SlumbotError` if called before any successful
        ``new_hand`` (no token held).
        """
        if self._token is None:
            raise SlumbotError(
                "act() called before new_hand(); no session token held"
            )
        body: dict[str, Any] = {"token": self._token, "incr": incr}
        return self._post("/api/act", body)

    def _post(self, path: str, body: dict[str, Any]) -> SlumbotResponse:
        """Single HTTPS POST + response parsing + token refresh.

        URL is always ``https://<host><path>``. Errors propagate per
        the policy in the module docstring; the response token is
        cached as the new ``self._token`` on success.
        """
        url = f"https://{self.host}{path}"
        resp = requests.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        parsed = _parse_response(data)
        # Token is optional (M4.4 live verify): only update when the
        # server provides a fresh value. Otherwise the previously-held
        # token continues to authenticate subsequent requests.
        if parsed.token is not None:
            self._token = parsed.token
        return parsed
