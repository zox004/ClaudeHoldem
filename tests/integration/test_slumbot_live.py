"""Live integration tests against ``slumbot.com`` — Phase 4 M4.4.

Marked ``@pytest.mark.live`` and skipped by default. Run explicitly::

    uv run pytest -m live tests/integration/test_slumbot_live.py

These tests touch the real Slumbot API. Each test plays exactly one
hand and is therefore extremely cheap (~1-2 seconds end-to-end) but
should **not** be left running in CI — they exercise an external
service we don't control.

Coverage:
- ``test_anonymous_new_hand_smoke``: anonymous-mode `/api/new_hand` round
  trip. Validates host reachability + JSON contract + token issuance.
- ``test_one_hand_uniform_smoke``: full one-hand smoke through
  :class:`SlumbotHarness` with uniform strategy. Validates the M4.1
  + M4.2 + M4.3 stack against a real Slumbot session.

Failures here typically mean: (a) Slumbot API spec drift, (b) network
issue, (c) regression in our protocol layer that mocks could not
catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.eval.slumbot_client import SlumbotClient
from poker_ai.eval.slumbot_harness import SlumbotHarness
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLGame,
    AbstractedHUNLState,
)


def _uniform_strategy(_state: AbstractedHUNLState) -> np.ndarray:
    return np.full(6, 1.0 / 6.0, dtype=np.float64)


@pytest.mark.live
def test_anonymous_new_hand_smoke() -> None:
    """Bare ``/api/new_hand`` against slumbot.com: returns a session
    token, a valid client_pos, two hole cards."""
    client = SlumbotClient()
    resp = client.new_hand()
    assert isinstance(resp.token, str) and len(resp.token) > 0
    assert resp.client_pos in (0, 1)
    assert len(resp.hole_cards) == 2
    for card in resp.hole_cards:
        assert isinstance(card, str)
        assert len(card) == 2


@pytest.mark.live
def test_one_hand_uniform_smoke() -> None:
    """Full one-hand smoke via SlumbotHarness, uniform strategy.

    Runs end-to-end M4.1 + M4.2 + M4.3 stack against the real Slumbot
    server. Retries up to 10 hands looking for one that completes
    without a known-edge-case error (M4.2 nearest-bucket dispatch can
    diverge from the server when the server's raw raise sizes don't
    cleanly map onto our 4-size abstracted grid; that is a known M4.2
    (i) limitation, registered as asset #24 candidate / Phase 5
    Schnizlein 2009 hook).

    The smoke succeeds as soon as any hand completes cleanly. If none
    of 10 attempts succeed, that itself is a strong signal of a
    transport/protocol regression (test fails).
    """
    client = SlumbotClient()
    game = AbstractedHUNLGame(
        n_buckets=10,
        n_trials=200,
        postflop_mc_trials=30,
        postflop_threshold_sample_size=80,
        seed=42,
    )
    harness = SlumbotHarness(client)
    rng = np.random.default_rng(42)

    last_exc: BaseException | None = None
    for _attempt in range(10):
        try:
            record = harness.play_one_hand(game, _uniform_strategy, rng)
        except (ValueError, Exception) as exc:   # noqa: BLE001
            last_exc = exc
            continue
        assert record.client_pos in (0, 1)
        assert isinstance(record.sequence, str)
        assert record.slumbot_winnings is not None
        return
    raise AssertionError(
        f"all 10 attempts failed; last error: {last_exc!r}"
    )
