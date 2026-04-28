"""Phase 4 M4.5.1 — Pilot helpers (TDD red-first).

Locks in three contracts the pilot script depends on:

* ``StrategyWithMissCounter`` — wraps a trained strategy dict; on
  ``KeyError`` returns uniform 6-vec and counts the miss separately
  from divergence (3-mode separation: replay-divergence / strategy-miss
  / transport).
* ``classify_failure_mode(exc)`` — maps a per-hand exception to one of
  three labels: ``"replay-divergence"``, ``"harness-desync"``,
  ``"transport"`` (or ``"unknown"`` as catch-all). Uniform-fallback
  miss never reaches this layer.
* ``decide_path(divergence_rate)`` — path A / ambiguous / B based on
  user-spec thresholds (< 5%, 5-15%, > 15%).
"""

from __future__ import annotations

import numpy as np
import pytest
import requests

from poker_ai.eval.slumbot_client import SlumbotError

from experiments.phase4_m45_pilot import (
    StrategyWithMissCounter,
    classify_failure_mode,
    decide_path,
)


# ---------------------------------------------------------------------------
# StrategyWithMissCounter
# ---------------------------------------------------------------------------
class _FakeState:
    """Minimal state with ``infoset_key`` as a ``@property`` — matches
    AbstractedHUNLState's actual API (M4.5.1 smoke caught the
    parens-vs-property confusion as claude self-audit #26 / 자산 #22
    6번째 instance).
    """

    def __init__(self, key: str) -> None:
        self._key = key

    @property
    def infoset_key(self) -> str:
        return self._key


def _toy_dict() -> dict[str, np.ndarray]:
    return {
        "A|0::": np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0]),
        "B|0::": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    }


def test_miss_counter_returns_dict_value_on_hit() -> None:
    s = StrategyWithMissCounter(_toy_dict())
    out = s(_FakeState("A|0::"))
    np.testing.assert_array_equal(out, _toy_dict()["A|0::"])
    assert s.n_misses == 0
    assert s.n_lookups == 1


def test_miss_counter_returns_uniform_on_miss() -> None:
    s = StrategyWithMissCounter(_toy_dict())
    out = s(_FakeState("Z|3:99:rrr/cc/c/"))
    expected = np.full(6, 1.0 / 6.0, dtype=np.float64)
    np.testing.assert_allclose(out, expected)
    assert s.n_misses == 1


def test_miss_counter_tracks_unique_missed_keys() -> None:
    s = StrategyWithMissCounter(_toy_dict())
    s(_FakeState("X|1::r"))
    s(_FakeState("X|1::r"))   # same miss key 2x
    s(_FakeState("Y|2::cc"))
    assert s.n_misses == 3
    assert s.n_unique_missed_keys == 2


def test_miss_counter_returns_six_vector_dtype_float64() -> None:
    """Harness expects shape (6,) float64 for both hit and miss paths."""
    s = StrategyWithMissCounter(_toy_dict())
    hit = s(_FakeState("A|0::"))
    miss = s(_FakeState("absent"))
    for arr in (hit, miss):
        assert arr.shape == (6,)
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# classify_failure_mode
# ---------------------------------------------------------------------------
def test_classify_value_error_is_replay_divergence() -> None:
    """``replay_sequence`` raises ``ValueError`` on illegal action /
    nearest-bucket dispatch divergence (M4.2 known limitation,
    asset #24 candidate).
    """
    assert classify_failure_mode(ValueError("illegal raise size")) == (
        "replay-divergence"
    )


def test_classify_slumbot_error_is_harness_desync() -> None:
    """``SlumbotError`` is raised by the harness on state-machine
    desyncs (winnings-vs-terminal mismatch, wrong-turn) or by the
    client on server ``error_msg``. Both buckets count as the same
    cluster for path attribution.
    """
    assert classify_failure_mode(SlumbotError("desync: wrong turn")) == (
        "harness-desync"
    )


def test_classify_http_error_is_transport() -> None:
    """``requests.HTTPError`` after retry exhaustion = transport."""
    assert classify_failure_mode(requests.HTTPError("502 bad gateway")) == (
        "transport"
    )


def test_classify_unknown_exception_is_unknown() -> None:
    """Defensive: anything else is bucketed but flagged for triage."""
    assert classify_failure_mode(RuntimeError("???")) == "unknown"


# ---------------------------------------------------------------------------
# decide_path
# ---------------------------------------------------------------------------
def test_decide_path_below_5pct_is_A() -> None:
    """< 5% divergence → trained strategy precise enough → production."""
    assert decide_path(0.0) == "A"
    assert decide_path(0.04) == "A"
    assert decide_path(0.0499) == "A"


def test_decide_path_between_5_and_15_is_ambiguous() -> None:
    assert decide_path(0.05) == "ambiguous"
    assert decide_path(0.10) == "ambiguous"
    assert decide_path(0.149) == "ambiguous"


def test_decide_path_15_or_above_is_B() -> None:
    """≥ 15% divergence → Schnizlein 2009 introduction urgent."""
    assert decide_path(0.15) == "B"
    assert decide_path(0.30) == "B"
    assert decide_path(1.0) == "B"


def test_decide_path_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="divergence_rate"):
        decide_path(-0.01)
    with pytest.raises(ValueError, match="divergence_rate"):
        decide_path(1.01)
