"""Phase 4 M4.5.0 — Strategy persist infrastructure (TDD red-first).

Locks in the pickle artifact contract that ``phase4_m45_train_strategies``
emits and that downstream M4.5.1 mini-pilot / M4.5.3 production runs
will load. The script itself is allowed to be imported from
``experiments.phase4_m45_train_strategies`` (sibling pattern used by
existing experiments). Helpers under test:

* ``dump_strategy(strategy, *, seed, T, game_config, n_infosets_by_round, out_path)``
* ``load_strategy(path) -> dict``
* ``count_infosets_by_round(strategy) -> dict[int, int]``

Schema (frozen here on purpose — downstream consumers depend on it):

    {
        "seed": int,
        "T": int,
        "game_config": {
            "n_buckets": int,
            "n_trials": int,
            "postflop_mc_trials": int,
            "postflop_threshold_sample_size": int,
            "n_actions": int,
            "epsilon": float,
            "starting_stack_bb": int,   # MUST equal STARTING_STACK_BB (200)
        },
        "strategy": dict[str, np.ndarray],
        "n_infosets_by_round": {0: int, 1: int, 2: int, 3: int},
    }
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest  # noqa: F401  (used by tmp_path fixture + raises)

from poker_ai.games.hunl_state import STARTING_STACK_BB

from experiments.phase4_m45_train_strategies import (
    count_infosets_by_round,
    dump_strategy,
    load_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _toy_strategy() -> dict[str, np.ndarray]:
    """Tiny deterministic strategy with infoset keys covering all 4
    rounds, using the AbstractedHUNLGame infoset-key contract:

        ``"<hole_bucket>|<round>:<board_segment>:<history>"``

    Round 0 entries have empty board segment.
    """
    rng = np.random.default_rng(0)

    def _row() -> np.ndarray:
        v = rng.random(6)
        return v / v.sum()

    return {
        # round 0 — preflop, 3 entries
        "3|0::": _row(),
        "7|0::r": _row(),
        "12|0::rc": _row(),
        # round 1 — flop, 2 entries
        "3|1:8:rc/": _row(),
        "7|1:14:rc/r": _row(),
        # round 2 — turn, 1 entry
        "3|2:21:rc/rc/": _row(),
        # round 3 — river, 1 entry
        "3|3:33:rc/rc/rc/": _row(),
    }


def _toy_game_config() -> dict[str, object]:
    return {
        "n_buckets": 50,
        "n_trials": 10000,
        "postflop_mc_trials": 300,
        "postflop_threshold_sample_size": 10000,
        "n_actions": 6,
        "epsilon": 0.05,
        "starting_stack_bb": STARTING_STACK_BB,
    }


# ---------------------------------------------------------------------------
# count_infosets_by_round
# ---------------------------------------------------------------------------
def test_count_infosets_by_round_returns_all_four_keys() -> None:
    counts = count_infosets_by_round(_toy_strategy())
    assert set(counts.keys()) == {0, 1, 2, 3}


def test_count_infosets_by_round_correct_counts() -> None:
    counts = count_infosets_by_round(_toy_strategy())
    assert counts == {0: 3, 1: 2, 2: 1, 3: 1}


def test_count_infosets_by_round_handles_empty_strategy() -> None:
    counts = count_infosets_by_round({})
    assert counts == {0: 0, 1: 0, 2: 0, 3: 0}


def test_count_infosets_by_round_skips_malformed_keys() -> None:
    # Malformed keys must not crash the counter; baseline script uses
    # the same skip-silently contract.
    strategy = dict(_toy_strategy())
    strategy["this-is-not-a-valid-infoset-key"] = np.array([1.0])
    strategy["bad|format"] = np.array([1.0])
    counts = count_infosets_by_round(strategy)
    assert counts == {0: 3, 1: 2, 2: 1, 3: 1}


# ---------------------------------------------------------------------------
# dump_strategy / load_strategy round-trip
# ---------------------------------------------------------------------------
def test_dump_strategy_writes_pickle_file(tmp_path: Path) -> None:
    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()
    counts = count_infosets_by_round(strategy)
    written = dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=counts,
        out_path=out,
    )
    assert written == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_load_strategy_round_trip_preserves_schema(tmp_path: Path) -> None:
    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()
    counts = count_infosets_by_round(strategy)
    dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=counts,
        out_path=out,
    )
    loaded = load_strategy(out)

    assert set(loaded.keys()) == {
        "seed",
        "T",
        "game_config",
        "strategy",
        "n_infosets_by_round",
    }
    assert loaded["seed"] == 42
    assert loaded["T"] == 100_000
    assert loaded["n_infosets_by_round"] == counts


def test_load_strategy_preserves_array_dtype_and_shape(tmp_path: Path) -> None:
    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()
    dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=count_infosets_by_round(strategy),
        out_path=out,
    )
    loaded = load_strategy(out)
    assert set(loaded["strategy"].keys()) == set(strategy.keys())
    for key, arr in strategy.items():
        loaded_arr = loaded["strategy"][key]
        assert isinstance(loaded_arr, np.ndarray)
        assert loaded_arr.shape == arr.shape
        assert loaded_arr.dtype == arr.dtype
        np.testing.assert_array_equal(loaded_arr, arr)


# ---------------------------------------------------------------------------
# starting_stack_bb lock-in (M4.0 mentor #9 fact-transfer guard)
# ---------------------------------------------------------------------------
def test_game_config_starting_stack_bb_equals_constant(tmp_path: Path) -> None:
    """Artifact metadata MUST mark the stack convention. Prevents the
    100 BB ↔ 200 BB confusion that triggered mentor #9 self-correction
    at M4.0 entry.
    """
    out = tmp_path / "seed42.pkl"
    dump_strategy(
        _toy_strategy(),
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=count_infosets_by_round(_toy_strategy()),
        out_path=out,
    )
    loaded = load_strategy(out)
    assert "starting_stack_bb" in loaded["game_config"]
    assert loaded["game_config"]["starting_stack_bb"] == STARTING_STACK_BB
    assert loaded["game_config"]["starting_stack_bb"] == 200


# ---------------------------------------------------------------------------
# M4.5.0a — α (lock-serialized dump) + atomic rename
# ---------------------------------------------------------------------------
def test_dump_strategy_accepts_lock_kwarg(tmp_path: Path) -> None:
    """``lock=`` kwarg accepted; round-trip works under a real
    threading lock. Used by the M4.5.0a Manager.Lock injection from
    the spawn-pool worker initializer.
    """
    import threading

    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()
    dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=count_infosets_by_round(strategy),
        out_path=out,
        lock=threading.Lock(),
    )
    loaded = load_strategy(out)
    assert loaded["seed"] == 42


def test_dump_strategy_uses_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic write invariant: ``dump_strategy`` writes a .tmp sibling
    first and ``Path.rename``-s it onto ``out_path``. Prevents the
    M4.5.0 1차-시도 failure mode where killing mid-dump left half-written
    .pkl files indistinguishable from valid ones.
    """
    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()

    rename_calls: list[tuple[Path, Path]] = []
    original_rename = Path.rename

    def _capturing_rename(self: Path, target: Path) -> Path:
        rename_calls.append((self, Path(target)))
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _capturing_rename)

    dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=count_infosets_by_round(strategy),
        out_path=out,
    )

    assert len(rename_calls) == 1, (
        f"expected exactly 1 atomic rename call, got {len(rename_calls)}"
    )
    src, dst = rename_calls[0]
    assert dst == out
    assert src.suffix == ".tmp" or src.name.endswith(".tmp"), (
        f"source must be a .tmp sibling, got {src}"
    )
    assert out.is_file()


def test_dump_strategy_no_tmp_leftover_after_success(tmp_path: Path) -> None:
    """After a successful dump, no ``.tmp`` sibling lingers in the dir."""
    out = tmp_path / "seed42.pkl"
    strategy = _toy_strategy()
    dump_strategy(
        strategy,
        seed=42,
        T=100_000,
        game_config=_toy_game_config(),
        n_infosets_by_round=count_infosets_by_round(strategy),
        out_path=out,
    )
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"unexpected .tmp leftover: {leftover}"


def test_dump_strategy_rejects_wrong_starting_stack_bb(tmp_path: Path) -> None:
    """Defensive: callers cannot accidentally dump under a non-200 stack
    config. M4.0 lesson: stack drift between training-time and
    benchmark-time is silent unless asserted.
    """
    bad_config = dict(_toy_game_config())
    bad_config["starting_stack_bb"] = 100
    with pytest.raises(ValueError, match="starting_stack_bb"):
        dump_strategy(
            _toy_strategy(),
            seed=42,
            T=100_000,
            game_config=bad_config,
            n_infosets_by_round=count_infosets_by_round(_toy_strategy()),
            out_path=tmp_path / "bad.pkl",
        )
