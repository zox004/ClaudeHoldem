"""Phase 4 M4.5.0 — Strategy persist infrastructure.

Trains MCCFR ExternalSampling against ``AbstractedHUNLGame`` for each
seed at production scale (n_buckets=50, n_trials=10000,
postflop_mc_trials=300, postflop_threshold_sample_size=10000) and
persists the averaged strategy to disk so M4.5.1+ can load it without
re-training.

This is *infra only* — no LBR, no occupancy plot, no aggregation. The
M3.4 / M4.0 baseline scripts already cover those measurements; M4.5.0
exists because those scripts consume ``trainer.average_strategy()``
in-process and never write it to disk.

Pickle artifact contract (frozen — downstream consumers depend on it)::

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

Run::

    uv run python -m experiments.phase4_m45_train_strategies
    uv run python -m experiments.phase4_m45_train_strategies T=1000          # smoke
    uv run python -m experiments.phase4_m45_train_strategies parallel=false
    uv run python -m experiments.phase4_m45_train_strategies seeds=[42]
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.mccfr import MCCFRExternalSampling
from poker_ai.games.hunl_abstraction import AbstractedHUNLGame
from poker_ai.games.hunl_state import STARTING_STACK_BB

log = logging.getLogger(__name__)

# Cross-seed audit threshold — M4.5.0 weak suggestion #1. Per-round
# infoset count spread (max-min)/mean across seeds; > 5% surfaces a
# warning so downstream M4.5.1 can decide whether to investigate
# abstractor non-determinism (none expected, but artifact-level guard).
_INFOSET_SPREAD_AUDIT_THRESHOLD = 0.05

# M4.5.0a: spawn-pool-shared lock for dump serialization. Set by
# ``_worker_init`` via ``initargs``; ``None`` if running without a pool
# or in tests.
_DUMP_LOCK: ContextManager[Any] | None = None


def _worker_init(lock: ContextManager[Any]) -> None:
    """Spawn-pool worker initializer. Stores the shared
    ``multiprocessing.Manager.Lock`` (or any context-manager lock) in a
    module-global so ``_run_seed`` can pass it to ``dump_strategy``
    without threading it through ``cfg_dict`` (Manager proxies are not
    OmegaConf-serializable).
    """
    global _DUMP_LOCK
    _DUMP_LOCK = lock


# =============================================================================
# Schema helpers (unit-tested)
# =============================================================================
def count_infosets_by_round(
    strategy: dict[str, np.ndarray],
) -> dict[int, int]:
    """Counts strategy infosets per betting round (0=preflop, 1=flop,
    2=turn, 3=river).

    Infoset key contract (AbstractedHUNLGame):
        ``"<hole_bucket>|<round>:<board_segment>:<history>"``

    Malformed keys are skipped silently — same contract as the M3.4
    baseline ``_compute_bucket_occupancy`` parser.
    """
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for key in strategy.keys():
        try:
            _head, rest = key.split("|", 1)
            round_part, _board_part, _hist = rest.split(":", 2)
            round_idx = int(round_part)
        except (ValueError, IndexError):
            continue
        if round_idx in counts:
            counts[round_idx] += 1
    return counts


def dump_strategy(
    strategy: dict[str, np.ndarray],
    *,
    seed: int,
    T: int,
    game_config: dict[str, Any],
    n_infosets_by_round: dict[int, int],
    out_path: Path,
    lock: ContextManager[Any] | None = None,
) -> Path:
    """Pickles the artifact dict to ``out_path`` atomically. Validates
    that ``game_config["starting_stack_bb"] == STARTING_STACK_BB`` to
    prevent the 100 BB ↔ 200 BB confusion that triggered mentor #9
    self-correction at M4.0 entry.

    M4.5.0a contract:

    * Atomic write — pickles to ``<out_path>.tmp`` first, fsync, then
      ``Path.rename`` onto ``out_path``. Truncated ``.pkl`` files are
      structurally impossible (only ``.tmp`` files can be partial).
    * ``lock`` (optional) — when supplied, the entire dump (write +
      fsync + rename) runs inside ``with lock:``. Used by the M4.5.0a
      spawn-pool to serialize 5 worker dumps so unified-memory swap
      pressure (root cause of the 1차-시도 3/5 truncation) cannot
      recur.
    """
    stack_bb = game_config.get("starting_stack_bb")
    if stack_bb != STARTING_STACK_BB:
        raise ValueError(
            f"game_config starting_stack_bb={stack_bb!r} does not match "
            f"STARTING_STACK_BB={STARTING_STACK_BB} (M4.0 stack lock-in)"
        )
    artifact = {
        "seed": int(seed),
        "T": int(T),
        "game_config": dict(game_config),
        "strategy": strategy,
        "n_infosets_by_round": dict(n_infosets_by_round),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    cm: ContextManager[Any] = lock if lock is not None else nullcontext()
    with cm:
        with tmp_path.open("wb") as fh:
            pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.rename(out_path)
    return out_path


def load_strategy(path: Path) -> dict[str, Any]:
    """Loads and returns the pickled artifact dict."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


# =============================================================================
# Per-seed worker
# =============================================================================
def _build_game_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_buckets": int(cfg_dict["n_buckets"]),
        "n_trials": int(cfg_dict["n_trials"]),
        "postflop_mc_trials": int(cfg_dict["postflop_mc_trials"]),
        "postflop_threshold_sample_size": int(
            cfg_dict["postflop_threshold_sample_size"]
        ),
        "n_actions": int(cfg_dict["n_actions"]),
        "epsilon": float(cfg_dict["epsilon"]),
        "starting_stack_bb": STARTING_STACK_BB,
    }


def _run_seed(seed: int, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Single-seed train+dump worker. Picklable for
    :class:`multiprocessing.Pool`.
    """
    game_config = _build_game_config(cfg_dict)
    T = int(cfg_dict["T"])
    out_dir = Path(cfg_dict["out_dir"])
    out_path = out_dir / f"seed{seed}.pkl"

    # Stage 1: game setup.
    setup_start = time.perf_counter()
    game = AbstractedHUNLGame(
        n_buckets=game_config["n_buckets"],
        n_trials=game_config["n_trials"],
        postflop_mc_trials=game_config["postflop_mc_trials"],
        postflop_threshold_sample_size=game_config[
            "postflop_threshold_sample_size"
        ],
        seed=seed,
    )
    setup_t = time.perf_counter() - setup_start
    log.info("[seed=%d] game setup: %.1fs", seed, setup_t)

    # Stage 2: MCCFR train.
    rng = np.random.default_rng(seed)
    trainer = MCCFRExternalSampling(
        game=game,
        n_actions=game_config["n_actions"],
        rng=rng,
        epsilon=game_config["epsilon"],
    )
    train_start = time.perf_counter()
    trainer.train(iterations=T)
    train_t = time.perf_counter() - train_start

    # Stage 3: snapshot + persist.
    strategy = trainer.average_strategy()
    counts = count_infosets_by_round(strategy)
    written = dump_strategy(
        strategy,
        seed=seed,
        T=T,
        game_config=game_config,
        n_infosets_by_round=counts,
        out_path=out_path,
        lock=_DUMP_LOCK,
    )
    pickle_size_mb = written.stat().st_size / (1024.0 * 1024.0)

    log.info(
        "[seed=%d] T=%d  n_infosets=%d (preflop=%d, flop=%d, turn=%d, river=%d)  "
        "wall_mccfr=%.1fs  pickle=%.1fMB  → %s",
        seed, T, sum(counts.values()),
        counts[0], counts[1], counts[2], counts[3],
        train_t, pickle_size_mb, written,
    )

    return {
        "seed": seed,
        "setup_t_s": setup_t,
        "train_t_s": train_t,
        "n_infosets_total": sum(counts.values()),
        "n_infosets_by_round": counts,
        "pickle_path": str(written),
        "pickle_size_mb": pickle_size_mb,
    }


# =============================================================================
# Aggregation + audit
# =============================================================================
def _audit_cross_seed_spread(
    per_seed: list[dict[str, Any]],
    threshold: float = _INFOSET_SPREAD_AUDIT_THRESHOLD,
) -> dict[int, dict[str, float]]:
    """Per-round (max-min)/mean infoset count spread across seeds.
    Logs WARNING for any round exceeding ``threshold``.
    """
    audit: dict[int, dict[str, float]] = {}
    for r in (0, 1, 2, 3):
        counts = np.array(
            [run["n_infosets_by_round"][r] for run in per_seed],
            dtype=np.float64,
        )
        if counts.size == 0:
            continue
        mean_c = float(counts.mean())
        spread = (
            float(counts.max() - counts.min()) / mean_c if mean_c > 0 else 0.0
        )
        audit[r] = {
            "mean": mean_c,
            "min": float(counts.min()),
            "max": float(counts.max()),
            "spread": spread,
        }
        if spread > threshold:
            log.warning(
                "[audit] round %d cross-seed infoset-count spread %.2f%% "
                "exceeds %.2f%% threshold (min=%d, max=%d, mean=%.1f)",
                r, spread * 100, threshold * 100,
                int(counts.min()), int(counts.max()), mean_c,
            )
    return audit


# =============================================================================
# Main
# =============================================================================
def _run_all_seeds(cfg: DictConfig, out_dir: Path) -> list[dict[str, Any]]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    cfg_dict["out_dir"] = str(out_dir)
    seeds = list(cfg.seeds)

    if bool(cfg.parallel):
        log.info(
            "Running %d seeds in parallel (multiprocessing.Pool, "
            "Manager.Lock-serialized dumps)",
            len(seeds),
        )
        ctx = mp.get_context("spawn")
        # Manager.Lock proxy crosses spawn-process boundaries. Workers
        # use it via the module-global ``_DUMP_LOCK`` set in
        # ``_worker_init`` so the entire pickle.dump+fsync+rename for
        # one seed is exclusive — eliminates the 1차-시도 root cause
        # (5×1.1GB concurrent dumps swap-bound on 16GB unified memory).
        with mp.Manager() as manager:
            dump_lock = manager.Lock()
            with ctx.Pool(
                len(seeds),
                initializer=_worker_init,
                initargs=(dump_lock,),
            ) as pool:
                results = pool.starmap(
                    _run_seed, [(int(s), cfg_dict) for s in seeds]
                )
    else:
        log.info("Running %d seeds sequentially", len(seeds))
        results = [_run_seed(int(s), cfg_dict) for s in seeds]
    return list(results)


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="phase4_m45_train_strategies",
)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    per_seed = _run_all_seeds(cfg, out_dir)
    wall_total = time.perf_counter() - wall_start

    audit = _audit_cross_seed_spread(per_seed)

    log.info("====== M4.5.0 train+dump summary ======")
    for r in per_seed:
        log.info(
            "  seed=%d  T=%d  n_infosets=%d  train=%.1fs  pickle=%.1fMB",
            r["seed"], int(cfg.T), r["n_infosets_total"],
            r["train_t_s"], r["pickle_size_mb"],
        )
    log.info("====== Cross-seed infoset-count audit ======")
    round_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    for r_idx, stats in audit.items():
        log.info(
            "  %s: mean=%.1f  min=%d  max=%d  spread=%.2f%%",
            round_names[r_idx], stats["mean"], int(stats["min"]),
            int(stats["max"]), stats["spread"] * 100,
        )
    log.info(
        "Wall total: %.1fs (%.2f min)  out_dir=%s",
        wall_total, wall_total / 60.0, out_dir,
    )


if __name__ == "__main__":
    main()
