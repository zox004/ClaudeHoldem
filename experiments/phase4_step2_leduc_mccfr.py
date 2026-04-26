"""Phase 4 Step 2 — Leduc MCCFR + abstraction validate.

Forks ``phase2_leduc_mccfr.py`` with one substantive change: the game
factory selects between :class:`LeducPoker` (raw, Phase 2 reproduction)
and :class:`AbstractedLeducPoker` (Pluribus-path validation) via
``cfg.game_choice``. Everything else — MCCFR algorithm, 5-seed
multiprocessing harness, exploitability eval, W&B logging — is reused
unchanged. The wrapper game implements ``GameProtocol`` so the MCCFR
trainer needs no modification (Phase 2 design lock holds).

Two configurations:

- ``game_choice=raw`` — :class:`LeducPoker`. Step 2a: re-runs Phase 2's
  100k × 5-seed MCCFR to verify reproducibility before introducing
  abstraction.
- ``game_choice=abstracted_2`` — :class:`AbstractedLeducPoker(n_buckets=2)`.
  Step 2b: 192-infoset abstraction (vs raw 288), 33 % infoset reduction.
  Pass criterion ``< 5 mbb/g`` ⇒ Pluribus path commit for Phase 4 HUNL.

A third option ``game_choice=abstracted_3`` is provided for wrapper
identity sanity (3-bucket on Leduc is the identity map, see
:mod:`poker_ai.games.leduc_abstraction` docstring) — not used by the
default Step 2 schedule.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.mccfr import MCCFRExternalSampling
from poker_ai.eval.exploitability import exploitability_mbb
from poker_ai.games.leduc import LeducPoker
from poker_ai.games.leduc_abstraction import AbstractedLeducPoker

log = logging.getLogger(__name__)


def _make_game(game_choice: str) -> Any:
    """Factory: instantiates the GameProtocol-compatible game requested
    by ``cfg.game_choice``. Defined as a top-level function so the seed
    workers (spawned via ``multiprocessing``) can re-import it."""
    if game_choice == "raw":
        return LeducPoker()
    if game_choice == "abstracted_2":
        return AbstractedLeducPoker(n_buckets=2)
    if game_choice == "abstracted_3":
        return AbstractedLeducPoker(n_buckets=3)
    raise ValueError(
        f"game_choice must be one of {{'raw', 'abstracted_2', 'abstracted_3'}}, "
        f"got {game_choice!r}"
    )


def _should_log(iter_idx: int, log_every: int, dense_prefix: int) -> bool:
    if iter_idx <= dense_prefix:
        return True
    return iter_idx % log_every == 0


def _run_seed(seed: int, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Single-seed MCCFR training, mirrors phase2_leduc_mccfr._run_seed."""
    iterations = int(cfg_dict["iterations"])
    n_actions = int(cfg_dict["n_actions"])
    big_blind = float(cfg_dict["big_blind"])
    epsilon = float(cfg_dict["epsilon"])
    exit_threshold = float(cfg_dict["exit_criterion_mbb"])
    log_every = int(cfg_dict["log_every"])
    dense_prefix = int(cfg_dict["dense_prefix"])
    game_choice = str(cfg_dict["game_choice"])
    wandb_cfg = cfg_dict["wandb"]

    run_name = f"step2-{game_choice}-seed{seed}"
    run = wandb.init(
        project=wandb_cfg["project"],
        name=run_name,
        tags=list(wandb_cfg["tags"]) + [f"seed{seed}", game_choice],
        config=cfg_dict,
        reinit=True,
        mode=wandb_cfg["mode"],
    )
    assert run is not None

    rng = np.random.default_rng(seed)
    game = _make_game(game_choice)
    trainer = MCCFRExternalSampling(
        game=game, n_actions=n_actions, rng=rng, epsilon=epsilon
    )

    steps: list[int] = []
    expl_mbb: list[float] = []
    iters_to_exit: int | None = None

    loop_start = time.perf_counter()
    for t in range(1, iterations + 1):
        trainer.train(1)
        if not _should_log(t, log_every, dense_prefix):
            continue
        avg_strategy = trainer.average_strategy()
        expl_mbb_val = exploitability_mbb(game, avg_strategy, big_blind=big_blind)
        if iters_to_exit is None and expl_mbb_val < exit_threshold:
            iters_to_exit = t
        wandb.log({"exploitability_mbb": expl_mbb_val}, step=t)
        steps.append(t)
        expl_mbb.append(expl_mbb_val)

    total_sec = time.perf_counter() - loop_start
    final_expl_mbb = expl_mbb[-1]

    wandb.summary.update(
        {
            "final_exploitability_mbb": final_expl_mbb,
            "exit_criterion_met": final_expl_mbb < exit_threshold,
            "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
            "total_seconds": total_sec,
            "seed": seed,
            "game_choice": game_choice,
        }
    )
    wandb.finish()

    return {
        "seed": seed,
        "steps": steps,
        "expl_mbb": expl_mbb,
        "final_expl_mbb": final_expl_mbb,
        "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
        "total_seconds": total_sec,
    }


def _run_all_seeds(cfg: DictConfig) -> list[dict[str, Any]]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    seeds = list(cfg.seeds)
    if bool(cfg.parallel):
        log.info("Running %d seeds in parallel (multiprocessing.Pool)", len(seeds))
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(seeds)) as pool:
            results = pool.starmap(
                _run_seed, [(int(s), cfg_dict) for s in seeds]
            )
    else:
        log.info("Running %d seeds sequentially", len(seeds))
        results = [_run_seed(int(s), cfg_dict) for s in seeds]
    return list(results)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ref_steps = np.asarray(results[0]["steps"])
    mbb_arr = np.array([r["expl_mbb"] for r in results], dtype=np.float64)
    return {
        "steps": ref_steps,
        "mean": mbb_arr.mean(axis=0),
        "std": mbb_arr.std(axis=0),
        "min": mbb_arr.min(axis=0),
        "max": mbb_arr.max(axis=0),
    }


def _plot(
    agg: dict[str, np.ndarray],
    per_seed: list[dict[str, Any]],
    cfg: DictConfig,
    save_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    steps = agg["steps"]
    ax.loglog(steps, agg["mean"], marker=".", markersize=3, label=f"{cfg.game_choice} mean")
    ax.fill_between(steps, agg["min"], agg["max"], alpha=0.15, label="min / max")
    for r in per_seed:
        ax.plot(r["steps"], r["expl_mbb"], alpha=0.25, linewidth=0.7)
    ax.axhline(
        y=float(cfg.exit_criterion_mbb), color="red", linestyle="--",
        label=f"GREEN threshold {cfg.exit_criterion_mbb} mbb/g",
    )
    ax.axhline(
        y=float(cfg.phase2_baseline_mbb), color="purple", linestyle=":",
        alpha=0.6,
        label=f"Phase 2 baseline {cfg.phase2_baseline_mbb} mbb/g",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("exploitability (mbb/g)")
    ax.set_title(
        f"Phase 4 Step 2 — Leduc MCCFR ({cfg.game_choice}) "
        f"5 seeds × {cfg.iterations}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="phase4_step2_leduc_mccfr",
)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    wall_start = time.perf_counter()

    # Sanity: reject unknown game_choice early.
    _make_game(str(cfg.game_choice))

    results = _run_all_seeds(cfg)
    agg = _aggregate(results)
    total_sec = time.perf_counter() - wall_start

    finals = [r["final_expl_mbb"] for r in results]
    final_mean = float(np.mean(finals))
    final_std = float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0
    final_min = float(np.min(finals))
    final_max = float(np.max(finals))

    log.info(
        "FINAL game=%s 5-seed mean=%.4f mbb/g std=%.4f min=%.4f max=%.4f wall=%.1fs",
        cfg.game_choice, final_mean, final_std, final_min, final_max, total_sec,
    )
    for r in results:
        log.info(
            "  seed=%d final=%.4f mbb/g iters_to_exit=%d wall=%.1fs",
            r["seed"], r["final_expl_mbb"],
            r["iters_to_exit"], r["total_seconds"],
        )

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name=f"step2-{cfg.game_choice}-summary",
        tags=list(cfg.wandb.tags) + ["summary", str(cfg.game_choice)],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / f"phase4_step2_{cfg.game_choice}.png"
    fig = _plot(agg, results, cfg, png_path)
    wandb.log({"convergence": wandb.Image(str(png_path))})
    wandb.summary.update(
        {
            "final_mean_mbb": final_mean,
            "final_std_mbb": final_std,
            "final_min_mbb": final_min,
            "final_max_mbb": final_max,
            "exit_met": final_mean < float(cfg.exit_criterion_mbb),
            "total_seconds": total_sec,
            "game_choice": str(cfg.game_choice),
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
