"""Phase 2 Week 2 Day 6 — Leduc MCCFR (External Sampling) 5-seed convergence.

Per-seed worker: each runs an independent MCCFR training trajectory under a
distinct ``np.random.Generator`` and logs snapshots to its own W&B run.
Parent process then aggregates (mean / std / max / min per snapshot step)
and publishes a summary W&B run with the combined plot.

Structure: ~90% reuse from ``phase2_leduc_cfr_plus.py``. Divergences:
- Loop over ``cfg.seeds`` (5-seed) instead of single seed
- ``MCCFRExternalSampling`` with ``rng`` injection + ``epsilon``
- ``multiprocessing.Pool`` optional parallelism (``cfg.parallel``)
- Summary plot shows mean convergence + variance band (±std)

Run:
    uv run python -m experiments.phase2_leduc_mccfr                  # full 100k × 5, parallel
    uv run python -m experiments.phase2_leduc_mccfr iterations=1000  # smoke
    uv run python -m experiments.phase2_leduc_mccfr parallel=false   # sequential (debug)
    uv run python -m experiments.phase2_leduc_mccfr wandb.mode=disabled
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
from poker_ai.eval.exploitability import exploitability, exploitability_mbb
from poker_ai.games.leduc import LeducPoker

log = logging.getLogger(__name__)


def should_log(iter_idx: int, log_every: int, dense_prefix: int) -> bool:
    if iter_idx <= dense_prefix:
        return True
    return iter_idx % log_every == 0


def _run_seed(
    seed: int,
    cfg_dict: dict[str, Any],
) -> dict[str, Any]:
    """Single-seed MCCFR training + W&B logging. Runs in its own process when
    ``cfg.parallel`` is ``True``.

    ``cfg_dict`` is a plain dict (picklable) of the resolved Hydra config.
    """
    iterations = int(cfg_dict["iterations"])
    n_actions = int(cfg_dict["n_actions"])
    big_blind = float(cfg_dict["big_blind"])
    epsilon = float(cfg_dict["epsilon"])
    exit_threshold = float(cfg_dict["exit_criterion_mbb"])
    log_every = int(cfg_dict["log_every"])
    dense_prefix = int(cfg_dict["dense_prefix"])
    wandb_cfg = cfg_dict["wandb"]

    run_name = f"leduc-mccfr-seed{seed}"
    run = wandb.init(
        project=wandb_cfg["project"],
        name=run_name,
        tags=list(wandb_cfg["tags"]) + [f"seed{seed}", "per-seed"],
        config=cfg_dict,
        reinit=True,
        mode=wandb_cfg["mode"],
    )
    assert run is not None

    rng = np.random.default_rng(seed)
    game = LeducPoker()
    trainer = MCCFRExternalSampling(
        game=game, n_actions=n_actions, rng=rng, epsilon=epsilon
    )

    steps: list[int] = []
    expl_chips: list[float] = []
    expl_mbb: list[float] = []
    game_values: list[float] = []
    iters_to_exit: int | None = None

    loop_start = time.perf_counter()
    for t in range(1, iterations + 1):
        step_start = time.perf_counter()
        trainer.train(1)
        step_sec = time.perf_counter() - step_start

        if not should_log(t, log_every, dense_prefix):
            continue

        avg_strategy = trainer.average_strategy()
        expl_val = exploitability(game, avg_strategy)
        expl_mbb_val = exploitability_mbb(game, avg_strategy, big_blind=big_blind)
        game_val = trainer.game_value()

        if iters_to_exit is None and expl_mbb_val < exit_threshold:
            iters_to_exit = t

        wandb.log(
            {
                "exploitability_chips": expl_val,
                "exploitability_mbb": expl_mbb_val,
                "game_value": game_val,
                "iter_per_sec": 1.0 / step_sec if step_sec > 0 else float("inf"),
            },
            step=t,
        )
        steps.append(t)
        expl_chips.append(expl_val)
        expl_mbb.append(expl_mbb_val)
        game_values.append(game_val)

    total_sec = time.perf_counter() - loop_start
    final_expl_mbb = expl_mbb[-1]
    final_game_value = game_values[-1]

    wandb.summary.update(
        {
            "final_exploitability_mbb": final_expl_mbb,
            "final_exploitability_chips": expl_chips[-1],
            "final_game_value": final_game_value,
            "exit_criterion_met": final_expl_mbb < exit_threshold,
            "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
            "total_seconds": total_sec,
            "seed": seed,
        }
    )
    wandb.finish()

    return {
        "seed": seed,
        "steps": steps,
        "expl_chips": expl_chips,
        "expl_mbb": expl_mbb,
        "game_values": game_values,
        "final_expl_mbb": final_expl_mbb,
        "final_game_value": final_game_value,
        "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
        "total_seconds": total_sec,
    }


def _run_all_seeds(cfg: DictConfig) -> list[dict[str, Any]]:
    """Run all seeds either in parallel (multiprocessing.Pool) or sequentially."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    seeds = list(cfg.seeds)

    if bool(cfg.parallel):
        log.info("Running %d seeds in parallel (multiprocessing.Pool)", len(seeds))
        # spawn context is safer with W&B/CUDA; for CPU-only CFR, fork is fine.
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(seeds)) as pool:
            results = pool.starmap(
                _run_seed,
                [(int(s), cfg_dict) for s in seeds],
            )
    else:
        log.info("Running %d seeds sequentially", len(seeds))
        results = [_run_seed(int(s), cfg_dict) for s in seeds]

    return list(results)


def _aggregate_histories(
    results: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Align per-seed snapshots by step and compute mean/std/min/max at each step.

    Assumes all seeds use identical log cadence → identical step vectors.
    """
    ref_steps = np.asarray(results[0]["steps"])
    for r in results[1:]:
        if np.array_equal(np.asarray(r["steps"]), ref_steps):
            continue
        raise ValueError("per-seed step vectors diverged; identical cadence required")

    mbb_arr = np.array([r["expl_mbb"] for r in results], dtype=np.float64)
    return {
        "steps": ref_steps,
        "expl_mbb_mean": mbb_arr.mean(axis=0),
        "expl_mbb_std": mbb_arr.std(axis=0),
        "expl_mbb_min": mbb_arr.min(axis=0),
        "expl_mbb_max": mbb_arr.max(axis=0),
    }


def _plot_convergence(
    aggregate: dict[str, np.ndarray],
    per_seed: list[dict[str, Any]],
    cfg: DictConfig,
    save_path: Path,
) -> plt.Figure:
    steps = aggregate["steps"]
    mean = aggregate["expl_mbb_mean"]
    std = aggregate["expl_mbb_std"]
    low = aggregate["expl_mbb_min"]
    high = aggregate["expl_mbb_max"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot A: log-log mean convergence (expected slope ≈ −0.5 for MCCFR).
    ax_a = axes[0]
    ax_a.loglog(steps, mean, marker=".", linewidth=1.3, markersize=3, label="MCCFR mean")
    ax_a.fill_between(steps, low, high, alpha=0.15, label="min / max")
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("exploitability (mbb/g)")
    ax_a.set_title("MCCFR 5-seed Convergence (log-log)")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=8)

    # Plot B: linear mean ± std with reference lines (Vanilla / CFR+ / Exit #1).
    ax_b = axes[1]
    ax_b.plot(steps, mean, linewidth=1.4, label="MCCFR mean")
    ax_b.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
    for r in per_seed:
        ax_b.plot(r["steps"], r["expl_mbb"], alpha=0.3, linewidth=0.7)
    ax_b.axhline(
        y=float(cfg.exit_criterion_mbb),
        color="red",
        linestyle="--",
        label=f"Exit Criterion ({cfg.exit_criterion_mbb} mbb/g)",
    )
    ax_b.axhline(
        y=float(cfg.vanilla_baseline_mbb),
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"Vanilla 100k ({cfg.vanilla_baseline_mbb} mbb/g)",
    )
    ax_b.axhline(
        y=float(cfg.cfr_plus_baseline_mbb),
        color="purple",
        linestyle=":",
        alpha=0.7,
        label=f"CFR+ 2k ({cfg.cfr_plus_baseline_mbb} mbb/g)",
    )
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("exploitability (mbb/g)")
    ax_b.set_title("MCCFR vs baselines")
    ax_b.set_yscale("log")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"Leduc MCCFR ES — 5 seeds × {cfg.iterations} iter  "
        f"ε={cfg.epsilon}  bb={cfg.big_blind}",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase2_leduc_mccfr")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    wall_start = time.perf_counter()
    per_seed = _run_all_seeds(cfg)
    wall_total = time.perf_counter() - wall_start

    final_expls = [float(r["final_expl_mbb"]) for r in per_seed]
    mean_final = float(np.mean(final_expls))
    std_final = float(np.std(final_expls))
    min_final = float(np.min(final_expls))
    max_final = float(np.max(final_expls))
    exit_met = mean_final < float(cfg.exit_criterion_mbb)

    vanilla_baseline = float(cfg.vanilla_baseline_mbb)
    cfr_plus_baseline = float(cfg.cfr_plus_baseline_mbb)
    speedup_vs_vanilla = (
        vanilla_baseline / mean_final if mean_final > 0 else float("inf")
    )

    log.info(
        "5-seed summary @ %d iter:  mean=%.4f  std=%.4f  min=%.4f  max=%.4f  "
        "exit_met=%s  wall=%.1fs  speedup_vs_vanilla=%.2fx",
        int(cfg.iterations),
        mean_final,
        std_final,
        min_final,
        max_final,
        exit_met,
        wall_total,
        speedup_vs_vanilla,
    )
    log.info("Per-seed final expl (mbb/g): %s", final_expls)
    log.info(
        "Exit #1 (<1.0 mbb/g): %s  |  CFR+ baseline (%.4f) beaten: %s",
        "PASS" if exit_met else "FAIL",
        cfr_plus_baseline,
        "YES" if mean_final < cfr_plus_baseline else "NO",
    )

    aggregate = _aggregate_histories(per_seed)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "leduc_mccfr_5seed_convergence.png"
    fig = _plot_convergence(aggregate, per_seed, cfg, png_path)
    log.info("saved figure: %s", png_path)

    # Summary run rolls all seeds into one W&B record.
    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="leduc-mccfr-5seed-summary",
        tags=list(cfg.wandb.tags) + ["summary"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None
    wandb.log({"convergence_figure": wandb.Image(str(png_path))})
    wandb.summary.update(
        {
            "mean_final_exploitability_mbb": mean_final,
            "std_final_exploitability_mbb": std_final,
            "min_final_exploitability_mbb": min_final,
            "max_final_exploitability_mbb": max_final,
            "exit_criterion_met": exit_met,
            "wall_total_seconds": wall_total,
            "speedup_vs_vanilla_100k": speedup_vs_vanilla,
            "seeds": list(cfg.seeds),
            "parallel": bool(cfg.parallel),
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
