"""Phase 2 Week 1 Day 5+ — Leduc Hold'em CFR+ convergence experiment.

CFR+ (Tammelin 2014) on Leduc. Same harness shape as phase2_leduc_vanilla.py
(~95% reuse); swaps the trainer class and adjusts thresholds.

Empirical hook: Day 5 audit established CFR+ reaches 0.042 mbb/g at T=2k
(151× speedup vs Vanilla's 1.48 mbb/g @ 100k baseline). This 100k run
verifies Tammelin Figure 2 extrapolation (sub-0.05 mbb/g @ 100k) + formally
closes Phase 2 Exit #1 (< 1 mbb/g — rescued via CFR+) and Exit #2 (5-10×
speedup — expected 100× actual).

Run:
    uv run python -m experiments.phase2_leduc_cfr_plus
    uv run python -m experiments.phase2_leduc_cfr_plus iterations=1000  # smoke
    uv run python -m experiments.phase2_leduc_cfr_plus wandb.mode=disabled
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.eval.exploitability import exploitability, exploitability_mbb
from poker_ai.games.leduc import LeducPoker

log = logging.getLogger(__name__)


def should_log(iter_idx: int, log_every: int, dense_prefix: int) -> bool:
    """Dense prefix + sparse tail (matches Phase 2 Vanilla harness)."""
    if iter_idx <= dense_prefix:
        return True
    return iter_idx % log_every == 0


def _run_training(cfg: DictConfig) -> dict[str, np.ndarray | float | int | bool]:
    """Train CFR+ on Leduc and return the convergence history."""
    run_name = f"leduc-cfr-plus-seed{cfg.seed}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        tags=list(cfg.wandb.tags) + [f"seed{cfg.seed}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert run is not None

    np.random.seed(int(cfg.seed))
    game = LeducPoker()
    trainer = CFRPlus(game=game, n_actions=int(cfg.n_actions))
    big_blind = float(cfg.big_blind)

    steps: list[int] = []
    expl_chips: list[float] = []
    expl_mbb: list[float] = []
    game_values: list[float] = []

    iters_to_exit: int | None = None
    exit_threshold = float(cfg.exit_criterion_mbb)

    loop_start = time.perf_counter()
    for t in range(1, int(cfg.iterations) + 1):
        step_start = time.perf_counter()
        trainer.train(1)
        step_sec = time.perf_counter() - step_start

        if not should_log(t, int(cfg.log_every), int(cfg.dense_prefix)):
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
    exit_criterion_met = final_expl_mbb < exit_threshold
    vanilla_baseline_mbb = float(cfg.vanilla_baseline_mbb)
    speedup_vs_vanilla = vanilla_baseline_mbb / final_expl_mbb if final_expl_mbb > 0 else float("inf")

    log.info(
        "iterations=%d  final_expl=%.6f mbb/g  game_value=%.6f  "
        "exit=%s  iters_to_exit=%s  elapsed=%.1fs  speedup_vs_vanilla=%.2fx",
        int(cfg.iterations),
        final_expl_mbb,
        final_game_value,
        exit_criterion_met,
        iters_to_exit,
        total_sec,
        speedup_vs_vanilla,
    )

    # Snapshots + slope (for Tammelin-compare + Exit #2 speedup at benchmark points).
    steps_arr = np.asarray(steps)
    mbb_arr = np.asarray(expl_mbb)
    log.info("CFR+ O(1/T?) snapshots + Vanilla-baseline speedup:")
    for snap in (100, 1000, 10000, int(cfg.iterations)):
        hits = np.where(steps_arr == snap)[0]
        if hits.size:
            idx = int(hits[0])
            speedup_at_snap = vanilla_baseline_mbb / mbb_arr[idx] if mbb_arr[idx] > 0 else float("inf")
            log.info(
                "  iter=%6d  expl=%.6f mbb/g  speedup=%.2fx vs Vanilla-100k-baseline",
                snap, mbb_arr[idx], speedup_at_snap,
            )

    wandb.summary.update(
        {
            "final_exploitability_mbb": final_expl_mbb,
            "final_exploitability_chips": expl_chips[-1],
            "final_game_value": final_game_value,
            "exit_criterion_met": exit_criterion_met,
            "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
            "total_seconds": total_sec,
            "speedup_vs_vanilla_100k_baseline": speedup_vs_vanilla,
            "vanilla_baseline_mbb": vanilla_baseline_mbb,
        }
    )
    wandb.finish()

    return {
        "steps": np.asarray(steps),
        "expl_chips": np.asarray(expl_chips),
        "expl_mbb": np.asarray(expl_mbb),
        "game_values": np.asarray(game_values),
        "final_expl_mbb": final_expl_mbb,
        "final_game_value": final_game_value,
        "exit_criterion_met": exit_criterion_met,
        "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
        "speedup_vs_vanilla": speedup_vs_vanilla,
    }


def _plot_convergence(
    history: dict[str, np.ndarray | float | int | bool],
    cfg: DictConfig,
    save_path: Path,
) -> plt.Figure:
    steps = history["steps"]
    expl_chips = history["expl_chips"]
    expl_mbb = history["expl_mbb"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot A: log-log. CFR+ is expected to show slope near −1 (O(1/T))
    # versus Vanilla's O(1/√T) slope near −0.5.
    ax_a = axes[0]
    ax_a.loglog(steps, expl_chips, marker=".", linewidth=1.3, markersize=3, label="CFR+")
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("exploitability (chips/g)")
    ax_a.set_title("CFR+ Convergence (log-log)")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=8)

    # Plot B: linear mbb/g with Exit Criterion + Vanilla baseline reference.
    ax_b = axes[1]
    ax_b.plot(steps, expl_mbb, linewidth=1.4, label="CFR+ exploitability")
    ax_b.axhline(
        y=float(cfg.exit_criterion_mbb),
        color="red",
        linestyle="--",
        label=f"Exit Criterion (CFR+ target {cfg.exit_criterion_mbb} mbb/g)",
    )
    ax_b.axhline(
        y=float(cfg.vanilla_baseline_mbb),
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"Vanilla 100k baseline ({cfg.vanilla_baseline_mbb} mbb/g)",
    )
    ax_b.axhline(
        y=float(cfg.expected_final_mbb),
        color="green",
        linestyle=":",
        alpha=0.5,
        label=f"Expected CFR+ @ 100k (~{cfg.expected_final_mbb} mbb/g)",
    )
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("exploitability (mbb/g)")
    ax_b.set_title("Exit #1 + Exit #2 tracking")
    ax_b.set_yscale("log")  # Log scale to show CFR+ dominance clearly
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Leduc CFR+ — seed={cfg.seed}  iters={cfg.iterations}  bb={cfg.big_blind}",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase2_leduc_cfr_plus")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    history = _run_training(cfg)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "leduc_cfr_plus_convergence.png"
    fig = _plot_convergence(history, cfg, png_path)
    log.info("saved figure: %s", png_path)

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="leduc-cfr-plus-summary",
        tags=list(cfg.wandb.tags) + ["summary"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None
    wandb.log({"convergence_figure": wandb.Image(str(png_path))})
    wandb.summary.update(
        {
            "final_exploitability_mbb": float(history["final_expl_mbb"]),
            "final_game_value": float(history["final_game_value"]),
            "exit_criterion_met": bool(history["exit_criterion_met"]),
            "iters_to_exit": int(history["iters_to_exit"]),
            "speedup_vs_vanilla": float(history["speedup_vs_vanilla"]),
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
