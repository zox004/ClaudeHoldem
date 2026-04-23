"""Phase 2 Week 1 Day 4 — Leduc Hold'em Vanilla CFR convergence experiment.

Trains :class:`VanillaCFR` on Leduc for ``cfg.iterations`` cycles, logging
exploitability (chips/g and mbb/g with ``big_blind=2``) and P1 game value
at a mix of dense-prefix and sparse-tail cadences. Reaches the Phase 2
Exit Criterion #1 (< 1 mbb/g @ 100k iter) and records the first iter at
which the threshold is crossed.

Structure reuse
---------------
~90% structural reuse from ``experiments.phase1_kuhn_vanilla`` (Phase 1):
    - Hydra @main + DictConfig
    - wandb.init/finish pattern, separate summary run with embedded image
    - matplotlib figure save under hydra run dir
    - ``should_log`` dense-prefix + sparse-tail cadence

Divergences from Phase 1 Kuhn harness:
    - Game: ``KuhnPoker`` → ``LeducPoker``; ``n_actions=2`` → ``n_actions=3``
    - ``big_blind`` parameter: Kuhn's 1 chip → Leduc's 2 chips (round-1 bet
      size convention); passed through ``exploitability_mbb``
    - Expected runtime: ~2.5h on M1 Pro at 100k iter (vs Kuhn 3.3s)
    - ``log_every=500`` (vs Kuhn 100): Leduc's 288 infosets are decaying at
      similar O(1/√T) rate, but 100k × traversal is much longer → fewer
      snapshots amortize cost. ~200 data points total.

Run:
    uv run python -m experiments.phase2_leduc_vanilla
    uv run python -m experiments.phase2_leduc_vanilla iterations=1000  # smoke
    uv run python -m experiments.phase2_leduc_vanilla wandb.mode=disabled
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

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import exploitability, exploitability_mbb
from poker_ai.games.leduc import LeducPoker

log = logging.getLogger(__name__)


def should_log(iter_idx: int, log_every: int, dense_prefix: int) -> bool:
    """Log every step within the dense prefix; then every ``log_every`` iters.

    ``iter_idx`` is 1-indexed (1..iterations). The dense prefix captures the
    steep initial exploitability decay that would collapse to a single point
    under a pure ``log_every=500`` cadence.
    """
    if iter_idx <= dense_prefix:
        return True
    return iter_idx % log_every == 0


def _run_training(cfg: DictConfig) -> dict[str, np.ndarray | float | int | bool]:
    """Train Vanilla CFR on Leduc and return the convergence history."""
    run_name = f"leduc-vanilla-seed{cfg.seed}"
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
    trainer = VanillaCFR(game=game, n_actions=int(cfg.n_actions))
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

    log.info(
        "iterations=%d  final_expl=%.4f mbb/g  game_value=%.6f  "
        "exit=%s  iters_to_exit=%s  elapsed=%.1fs",
        int(cfg.iterations),
        final_expl_mbb,
        final_game_value,
        exit_criterion_met,
        iters_to_exit,
        total_sec,
    )

    # O(1/√T) sanity snapshots — data points on the log-log curve.
    steps_arr = np.asarray(steps)
    mbb_arr = np.asarray(expl_mbb)
    log.info("O(1/sqrt(T)) snapshots:")
    for snap in (100, 1000, 10000, int(cfg.iterations)):
        hits = np.where(steps_arr == snap)[0]
        if hits.size:
            idx = int(hits[0])
            log.info(
                "  iter=%6d  expl=%.4f mbb/g  game_value=%+.6f",
                snap, mbb_arr[idx], game_values[idx],
            )
    wandb.summary.update(
        {
            "final_exploitability_mbb": final_expl_mbb,
            "final_exploitability_chips": expl_chips[-1],
            "final_game_value": final_game_value,
            "exit_criterion_met": exit_criterion_met,
            "iters_to_exit": iters_to_exit if iters_to_exit is not None else -1,
            "total_seconds": total_sec,
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

    # Plot A: log-log — O(1/√T) appears as a straight line with slope -0.5.
    ax_a = axes[0]
    ax_a.loglog(steps, expl_chips, marker=".", linewidth=1.3, markersize=3)
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("exploitability (chips/g)")
    ax_a.set_title("Convergence (log-log) — O(1/√T) as slope −0.5")
    ax_a.grid(True, which="both", alpha=0.3)

    # Plot B: linear mbb/g with Exit Criterion + expected-final reference lines.
    ax_b = axes[1]
    ax_b.plot(steps, expl_mbb, linewidth=1.4, label="exploitability")
    ax_b.axhline(
        y=float(cfg.exit_criterion_mbb),
        color="red",
        linestyle="--",
        label=f"Exit Criterion ({cfg.exit_criterion_mbb} mbb/g)",
    )
    ax_b.axhline(
        y=float(cfg.expected_final_mbb),
        color="green",
        linestyle=":",
        alpha=0.5,
        label=f"Expected final (~{cfg.expected_final_mbb} mbb/g)",
    )
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("exploitability (mbb/g)")
    ax_b.set_title("Phase 2 Exit Criterion #1 tracking")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Leduc Vanilla CFR — seed={cfg.seed}  iters={cfg.iterations}  bb={cfg.big_blind}",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase2_leduc")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    history = _run_training(cfg)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "leduc_vanilla_convergence.png"
    fig = _plot_convergence(history, cfg, png_path)
    log.info("saved figure: %s", png_path)

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="leduc-vanilla-summary",
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
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
