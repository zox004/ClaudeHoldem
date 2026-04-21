"""Phase 1 Week 1 — RPS self-play convergence experiment.

Runs ``num_players=2`` symmetric self-play with :class:`RegretMatcher` for each
seed in ``cfg.seeds``. Logs per-step L-infinity and L1 distance between each
player's time-averaged strategy and the uniform Nash ``(1/3, 1/3, 1/3)``.

Per-seed W&B runs are separate to enable side-by-side comparison in the
dashboard; a final "summary" run uploads a 3-subplot convergence figure and a
PNG is also saved locally under the Hydra run directory.

Run:
    uv run python -m experiments.phase1_rps_convergence
    uv run python -m experiments.phase1_rps_convergence iterations=100  # dry
    uv run python -m experiments.phase1_rps_convergence wandb.mode=disabled
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.regret_matching import RegretMatcher

log = logging.getLogger(__name__)

# RPS payoff matrix from the row player's perspective: PAYOFF[my][opp].
# Rock beats scissors, paper beats rock, scissors beats paper.
_PAYOFF = np.array(
    [
        [0, -1, 1],   # ROCK
        [1, 0, -1],   # PAPER
        [-1, 1, 0],   # SCISSORS
    ],
    dtype=np.float64,
)


def rps_utilities(opp_action: int) -> np.ndarray:
    return _PAYOFF[:, opp_action].copy()


def _run_seed(
    seed: int,
    iterations: int,
    log_every: int,
    n_actions: int,
    cfg: DictConfig,
) -> dict[str, np.ndarray]:
    """Run a single self-play experiment and return its convergence history."""
    run_name = f"rps-seed{seed}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        tags=list(cfg.wandb.tags) + [f"seed{seed}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert run is not None

    m1 = RegretMatcher(n_actions=n_actions, rng=np.random.default_rng(seed))
    m2 = RegretMatcher(n_actions=n_actions, rng=np.random.default_rng(seed + 1))
    uniform = np.full(n_actions, 1.0 / n_actions)

    steps: list[int] = []
    p1_linf: list[float] = []
    p1_l1: list[float] = []
    p2_linf: list[float] = []
    p2_l1: list[float] = []

    for t in range(1, iterations + 1):
        a1 = m1.sample_action()
        a2 = m2.sample_action()
        m1.update(rps_utilities(a2))
        m2.update(rps_utilities(a1))

        if t % log_every == 0 or t == iterations:
            avg1 = m1.average_strategy()
            avg2 = m2.average_strategy()
            d1 = np.abs(avg1 - uniform)
            d2 = np.abs(avg2 - uniform)
            row = {
                "step": t,
                "p1/l_inf": float(d1.max()),
                "p1/l1": float(d1.sum()),
                "p2/l_inf": float(d2.max()),
                "p2/l1": float(d2.sum()),
                "p1/rock": float(avg1[0]),
                "p1/paper": float(avg1[1]),
                "p1/scissors": float(avg1[2]),
            }
            wandb.log(row, step=t)
            steps.append(t)
            p1_linf.append(row["p1/l_inf"])
            p1_l1.append(row["p1/l1"])
            p2_linf.append(row["p2/l_inf"])
            p2_l1.append(row["p2/l1"])

    final1 = m1.average_strategy()
    final2 = m2.average_strategy()
    log.info(
        "seed=%d final avg  P1=%s  P2=%s  |uniform|_1 P1=%.4f P2=%.4f",
        seed, np.round(final1, 4), np.round(final2, 4),
        float(np.abs(final1 - uniform).sum()),
        float(np.abs(final2 - uniform).sum()),
    )
    wandb.summary.update(
        {
            "final/p1_strategy": final1.tolist(),
            "final/p2_strategy": final2.tolist(),
            "final/p1_l1_to_uniform": float(np.abs(final1 - uniform).sum()),
            "final/p2_l1_to_uniform": float(np.abs(final2 - uniform).sum()),
        }
    )
    wandb.finish()

    return {
        "steps": np.asarray(steps),
        "p1_l_inf": np.asarray(p1_linf),
        "p1_l1": np.asarray(p1_l1),
        "p2_l_inf": np.asarray(p2_linf),
        "p2_l1": np.asarray(p2_l1),
        "final_p1": final1,
        "final_p2": final2,
    }


def _plot_convergence(
    histories: dict[int, dict[str, np.ndarray]],
    save_path: Path,
) -> plt.Figure:
    seeds = sorted(histories.keys())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds, strict=True):
        h = histories[seed]
        ax.plot(h["steps"], h["p1_l1"], label="P1 L1", linewidth=1.4)
        ax.plot(h["steps"], h["p2_l1"], label="P2 L1", linewidth=1.4, linestyle="--")
        ax.plot(h["steps"], h["p1_l_inf"], label="P1 L∞", linewidth=0.9, alpha=0.6)
        ax.plot(h["steps"], h["p2_l_inf"], label="P2 L∞", linewidth=0.9,
                alpha=0.6, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(f"seed {seed}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("|avg_strategy - uniform|")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("RPS self-play: time-averaged strategy → uniform Nash", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase1_rps")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    histories: dict[int, dict[str, np.ndarray]] = {}
    for seed in cfg.seeds:
        histories[int(seed)] = _run_seed(
            seed=int(seed),
            iterations=int(cfg.iterations),
            log_every=int(cfg.log_every),
            n_actions=int(cfg.n_actions),
            cfg=cfg,
        )

    # Hydra already chdir'd into the run dir when chdir=true; we kept chdir=false,
    # so retrieve the run dir explicitly.
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "rps_convergence.png"
    fig = _plot_convergence(histories, png_path)
    log.info("saved figure: %s", png_path)

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="rps-summary",
        tags=list(cfg.wandb.tags) + ["summary"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None
    wandb.log({"convergence_figure": wandb.Image(str(png_path))})
    for seed, h in histories.items():
        wandb.summary[f"final/seed{seed}/p1_l1"] = float(
            np.abs(h["final_p1"] - 1.0 / cfg.n_actions).sum()
        )
        wandb.summary[f"final/seed{seed}/p2_l1"] = float(
            np.abs(h["final_p2"] - 1.0 / cfg.n_actions).sum()
        )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
