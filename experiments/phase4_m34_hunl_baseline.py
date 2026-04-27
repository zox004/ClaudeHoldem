"""Phase 4 M3.4 — AbstractedHUNL 5-seed × 3-anchor LBR baseline.

Per-seed worker:
  1. Build :class:`AbstractedHUNLGame` with production-default abstractor
     params (n_buckets=50, n_trials=10000, postflop_mc_trials=300,
     postflop_threshold_sample_size=10000).
  2. MCCFRExternalSampling cumulative training across ``t_anchors``
     ({1k, 10k, 100k}); after each anchor, snapshot
     ``trainer.average_strategy()`` and run
     :func:`lbr_exploitability` (paired, n_samples=2000) on the
     resulting strategy.
  3. After the final anchor, parse ``infoset_keys`` and bucket the
     visit counts per (round, board_bucket) for occupancy-histogram
     framework reporting (M3.1 measurement, M3.4 framework).

Parent process aggregates 5-seed × 3-anchor LBR mean ± SE, applies the
mentor's bucket-occupancy 4-pattern framework, and emits one summary
W&B run.

Run:
    uv run python -m experiments.phase4_m34_hunl_baseline             # full
    uv run python -m experiments.phase4_m34_hunl_baseline t_anchors=[100,1000]
    uv run python -m experiments.phase4_m34_hunl_baseline parallel=false
    uv run python -m experiments.phase4_m34_hunl_baseline wandb.mode=disabled
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.mccfr import MCCFRExternalSampling
from poker_ai.eval.local_best_response import lbr_exploitability
from poker_ai.games.hunl_abstraction import AbstractedHUNLGame

log = logging.getLogger(__name__)


# =============================================================================
# Per-seed worker
# =============================================================================
def _run_seed(
    seed: int,
    cfg_dict: dict[str, Any],
) -> dict[str, Any]:
    """Single-seed M3.4 baseline run. Designed to be picklable for
    :class:`multiprocessing.Pool`.

    Stages: game setup → MCCFR cumulative-T training → per-anchor LBR
    measurement → final-strategy bucket-occupancy histogram. Each
    stage timed; results returned as a plain dict.
    """
    n_buckets = int(cfg_dict["n_buckets"])
    n_trials = int(cfg_dict["n_trials"])
    postflop_mc_trials = int(cfg_dict["postflop_mc_trials"])
    postflop_threshold_sample_size = int(cfg_dict["postflop_threshold_sample_size"])
    n_actions = int(cfg_dict["n_actions"])
    epsilon = float(cfg_dict["epsilon"])
    t_anchors = [int(t) for t in cfg_dict["t_anchors"]]
    n_samples_lbr = int(cfg_dict["n_samples_lbr"])
    lbr_paired = bool(cfg_dict["lbr_paired"])
    wandb_cfg = cfg_dict["wandb"]

    run_name = f"m34-seed{seed}"
    run = wandb.init(
        project=wandb_cfg["project"],
        name=run_name,
        tags=list(wandb_cfg["tags"]) + [f"seed{seed}", "per-seed"],
        config=cfg_dict,
        reinit=True,
        mode=wandb_cfg["mode"],
    )
    assert run is not None

    # --- Stage 1: game setup
    setup_start = time.perf_counter()
    game = AbstractedHUNLGame(
        n_buckets=n_buckets,
        n_trials=n_trials,
        postflop_mc_trials=postflop_mc_trials,
        postflop_threshold_sample_size=postflop_threshold_sample_size,
        seed=seed,
    )
    setup_t = time.perf_counter() - setup_start
    log.info("[seed=%d] game setup: %.1fs", seed, setup_t)

    # --- Stage 2: MCCFR cumulative training + per-anchor LBR
    rng_mccfr = np.random.default_rng(seed)
    trainer = MCCFRExternalSampling(
        game=game, n_actions=n_actions, rng=rng_mccfr, epsilon=epsilon
    )

    cumulative_t = 0
    anchor_results: list[dict[str, Any]] = []
    pf = game.postflop_abstractor

    for t_target in t_anchors:
        delta_t = t_target - cumulative_t
        train_start = time.perf_counter()
        trainer.train(iterations=delta_t)
        train_t = time.perf_counter() - train_start
        cumulative_t = t_target

        strategy = trainer.average_strategy()
        n_infosets = len(strategy)

        # Reset cache before LBR for clean per-anchor measurement.
        pf._cache_hits = 0
        pf._cache_misses = 0
        pf._cache.clear()

        rng_lbr = np.random.default_rng(seed + 1_000_000 + t_target)
        lbr_start = time.perf_counter()
        mean, se = lbr_exploitability(
            game,
            strategy,
            n_samples=n_samples_lbr,
            rng=rng_lbr,
            paired=lbr_paired,
        )
        lbr_t = time.perf_counter() - lbr_start
        cache_stats = pf.cache_stats()

        anchor = {
            "t": t_target,
            "lbr_mean_chips": float(mean),
            "lbr_se_chips": float(se),
            "lbr_mean_mbb": float(mean) * 1000.0 / 2.0,   # bb=2 in HUNL
            "lbr_se_mbb": float(se) * 1000.0 / 2.0,
            "n_infosets": n_infosets,
            "cache_hits": int(cache_stats["hits"]),
            "cache_misses": int(cache_stats["misses"]),
            "cache_hit_rate": float(cache_stats["hit_rate"]),
            "cache_size": len(pf._cache),
            "wall_mccfr_delta_s": train_t,
            "wall_lbr_s": lbr_t,
        }
        anchor_results.append(anchor)

        log.info(
            "[seed=%d] T=%d  LBR=%.4f±%.4f chips  (%.1f±%.1f mbb/g)  "
            "infosets=%d  hit_rate=%.2f%%  wall_mccfr=%.1fs  wall_lbr=%.1fs",
            seed, t_target, mean, se,
            mean * 1000.0 / 2.0, se * 1000.0 / 2.0,
            n_infosets, cache_stats["hit_rate"] * 100.0,
            train_t, lbr_t,
        )

        wandb.log(
            {
                f"lbr_mean_chips_t{t_target}": mean,
                f"lbr_se_chips_t{t_target}": se,
                f"lbr_mean_mbb_t{t_target}": mean * 1000.0 / 2.0,
                f"n_infosets_t{t_target}": n_infosets,
                f"cache_hit_rate_t{t_target}": cache_stats["hit_rate"],
                f"wall_mccfr_t{t_target}": train_t,
                f"wall_lbr_t{t_target}": lbr_t,
            },
            step=t_target,
        )

    # --- Stage 3: bucket occupancy histogram (final strategy)
    final_strategy = trainer.average_strategy()
    occupancy = _compute_bucket_occupancy(final_strategy, n_buckets=n_buckets)

    wandb.summary.update(
        {
            "seed": seed,
            "setup_t_s": setup_t,
            "final_n_infosets": len(final_strategy),
        }
    )
    wandb.finish()

    return {
        "seed": seed,
        "setup_t_s": setup_t,
        "anchors": anchor_results,
        "occupancy": {r: occ.tolist() for r, occ in occupancy.items()},
    }


def _compute_bucket_occupancy(
    strategy: dict[str, np.ndarray], n_buckets: int
) -> dict[int, np.ndarray]:
    """Returns ``{round_idx: histogram of size n_buckets}`` from
    parsing infoset_keys in the format
    ``"<hole_bucket>|<round>:<board_segment>:<history>"``.

    Round 0 (preflop) bins by ``hole_bucket``; rounds 1-3 bin by
    ``board_bucket`` (single integer, M3.1 contract). Keys that don't
    match the format are skipped silently — they should not appear in
    a properly-trained AbstractedHUNLGame strategy.
    """
    occupancy: dict[int, np.ndarray] = {
        r: np.zeros(n_buckets, dtype=np.int64) for r in (0, 1, 2, 3)
    }
    for key in strategy.keys():
        try:
            head, rest = key.split("|", 1)
            round_part, board_part, _hist = rest.split(":", 2)
            round_idx = int(round_part)
            if round_idx == 0 or board_part == "":
                bucket = int(head)
                if 0 <= bucket < n_buckets:
                    occupancy[0][bucket] += 1
            else:
                bucket = int(board_part)
                if 0 <= bucket < n_buckets and round_idx in (1, 2, 3):
                    occupancy[round_idx][bucket] += 1
        except (ValueError, IndexError):
            continue
    return occupancy


# =============================================================================
# Aggregation across seeds
# =============================================================================
def _aggregate_lbr(
    per_seed: list[dict[str, Any]], t_anchors: list[int]
) -> dict[int, dict[str, float]]:
    """Mean / std / SE across seeds at each T anchor."""
    out: dict[int, dict[str, float]] = {}
    for t in t_anchors:
        means = []
        ses = []
        for r in per_seed:
            for a in r["anchors"]:
                if a["t"] == t:
                    means.append(a["lbr_mean_chips"])
                    ses.append(a["lbr_se_chips"])
                    break
        means_arr = np.asarray(means, dtype=np.float64)
        ses_arr = np.asarray(ses, dtype=np.float64)
        n = len(means_arr)
        # Across-seed sample mean and SEM.
        mean_seeds = float(means_arr.mean())
        std_seeds = float(means_arr.std(ddof=1)) if n >= 2 else 0.0
        sem_seeds = float(std_seeds / np.sqrt(n)) if n >= 2 else 0.0
        # Combined SE: between-seed + average within-seed.
        within_var = float(np.mean(ses_arr ** 2))
        combined_se = float(np.sqrt(sem_seeds ** 2 + within_var / n))
        out[t] = {
            "mean_chips": mean_seeds,
            "std_seeds_chips": std_seeds,
            "sem_seeds_chips": sem_seeds,
            "within_se_chips": float(np.sqrt(within_var)),
            "combined_se_chips": combined_se,
            "mean_mbb": mean_seeds * 1000.0 / 2.0,
            "combined_se_mbb": combined_se * 1000.0 / 2.0,
            "per_seed_means_chips": means_arr.tolist(),
        }
    return out


def _aggregate_occupancy(
    per_seed: list[dict[str, Any]], n_buckets: int
) -> dict[int, np.ndarray]:
    """Sums occupancy across seeds per round."""
    out: dict[int, np.ndarray] = {
        r: np.zeros(n_buckets, dtype=np.int64) for r in (0, 1, 2, 3)
    }
    for r_data in per_seed:
        occ = r_data["occupancy"]
        for round_str, hist in occ.items():
            out[int(round_str)] += np.asarray(hist, dtype=np.int64)
    return out


# =============================================================================
# Bucket-occupancy framework (mentor 4-pattern, M3.4)
# =============================================================================
def _classify_occupancy(
    histogram: np.ndarray,
) -> dict[str, Any]:
    """Classifies a per-round bucket histogram against the M3.4 4-pattern
    framework.

    Patterns:
      "balanced"          — every bucket has > 1% of total
      "collapse"          — 1 or 2 buckets hold > 50% combined
      "sparse_acceptable" — some buckets < 0.1%, rest > 1% (no empties)
      "dead_buckets"      — at least one bucket has 0 visits

    Returns the diagnostic + summary statistics.
    """
    total = int(histogram.sum())
    if total == 0:
        return {
            "pattern": "no_visits",
            "total": 0,
            "max_share": 0.0,
            "min_share": 0.0,
            "n_below_0p1": 0,
            "n_empty": int((histogram == 0).sum()),
        }
    shares = histogram.astype(np.float64) / total
    max_share = float(shares.max())
    min_share = float(shares.min())
    n_empty = int((histogram == 0).sum())
    n_below_0p1 = int((shares < 0.001).sum())

    if n_empty > 0:
        pattern = "dead_buckets"
    else:
        # collapse: top 1-2 buckets hold > 50% combined.
        sorted_shares = np.sort(shares)[::-1]
        top2 = float(sorted_shares[:2].sum())
        if top2 > 0.5:
            pattern = "collapse"
        elif min_share >= 0.01:
            pattern = "balanced"
        else:
            pattern = "sparse_acceptable"
    return {
        "pattern": pattern,
        "total": total,
        "max_share": max_share,
        "min_share": min_share,
        "n_below_0p1": n_below_0p1,
        "n_empty": n_empty,
    }


# =============================================================================
# Plot
# =============================================================================
def _plot_lbr_trend(
    aggregate: dict[int, dict[str, float]],
    t_anchors: list[int],
    save_path: Path,
) -> plt.Figure:
    """Log-x plot of 5-seed LBR mean ± combined SE across T anchors."""
    means = [aggregate[t]["mean_mbb"] for t in t_anchors]
    ses = [aggregate[t]["combined_se_mbb"] for t in t_anchors]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        t_anchors, means, yerr=ses, marker="o", linewidth=1.6,
        markersize=8, capsize=5, label="LBR (5-seed mean ± combined SE)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("MCCFR iterations (T)")
    ax.set_ylabel("LBR exploitability (mbb/g)")
    ax.set_title(
        "Phase 4 M3.4 — AbstractedHUNL LBR baseline\n"
        f"(n_buckets=50, MCCFR 5-seed × n_samples_lbr=2000)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _plot_occupancy(
    occupancy_total: dict[int, np.ndarray],
    n_buckets: int,
    save_path: Path,
) -> plt.Figure:
    """Bar plot of bucket occupancy per round (5-seed sum)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    round_labels = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    for round_idx, ax in enumerate(axes.flat):
        hist = occupancy_total[round_idx]
        total = hist.sum()
        if total == 0:
            ax.set_title(f"{round_labels[round_idx]} — no visits")
            ax.bar(range(n_buckets), hist, alpha=0.7)
        else:
            shares = hist.astype(np.float64) / total
            ax.bar(range(n_buckets), shares, alpha=0.7)
            classification = _classify_occupancy(hist)
            ax.set_title(
                f"{round_labels[round_idx]} ({total:,} visits, "
                f"pattern={classification['pattern']})"
            )
            ax.axhline(0.01, linestyle="--", color="red", alpha=0.4,
                       label="1% threshold")
            ax.axhline(0.001, linestyle=":", color="orange", alpha=0.4,
                       label="0.1% threshold")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("bucket index")
        ax.set_ylabel("visit share")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Phase 4 M3.4 — Bucket occupancy histogram (5-seed sum)",
        y=1.0, fontsize=11,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# Main
# =============================================================================
def _run_all_seeds(cfg: DictConfig) -> list[dict[str, Any]]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    seeds = list(cfg.seeds)

    if bool(cfg.parallel):
        log.info("Running %d seeds in parallel (multiprocessing.Pool)", len(seeds))
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


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="phase4_m34_hunl_baseline",
)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    wall_start = time.perf_counter()
    per_seed = _run_all_seeds(cfg)
    wall_total = time.perf_counter() - wall_start

    t_anchors = [int(t) for t in cfg.t_anchors]
    aggregate = _aggregate_lbr(per_seed, t_anchors)
    occupancy_total = _aggregate_occupancy(per_seed, int(cfg.n_buckets))

    # Bucket-occupancy framework classification per round.
    classifications: dict[int, dict[str, Any]] = {}
    for r in (0, 1, 2, 3):
        classifications[r] = _classify_occupancy(occupancy_total[r])

    # T-trend monotonic check (assets #11 + #18).
    means_chain = [aggregate[t]["mean_chips"] for t in t_anchors]
    ratios = []
    for i in range(len(means_chain) - 1):
        if means_chain[i + 1] != 0:
            ratios.append(means_chain[i] / means_chain[i + 1])
    t_max = t_anchors[-1]
    t_min = t_anchors[0]
    overall_ratio = (
        means_chain[0] / means_chain[-1]
        if means_chain[-1] != 0
        else float("inf")
    )

    log.info("====== M3.4 5-seed × %d-anchor summary ======", len(t_anchors))
    for t in t_anchors:
        a = aggregate[t]
        log.info(
            "  T=%d  mean=%.4f chips (%.1f mbb/g)  combined_SE=%.4f  "
            "per_seed=%s",
            t, a["mean_chips"], a["mean_mbb"], a["combined_se_chips"],
            [f"{v:+.3f}" for v in a["per_seed_means_chips"]],
        )
    log.info("Asset #11 / #18 trend: T=%d→%d ratio=%.2fx  (≥3× target)",
             t_min, t_max, overall_ratio)
    log.info("Wall total: %.1fs (%.2f min)", wall_total, wall_total / 60.0)

    # Bucket occupancy log
    log.info("====== Bucket occupancy classification ======")
    round_labels = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    for r, c in classifications.items():
        log.info(
            "  %s: pattern=%s  total=%d  max_share=%.3f  min_share=%.4f  "
            "below_0.1%%=%d  empty=%d",
            round_labels[r], c["pattern"], c["total"],
            c["max_share"], c["min_share"],
            c["n_below_0p1"], c["n_empty"],
        )

    # Plots.
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    lbr_png = run_dir / "lbr_trend.png"
    occ_png = run_dir / "bucket_occupancy.png"
    fig_lbr = _plot_lbr_trend(aggregate, t_anchors, lbr_png)
    fig_occ = _plot_occupancy(occupancy_total, int(cfg.n_buckets), occ_png)
    log.info("saved: %s", lbr_png)
    log.info("saved: %s", occ_png)

    # Summary W&B run.
    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="m34-summary",
        tags=list(cfg.wandb.tags) + ["summary"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None

    flat_summary: dict[str, Any] = {
        "wall_total_s": wall_total,
        "asset_11_18_overall_ratio": overall_ratio,
        "n_seeds": len(cfg.seeds),
    }
    for t in t_anchors:
        a = aggregate[t]
        flat_summary[f"lbr_mean_chips_t{t}"] = a["mean_chips"]
        flat_summary[f"lbr_combined_se_chips_t{t}"] = a["combined_se_chips"]
        flat_summary[f"lbr_mean_mbb_t{t}"] = a["mean_mbb"]
    for r, c in classifications.items():
        flat_summary[f"occupancy_pattern_round{r}"] = c["pattern"]
        flat_summary[f"occupancy_max_share_round{r}"] = c["max_share"]
        flat_summary[f"occupancy_n_empty_round{r}"] = c["n_empty"]

    wandb.log({"lbr_trend_figure": wandb.Image(str(lbr_png)),
               "occupancy_figure": wandb.Image(str(occ_png))})
    wandb.summary.update(flat_summary)
    wandb.finish()
    plt.close(fig_lbr)
    plt.close(fig_occ)


if __name__ == "__main__":
    main()
