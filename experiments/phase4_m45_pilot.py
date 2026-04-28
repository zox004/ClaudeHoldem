"""Phase 4 M4.5.1 — Trained-strategy mini-pilot vs Slumbot.

5-thread × 1k hand × 5-seed parallel pilot. Loads each seed's
trained strategy pickle (M4.5.0a artifacts) and plays against
``slumbot.com`` to measure the **trained** divergence rate (vs
M4.4 uniform baseline 17%) and decide the production path:

* path A (mean divergence < 5%) — trained strategy precise enough,
  proceed to M4.5.3-A 10k production
* ambiguous (5-15%) — mechanism attribution decides; if dominant
  failure mode is ``strategy-miss`` → path A (more train solves it),
  if dominant is ``replay-divergence`` → path B (Schnizlein 2009)
* path B (≥ 15%) — Schnizlein 2009 probabilistic state translation
  introduction (asset #24 candidate) urgent

Three failure modes tracked separately (mentor + claude push-back
M4.5.1 entry):

* ``replay-divergence`` — ``ValueError`` from ``replay_sequence``,
  M4.2 nearest-bucket dispatch bias
* ``harness-desync`` — ``SlumbotError`` from harness state-machine
  checks or server ``error_msg``
* ``transport`` — ``requests.HTTPError`` after retry exhaustion

``StrategyWithMissCounter`` separates **strategy-miss** (uniform
fallback, hand still completes) from **divergence** (hand fails).
This was the spec gap claude caught at M4.5.1 entry.

Run::

    uv run python -m experiments.phase4_m45_pilot
    uv run python -m experiments.phase4_m45_pilot n_hands=10              # smoke
    uv run python -m experiments.phase4_m45_pilot seeds=[42] n_hands=5
"""

from __future__ import annotations

import logging
import pickle
import resource
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import requests
from omegaconf import DictConfig, OmegaConf

from poker_ai.eval.slumbot_client import SlumbotClient, SlumbotError
from poker_ai.eval.slumbot_harness import (
    HandRecord,
    SlumbotHarness,
    mbb_per_hand_winrate,
)
from poker_ai.games.hunl_abstraction import (
    AbstractedHUNLGame,
    AbstractedHUNLState,
)

log = logging.getLogger(__name__)


# Path-decision thresholds (M4.5.1 spec, user-confirmed).
_PATH_A_MAX = 0.05   # < 5% → A
_PATH_B_MIN = 0.15   # ≥ 15% → B


# =============================================================================
# StrategyWithMissCounter — separates strategy-miss from divergence
# =============================================================================
class StrategyWithMissCounter:
    """Wraps a trained strategy ``dict[str, np.ndarray]`` with miss
    accounting. On ``KeyError`` returns a uniform 6-vector (harness
    masks + renormalises legal actions) and increments
    :attr:`n_misses`. The hand continues — strategy-miss is **not** a
    failure, just an abstraction-coverage observation.

    Thread-safety: a single instance is *not* shared across threads in
    M4.5.1 (each thread gets its own seed → its own instance), so
    counter state has no contention.
    """

    __slots__ = (
        "_strategy",
        "_missed_keys",
        "n_lookups",
        "n_misses",
    )

    def __init__(self, strategy: dict[str, np.ndarray]) -> None:
        self._strategy = strategy
        self._missed_keys: set[str] = set()
        self.n_lookups: int = 0
        self.n_misses: int = 0

    @property
    def n_unique_missed_keys(self) -> int:
        return len(self._missed_keys)

    def __call__(self, state: AbstractedHUNLState) -> np.ndarray:
        # AbstractedHUNLState.infoset_key is a @property (no parens).
        key = state.infoset_key
        self.n_lookups += 1
        try:
            value = self._strategy[key]
        except KeyError:
            self.n_misses += 1
            self._missed_keys.add(key)
            return np.full(6, 1.0 / 6.0, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)


# =============================================================================
# Failure mode classifier — 3-mode + unknown
# =============================================================================
def classify_failure_mode(exc: BaseException) -> str:
    """Maps a per-hand exception to one of:

    * ``"replay-divergence"`` — :class:`ValueError` from
      ``replay_sequence`` (M4.2 nearest-bucket dispatch bias)
    * ``"harness-desync"`` — :class:`SlumbotError` from harness or
      client (state-machine desync, server ``error_msg``)
    * ``"transport"`` — :class:`requests.HTTPError` (retry exhaustion)
    * ``"unknown"`` — anything else; flagged for triage

    isinstance order matters because ``SlumbotError`` and
    ``HTTPError`` are not ``ValueError`` subclasses, but checking
    them first guards against accidental subclass changes upstream.
    """
    if isinstance(exc, SlumbotError):
        return "harness-desync"
    if isinstance(exc, requests.HTTPError):
        return "transport"
    if isinstance(exc, ValueError):
        return "replay-divergence"
    return "unknown"


# =============================================================================
# Path decision
# =============================================================================
def decide_path(divergence_rate: float) -> str:
    """Returns ``"A"`` (< 5%), ``"ambiguous"`` (5-15%), or ``"B"``
    (≥ 15%). Raises :class:`ValueError` on out-of-range input.
    """
    if not 0.0 <= divergence_rate <= 1.0:
        raise ValueError(
            f"divergence_rate must be in [0, 1]; got {divergence_rate}"
        )
    if divergence_rate < _PATH_A_MAX:
        return "A"
    if divergence_rate < _PATH_B_MIN:
        return "ambiguous"
    return "B"


# =============================================================================
# Per-seed worker (runs in ThreadPoolExecutor)
# =============================================================================
@dataclass
class SeedResult:
    seed: int
    n_attempted: int
    n_success: int
    failure_modes: dict[str, int]
    strategy_miss_count: int
    strategy_miss_unique_keys: int
    n_lookups: int
    win_chips: list[int]   # per-success-hand client-side chip utilities
    setup_t_s: float
    train_t_s: float = 0.0  # placeholder — not measured here
    pilot_wall_s: float = 0.0
    failure_tracebacks: list[str] = field(default_factory=list)


def _run_seed(
    seed: int,
    strategy_dict: dict[str, np.ndarray],
    cfg_dict: dict[str, Any],
) -> SeedResult:
    """Single-seed pilot worker. Builds its own AbstractedHUNLGame +
    SlumbotClient (token isolation per thread). Sequential
    ``play_one_hand`` loop; per-hand ``hand_sleep_s`` between hands.
    """
    n_hands = int(cfg_dict["n_hands"])
    hand_sleep_s = float(cfg_dict["hand_sleep_s"])
    n_buckets = int(cfg_dict["n_buckets"])
    n_trials = int(cfg_dict["n_trials"])
    postflop_mc_trials = int(cfg_dict["postflop_mc_trials"])
    postflop_threshold_sample_size = int(
        cfg_dict["postflop_threshold_sample_size"]
    )

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

    client = SlumbotClient()
    harness = SlumbotHarness(
        client,
        max_retries=int(cfg_dict["max_retries"]),
        backoff_base=float(cfg_dict["backoff_base"]),
        max_backoff=float(cfg_dict["max_backoff"]),
    )
    strategy_fn = StrategyWithMissCounter(strategy_dict)
    rng = np.random.default_rng(seed)

    failure_modes: dict[str, int] = {
        "replay-divergence": 0,
        "harness-desync": 0,
        "transport": 0,
        "unknown": 0,
    }
    win_chips: list[int] = []
    failure_tracebacks: list[str] = []

    pilot_start = time.perf_counter()
    for i in range(n_hands):
        try:
            rec: HandRecord = harness.play_one_hand(game, strategy_fn, rng)
            win_chips.append(rec.our_utility_chips)
        except Exception as exc:   # noqa: BLE001 — intentional broad catch
            mode = classify_failure_mode(exc)
            failure_modes[mode] += 1
            if mode == "unknown" and len(failure_tracebacks) < 5:
                failure_tracebacks.append(
                    f"hand {i} ({type(exc).__name__}): {exc}\n"
                    f"{traceback.format_exc()}"
                )
        if i < n_hands - 1 and hand_sleep_s > 0.0:
            time.sleep(hand_sleep_s)
        # Lightweight progress log every 100 hands.
        if (i + 1) % 100 == 0:
            log.info(
                "[seed=%d] %d/%d hands  success=%d  modes=%s  "
                "miss_count=%d (unique=%d)",
                seed, i + 1, n_hands, len(win_chips), dict(failure_modes),
                strategy_fn.n_misses, strategy_fn.n_unique_missed_keys,
            )
    pilot_wall = time.perf_counter() - pilot_start

    return SeedResult(
        seed=seed,
        n_attempted=n_hands,
        n_success=len(win_chips),
        failure_modes=failure_modes,
        strategy_miss_count=strategy_fn.n_misses,
        strategy_miss_unique_keys=strategy_fn.n_unique_missed_keys,
        n_lookups=strategy_fn.n_lookups,
        win_chips=win_chips,
        setup_t_s=setup_t,
        pilot_wall_s=pilot_wall,
        failure_tracebacks=failure_tracebacks,
    )


# =============================================================================
# Main
# =============================================================================
def _rss_mb() -> float:
    """Process RSS in MB. macOS reports ru_maxrss in *bytes*, Linux in
    *KB* — split via ``sys.platform`` per resource(3) docs.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0   # Linux KB → MB


def _load_all_strategies(
    strategy_dir: Path, seeds: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    """Loads ``seed{S}.pkl`` for each seed and returns ``{seed: dict}``.
    Logs total RSS after loading (M4.5.1 weak hook #7).
    """
    strategies: dict[int, dict[str, np.ndarray]] = {}
    for s in seeds:
        path = strategy_dir / f"seed{s}.pkl"
        log.info("loading %s (%.0f MB)...", path.name,
                 path.stat().st_size / (1024.0 * 1024.0))
        with path.open("rb") as fh:
            artifact = pickle.load(fh)
        strategies[s] = artifact["strategy"]
        log.info(
            "[seed=%d] loaded n_infosets=%d  T=%d  stack_bb=%d  rss=%.0f MB",
            s, len(artifact["strategy"]), artifact["T"],
            artifact["game_config"]["starting_stack_bb"], _rss_mb(),
        )
    return strategies


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="phase4_m45_pilot",
)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    seeds = [int(s) for s in cfg.seeds]
    strategy_dir = Path(cfg.strategy_dir)
    if not strategy_dir.is_dir():
        raise FileNotFoundError(f"strategy_dir not found: {strategy_dir}")

    strategies = _load_all_strategies(strategy_dir, seeds)
    log.info("post-load total RSS: %.0f MB", _rss_mb())

    wall_start = time.perf_counter()
    results: list[SeedResult] = []
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        futures = {
            pool.submit(_run_seed, s, strategies[s], cfg_dict): s
            for s in seeds
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_total = time.perf_counter() - wall_start

    results.sort(key=lambda r: r.seed)

    # ------------------------------------------------------------------ aggr
    total_attempted = sum(r.n_attempted for r in results)
    total_success = sum(r.n_success for r in results)
    total_failures = total_attempted - total_success
    divergence_rate = (
        total_failures / total_attempted if total_attempted > 0 else 0.0
    )
    aggregated_modes: dict[str, int] = {
        mode: sum(r.failure_modes.get(mode, 0) for r in results)
        for mode in ("replay-divergence", "harness-desync", "transport", "unknown")
    }
    dominant_mode = (
        max(aggregated_modes, key=lambda k: aggregated_modes[k])
        if any(aggregated_modes.values()) else "none"
    )

    # Winrate (secondary informational, M4.5.1 spec push-back).
    all_records: list[HandRecord] = []
    for r in results:
        for chips in r.win_chips:
            all_records.append(
                HandRecord(
                    deal=(0,) * 9,
                    sequence="",
                    client_pos=-1,
                    our_utility_chips=chips,
                    slumbot_winnings=0,
                    sync_check=True,
                )
            )
    if all_records:
        mean_mbb, se_mbb = mbb_per_hand_winrate(all_records)
        ci95_mbb = 1.96 * se_mbb
    else:
        mean_mbb, se_mbb, ci95_mbb = 0.0, 0.0, 0.0

    path = decide_path(divergence_rate)

    # ------------------------------------------------------------------ log
    log.info("===== M4.5.1 5-seed pilot summary =====")
    for r in results:
        rate = (r.n_attempted - r.n_success) / r.n_attempted if r.n_attempted else 0.0
        log.info(
            "  seed=%d  attempted=%d  success=%d  divergence=%.2f%%  "
            "modes=%s  strategy_miss=%d (unique=%d)  pilot_wall=%.1fs",
            r.seed, r.n_attempted, r.n_success, rate * 100,
            dict(r.failure_modes), r.strategy_miss_count,
            r.strategy_miss_unique_keys, r.pilot_wall_s,
        )
    log.info(
        "TOTAL  attempted=%d  success=%d  divergence_rate=%.2f%%  modes=%s",
        total_attempted, total_success, divergence_rate * 100, aggregated_modes,
    )
    log.info("dominant failure mode: %s", dominant_mode)
    log.info(
        "winrate (secondary): mean=%.1f mbb/hand  SE=%.1f  95%%CI=±%.1f  n=%d",
        mean_mbb, se_mbb, ci95_mbb, len(all_records),
    )
    log.info("path decision: %s  (thresholds: A<5%%, B≥15%%)", path)
    if path == "ambiguous":
        log.info(
            "ambiguous → mechanism-attribution: dominant=%s. "
            "spec rule: strategy-miss-dominant → A (more train), "
            "replay-divergence-dominant → B (Schnizlein 2009).",
            dominant_mode,
        )
    log.info(
        "wall total: %.1fs (%.2f min)  final RSS: %.0f MB",
        wall_total, wall_total / 60.0, _rss_mb(),
    )

    # Surface unknown-mode tracebacks (Hook 3 from M4.4 carryover).
    for r in results:
        for tb in r.failure_tracebacks:
            log.warning("[seed=%d unknown-mode failure]\n%s", r.seed, tb)


if __name__ == "__main__":
    main()
