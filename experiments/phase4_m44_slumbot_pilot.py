"""Phase 4 M4.4 — Slumbot live pilot run (infrastructure validation).

End-to-end smoke against ``slumbot.com``: 30 hands with a uniform
strategy, per-hand structured logging, and a session summary that
exposes the three audit hooks the mock-only suite (M4.1-M4.3) cannot:

- **Hook 1** — rate-limit detection: per-hand wall-clock + retry
  trigger counts.
- **Hook 2** — token rotation reality check: whether the server-side
  token actually rotates (vs only documented as such).
- **Hook 3** — sequence-parsing edge cases: any hand that raises
  during ``replay_sequence`` is captured (raw sequence + traceback)
  and the session continues, so a single weird Slumbot string does
  not abort the whole pilot.

Trained-strategy validation is intentionally NOT in M4.4 scope — that
moved to M4.5 production. M4.4 keeps the pilot focused on the
transport / protocol / harness stack alone.

Run::

    uv run python -m experiments.phase4_m44_slumbot_pilot
    uv run python -m experiments.phase4_m44_slumbot_pilot n_hands=5
    uv run python -m experiments.phase4_m44_slumbot_pilot hand_sleep_s=2.0
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from poker_ai.eval.slumbot_client import SlumbotClient
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


@dataclass
class HandLog:
    """Per-hand structured log entry. Mutable for in-place fill."""

    idx: int
    wall_clock_s: float = 0.0
    sequence: str = ""
    client_pos: int = -1
    slumbot_winnings: int | None = None
    sync_check: bool | None = None
    token_before: str | None = None
    token_after: str | None = None
    error_kind: str | None = None
    error_msg: str | None = None
    error_traceback: str | None = None


def uniform_strategy(_state: AbstractedHUNLState) -> np.ndarray:
    """Strategy returning uniform 6-vector. Harness applies legal mask."""
    return np.full(6, 1.0 / 6.0, dtype=np.float64)


def _summarise(
    logs: list[HandLog],
    cfg: DictConfig,
    total_wall_s: float,
) -> dict[str, Any]:
    """Aggregates pilot results into a summary dict."""
    successes = [h for h in logs if h.error_kind is None]
    failures = [h for h in logs if h.error_kind is not None]

    pos0 = sum(1 for h in successes if h.client_pos == 0)
    pos1 = sum(1 for h in successes if h.client_pos == 1)
    sync_ok = sum(1 for h in successes if h.sync_check is True)
    sync_false = sum(1 for h in successes if h.sync_check is False)

    # Token rotation (Hook 2): count distinct (before, after) pairs
    # where after != before.
    rotations = sum(
        1
        for h in successes
        if h.token_before
        and h.token_after
        and h.token_before != h.token_after
    )

    wall_clocks = [h.wall_clock_s for h in successes]
    avg_wall = float(np.mean(wall_clocks)) if wall_clocks else 0.0
    median_wall = float(np.median(wall_clocks)) if wall_clocks else 0.0
    max_wall = float(np.max(wall_clocks)) if wall_clocks else 0.0

    # Winrate from successful hands only.
    successful_records = [
        HandRecord(
            deal=(0,) * 9,   # placeholder for aggregation only
            sequence=h.sequence,
            client_pos=h.client_pos,
            our_utility_chips=int(round((h.slumbot_winnings or 0) / 50)),
            slumbot_winnings=h.slumbot_winnings or 0,
            sync_check=bool(h.sync_check),
        )
        for h in successes
        if h.slumbot_winnings is not None
    ]
    if successful_records:
        mean_mbb, se_mbb = mbb_per_hand_winrate(successful_records)
    else:
        mean_mbb, se_mbb = 0.0, 0.0

    return {
        "n_hands_requested": int(cfg.n_hands),
        "n_hands_attempted": len(logs),
        "n_success": len(successes),
        "n_failure": len(failures),
        "wall_clock_total_s": total_wall_s,
        "hand_avg_s": avg_wall,
        "hand_median_s": median_wall,
        "hand_max_s": max_wall,
        "client_pos_0_count": pos0,
        "client_pos_1_count": pos1,
        "sync_check_pass": sync_ok,
        "sync_check_false": sync_false,
        "token_rotations": rotations,
        "mean_winrate_mbb_per_hand": mean_mbb,
        "se_winrate_mbb_per_hand": se_mbb,
        "failure_kinds": [h.error_kind for h in failures],
    }


def _save_logs(
    logs: list[HandLog],
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    """Writes per-hand JSONL + session summary YAML for post-hoc audit."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    hand_path = out_dir / "hand_logs.jsonl"
    with hand_path.open("w") as fh:
        for h in logs:
            fh.write(json.dumps(asdict(h), default=str) + "\n")

    summary_path = out_dir / "session_summary.yaml"
    summary_path.write_text(OmegaConf.to_yaml(OmegaConf.create(summary)))

    log.info("saved per-hand log: %s", hand_path)
    log.info("saved session summary: %s", summary_path)


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="phase4_m44_slumbot_pilot",
)
def main(cfg: DictConfig) -> None:
    """Live pilot run entry point. Single-threaded; sequential hands.

    Network-bound — wall-clock is dominated by Slumbot HTTP latency,
    not by local CPU. Conservative ``hand_sleep_s`` between hands
    (Hook 1 mitigation).
    """
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    client = SlumbotClient()
    harness = SlumbotHarness(
        client,
        max_retries=int(cfg.max_retries),
        backoff_base=float(cfg.backoff_base),
        max_backoff=float(cfg.max_backoff),
    )
    log.info("building AbstractedHUNLGame (production scale)...")
    setup_t0 = time.perf_counter()
    game = AbstractedHUNLGame(
        n_buckets=int(cfg.n_buckets),
        n_trials=int(cfg.n_trials),
        postflop_mc_trials=int(cfg.postflop_mc_trials),
        postflop_threshold_sample_size=int(cfg.postflop_threshold_sample_size),
        seed=int(cfg.seed),
    )
    log.info("game setup: %.1fs", time.perf_counter() - setup_t0)

    rng = np.random.default_rng(int(cfg.seed))
    n_hands = int(cfg.n_hands)
    hand_sleep_s = float(cfg.hand_sleep_s)

    logs: list[HandLog] = []
    session_t0 = time.perf_counter()
    for i in range(n_hands):
        entry = HandLog(idx=i, token_before=client.token)
        t0 = time.perf_counter()
        try:
            rec = harness.play_one_hand(game, uniform_strategy, rng)
            entry.wall_clock_s = time.perf_counter() - t0
            entry.sequence = rec.sequence
            entry.client_pos = rec.client_pos
            entry.slumbot_winnings = rec.slumbot_winnings
            entry.sync_check = rec.sync_check
            entry.token_after = client.token
            log.info(
                "hand %d/%d  wall=%.2fs  pos=%d  seq=%s  win=%d  sync=%s",
                i + 1, n_hands, entry.wall_clock_s, rec.client_pos,
                rec.sequence, rec.slumbot_winnings, rec.sync_check,
            )
        except Exception as exc:   # noqa: BLE001  (intentional broad catch)
            entry.wall_clock_s = time.perf_counter() - t0
            entry.error_kind = type(exc).__name__
            entry.error_msg = str(exc)
            entry.error_traceback = traceback.format_exc()
            entry.token_after = client.token
            log.error(
                "hand %d/%d FAILED (%s): %s",
                i + 1, n_hands, entry.error_kind, entry.error_msg,
            )
        logs.append(entry)
        if i < n_hands - 1 and hand_sleep_s > 0.0:
            time.sleep(hand_sleep_s)
    session_wall = time.perf_counter() - session_t0

    summary = _summarise(logs, cfg, session_wall)
    log.info("===== M4.4 Pilot Summary =====")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)

    out_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    _save_logs(logs, summary, out_dir)

    # Surface failure tracebacks at the end (Hook 3).
    failures = [h for h in logs if h.error_kind is not None]
    if failures:
        log.warning("===== Failed hands (%d) =====", len(failures))
        for h in failures:
            log.warning(
                "hand %d (%s): %s\n%s",
                h.idx, h.error_kind, h.error_msg, h.error_traceback,
            )


if __name__ == "__main__":
    main()
