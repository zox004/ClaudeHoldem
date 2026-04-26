"""Unit tests for postflop board abstraction (Phase 4 M3.1).

These tests are written **before** the ``PostflopBoardAbstractor``
implementation lands in :mod:`poker_ai.games.hunl_abstraction`. They
encode the design contract agreed with the mentor:

- Round-aware single class (flop=1, turn=2, river=3); preflop is out
  of scope and must raise.
- Lazy cache keyed by canonical (hole, board, round_idx) signature
  with ``cache_stats()`` reporting.
- Same E[HS²]-percentile pattern as M2's :class:`HUNLCardAbstractor`,
  but conditioned on the public board.
- ``AbstractedHUNLState.infoset_key`` switches the postflop board
  representation from the raw card-id comma list (M2) to the bucket
  index produced by this abstractor (M3.1). Preflop key is unchanged.

TDD intent: every test below currently fails because
``PostflopBoardAbstractor`` does not yet exist and
``AbstractedHUNLState.infoset_key`` still emits raw board strings.
The implementation in M3.1 must make the entire file go green.
"""

from __future__ import annotations

import numpy as np
import pytest


# Card-id helper for fixtures (mirrors test_hunl_abstraction.py).
_R = {c: i for i, c in enumerate("23456789TJQKA")}
_S = {c: i for i, c in enumerate("cdhs")}


def cid(s: str) -> int:
    return _R[s[0]] * 4 + _S[s[1]]


def _disjoint_filler(used: set[int], n: int) -> list[int]:
    """Returns n distinct card-ids not in ``used``."""
    out: list[int] = []
    for c in range(52):
        if c in used:
            continue
        out.append(c)
        if len(out) == n:
            return out
    raise ValueError("not enough cards left to fill")


# Reduced configuration used across the file to keep precompute fast.
# threshold_sample_size must be ≥ n_buckets (otherwise some buckets get
# zero training samples). 60 with 10 buckets and 50 mc_trials brings
# the per-round precompute well under a second on M1 Pro.
_FAST_KW = dict(n_buckets=10, mc_trials=50, threshold_sample_size=60, seed=42)


# =============================================================================
# Construction + thresholds derivation
# =============================================================================
class TestPostflopBoardAbstractorConstruction:
    def test_constructs_with_default_args(self) -> None:
        """Smoke: construction with the fast test kwargs succeeds."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        ab = PostflopBoardAbstractor(**_FAST_KW)
        assert ab.n_buckets == 10
        assert ab.mc_trials == 50

    def test_thresholds_for_three_rounds(self) -> None:
        """Three rounds (flop, turn, river) get their own threshold list."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        ab = PostflopBoardAbstractor(**_FAST_KW)
        # Implementation may expose thresholds as a dict keyed by round
        # index (1=flop, 2=turn, 3=river). Test by attribute presence.
        assert hasattr(ab, "thresholds")
        for r in (1, 2, 3):
            t = ab.thresholds[r]
            # Each round should hold n_buckets-1 cut points (or n_buckets
            # depending on convention); minimum requirement is monotone
            # non-decreasing and length within {n_buckets-1, n_buckets}.
            assert len(t) in (ab.n_buckets - 1, ab.n_buckets)
            for i in range(len(t) - 1):
                assert t[i] <= t[i + 1], "thresholds must be non-decreasing"

    def test_invalid_n_buckets_raises(self) -> None:
        """n_buckets ∉ [1, 169] is rejected."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        with pytest.raises(ValueError, match="n_buckets"):
            PostflopBoardAbstractor(
                n_buckets=0, mc_trials=50, threshold_sample_size=60, seed=42
            )
        with pytest.raises(ValueError, match="n_buckets"):
            PostflopBoardAbstractor(
                n_buckets=200, mc_trials=50, threshold_sample_size=60, seed=42
            )

    def test_invalid_mc_trials_raises(self) -> None:
        """mc_trials < 1 is rejected."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        with pytest.raises(ValueError, match="mc_trials"):
            PostflopBoardAbstractor(
                n_buckets=10, mc_trials=0, threshold_sample_size=60, seed=42
            )

    def test_invalid_threshold_sample_size_raises(self) -> None:
        """threshold_sample_size < n_buckets is rejected (need ≥ 1
        sample per bucket to compute percentile cuts)."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        with pytest.raises(ValueError, match="threshold_sample_size"):
            PostflopBoardAbstractor(
                n_buckets=10, mc_trials=50, threshold_sample_size=5, seed=42
            )


# =============================================================================
# bucket() round dispatch
# =============================================================================
class TestPostflopBoardAbstractorBucket:
    @pytest.fixture(scope="class")
    def ab(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        return PostflopBoardAbstractor(**_FAST_KW)

    def test_flop_bucket_in_range(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Flop (round=1, 3-card board) returns a bucket in [0, n_buckets)."""
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"))
        b = ab.bucket(hole, board, 1)
        assert 0 <= b < ab.n_buckets

    def test_turn_bucket_in_range(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Turn (round=2, 4-card board) returns a bucket in [0, n_buckets)."""
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"), cid("Td"))
        b = ab.bucket(hole, board, 2)
        assert 0 <= b < ab.n_buckets

    def test_river_bucket_in_range(self, ab) -> None:  # type: ignore[no-untyped-def]
        """River (round=3, 5-card board) returns a bucket in [0, n_buckets)."""
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"), cid("Td"), cid("Qs"))
        b = ab.bucket(hole, board, 3)
        assert 0 <= b < ab.n_buckets

    def test_preflop_round_idx_raises(self, ab) -> None:  # type: ignore[no-untyped-def]
        """round_idx=0 (preflop) is out of scope for this abstractor."""
        hole = (cid("As"), cid("Kh"))
        with pytest.raises(ValueError, match="round"):
            ab.bucket(hole, (), 0)

    def test_invalid_round_idx_raises(self, ab) -> None:  # type: ignore[no-untyped-def]
        """round_idx ∉ {1, 2, 3} is rejected."""
        hole = (cid("As"), cid("Kh"))
        with pytest.raises(ValueError, match="round"):
            ab.bucket(hole, (cid("2c"), cid("7d"), cid("9h")), 4)


# =============================================================================
# Lazy cache behaviour
# =============================================================================
class TestPostflopBoardAbstractorCache:
    @pytest.fixture(scope="class")
    def ab(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        return PostflopBoardAbstractor(**_FAST_KW)

    def test_repeat_call_hits_cache(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Same (hole, board, round) twice → second call increments hits."""
        # Use a fresh abstractor so cache stats are clean.
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"))
        local.bucket(hole, board, 1)
        local.bucket(hole, board, 1)
        stats = local.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_distinct_inputs_increment_misses(self) -> None:
        """Two distinct (hole, board, round) triples → both miss."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        h1 = (cid("As"), cid("Kh"))
        h2 = (cid("Qs"), cid("Qh"))
        board = (cid("2c"), cid("7d"), cid("9h"))
        local.bucket(h1, board, 1)
        local.bucket(h2, board, 1)
        stats = local.cache_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0

    def test_cache_returns_consistent_value(self) -> None:
        """A cached lookup returns the exact same bucket as the first
        compute — caching must not perturb the assignment."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"))
        b_first = local.bucket(hole, board, 1)
        b_second = local.bucket(hole, board, 1)
        assert b_first == b_second


# =============================================================================
# cache_stats() property
# =============================================================================
class TestPostflopBoardAbstractorCacheStats:
    def test_initial_stats_zero(self) -> None:
        """A fresh abstractor reports hits=0, misses=0, total=0, hit_rate=0.0."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        s = local.cache_stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["total"] == 0
        assert s["hit_rate"] == 0.0

    def test_one_miss_one_hit_gives_50pct(self) -> None:
        """1 miss + 1 hit → hit_rate=0.5."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        hole = (cid("As"), cid("Kh"))
        board = (cid("2c"), cid("7d"), cid("9h"))
        local.bucket(hole, board, 1)   # miss
        local.bucket(hole, board, 1)   # hit
        s = local.cache_stats()
        assert s["total"] == 2
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_only_misses_gives_zero_hit_rate(self) -> None:
        """5 misses + 0 hits → hit_rate=0.0."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        local = PostflopBoardAbstractor(**_FAST_KW)
        hole = (cid("As"), cid("Kh"))
        # Five distinct boards.
        for board in [
            (cid("2c"), cid("7d"), cid("9h")),
            (cid("3c"), cid("7d"), cid("9h")),
            (cid("4c"), cid("7d"), cid("9h")),
            (cid("5c"), cid("7d"), cid("9h")),
            (cid("6c"), cid("7d"), cid("9h")),
        ]:
            local.bucket(hole, board, 1)
        s = local.cache_stats()
        assert s["misses"] == 5
        assert s["hits"] == 0
        assert s["hit_rate"] == 0.0


# =============================================================================
# Determinism
# =============================================================================
class TestPostflopBoardAbstractorDeterminism:
    def test_same_seed_same_thresholds(self) -> None:
        """Same seed → identical threshold lists for all three rounds."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        a = PostflopBoardAbstractor(**_FAST_KW)
        b = PostflopBoardAbstractor(**_FAST_KW)
        for r in (1, 2, 3):
            assert list(a.thresholds[r]) == list(b.thresholds[r])

    def test_same_seed_same_bucket(self) -> None:
        """Same seed + same input → same bucket index."""
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        a = PostflopBoardAbstractor(**_FAST_KW)
        b = PostflopBoardAbstractor(**_FAST_KW)
        hole = (cid("As"), cid("Ah"))
        board = (cid("2c"), cid("3d"), cid("4h"))
        assert a.bucket(hole, board, 1) == b.bucket(hole, board, 1)


# =============================================================================
# Extreme hands stable (audit #20 transfer from M2)
# =============================================================================
class TestPostflopBoardAbstractorExtremeHands:
    @pytest.fixture(scope="class")
    def ab(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        # Slightly higher mc_trials so quad-vs-trash is stably separated.
        return PostflopBoardAbstractor(
            n_buckets=10, mc_trials=200, threshold_sample_size=200, seed=42
        )

    def test_quads_lands_in_top_bucket(self, ab) -> None:  # type: ignore[no-untyped-def]
        """AsAh on flop AdAc2s (four-of-a-kind aces) → top bucket."""
        hole = (cid("As"), cid("Ah"))
        board = (cid("Ad"), cid("Ac"), cid("2s"))
        b = ab.bucket(hole, board, 1)
        assert b >= ab.n_buckets - 2, (
            f"quads should be near top bucket; got {b}/{ab.n_buckets}"
        )

    def test_dominated_lands_in_bottom_bucket(self, ab) -> None:  # type: ignore[no-untyped-def]
        """2c3d on flop KsKhKd (kings full beating any board pair) →
        bottom bucket because the hole barely contributes."""
        hole = (cid("2c"), cid("3d"))
        board = (cid("Ks"), cid("Kh"), cid("Kd"))
        b = ab.bucket(hole, board, 1)
        assert b <= 1, (
            f"dominated junk should be near bottom bucket; got {b}"
        )

    def test_extreme_hands_have_separation(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Top extreme bucket and bottom extreme bucket differ by at
        least n_buckets - 5 (transfer of M2 audit #20 invariant)."""
        top_hole = (cid("As"), cid("Ah"))
        top_board = (cid("Ad"), cid("Ac"), cid("2s"))
        bot_hole = (cid("2c"), cid("3d"))
        bot_board = (cid("Ks"), cid("Kh"), cid("Kd"))
        top_b = ab.bucket(top_hole, top_board, 1)
        bot_b = ab.bucket(bot_hole, bot_board, 1)
        assert top_b - bot_b >= ab.n_buckets - 5


# =============================================================================
# Input validation
# =============================================================================
class TestPostflopBoardAbstractorValidation:
    @pytest.fixture(scope="class")
    def ab(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor

        return PostflopBoardAbstractor(**_FAST_KW)

    def test_wrong_hole_length_raises(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Hole must contain exactly 2 cards."""
        with pytest.raises(ValueError, match="hole"):
            ab.bucket((cid("As"),), (cid("2c"), cid("7d"), cid("9h")), 1)
        with pytest.raises(ValueError, match="hole"):
            ab.bucket(
                (cid("As"), cid("Kh"), cid("Qd")),
                (cid("2c"), cid("7d"), cid("9h")),
                1,
            )

    def test_wrong_board_length_for_round_raises(self, ab) -> None:  # type: ignore[no-untyped-def]
        """round=1 expects 3-card board; passing 4 cards is invalid."""
        with pytest.raises(ValueError, match="board"):
            ab.bucket(
                (cid("As"), cid("Kh")),
                (cid("2c"), cid("7d"), cid("9h"), cid("Td")),
                1,
            )

    def test_duplicate_card_raises(self, ab) -> None:  # type: ignore[no-untyped-def]
        """Duplicate card across hole+board is rejected."""
        with pytest.raises(ValueError, match="duplicate|cards|differ"):
            ab.bucket(
                (cid("As"), cid("Kh")),
                (cid("As"), cid("7d"), cid("9h")),   # As repeated
                1,
            )


# =============================================================================
# AbstractedHUNLState.infoset_key change (M3.1 contract)
# =============================================================================
class TestAbstractedHUNLStateInfosetKeyM31:
    @pytest.fixture(scope="class")
    def game(self):  # type: ignore[no-untyped-def]
        from poker_ai.games.hunl_abstraction import AbstractedHUNLGame

        # AbstractedHUNLGame is expected to wire in a PostflopBoardAbstractor
        # automatically (M3.1.9 integration). Construction args mirror M2.
        return AbstractedHUNLGame(n_buckets=10, n_trials=200, seed=42)

    def _drive_to_postflop(self, game, deal):  # type: ignore[no-untyped-def]
        """Helper: SB CALL + BB CHECK closes preflop, advancing to flop."""
        from poker_ai.games.hunl_abstraction import AbstractedHUNLAction

        state = game.state_from_deal(deal)
        # Preflop: SB (player 1) CALL.
        state = state.next_state(AbstractedHUNLAction.CALL)
        # Preflop: BB (player 0) CHECK (modeled as CALL when no bet open).
        state = state.next_state(AbstractedHUNLAction.CALL)
        return state

    def test_preflop_key_unchanged(self, game) -> None:  # type: ignore[no-untyped-def]
        """Preflop infoset_key still uses the M2 layout
        ``"<bucket>|0::<history>"`` (board empty) — no postflop bucket
        appears at preflop."""
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        key = state.infoset_key
        # Layout: "<hole_bucket>|0:<board_repr>:<hist>" with empty board.
        head, rest = key.split("|", 1)
        round_part, board_part, _hist_part = rest.split(":", 2)
        assert head.isdigit()
        assert round_part == "0"
        assert board_part == "", (
            f"preflop key board segment must be empty, got {board_part!r}"
        )

    def test_flop_key_uses_board_bucket(self, game) -> None:  # type: ignore[no-untyped-def]
        """Postflop board segment is now a single integer bucket index
        (M3.1 contract), not a comma-list of card-ids."""
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        flop_state = self._drive_to_postflop(game, deal)
        # Sanity: we should be at round 1 with 3 board cards visible.
        assert flop_state._raw.current_round == 1
        assert len(flop_state._raw.board_cards) == 3

        key = flop_state.infoset_key
        _, rest = key.split("|", 1)
        round_part, board_part, _hist = rest.split(":", 2)
        assert round_part == "1"
        # board_part must be a single integer (not a comma-list).
        assert "," not in board_part, (
            f"flop board segment must be a single bucket index, got "
            f"{board_part!r}"
        )
        assert board_part.isdigit()
        # And it must be < n_buckets of the postflop abstractor.
        assert 0 <= int(board_part) < 10

    def test_turn_and_river_key_use_board_bucket(self, game) -> None:  # type: ignore[no-untyped-def]
        """Same single-int board segment contract applies to turn & river."""
        from poker_ai.games.hunl_abstraction import AbstractedHUNLAction

        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        # Drive to flop, then close flop with check-check to reach turn.
        state = self._drive_to_postflop(game, deal)
        state = state.next_state(AbstractedHUNLAction.CALL)   # BB check
        state = state.next_state(AbstractedHUNLAction.CALL)   # SB check → turn
        assert state._raw.current_round == 2
        key_turn = state.infoset_key
        _, rest = key_turn.split("|", 1)
        round_part, board_part, _hist = rest.split(":", 2)
        assert round_part == "2"
        assert "," not in board_part and board_part.isdigit()

        # Close turn → river.
        state = state.next_state(AbstractedHUNLAction.CALL)
        state = state.next_state(AbstractedHUNLAction.CALL)
        assert state._raw.current_round == 3
        key_river = state.infoset_key
        _, rest = key_river.split("|", 1)
        round_part, board_part, _hist = rest.split(":", 2)
        assert round_part == "3"
        assert "," not in board_part and board_part.isdigit()

    def test_m2_compat_break_is_visible(self, game) -> None:  # type: ignore[no-untyped-def]
        """M3 postflop key must NOT match the legacy M2 raw card-id-list
        layout. This is the explicit compatibility-break test the design
        spec calls out: existing M2 trainer strategies cannot be loaded
        into an M3 trainer because their keys differ."""
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        flop_state = self._drive_to_postflop(game, deal)
        m3_key = flop_state.infoset_key
        # Reconstruct what an M2 key would look like for the same state.
        raw_board = flop_state._raw.board_cards
        m2_board_str = ",".join(str(c) for c in raw_board)
        assert m2_board_str not in m3_key, (
            "M3 key must not embed the raw card-id list (compat break)"
        )

    def test_same_raw_board_yields_same_key(self, game) -> None:  # type: ignore[no-untyped-def]
        """Trivial sanity: identical deals → identical keys. Holds in
        both M2 and M3; not a real abstraction-collapse check (see
        :meth:`test_different_raw_boards_same_bucket_collapse` for the
        real one)."""
        deal_a = (cid("As"), cid("Ah"), cid("2c"), cid("3d"),
                  cid("Ts"), cid("Td"), cid("Th"), cid("4c"), cid("5h"))
        deal_b = deal_a
        sa = self._drive_to_postflop(game, deal_a)
        sb = self._drive_to_postflop(game, deal_b)
        assert sa.infoset_key == sb.infoset_key

    def test_different_raw_boards_same_bucket_collapse(self, game) -> None:  # type: ignore[no-untyped-def]
        """Two flop states with the SAME P0 hole and the SAME betting
        history but with DIFFERENT raw flop boards that land in the
        SAME postflop bucket must produce IDENTICAL infoset_keys. This
        is the abstraction-aliasing invariant M3 actually buys us — and
        the contract that fails under M2's raw-board-string key.

        Strategy: enumerate flops over a small candidate set, group by
        the postflop abstractor's bucket, and pick any group with ≥ 2
        distinct flops. If no collision exists in the candidate set
        (very unlikely with 10 buckets), the test ``pytest.skip``s with
        a hint to widen the candidate set."""
        # Fixed P0 hole cards.
        p0_hole = (cid("As"), cid("Ah"))
        # P1 hole cards distinct from P0 but otherwise arbitrary; held
        # constant so only the flop differs across the two probed deals.
        p1_hole = (cid("2c"), cid("3d"))
        used_holes = set(p0_hole) | set(p1_hole)
        # Candidate flops: vary the first card across all remaining 48
        # cards, fix the other two flop cards, fix turn/river. This
        # produces up to 48 candidate flops; we group by bucket and pick
        # the first group with at least two members.
        fixed_flop_2 = cid("Td")
        fixed_flop_3 = cid("Th")
        fixed_turn = cid("4c")
        fixed_river = cid("5h")
        used_static = used_holes | {fixed_flop_2, fixed_flop_3,
                                    fixed_turn, fixed_river}
        abstractor = game.postflop_abstractor
        bucket_to_flops: dict[int, list[int]] = {}
        for c in range(52):
            if c in used_static:
                continue
            board = (c, fixed_flop_2, fixed_flop_3)
            try:
                b = abstractor.bucket(p0_hole, board, 1)
            except ValueError:
                continue
            bucket_to_flops.setdefault(b, []).append(c)
        # Find a collision bucket.
        collisions = [flops for flops in bucket_to_flops.values()
                      if len(flops) >= 2]
        if not collisions:
            pytest.skip(
                "no collision in candidate set; widen candidates if this "
                "trips repeatedly"
            )
        flop_a, flop_b = collisions[0][0], collisions[0][1]
        deal_a = (p0_hole[0], p0_hole[1], p1_hole[0], p1_hole[1],
                  flop_a, fixed_flop_2, fixed_flop_3,
                  fixed_turn, fixed_river)
        deal_b = (p0_hole[0], p0_hole[1], p1_hole[0], p1_hole[1],
                  flop_b, fixed_flop_2, fixed_flop_3,
                  fixed_turn, fixed_river)
        sa = self._drive_to_postflop(game, deal_a)
        sb = self._drive_to_postflop(game, deal_b)
        # Sanity: the two raw boards are actually different.
        assert sa._raw.board_cards != sb._raw.board_cards
        # The contract: keys collapse despite distinct raw boards.
        assert sa.infoset_key == sb.infoset_key, (
            f"M3 abstraction-collapse contract violated: keys differ "
            f"despite same bucket\n  sa.key={sa.infoset_key}\n  "
            f"sb.key={sb.infoset_key}"
        )


# =============================================================================
# AbstractedHUNLGame integration with the postflop abstractor
# =============================================================================
class TestAbstractedHUNLGameWithPostflopAbstractor:
    def test_game_exposes_postflop_abstractor(self) -> None:
        """AbstractedHUNLGame must auto-construct a postflop abstractor
        and expose it as ``postflop_abstractor`` (or similar attribute)
        so M3.4 integration tests can introspect cache hit rate."""
        from poker_ai.games.hunl_abstraction import AbstractedHUNLGame

        game = AbstractedHUNLGame(n_buckets=10, n_trials=200, seed=42)
        assert hasattr(game, "postflop_abstractor"), (
            "AbstractedHUNLGame must expose a postflop_abstractor attribute"
        )
        # Attribute should be a PostflopBoardAbstractor instance.
        from poker_ai.games.hunl_abstraction import PostflopBoardAbstractor
        assert isinstance(game.postflop_abstractor, PostflopBoardAbstractor)

    def test_postflop_state_drives_cache_hits(self) -> None:
        """Smoke: walking from preflop to flop and reading infoset_key
        twice on the same state should bump the postflop abstractor's
        cache stats. This is the integration handle MCCFR will rely on."""
        from poker_ai.games.hunl_abstraction import (
            AbstractedHUNLAction,
            AbstractedHUNLGame,
        )

        game = AbstractedHUNLGame(n_buckets=10, n_trials=200, seed=42)
        deal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        state = game.state_from_deal(deal)
        state = state.next_state(AbstractedHUNLAction.CALL)   # SB call
        state = state.next_state(AbstractedHUNLAction.CALL)   # BB check → flop
        # First read: cache miss.
        _ = state.infoset_key
        # Second read on the same state: cache hit.
        _ = state.infoset_key
        stats = game.postflop_abstractor.cache_stats()
        assert stats["total"] >= 2
        assert stats["hits"] >= 1
