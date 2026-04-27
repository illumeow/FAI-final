import random
import time
from typing import Any

from .game_core import GameCore
from .simulation_player_v2 import (
    _heuristic_key,
    _row_sums_from_board,
    _rollout_total_score,
    _sample_opp_hands,
)


def _mean_score(totals: dict[int, float], counts: dict[int, int], card: int) -> float:
    if counts[card] == 0:
        return float("inf")
    return totals[card] / counts[card]


class SimulationPlayerV3:
    """
    V2's faster rollout kernel (lookup table, incremental row sums, no
    redundant sorts) combined with V1's phase-2 strategy: paired CRN warmup
    followed by round-robin over the best-mean ordering with periodic pruning
    of weak candidates.
    """

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx
        self.core = GameCore(player_idx)
        self.rng = random.Random()
        self.time_budget_sec = 0.92
        self.min_paired_iters = 4
        self.prune_floor = 3

    def action(self, hand: list[int], history: dict[str, Any]) -> int:
        if len(hand) == 1:
            return hand[0]

        deadline = time.perf_counter() + self.time_budget_sec

        board = history["board"]
        n_players = len(history["scores"])
        n_opponents = n_players - 1
        rounds_left = len(hand)
        unseen = self.core.build_unseen_pool(hand, history)

        candidates: list[int] = list(hand)
        totals: dict[int, float] = {c: 0.0 for c in candidates}
        counts: dict[int, int] = {c: 0 for c in candidates}
        my_pid = self.player_idx

        # Phase 1 (CRN warmup): same opp_hand evaluated against every candidate.
        for _ in range(self.min_paired_iters):
            if time.perf_counter() >= deadline:
                break
            opp_hands = _sample_opp_hands(self.rng, unseen, n_opponents, rounds_left)
            for c in candidates:
                if time.perf_counter() >= deadline:
                    break
                totals[c] += _rollout_total_score(board, hand, c, opp_hands, my_pid)
                counts[c] += 1

        # Phase 2: round-robin over best-mean ordering, periodic prune.
        active = candidates[:]
        pulls = 0
        prune_period = max(12, len(candidates) * 2)
        while active and time.perf_counter() < deadline:
            active.sort(
                key=lambda c: (_mean_score(totals, counts, c), counts[c], c),
            )
            chosen = active[pulls % len(active)]
            opp_hands = _sample_opp_hands(self.rng, unseen, n_opponents, rounds_left)
            totals[chosen] += _rollout_total_score(board, hand, chosen, opp_hands, my_pid)
            counts[chosen] += 1
            pulls += 1

            if pulls % prune_period == 0 and len(active) > self.prune_floor:
                ranked = sorted(active, key=lambda x: (_mean_score(totals, counts, x), x))
                keep = max(self.prune_floor, len(active) // 2)
                active = ranked[:keep]

        row_sums_now = _row_sums_from_board(board)
        return min(
            candidates,
            key=lambda c: (
                _mean_score(totals, counts, c),
                _heuristic_key(board, row_sums_now, c),
            ),
        )
