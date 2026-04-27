import time
from typing import Any

from .game_core import GameCore

class SimulationRolloutEvaluator:
    def __init__(self, core: GameCore, player_idx: int) -> None:
        self.core = core
        self.player_idx = player_idx

    """
    Given state (hand, board) and action (fixed_first_card)
    Given sampled opponent hands (opp_hands)
    Simulate the rest of the game with greedy policies and return my total score at the end
    """
    def rollout_total_score(
        self,
        board: list[list[int]],
        my_hand: list[int],
        fixed_first_card: int,
        opp_hands: list[list[int]],
    ) -> int:
        sim_board = [row.copy() for row in board]
        me = sorted(my_hand)
        me.remove(fixed_first_card)

        opp = [h.copy() for h in opp_hands]
        rounds_left = len(my_hand)
        my_total_score = 0

        for r in range(rounds_left):
            if r == 0:
                my_card = fixed_first_card
            else:
                my_card = self.core.greedy_pick(me, sim_board)
                me.remove(my_card)

            played = [(my_card, self.player_idx)]

            for i, oh in enumerate(opp):
                if not oh:
                    continue
                oc = self.core.greedy_pick(oh, sim_board)
                oh.remove(oc)
                played.append((oc, -1000 - i))

            played.sort(key=lambda x: x[0])
            for card, pid in played:
                gained = self.core.place_card(sim_board, card)
                if pid == self.player_idx:
                    my_total_score += gained

        return my_total_score


class SimulationStats:
    @staticmethod
    def mean_score(totals: dict[int, float], counts: dict[int, int], card: int) -> float:
        if counts[card] == 0:
            return float("inf")
        return totals[card] / counts[card]


class SimulationPlayer:
    """
    Time-budgeted Monte Carlo player.

    It samples hidden opponent hands (determinization) and rolls out the
    remaining round sequence with lightweight greedy policies.
    """

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx
        self.core = GameCore(player_idx)
        self.rollout_evaluator = SimulationRolloutEvaluator(self.core, player_idx)
        self.time_budget_sec = 0.92
        self.min_samples_per_card = 4

    def action(self, hand: list[int], history: dict[str, Any]) -> int:
        if len(hand) == 1:
            return hand[0]

        deadline = time.perf_counter() + self.time_budget_sec

        board = history["board"]
        n_players = len(history["scores"])
        n_opponents = n_players - 1
        rounds_left = len(hand)
        unseen = self.core.build_unseen_pool(hand, history)

        candidates: list[int] = hand
        totals: dict[int, float] = {c: 0.0 for c in candidates}
        counts: dict[int, int] = {c: 0 for c in candidates}

        # Phase 1: guarantee a minimum number of samples per candidate.
        for c in candidates:
            for _ in range(self.min_samples_per_card):
                if time.perf_counter() >= deadline:
                    break
                opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)
                score = self.rollout_evaluator.rollout_total_score(board, hand, c, opp_hands)
                totals[c] += score
                counts[c] += 1

        # Phase 2: allocate remaining time adaptively (focus on best candidates).
        active = candidates[:]
        pulls = 0
        while active and time.perf_counter() < deadline:
            # Optimistic exploration: low mean score first, then low sample count.
            active.sort(
                key=lambda c: (
                    SimulationStats.mean_score(totals, counts, c),
                    counts[c],
                    c,
                )
            )

            c = active[pulls % len(active)]
            opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)
            score = self.rollout_evaluator.rollout_total_score(board, hand, c, opp_hands)
            totals[c] += score
            counts[c] += 1
            pulls += 1

            # Periodically prune weak candidates to spend time on contenders.
            if pulls % max(12, len(candidates) * 2) == 0 and len(active) > 3:
                ranked = sorted(
                    active,
                    key=lambda x: (SimulationStats.mean_score(totals, counts, x), x),
                )
                keep = max(3, len(active) // 2)
                active = ranked[:keep]

        best_card = min(
            candidates,
            key=lambda c: (
                SimulationStats.mean_score(totals, counts, c),
                self.core.heuristic_card_key(board, c),
            ),
        )
        return best_card
