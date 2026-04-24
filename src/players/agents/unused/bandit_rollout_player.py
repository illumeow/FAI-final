import math
import time

from .core_utils import SimpleAgentCore


class BanditRolloutPlayer:
    """
    Extra idea: UCB bandit over candidate cards with rollout evaluation.
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.core = SimpleAgentCore(player_idx, seed_offset=21013)
        self.time_budget_sec = 0.86
        self.max_actions = 7
        self.ucb_c = 1.20

    def _candidate_subset(self, hand, board):
        return sorted(hand, key=lambda c: self.core.heuristic_card_key(board, c))[
            : self.max_actions
        ]

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen = self.core.build_unseen_pool(hand, history)
        rounds_left = len(hand)
        deadline = time.perf_counter() + self.time_budget_sec

        actions = self._candidate_subset(hand, board)
        pulls = {a: 0 for a in actions}
        loss_sum = {a: 0.0 for a in actions}

        # Warm start: one rollout per action.
        for a in actions:
            if time.perf_counter() >= deadline:
                break
            opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)
            loss = self.core.rollout_total_penalty(board, hand, a, opp_hands)
            pulls[a] += 1
            loss_sum[a] += float(loss)

        total_pulls = sum(pulls.values())

        while actions and time.perf_counter() < deadline:
            total_pulls = max(1, total_pulls)

            def ucb_value(a):
                if pulls[a] == 0:
                    return float("inf")
                mean_reward = -(loss_sum[a] / pulls[a])
                explore = self.ucb_c * math.sqrt(math.log(total_pulls + 1.0) / pulls[a])
                return mean_reward + explore

            chosen = max(actions, key=ucb_value)
            opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)
            loss = self.core.rollout_total_penalty(board, hand, chosen, opp_hands)
            pulls[chosen] += 1
            loss_sum[chosen] += float(loss)
            total_pulls += 1

        return min(
            actions,
            key=lambda a: (
                (loss_sum[a] / pulls[a]) if pulls[a] else float("inf"),
                self.core.heuristic_card_key(board, a),
            ),
        )
