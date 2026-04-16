import time

from .core_utils import SimpleAgentCore


class CFRPlayer:
    """
    Online CFR-style regret matching over a reduced candidate set.
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.core = SimpleAgentCore(player_idx, seed_offset=13001)
        self.time_budget_sec = 0.86
        self.max_actions = 7

    def _candidate_subset(self, hand, board):
        ranked = sorted(hand, key=lambda c: self.core.heuristic_card_key(board, c))
        return ranked[: min(self.max_actions, len(ranked))]

    def _strategy_from_regret(self, regrets, actions):
        positives = [max(0.0, regrets[a]) for a in actions]
        denom = sum(positives)
        if denom <= 1e-12:
            p = 1.0 / len(actions)
            return {a: p for a in actions}
        return {a: max(0.0, regrets[a]) / denom for a in actions}

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen = self.core.build_unseen_pool(hand, history)
        rounds_left = len(hand)

        actions = self._candidate_subset(hand, board)
        regrets = {a: 0.0 for a in actions}
        strategy_sum = {a: 0.0 for a in actions}
        utility_sum = {a: 0.0 for a in actions}
        utility_count = {a: 0 for a in actions}

        deadline = time.perf_counter() + self.time_budget_sec

        while time.perf_counter() < deadline:
            strategy = self._strategy_from_regret(regrets, actions)

            opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)

            utilities = {}
            for a in actions:
                loss = self.core.rollout_total_penalty(board, hand, a, opp_hands)
                util = -float(loss)
                utilities[a] = util
                utility_sum[a] += util
                utility_count[a] += 1

            node_util = sum(strategy[a] * utilities[a] for a in actions)

            for a in actions:
                regrets[a] += utilities[a] - node_util
                strategy_sum[a] += strategy[a]

        if sum(strategy_sum.values()) > 0:
            avg_strategy = {
                a: strategy_sum[a] / sum(strategy_sum.values())
                for a in actions
            }
            return max(actions, key=lambda a: (avg_strategy[a], utility_sum[a] / max(1, utility_count[a]), -a))

        return min(actions, key=lambda a: self.core.heuristic_card_key(board, a))
