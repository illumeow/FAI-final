import time
from collections.abc import Iterator
from itertools import combinations
from math import comb
from typing import Any

from .game_core import GameCore

class CFRRolloutEvaluator:
    def __init__(self, core: GameCore, player_idx: int) -> None:
        self.core = core
        self.player_idx = player_idx

    def pick_opponent_card(
        self,
        opp_hand: list[int],
        board: list[list[int]],
        epsilon: float,
    ) -> int:
        if epsilon > 0.0 and self.core.rng.random() < epsilon:
            return self.core.rng.choice(opp_hand)
        return self.core.greedy_pick(opp_hand, board)

    def rollout_total_score(
        self,
        board: list[list[int]],
        my_hand: list[int],
        fixed_first_card: int,
        opp_hands: list[list[int]],
        opponent_epsilon: float,
    ) -> int:
        sim_board = [row.copy() for row in board]
        me = sorted(my_hand)
        me.remove(fixed_first_card)
        opp = [h.copy() for h in opp_hands]
        rounds_left = len(my_hand)
        my_total_score = 0

        for round_idx in range(rounds_left):
            if round_idx == 0:
                my_card = fixed_first_card
            else:
                my_card = self.core.greedy_pick(me, sim_board)
                me.remove(my_card)

            played = [(my_card, self.player_idx)]
            for i, oh in enumerate(opp):
                if not oh:
                    continue
                oc = self.pick_opponent_card(oh, sim_board, opponent_epsilon)
                oh.remove(oc)
                played.append((oc, -1000 - i))

            played.sort(key=lambda x: x[0])
            for card, pid in played:
                gained = self.core.place_card(sim_board, card)
                if pid == self.player_idx:
                    my_total_score += gained

        return my_total_score


class HandAssignmentUtils:
    @staticmethod
    def count_hand_assignments(unseen_len: int, n_opponents: int, rounds_left: int) -> int:
        if unseen_len != n_opponents * rounds_left:
            return 0
        total = 1
        remaining = unseen_len
        for _ in range(n_opponents):
            total *= comb(remaining, rounds_left)
            remaining -= rounds_left
        return total

    @staticmethod
    def iter_hand_assignments(
        unseen_cards: list[int],
        n_opponents: int,
        rounds_left: int,
    ) -> Iterator[list[list[int]]]:
        def rec(remaining: tuple[int, ...], k: int) -> Iterator[list[list[int]]]:
            if k == 0:
                yield []
                return

            for combo in combinations(remaining, rounds_left):
                combo_set = set(combo)
                rest = tuple(c for c in remaining if c not in combo_set)
                for tail in rec(rest, k - 1):
                    yield [sorted(combo)] + tail

        yield from rec(tuple(sorted(unseen_cards)), n_opponents)


class CFRPlusPlayer:
    """
    Stronger CFR variant with:
    1) Regret Matching Plus (RM+)
    2) Sacrifice-aware action pruning
    3) Epsilon-greedy opponent rollout policy
    4) Exact endgame solver for small remaining horizons
    5) Adaptive time management based on iteration speed
    """

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx
        self.core = GameCore(player_idx)
        self.rollout_evaluator = CFRRolloutEvaluator(self.core, player_idx)
        self.hand_assignment_utils = HandAssignmentUtils()

        self.time_budget_sec = 0.95
        self.max_actions = 7
        self.max_actions_with_sacrifice = 8

        self.opponent_epsilon = 0.05

        self.exact_endgame_rounds = 3
        self.exact_endgame_max_assignments = 5000

        self.min_guard_sec = 0.015
        self.iter_safety_mult = 1.20
        self.avg_iter_sec = 0.004
        self.regret_decay = 1.0

    def candidate_subset(self, hand: list[int], board: list[list[int]]) -> list[int]:
        ordered = sorted(hand, key=lambda c: self.core.heuristic_card_key(board, c))
        mandatory = {hand[0], hand[-1]}

        min_row_end = min(row[-1] for row in board)
        forced_take_cards = [c for c in hand if c < min_row_end]
        if forced_take_cards:
            mandatory.add(forced_take_cards[0])
            mandatory.add(forced_take_cards[-1])

        risky_reset_cards = []
        for card in hand:
            fit_idx = self.core.best_fit_row(board, card)
            if fit_idx != -1 and len(board[fit_idx]) >= 4:
                risky_reset_cards.append(card)
        risky_pick = sorted(
            risky_reset_cards,
            key=lambda c: self.core.heuristic_card_key(board, c),
        )
        if risky_pick:
            mandatory.add(risky_pick[0])

        selected = sorted(mandatory, key=lambda c: self.core.heuristic_card_key(board, c))
        selected_set = set(selected)
        for card in ordered:
            if len(selected) >= self.max_actions_with_sacrifice:
                break
            if card in selected_set:
                continue
            selected.append(card)
            selected_set.add(card)

        return selected

    def strategy_from_regret_plus(
        self,
        regrets: dict[int, float],
        actions: list[int],
    ) -> dict[int, float]:
        positives = [max(0.0, regrets[a]) for a in actions]
        denom = sum(positives)
        if denom <= 1e-12:
            p = 1.0 / len(actions)
            return {a: p for a in actions}
        return {a: max(0.0, regrets[a]) / denom for a in actions}

    def exact_endgame_expected_loss(
        self,
        board: list[list[int]],
        hand: list[int],
        first_card: int,
        unseen: list[int],
        n_opponents: int,
        rounds_left: int,
        assignment_count: int,
    ) -> float:
        total_loss = 0.0
        for opp_hands in self.hand_assignment_utils.iter_hand_assignments(
            unseen,
            n_opponents,
            rounds_left,
        ):
            total_loss += float(
                self.rollout_evaluator.rollout_total_score(
                    board,
                    hand,
                    first_card,
                    opp_hands,
                    opponent_epsilon=0.0,
                )
            )
        return total_loss / float(assignment_count)

    def can_iterate(self, deadline: float) -> bool:
        remaining = deadline - time.perf_counter()
        reserve = max(self.min_guard_sec, self.avg_iter_sec * self.iter_safety_mult)
        return remaining > reserve

    def action(self, hand: list[int], history: dict[str, Any]) -> int:
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen = self.core.build_unseen_pool(hand, history)
        rounds_left = len(hand)

        actions = self.candidate_subset(hand, board)
        if len(actions) == 1:
            return actions[0]

        use_exact_endgame = (
            rounds_left <= self.exact_endgame_rounds
            and len(unseen) == n_opponents * rounds_left
        )
        assignment_count = 0
        if use_exact_endgame:
            assignment_count = self.hand_assignment_utils.count_hand_assignments(
                len(unseen),
                n_opponents,
                rounds_left,
            )
            use_exact_endgame = 0 < assignment_count <= self.exact_endgame_max_assignments

        if use_exact_endgame:
            exact_losses: dict[int, float] = {
                action: self.exact_endgame_expected_loss(
                    board,
                    hand,
                    action,
                    unseen,
                    n_opponents,
                    rounds_left,
                    assignment_count,
                )
                for action in actions
            }
            return min(
                actions,
                key=lambda a: (exact_losses[a], self.core.heuristic_card_key(board, a)),
            )

        regrets = {a: 0.0 for a in actions}
        strategy_sum = {a: 0.0 for a in actions}
        utility_sum = {a: 0.0 for a in actions}
        utility_count = {a: 0 for a in actions}

        deadline = time.perf_counter() + self.time_budget_sec
        iteration = 0

        while self.can_iterate(deadline):
            iteration += 1
            iter_start = time.perf_counter()

            strategy = self.strategy_from_regret_plus(regrets, actions)
            opp_hands = self.core.sample_opponent_hands(unseen, n_opponents, rounds_left)

            utilities: dict[int, float] = {}
            for action in actions:
                loss = self.rollout_evaluator.rollout_total_score(
                    board,
                    hand,
                    action,
                    opp_hands,
                    opponent_epsilon=self.opponent_epsilon,
                )
                util = -float(loss)
                utilities[action] = util
                utility_sum[action] += util
                utility_count[action] += 1

            node_util = sum(strategy[a] * utilities[a] for a in actions)
            for action in actions:
                updated = self.regret_decay * regrets[action] + utilities[action] - node_util
                regrets[action] = max(0.0, updated)
                strategy_sum[action] += iteration * strategy[action]

            iter_cost = time.perf_counter() - iter_start
            self.avg_iter_sec = 0.85 * self.avg_iter_sec + 0.15 * iter_cost

        strategy_mass = sum(strategy_sum.values())
        if strategy_mass > 0.0:
            avg_strategy = {a: strategy_sum[a] / strategy_mass for a in actions}
            return max(
                actions,
                key=lambda a: (
                    avg_strategy[a],
                    utility_sum[a] / max(1, utility_count[a]),
                    -a,
                ),
            )

        return min(actions, key=lambda a: self.core.heuristic_card_key(board, a))
