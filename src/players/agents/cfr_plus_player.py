import random
import time
from collections.abc import Iterator
from itertools import combinations
from math import comb
from typing import Any


_N_CARDS = 104


def _build_card_score_table(n_cards: int = _N_CARDS) -> list[int]:
    table = [0] * (n_cards + 1)
    for c in range(1, n_cards + 1):
        if c % 55 == 0:
            table[c] = 7
        elif c % 11 == 0:
            table[c] = 5
        elif c % 10 == 0:
            table[c] = 3
        elif c % 5 == 0:
            table[c] = 2
        else:
            table[c] = 1
    return table


_CARD_SCORE = _build_card_score_table()


def _row_sums_from_board(board: list[list[int]]) -> list[int]:
    return [sum(_CARD_SCORE[c] for c in row) for row in board]


def _place_card(board: list[list[int]], row_sums: list[int], card: int) -> int:
    best_idx = -1
    best_last = -1
    for i, row in enumerate(board):
        last = row[-1]
        if last < card and last > best_last:
            best_last = last
            best_idx = i

    if best_idx != -1:
        if len(board[best_idx]) >= 5:
            incurred = row_sums[best_idx]
            board[best_idx] = [card]
            row_sums[best_idx] = _CARD_SCORE[card]
            return incurred
        board[best_idx].append(card)
        row_sums[best_idx] += _CARD_SCORE[card]
        return 0

    force_take_idx = 0
    best_key = (row_sums[0], len(board[0]), 0)
    for i in range(1, len(board)):
        k = (row_sums[i], len(board[i]), i)
        if k < best_key:
            best_key = k
            force_take_idx = i

    incurred = row_sums[force_take_idx]
    board[force_take_idx] = [card]
    row_sums[force_take_idx] = _CARD_SCORE[card]
    return incurred


def _heuristic_key(
    board: list[list[int]], row_sums: list[int], card: int
) -> tuple[int, int, int, int, int]:
    best_idx = -1
    best_last = -1
    for i, row in enumerate(board):
        last = row[-1]
        if last < card and last > best_last:
            best_last = last
            best_idx = i

    if best_idx == -1:
        force_take_idx = 0
        best_key = (row_sums[0], len(board[0]), 0)
        for i in range(1, len(board)):
            k = (row_sums[i], len(board[i]), i)
            if k < best_key:
                best_key = k
                force_take_idx = i
        return (row_sums[force_take_idx], 1, 10**6, 1, card)

    delta = card - board[best_idx][-1]
    row_len = len(board[best_idx])
    score = row_sums[best_idx] if row_len >= 5 else 0
    return (score, 0, delta, row_len + 1, card)


def _greedy_pick(hand: list[int], board: list[list[int]], row_sums: list[int]) -> int:
    return min(hand, key=lambda c: _heuristic_key(board, row_sums, c))


def _best_fit_row(board: list[list[int]], card: int) -> int:
    best_idx = -1
    best_last = -1
    for i, row in enumerate(board):
        last = row[-1]
        if last < card and last > best_last:
            best_last = last
            best_idx = i
    return best_idx


def _build_unseen_pool(
    hand: list[int], history: dict[str, Any], n_cards: int = _N_CARDS
) -> list[int]:
    known = set(hand)
    for row in history["board"]:
        known.update(row)
    for past_action in history["history_matrix"]:
        known.update(past_action)
    for past_board in history["board_history"]:
        for row in past_board:
            known.update(row)
    return [c for c in range(1, n_cards + 1) if c not in known]


def _sample_opp_hands(
    rng: random.Random,
    unseen: list[int],
    n_opponents: int,
    rounds_left: int,
    n_cards: int = _N_CARDS,
) -> list[list[int]]:
    needed = n_opponents * rounds_left
    if needed == 0:
        return []
    if len(unseen) >= needed:
        sampled = rng.sample(unseen, needed)
    elif unseen:
        sampled = [rng.choice(unseen) for _ in range(needed)]
    else:
        sampled = [rng.randint(1, n_cards) for _ in range(needed)]
    hands: list[list[int]] = []
    for i in range(n_opponents):
        h = sampled[i * rounds_left : (i + 1) * rounds_left]
        h.sort()
        hands.append(h)
    return hands


def _pick_opponent_card(
    rng: random.Random,
    opp_hand: list[int],
    board: list[list[int]],
    row_sums: list[int],
    epsilon: float,
) -> int:
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(opp_hand)
    return _greedy_pick(opp_hand, board, row_sums)


def _rollout_total_score(
    rng: random.Random,
    board: list[list[int]],
    my_hand: list[int],
    fixed_first_card: int,
    opp_hands: list[list[int]],
    my_pid: int,
    opponent_epsilon: float,
) -> int:
    sim_board = [row.copy() for row in board]
    row_sums = _row_sums_from_board(sim_board)
    my_remaining = my_hand.copy()
    my_remaining.remove(fixed_first_card)
    opp_remaining = [h.copy() for h in opp_hands]

    rounds_left = len(my_hand)
    my_total_score = 0

    for r in range(rounds_left):
        if r == 0:
            my_card = fixed_first_card
        else:
            my_card = _greedy_pick(my_remaining, sim_board, row_sums)
            my_remaining.remove(my_card)

        played: list[tuple[int, int]] = [(my_card, my_pid)]
        for i, opp_hand in enumerate(opp_remaining):
            if not opp_hand:
                continue
            opp_card = _pick_opponent_card(rng, opp_hand, sim_board, row_sums, opponent_epsilon)
            opp_hand.remove(opp_card)
            played.append((opp_card, -1000 - i))

        played.sort(key=lambda x: x[0])
        for card, pid in played:
            gained = _place_card(sim_board, row_sums, card)
            if pid == my_pid:
                my_total_score += gained

    return my_total_score


def _count_hand_assignments(unseen_len: int, n_opponents: int, rounds_left: int) -> int:
    if unseen_len != n_opponents * rounds_left:
        return 0
    total = 1
    remaining = unseen_len
    for _ in range(n_opponents):
        total *= comb(remaining, rounds_left)
        remaining -= rounds_left
    return total


def _iter_hand_assignments(
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


def _strategy_from_regret_plus(
    regrets: dict[int, float],
    actions: list[int],
) -> dict[int, float]:
    positives = [max(0.0, regrets[a]) for a in actions]
    denom = sum(positives)
    if denom <= 1e-12:
        p = 1.0 / len(actions)
        return {a: p for a in actions}
    return {a: max(0.0, regrets[a]) / denom for a in actions}


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
        self.rng = random.Random()

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

    def candidate_subset(
        self, hand: list[int], board: list[list[int]], row_sums: list[int]
    ) -> list[int]:
        ordered = sorted(hand, key=lambda c: _heuristic_key(board, row_sums, c))
        mandatory = {hand[0], hand[-1]}

        min_row_end = min(row[-1] for row in board)
        forced_take_cards = [c for c in hand if c < min_row_end]
        if forced_take_cards:
            mandatory.add(forced_take_cards[0])
            mandatory.add(forced_take_cards[-1])

        risky_reset_cards = []
        for card in hand:
            fit_idx = _best_fit_row(board, card)
            if fit_idx != -1 and len(board[fit_idx]) >= 4:
                risky_reset_cards.append(card)
        risky_pick = sorted(
            risky_reset_cards,
            key=lambda c: _heuristic_key(board, row_sums, c),
        )
        if risky_pick:
            mandatory.add(risky_pick[0])

        selected = sorted(mandatory, key=lambda c: _heuristic_key(board, row_sums, c))
        selected_set = set(selected)
        for card in ordered:
            if len(selected) >= self.max_actions_with_sacrifice:
                break
            if card in selected_set:
                continue
            selected.append(card)
            selected_set.add(card)

        return selected

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
        for opp_hands in _iter_hand_assignments(unseen, n_opponents, rounds_left):
            total_loss += float(
                _rollout_total_score(
                    self.rng,
                    board,
                    hand,
                    first_card,
                    opp_hands,
                    self.player_idx,
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
        row_sums = _row_sums_from_board(board)
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen = _build_unseen_pool(hand, history)
        rounds_left = len(hand)

        actions = self.candidate_subset(hand, board, row_sums)
        if len(actions) == 1:
            return actions[0]

        use_exact_endgame = (
            rounds_left <= self.exact_endgame_rounds
            and len(unseen) == n_opponents * rounds_left
        )
        assignment_count = 0
        if use_exact_endgame:
            assignment_count = _count_hand_assignments(len(unseen), n_opponents, rounds_left)
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
                key=lambda a: (exact_losses[a], _heuristic_key(board, row_sums, a)),
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

            strategy = _strategy_from_regret_plus(regrets, actions)
            opp_hands = _sample_opp_hands(self.rng, unseen, n_opponents, rounds_left)

            utilities: dict[int, float] = {}
            for action in actions:
                loss = _rollout_total_score(
                    self.rng,
                    board,
                    hand,
                    action,
                    opp_hands,
                    self.player_idx,
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

        return min(actions, key=lambda a: _heuristic_key(board, row_sums, a))
