import math
import random
import time
from typing import Any

from .game_core import GameCore


def _build_card_score_table(n_cards: int = 104) -> list[int]:
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


def _sample_opp_hands(
    rng: random.Random,
    unseen: list[int],
    n_opponents: int,
    rounds_left: int,
    n_cards: int = 104,
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
    return [sampled[i * rounds_left : (i + 1) * rounds_left] for i in range(n_opponents)]


def _rollout_total_score(
    board: list[list[int]],
    my_hand: list[int],
    fixed_first_card: int,
    opp_hands: list[list[int]],
    my_pid: int,
) -> int:
    sim_board = [row.copy() for row in board]
    row_sums = _row_sums_from_board(sim_board)
    my_remaining = my_hand.copy()
    opp_remaining = [h.copy() for h in opp_hands]

    rounds_left = len(my_hand)
    my_total_score = 0

    for r in range(rounds_left):
        my_card = fixed_first_card if r == 0 else _greedy_pick(my_remaining, sim_board, row_sums)
        my_remaining.remove(my_card)

        played: list[tuple[int, int]] = [(my_card, my_pid)]
        for i, opp_hand in enumerate(opp_remaining):
            if not opp_hand:
                continue
            opp_card = _greedy_pick(opp_hand, sim_board, row_sums)
            opp_hand.remove(opp_card)
            played.append((opp_card, -1000 - i))

        played.sort(key=lambda x: x[0])
        for card, pid in played:
            gained = _place_card(sim_board, row_sums, card)
            if pid == my_pid:
                my_total_score += gained

    return my_total_score


class SimulationPlayerV2:
    """
    Time-budgeted determinized Monte Carlo player using paired (CRN) warmup
    followed by UCB1-style allocation on the remaining budget.
    """

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx
        self.core = GameCore(player_idx)
        self.rng = random.Random()
        self.time_budget_sec = 0.92
        self.min_paired_iters = 2
        self.ucb_c = 2.0

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

        # Phase 1 (CRN warmup): one opp_hand → every candidate, paired sampling.
        for _ in range(self.min_paired_iters):
            if time.perf_counter() >= deadline:
                break
            opp_hands = _sample_opp_hands(self.rng, unseen, n_opponents, rounds_left)
            for c in candidates:
                if time.perf_counter() >= deadline:
                    break
                totals[c] += _rollout_total_score(board, hand, c, opp_hands, my_pid)
                counts[c] += 1

        # Phase 2 (UCB1): minimize penalty → pick the lowest LCB candidate.
        sampled = [c for c in candidates if counts[c] > 0]
        total_pulls = sum(counts[c] for c in sampled)
        ucb_c = self.ucb_c
        while sampled and time.perf_counter() < deadline:
            log_total = math.log(max(total_pulls, 2))
            chosen = min(
                sampled,
                key=lambda c: totals[c] / counts[c]
                - ucb_c * math.sqrt(log_total / counts[c]),
            )
            opp_hands = _sample_opp_hands(self.rng, unseen, n_opponents, rounds_left)
            totals[chosen] += _rollout_total_score(board, hand, chosen, opp_hands, my_pid)
            counts[chosen] += 1
            total_pulls += 1

        row_sums_now = _row_sums_from_board(board)
        return min(
            candidates,
            key=lambda c: (
                totals[c] / counts[c] if counts[c] > 0 else float("inf"),
                _heuristic_key(board, row_sums_now, c),
            ),
        )
