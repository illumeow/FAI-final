import math
import random
import time
from typing import Any


_N_CARDS = 104
_N_ROWS = 4
_N_OPPONENTS = 3


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
    for i in range(1, _N_ROWS):
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
        for i in range(1, _N_ROWS):
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
    # Hot path. Inlines _heuristic_key and hoists row state — the board is
    # fixed while picking, so per-card recomputation of row[-1], len(row),
    # and the force-take row was wasteful. ~50% rollout speedup.
    row_last = [r[-1] for r in board]
    row_len = [len(r) for r in board]

    ft_idx = 0
    ft_tiebreak = (row_sums[0], row_len[0], 0)
    for i in range(1, _N_ROWS):
        k = (row_sums[i], row_len[i], i)
        if k < ft_tiebreak:
            ft_tiebreak = k
            ft_idx = i
    ft_score = row_sums[ft_idx]

    best_card = hand[0]
    best_kkey: tuple[int, int, int, int, int] | None = None
    for card in hand:
        best_idx = -1
        best_last = -1
        for i, last in enumerate(row_last):
            if last < card and last > best_last:
                best_last = last
                best_idx = i

        if best_idx == -1:
            kkey = (ft_score, 1, 1_000_000, 1, card)
        else:
            rl = row_len[best_idx]
            score = row_sums[best_idx] if rl >= 5 else 0
            kkey = (score, 0, card - best_last, rl + 1, card)

        if best_kkey is None or kkey < best_kkey:
            best_kkey = kkey
            best_card = card

    return best_card


def _lookahead_pick(
    hand: list[int],
    board: list[list[int]],
    row_sums: list[int],
    others_cards: list[int],
) -> int:
    # 1-ply lookahead: each candidate is evaluated by simulating sorted placement
    # against `others_cards` (the other opps' greedy picks; my card deliberately
    # excluded so my candidate choice doesn't bias the opp's decision). State is
    # tracked in flat arrays to avoid the per-candidate board.copy() cost.
    if len(hand) == 1:
        return hand[0]
    if not others_cards:
        return _greedy_pick(hand, board, row_sums)

    base_last = [r[-1] for r in board]
    base_len = [len(r) for r in board]
    base_score = list(row_sums)

    others_sorted = sorted(others_cards)

    best_c = hand[0]
    best_pen = -1
    for c in hand:
        last = base_last.copy()
        rlen = base_len.copy()
        score = base_score.copy()

        # Merge c into others_sorted in ascending order.
        oi = 0
        n_others = len(others_sorted)
        c_done = False
        pen = 0
        while oi < n_others or not c_done:
            if not c_done and (oi >= n_others or c <= others_sorted[oi]):
                card = c
                is_me = True
                c_done = True
            else:
                card = others_sorted[oi]
                is_me = False
                oi += 1

            best_idx = -1
            best_last = -1
            for i in range(_N_ROWS):
                li = last[i]
                if li < card and li > best_last:
                    best_last = li
                    best_idx = i

            if best_idx == -1:
                ft_idx = 0
                ft_key = (score[0], rlen[0], 0)
                for i in range(1, _N_ROWS):
                    k = (score[i], rlen[i], i)
                    if k < ft_key:
                        ft_key = k
                        ft_idx = i
                gained = score[ft_idx]
                last[ft_idx] = card
                rlen[ft_idx] = 1
                score[ft_idx] = _CARD_SCORE[card]
            elif rlen[best_idx] >= 5:
                gained = score[best_idx]
                last[best_idx] = card
                rlen[best_idx] = 1
                score[best_idx] = _CARD_SCORE[card]
            else:
                gained = 0
                last[best_idx] = card
                rlen[best_idx] += 1
                score[best_idx] += _CARD_SCORE[card]

            if is_me:
                pen += gained

        if best_pen == -1 or pen < best_pen:
            best_pen = pen
            best_c = c
    return best_c


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
    return [sampled[i * rounds_left : (i + 1) * rounds_left] for i in range(n_opponents)]


def _simulate_one_round_inplace(
    board: list[list[int]],
    row_sums: list[int],
    my_card: int,
    opp_remaining: list[list[int]],
    my_pid: int,
    opp_lookahead: bool = False,
) -> int:
    if opp_lookahead:
        n_opp = len(opp_remaining)
        # First pass: each opp's static greedy pick. Used as the proxy for
        # "what other opps will play" in the second pass.
        greedy_picks: list[int] = [-1] * n_opp
        for i, h in enumerate(opp_remaining):
            if h:
                greedy_picks[i] = _greedy_pick(h, board, row_sums)
        # Second pass: each opp picks via 1-ply best response against the other
        # opps' greedy picks. My card is excluded from their planning set.
        final_picks: list[int] = [-1] * n_opp
        for i, h in enumerate(opp_remaining):
            if not h:
                continue
            others = [greedy_picks[j] for j in range(n_opp) if j != i and greedy_picks[j] != -1]
            final_picks[i] = _lookahead_pick(h, board, row_sums, others)

        played: list[tuple[int, int]] = [(my_card, my_pid)]
        for i, h in enumerate(opp_remaining):
            if not h:
                continue
            c = final_picks[i]
            h.remove(c)
            played.append((c, -1000 - i))
    else:
        played = [(my_card, my_pid)]
        for i, opp_hand in enumerate(opp_remaining):
            if not opp_hand:
                continue
            opp_card = _greedy_pick(opp_hand, board, row_sums)
            opp_hand.remove(opp_card)
            played.append((opp_card, -1000 - i))

    played.sort(key=lambda x: x[0])
    my_total = 0
    for card, pid in played:
        gained = _place_card(board, row_sums, card)
        if pid == my_pid:
            my_total += gained
    return my_total


def _endgame_best_response(
    board: list[list[int]],
    row_sums: list[int],
    my_remaining: list[int],
    opp_remaining: list[list[int]],
    my_pid: int,
    opp_lookahead: bool = False,
) -> int:
    if not my_remaining:
        return 0
    if len(my_remaining) == 1:
        b = [row.copy() for row in board]
        rs = row_sums.copy()
        opps = [h.copy() for h in opp_remaining]
        return _simulate_one_round_inplace(b, rs, my_remaining[0], opps, my_pid, opp_lookahead)

    best = -1
    for idx, my_card in enumerate(my_remaining):
        b = [row.copy() for row in board]
        rs = row_sums.copy()
        opps = [h.copy() for h in opp_remaining]
        gained = _simulate_one_round_inplace(b, rs, my_card, opps, my_pid, opp_lookahead)
        sub_my = my_remaining[:idx] + my_remaining[idx + 1 :]
        future = _endgame_best_response(b, rs, sub_my, opps, my_pid, opp_lookahead)
        total = gained + future
        if idx == 0 or total < best:
            best = total
    return best


def _rollout_total_score(
    board: list[list[int]],
    my_hand: list[int],
    fixed_first_card: int,
    opp_hands: list[list[int]],
    my_pid: int,
    endgame_threshold: int = 4,
    opp_lookahead: bool = False,
) -> int:
    sim_board = [row.copy() for row in board]
    row_sums = _row_sums_from_board(sim_board)
    my_remaining = my_hand.copy()
    my_remaining.remove(fixed_first_card)
    opp_remaining = [h.copy() for h in opp_hands]

    rounds_left = len(my_hand)
    my_total = _simulate_one_round_inplace(
        sim_board, row_sums, fixed_first_card, opp_remaining, my_pid, opp_lookahead
    )

    if rounds_left == 1:
        return my_total

    if rounds_left <= endgame_threshold:
        my_total += _endgame_best_response(
            sim_board, row_sums, my_remaining, opp_remaining, my_pid, opp_lookahead
        )
        return my_total

    for _ in range(rounds_left - 1):
        my_card = _greedy_pick(my_remaining, sim_board, row_sums)
        my_remaining.remove(my_card)
        my_total += _simulate_one_round_inplace(
            sim_board, row_sums, my_card, opp_remaining, my_pid, opp_lookahead
        )
    return my_total


class SimulationPlayer:
    """
    Time-budgeted determinized Monte Carlo player using paired (CRN) warmup
    followed by UCB1-style allocation on the remaining budget.
    """

    def __init__(
        self,
        player_idx: int,
        ucb_c: float = 7.0,
        endgame_threshold: int = 4,
        opp_lookahead: bool = False,
    ) -> None:
        self.player_idx = player_idx
        self.rng = random.Random()
        self.time_budget_sec = 0.95
        self.min_paired_iters = 4
        self.ucb_c = ucb_c
        self.endgame_threshold = endgame_threshold
        self.opp_lookahead = opp_lookahead

    def action(self, hand: list[int], history: dict[str, Any]) -> int:
        if len(hand) == 1:
            return hand[0]

        deadline = time.perf_counter() + self.time_budget_sec

        board = history["board"]
        rounds_left = len(hand)
        unseen = _build_unseen_pool(hand, history)

        candidates: list[int] = list(hand)
        totals: dict[int, float] = {c: 0.0 for c in candidates}
        counts: dict[int, int] = {c: 0 for c in candidates}
        my_pid = self.player_idx

        # Phase 1 (CRN warmup): one opp_hand → every candidate, paired sampling.
        for _ in range(self.min_paired_iters):
            if time.perf_counter() >= deadline:
                break
            opp_hands = _sample_opp_hands(self.rng, unseen, _N_OPPONENTS, rounds_left)
            for c in candidates:
                if time.perf_counter() >= deadline:
                    break
                totals[c] += _rollout_total_score(
                    board, hand, c, opp_hands, my_pid, self.endgame_threshold, self.opp_lookahead
                )
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
            opp_hands = _sample_opp_hands(self.rng, unseen, _N_OPPONENTS, rounds_left)
            totals[chosen] += _rollout_total_score(
                board, hand, chosen, opp_hands, my_pid, self.endgame_threshold, self.opp_lookahead
            )
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
