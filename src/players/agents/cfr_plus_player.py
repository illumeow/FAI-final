import math
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


def _lookahead_pick(
    hand: list[int],
    board: list[list[int]],
    row_sums: list[int],
    others_cards: list[int],
) -> int:
    # 1-ply lookahead: each candidate evaluated against `others_cards` (other
    # opps' greedy picks plus optional phantom for me) by simulating sorted
    # placement on flat-array state copies — no per-candidate board.copy().
    if len(hand) == 1:
        return hand[0]
    if not others_cards:
        return _greedy_pick(hand, board, row_sums)

    n_rows = len(board)
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
            for i in range(n_rows):
                li = last[i]
                if li < card and li > best_last:
                    best_last = li
                    best_idx = i

            if best_idx == -1:
                ft_idx = 0
                ft_key = (score[0], rlen[0], 0)
                for i in range(1, n_rows):
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


def _build_opp_weights(
    history: dict[str, Any],
    unseen: list[int],
    my_pid: int,
    weight_sigma: float,
) -> list[list[float] | None] | None:
    # Per-opp Gaussian weights over `unseen`, centered on each opp's empirical
    # mean of past plays. None entries → uniform fallback for that opp.
    if weight_sigma <= 0.0 or not unseen:
        return None
    history_matrix = history["history_matrix"]
    n_players = len(history["scores"])
    opp_pids = [pid for pid in range(n_players) if pid != my_pid]
    inv_2s2 = 1.0 / (2.0 * weight_sigma * weight_sigma)
    weights: list[list[float] | None] = []
    for pid in opp_pids:
        played = [round_actions[pid] for round_actions in history_matrix
                  if round_actions[pid] > 0]
        if played:
            mean = sum(played) / len(played)
            weights.append([math.exp(-((c - mean) ** 2) * inv_2s2) for c in unseen])
        else:
            weights.append(None)
    return weights


def _sample_opp_hands(
    rng: random.Random,
    unseen: list[int],
    n_opponents: int,
    rounds_left: int,
    n_cards: int = _N_CARDS,
    opp_weights: list[list[float] | None] | None = None,
) -> list[list[int]]:
    needed = n_opponents * rounds_left
    if needed == 0:
        return []
    if opp_weights is None:
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

    pool = list(unseen)
    weights_per_opp = [list(w) if w is not None else None for w in opp_weights]
    hands = []
    for i in range(n_opponents):
        w = weights_per_opp[i]
        hand: list[int] = []
        if w is None or not pool:
            if len(pool) >= rounds_left:
                hand = rng.sample(pool, rounds_left)
                for c in hand:
                    idx = pool.index(c)
                    pool.pop(idx)
                    for ow in weights_per_opp:
                        if ow is not None:
                            ow.pop(idx)
            elif pool:
                hand = [rng.choice(pool) for _ in range(rounds_left)]
            else:
                hand = [rng.randint(1, n_cards) for _ in range(rounds_left)]
            hand.sort()
            hands.append(hand)
            continue

        for _ in range(min(rounds_left, len(pool))):
            idx = rng.choices(range(len(pool)), weights=w, k=1)[0]
            hand.append(pool[idx])
            pool.pop(idx)
            for ow in weights_per_opp:
                if ow is not None:
                    ow.pop(idx)
        while len(hand) < rounds_left:
            hand.append(rng.randint(1, n_cards))
        hand.sort()
        hands.append(hand)
    return hands


def _simulate_one_round_inplace(
    board: list[list[int]],
    row_sums: list[int],
    my_card: int,
    opp_remaining: list[list[int]],
    my_pid: int,
    opp_lookahead: bool = False,
    phantom_my_card: int = -1,
) -> int:
    if opp_lookahead:
        n_opp = len(opp_remaining)
        # First pass: each opp's static greedy pick — proxy for "what other
        # opps will play" in the second pass.
        greedy_picks: list[int] = [-1] * n_opp
        for i, h in enumerate(opp_remaining):
            if h:
                greedy_picks[i] = _greedy_pick(h, board, row_sums)
        # Second pass: each opp picks via 1-ply best response against the
        # other opps' greedy picks. My card is excluded from their planning
        # set unless `phantom_my_card` is supplied (a stand-in independent of
        # my actual candidate).
        final_picks: list[int] = [-1] * n_opp
        for i, h in enumerate(opp_remaining):
            if not h:
                continue
            others = [greedy_picks[j] for j in range(n_opp) if j != i and greedy_picks[j] != -1]
            if phantom_my_card != -1:
                others.append(phantom_my_card)
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
    use_phantom: bool = False,
) -> int:
    if not my_remaining:
        return 0
    if len(my_remaining) == 1:
        b = [row.copy() for row in board]
        rs = row_sums.copy()
        opps = [h.copy() for h in opp_remaining]
        phantom = my_remaining[0] if use_phantom else -1
        return _simulate_one_round_inplace(b, rs, my_remaining[0], opps, my_pid, opp_lookahead, phantom)

    best = -1
    for idx, my_card in enumerate(my_remaining):
        b = [row.copy() for row in board]
        rs = row_sums.copy()
        opps = [h.copy() for h in opp_remaining]
        # Phantom = a sibling card from my_remaining, so candidates are
        # evaluated against the same opp policy.
        phantom = my_remaining[(idx + 1) % len(my_remaining)] if use_phantom else -1
        gained = _simulate_one_round_inplace(b, rs, my_card, opps, my_pid, opp_lookahead, phantom)
        sub_my = my_remaining[:idx] + my_remaining[idx + 1 :]
        future = _endgame_best_response(b, rs, sub_my, opps, my_pid, opp_lookahead, use_phantom)
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
    use_phantom: bool = False,
    phantom_round1: int = -1,
) -> int:
    sim_board = [row.copy() for row in board]
    rs = _row_sums_from_board(sim_board)
    my_remaining = my_hand.copy()
    my_remaining.remove(fixed_first_card)
    opp_remaining = [h.copy() for h in opp_hands]

    rounds_left = len(my_hand)
    my_total = _simulate_one_round_inplace(
        sim_board, rs, fixed_first_card, opp_remaining, my_pid, opp_lookahead,
        phantom_round1 if use_phantom else -1,
    )

    if rounds_left == 1:
        return my_total

    if rounds_left <= endgame_threshold:
        my_total += _endgame_best_response(
            sim_board, rs, my_remaining, opp_remaining, my_pid, opp_lookahead, use_phantom
        )
        return my_total

    for _ in range(rounds_left - 1):
        my_card = _greedy_pick(my_remaining, sim_board, rs)
        my_remaining.remove(my_card)
        # Rounds 2+ are determined by greedy regardless of fixed_first_card,
        # so opps seeing my actual card here doesn't bias the comparison.
        my_total += _simulate_one_round_inplace(
            sim_board, rs, my_card, opp_remaining, my_pid, opp_lookahead,
            my_card if use_phantom else -1,
        )
    return my_total


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
    CFR+ regret-matching wrapped around determinized monte-carlo evaluation:
        - Regret Matching Plus (RM+) with linear strategy averaging.
        - Sacrifice-aware action pruning (`candidate_subset`).
        - Chance-sampled opp hands; common random numbers across candidates.
        - 1-ply lookahead opponents (toggleable) with phantom-card guard
          against my-card information leakage.
        - Optional Gaussian-weighted opp hand priors keyed on past plays.
        - Within-rollout endgame best-response over my remaining cards.
        - Exact endgame solver (full opp-hand enumeration) for small horizons.
        - Adaptive time guard from running iter cost.
    """

    def __init__(
        self,
        player_idx: int,
        opp_lookahead: bool = True,
        opp_weighted_sampling: bool = False,
        opp_lookahead_phantom: bool = False,
        weight_sigma: float = 20.0,
        endgame_threshold: int = 4,
    ) -> None:
        self.player_idx = player_idx
        self.rng = random.Random()

        self.time_budget_sec = 0.95
        self.max_actions = 7
        self.max_actions_with_sacrifice = 8

        self.exact_endgame_rounds = 3
        self.exact_endgame_max_assignments = 5000

        self.min_guard_sec = 0.015
        self.iter_safety_mult = 1.20
        self.avg_iter_sec = 0.004
        self.regret_decay = 1.0

        self.opp_lookahead = opp_lookahead
        self.opp_weighted_sampling = opp_weighted_sampling
        self.opp_lookahead_phantom = opp_lookahead_phantom
        self.weight_sigma = weight_sigma
        self.endgame_threshold = endgame_threshold

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
        use_phantom: bool,
        phantom_round1: int,
    ) -> float:
        total_loss = 0.0
        for opp_hands in _iter_hand_assignments(unseen, n_opponents, rounds_left):
            total_loss += float(
                _rollout_total_score(
                    board,
                    hand,
                    first_card,
                    opp_hands,
                    self.player_idx,
                    endgame_threshold=self.endgame_threshold,
                    opp_lookahead=self.opp_lookahead,
                    use_phantom=use_phantom,
                    phantom_round1=phantom_round1,
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
        my_pid = self.player_idx

        actions = self.candidate_subset(hand, board, row_sums)
        if len(actions) == 1:
            return actions[0]

        opp_weights = (
            _build_opp_weights(history, unseen, my_pid, self.weight_sigma)
            if self.opp_weighted_sampling
            else None
        )
        use_phantom = self.opp_lookahead and self.opp_lookahead_phantom

        use_exact_endgame = (
            rounds_left <= self.exact_endgame_rounds
            and len(unseen) == n_opponents * rounds_left
        )
        assignment_count = 0
        if use_exact_endgame:
            assignment_count = _count_hand_assignments(len(unseen), n_opponents, rounds_left)
            use_exact_endgame = 0 < assignment_count <= self.exact_endgame_max_assignments

        if use_exact_endgame:
            exact_losses: dict[int, float] = {}
            for idx, action in enumerate(actions):
                phantom = actions[(idx + 1) % len(actions)] if use_phantom else -1
                exact_losses[action] = self.exact_endgame_expected_loss(
                    board, hand, action, unseen, n_opponents, rounds_left,
                    assignment_count, use_phantom, phantom,
                )
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
            opp_hands = _sample_opp_hands(
                self.rng, unseen, n_opponents, rounds_left, opp_weights=opp_weights
            )
            phantom_base = self.rng.choice(hand) if use_phantom else -1

            utilities: dict[int, float] = {}
            for action in actions:
                if use_phantom and phantom_base == action:
                    others = [h for h in hand if h != action]
                    phantom = self.rng.choice(others) if others else -1
                else:
                    phantom = phantom_base
                loss = _rollout_total_score(
                    board, hand, action, opp_hands, my_pid,
                    endgame_threshold=self.endgame_threshold,
                    opp_lookahead=self.opp_lookahead,
                    use_phantom=use_phantom,
                    phantom_round1=phantom,
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
