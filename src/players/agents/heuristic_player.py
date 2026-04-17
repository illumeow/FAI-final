from math import comb


class HeuristicPlayer:
    """
    Improved heuristic player with:
    1) Expected immediate risk (models unknown opponent cards probabilistically)
    2) Greedy two-step lookahead over own remaining cards
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.n_cards = 104

    def _card_score(self, card):
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1

    def _row_score(self, row):
        return sum(self._card_score(card) for card in row)

    def _best_fit_row(self, board, card):
        best_idx = -1
        best_last = -1
        for i, row in enumerate(board):
            last = row[-1]
            if last < card and last > best_last:
                best_last = last
                best_idx = i
        return best_idx, best_last

    def _forced_take_row(self, board):
        return min(
            range(len(board)),
            key=lambda i: (self._row_score(board[i]), len(board[i]), i),
        )

    def _apply_card(self, board, card):
        new_board = [row.copy() for row in board]
        fit_idx, fit_last = self._best_fit_row(new_board, card)

        if fit_idx == -1:
            take_idx = self._forced_take_row(new_board)
            penalty = self._row_score(new_board[take_idx])
            new_board[take_idx] = [card]
            return new_board, penalty, (1, 10**6, 1)

        row_len = len(new_board[fit_idx])
        delta = card - fit_last
        if row_len >= 5:
            penalty = self._row_score(new_board[fit_idx])
            new_board[fit_idx] = [card]
            return new_board, penalty, (0, delta, 1)

        new_board[fit_idx].append(card)
        return new_board, 0, (0, delta, row_len + 1)

    def _build_unseen_pool(self, hand, history):
        known = set(hand)

        for row in history["board"]:
            known.update(row)

        for actions in history.get("history_matrix", []):
            known.update(actions)

        for past_board in history.get("board_history", []):
            for row in past_board:
                known.update(row)

        return [c for c in range(1, self.n_cards + 1) if c not in known]

    def _binom_tail(self, n, p, threshold):
        if threshold <= 0:
            return 1.0
        if threshold > n:
            return 0.0
        total = 0.0
        for k in range(threshold, n + 1):
            total += comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        return total

    @staticmethod
    def _between_probability(unseen, low, high):
        if not unseen or high - low <= 1:
            return 0.0
        between = 0
        for card in unseen:
            if low < card < high:
                between += 1
        return between / len(unseen)

    def _expected_current_round_penalty(self, board, card, unseen, n_opponents):
        fit_idx, fit_last = self._best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self._forced_take_row(board)
            return float(self._row_score(board[take_idx])), (1, 10**6, 1)

        row = board[fit_idx]
        row_len = len(row)
        row_score = self._row_score(row)
        delta = card - fit_last

        if row_len >= 5:
            if not unseen or n_opponents <= 0:
                return float(row_score), (0, delta, 1)
            p_between = self._between_probability(unseen, fit_last, card)
            p_stable = (1.0 - p_between) ** n_opponents
            return float(row_score) * p_stable, (0, delta, 1)

        if not unseen or n_opponents <= 0:
            return 0.0, (0, delta, row_len + 1)

        p_between = self._between_probability(unseen, fit_last, card)
        needed_before_me = 5 - row_len
        p_forced_take = self._binom_tail(n_opponents, p_between, needed_before_me)
        expected = float(row_score) * p_forced_take
        return expected, (0, delta, row_len + 1)

    def _future_greedy_cost(self, board, remaining_hand):
        """Two-step greedy estimate ignoring hidden-opponent permutations."""
        if not remaining_hand:
            return 0.0

        first_step = []
        for c in remaining_hand:
            next_board, penalty, meta = self._apply_card(board, c)
            forced_low, delta, post_len = meta
            local = (
                float(penalty)
                + forced_low * 0.8
                + (delta / 70.0)
                + (post_len / 14.0)
                - (self._card_score(c) / 8.0)
            )
            first_step.append((local, c, next_board))

        best, best_card, best_board = min(first_step, key=lambda x: x[0])

        if len(remaining_hand) == 1:
            return best

        # One extra greedy layer.
        rem2 = [x for x in remaining_hand if x != best_card]
        if not rem2:
            return best

        second = float("inf")
        for c in rem2:
            _, p2, (f2, d2, l2) = self._apply_card(best_board, c)
            v2 = float(p2) + f2 * 0.8 + (d2 / 75.0) + (l2 / 15.0)
            if v2 < second:
                second = v2

        return best + 0.55 * second

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        unseen = self._build_unseen_pool(hand, history)
        n_opponents = max(0, len(history["scores"]) - 1)

        def decision_key(card):
            expected_now, meta = self._expected_current_round_penalty(
                board,
                card,
                unseen,
                n_opponents,
            )
            forced_low, delta, post_len = meta

            next_board, _, _ = self._apply_card(board, card)
            remaining = [c for c in hand if c != card]
            future = self._future_greedy_cost(next_board, remaining)

            key = (
                expected_now + future,
                expected_now,
                forced_low,
                delta,
                post_len,
                -self._card_score(card),
                card,
            )
            return key

        return min(hand, key=decision_key)
