from math import comb


class BitwiseSearchPlayer:
    """
    Bitset-based fast risk search using bit_count interval queries.
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

    def _build_unseen_mask(self, hand, history):
        known = set(hand)
        for row in history["board"]:
            known.update(row)
        for actions in history.get("history_matrix", []):
            known.update(actions)
        for past_board in history.get("board_history", []):
            for row in past_board:
                known.update(row)

        mask = 0
        for c in range(1, self.n_cards + 1):
            if c not in known:
                mask |= (1 << c)
        return mask

    def _between_count(self, mask, low, high):
        if high - low <= 1:
            return 0
        interval_mask = ((1 << high) - 1) ^ ((1 << (low + 1)) - 1)
        return (mask & interval_mask).bit_count()

    def _binom_tail(self, n, p, threshold):
        if threshold <= 0:
            return 1.0
        if threshold > n:
            return 0.0
        total = 0.0
        for k in range(threshold, n + 1):
            total += comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        return total

    def _expected_now(self, board, card, unseen_mask, unseen_cnt, n_opponents):
        fit_idx, fit_last = self._best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self._forced_take_row(board)
            return float(self._row_score(board[take_idx])), (1, 10**6, 1)

        row = board[fit_idx]
        row_len = len(row)
        row_score = self._row_score(row)
        delta = card - fit_last

        if unseen_cnt <= 0 or n_opponents <= 0:
            immediate = float(row_score) if row_len >= 5 else 0.0
            return immediate, (0, delta, row_len + 1)

        between = self._between_count(unseen_mask, fit_last, card)
        p_between = between / unseen_cnt

        if row_len >= 5:
            p_stable = (1.0 - p_between) ** n_opponents
            return float(row_score) * p_stable, (0, delta, 1)

        needed_before_me = 5 - row_len
        p_forced_take = self._binom_tail(n_opponents, p_between, needed_before_me)
        return float(row_score) * p_forced_take, (0, delta, row_len + 1)

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen_mask = self._build_unseen_mask(hand, history)
        unseen_cnt = unseen_mask.bit_count()

        best_card = hand[0]
        best_key = None

        for card in hand:
            expected, meta = self._expected_now(
                board,
                card,
                unseen_mask,
                unseen_cnt,
                n_opponents,
            )
            forced_low, delta, post_len = meta

            next_board, _, _ = self._apply_card(board, card)
            remaining = [c for c in hand if c != card]

            future_risk = 0.0
            if remaining:
                # One-step fast lookahead with the same bitwise risk model.
                nxt = []
                for nc in remaining:
                    enow, _ = self._expected_now(
                        next_board,
                        nc,
                        unseen_mask,
                        unseen_cnt,
                        n_opponents,
                    )
                    nxt.append(enow)
                future_risk = min(nxt)

            key = (
                expected + 0.60 * future_risk,
                expected,
                forced_low,
                delta,
                post_len,
                card,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_card = card

        return best_card
