class HeuristicPlayer:
    def __init__(self, player_idx):
        self.player_idx = player_idx

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
        best_row = -1
        best_last = -1
        for i, row in enumerate(board):
            last_card = row[-1]
            if last_card < card and last_card > best_last:
                best_last = last_card
                best_row = i
        return best_row, best_last

    def _forced_take_row(self, board):
        return min(
            range(len(board)),
            key=lambda i: (self._row_score(board[i]), len(board[i]), i),
        )

    def _score_card(self, board, card):
        fit_row, fit_last = self._best_fit_row(board, card)

        if fit_row == -1:
            # Card is lower than all row ends, so we must take a row.
            take_row = self._forced_take_row(board)
            immediate_penalty = self._row_score(board[take_row])
            forced_low = 1
            delta = 10**6
            post_row_len = 1
        else:
            forced_low = 0
            delta = card - fit_last
            post_row_len = len(board[fit_row]) + 1
            if len(board[fit_row]) >= 5:
                immediate_penalty = self._row_score(board[fit_row])
                post_row_len = 1
            else:
                immediate_penalty = 0

        # Prefer discarding high bullhead cards when move is otherwise safe.
        safe_dump_bonus = -self._card_score(card) if immediate_penalty == 0 else 0

        return (
            immediate_penalty,
            forced_low,
            delta,
            post_row_len,
            safe_dump_bonus,
            card,
        )

    def action(self, hand, history):
        board = history["board"]
        return min(hand, key=lambda c: self._score_card(board, c))
