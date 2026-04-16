import random


class SimulationPlayer:
    def __init__(self, player_idx, simulations=96, n_cards=104, seed=None):
        self.player_idx = player_idx
        self.simulations = int(simulations)
        self.n_cards = int(n_cards)
        self.rng = random.Random(seed if seed is not None else (9973 + player_idx))

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

    def _place_card(self, board, card):
        fit_row = -1
        fit_last = -1
        for i, row in enumerate(board):
            last_card = row[-1]
            if last_card < card and last_card > fit_last:
                fit_last = last_card
                fit_row = i

        if fit_row != -1:
            if len(board[fit_row]) >= 5:
                incurred = self._row_score(board[fit_row])
                board[fit_row] = [card]
                return incurred
            board[fit_row].append(card)
            return 0

        take_row = min(
            range(len(board)),
            key=lambda i: (self._row_score(board[i]), len(board[i]), i),
        )
        incurred = self._row_score(board[take_row])
        board[take_row] = [card]
        return incurred

    def _heuristic_tiebreak(self, board, card):
        fit_row = -1
        fit_last = -1
        for i, row in enumerate(board):
            last_card = row[-1]
            if last_card < card and last_card > fit_last:
                fit_last = last_card
                fit_row = i

        if fit_row == -1:
            take_row = min(
                range(len(board)),
                key=lambda i: (self._row_score(board[i]), len(board[i]), i),
            )
            return (self._row_score(board[take_row]), 1, 10**6, card)

        immediate = self._row_score(board[fit_row]) if len(board[fit_row]) >= 5 else 0
        return (immediate, 0, card - fit_last, card)

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

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_players = len(history["scores"])
        n_opponents = max(0, n_players - 1)
        unseen = self._build_unseen_pool(hand, history)

        best_card = hand[0]
        best_key = None

        for my_card in hand:
            total_penalty = 0.0

            for _ in range(self.simulations):
                if n_opponents == 0:
                    opp_cards = []
                elif len(unseen) >= n_opponents:
                    opp_cards = self.rng.sample(unseen, n_opponents)
                elif unseen:
                    opp_cards = [self.rng.choice(unseen) for _ in range(n_opponents)]
                else:
                    # Extremely defensive fallback.
                    opp_cards = [self.rng.randint(1, self.n_cards) for _ in range(n_opponents)]

                round_cards = [(my_card, self.player_idx)]
                round_cards.extend((oc, -1000 - i) for i, oc in enumerate(opp_cards))
                round_cards.sort(key=lambda x: x[0])

                sim_board = [row.copy() for row in board]
                my_penalty = 0
                for played_card, played_pid in round_cards:
                    incurred = self._place_card(sim_board, played_card)
                    if played_pid == self.player_idx:
                        my_penalty = incurred

                total_penalty += my_penalty

            expected_penalty = total_penalty / self.simulations
            tiebreak = self._heuristic_tiebreak(board, my_card)
            key = (expected_penalty,) + tiebreak

            if best_key is None or key < best_key:
                best_key = key
                best_card = my_card

        return best_card
