import random


class SimpleAgentCore:
    def __init__(self, player_idx, n_cards=104, seed_offset=0):
        self.player_idx = player_idx
        self.n_cards = n_cards
        self.rng = random.Random(seed_offset + 7919 * (player_idx + 1))

    def card_score(self, card):
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1

    def row_score(self, row):
        return sum(self.card_score(card) for card in row)

    def best_fit_row(self, board, card):
        best_idx = -1
        best_last = -1
        for i, row in enumerate(board):
            last = row[-1]
            if last < card and last > best_last:
                best_last = last
                best_idx = i
        return best_idx, best_last

    def forced_take_row(self, board):
        return min(
            range(len(board)),
            key=lambda i: (self.row_score(board[i]), len(board[i]), i),
        )

    def place_card(self, board, card):
        fit_idx, _ = self.best_fit_row(board, card)
        if fit_idx != -1:
            if len(board[fit_idx]) >= 5:
                incurred = self.row_score(board[fit_idx])
                board[fit_idx] = [card]
                return incurred
            board[fit_idx].append(card)
            return 0

        take_idx = self.forced_take_row(board)
        incurred = self.row_score(board[take_idx])
        board[take_idx] = [card]
        return incurred

    def heuristic_card_key(self, board, card):
        fit_idx, fit_last = self.best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self.forced_take_row(board)
            penalty = self.row_score(board[take_idx])
            return (penalty, 1, 10**6, 1, card)

        delta = card - fit_last
        row_len = len(board[fit_idx])
        penalty = self.row_score(board[fit_idx]) if row_len >= 5 else 0
        return (penalty, 0, delta, row_len + 1, card)

    def greedy_pick(self, hand, board):
        best = hand[0]
        best_key = self.heuristic_card_key(board, best)
        for card in hand[1:]:
            key = self.heuristic_card_key(board, card)
            if key < best_key:
                best_key = key
                best = card
        return best

    def build_unseen_pool(self, hand, history):
        known = set(hand)

        for row in history["board"]:
            known.update(row)

        for actions in history.get("history_matrix", []):
            known.update(actions)

        for past_board in history.get("board_history", []):
            for row in past_board:
                known.update(row)

        return [c for c in range(1, self.n_cards + 1) if c not in known]

    def sample_opponent_hands(self, unseen, n_opponents, rounds_left):
        needed = n_opponents * rounds_left
        if needed == 0:
            return []

        if len(unseen) >= needed:
            sampled = self.rng.sample(unseen, needed)
        elif unseen:
            sampled = [self.rng.choice(unseen) for _ in range(needed)]
        else:
            sampled = [self.rng.randint(1, self.n_cards) for _ in range(needed)]

        opp_hands = []
        idx = 0
        for _ in range(n_opponents):
            h = sampled[idx: idx + rounds_left]
            h.sort()
            opp_hands.append(h)
            idx += rounds_left
        return opp_hands

    def apply_round(self, board, my_card, opponent_cards, my_id):
        sim_board = [row.copy() for row in board]
        played = [(my_card, my_id)]
        played.extend((oc, -1000 - i) for i, oc in enumerate(opponent_cards))
        played.sort(key=lambda x: x[0])

        my_penalty = 0
        for card, pid in played:
            gained = self.place_card(sim_board, card)
            if pid == my_id:
                my_penalty += gained
        return sim_board, my_penalty

    def rollout_total_penalty(self, board, my_hand, fixed_first_card, opp_hands):
        sim_board = [row.copy() for row in board]
        me = sorted(my_hand)
        me.remove(fixed_first_card)

        opp = [h.copy() for h in opp_hands]
        rounds_left = len(my_hand)
        my_total_penalty = 0

        for r in range(rounds_left):
            if r == 0:
                my_card = fixed_first_card
            else:
                my_card = self.greedy_pick(me, sim_board)
                me.remove(my_card)

            played = [(my_card, self.player_idx)]
            for i, oh in enumerate(opp):
                if not oh:
                    continue
                oc = self.greedy_pick(oh, sim_board)
                oh.remove(oc)
                played.append((oc, -1000 - i))

            played.sort(key=lambda x: x[0])
            for card, pid in played:
                gained = self.place_card(sim_board, card)
                if pid == self.player_idx:
                    my_total_penalty += gained

        return my_total_penalty
