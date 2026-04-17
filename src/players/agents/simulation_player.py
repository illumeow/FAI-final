import time
import random


class SimulationPlayer:
    """
    Time-budgeted Monte Carlo player.

    It samples hidden opponent hands (determinization) and rolls out the
    remaining round sequence with lightweight greedy policies.
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.n_cards = 104
        self.rng = random.Random(40009 + player_idx)
        self.time_budget_sec = 0.92
        self.min_samples_per_card = 4

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

    def _place_card(self, board, card):
        fit_idx, _ = self._best_fit_row(board, card)
        if fit_idx != -1:
            if len(board[fit_idx]) >= 5:
                incurred = self._row_score(board[fit_idx])
                board[fit_idx] = [card]
                return incurred
            board[fit_idx].append(card)
            return 0

        take_idx = self._forced_take_row(board)
        incurred = self._row_score(board[take_idx])
        board[take_idx] = [card]
        return incurred

    def _heuristic_card_key(self, board, card):
        fit_idx, fit_last = self._best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self._forced_take_row(board)
            penalty = self._row_score(board[take_idx])
            return (penalty, 1, 10**6, 1, card)

        delta = card - fit_last
        row_len = len(board[fit_idx])
        penalty = self._row_score(board[fit_idx]) if row_len >= 5 else 0
        return (penalty, 0, delta, row_len + 1, card)

    def _greedy_pick(self, hand, board):
        return min(hand, key=lambda card: self._heuristic_card_key(board, card))

    @staticmethod
    def _mean_loss(totals, counts, card):
        if counts[card] == 0:
            return float("inf")
        return totals[card] / counts[card]

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

    def _sample_opponent_hands(self, unseen, n_opponents, rounds_left):
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

    def _rollout_total_penalty(self, board, my_hand, fixed_first_card, opp_hands):
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
                my_card = self._greedy_pick(me, sim_board)
                me.remove(my_card)

            played = [(my_card, self.player_idx)]

            for i, oh in enumerate(opp):
                if not oh:
                    continue
                oc = self._greedy_pick(oh, sim_board)
                oh.remove(oc)
                played.append((oc, -1000 - i))

            played.sort(key=lambda x: x[0])
            for card, pid in played:
                gained = self._place_card(sim_board, card)
                if pid == self.player_idx:
                    my_total_penalty += gained

        return my_total_penalty

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        deadline = time.perf_counter() + self.time_budget_sec

        board = history["board"]
        n_players = len(history["scores"])
        n_opponents = max(0, n_players - 1)
        rounds_left = len(hand)
        unseen = self._build_unseen_pool(hand, history)

        candidates = list(hand)
        totals = {c: 0.0 for c in candidates}
        counts = {c: 0 for c in candidates}

        # Phase 1: guarantee a minimum number of samples per candidate.
        for c in candidates:
            for _ in range(self.min_samples_per_card):
                if time.perf_counter() >= deadline:
                    break
                opp_hands = self._sample_opponent_hands(unseen, n_opponents, rounds_left)
                loss = self._rollout_total_penalty(board, hand, c, opp_hands)
                totals[c] += loss
                counts[c] += 1

        # Phase 2: allocate remaining time adaptively (focus on best candidates).
        active = candidates[:]
        pulls = 0
        while active and time.perf_counter() < deadline:
            # Optimistic exploration: low mean first, then low sample count.
            active.sort(
                key=lambda c: (
                    self._mean_loss(totals, counts, c),
                    counts[c],
                    c,
                )
            )

            c = active[pulls % len(active)]
            opp_hands = self._sample_opponent_hands(unseen, n_opponents, rounds_left)
            loss = self._rollout_total_penalty(board, hand, c, opp_hands)
            totals[c] += loss
            counts[c] += 1
            pulls += 1

            # Periodically prune weak candidates to spend time on contenders.
            if pulls % max(12, len(candidates) * 2) == 0 and len(active) > 3:
                ranked = sorted(
                    active,
                    key=lambda x: (self._mean_loss(totals, counts, x), x),
                )
                keep = max(3, len(active) // 2)
                active = ranked[:keep]

        best_card = min(
            candidates,
            key=lambda c: (
                self._mean_loss(totals, counts, c),
                self._heuristic_card_key(board, c),
            ),
        )
        return best_card
