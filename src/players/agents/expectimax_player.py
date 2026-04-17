import time

from .core_utils import SimpleAgentCore


class ExpectimaxPlayer:
    """
    Stochastic expectimax with sampled chance nodes.
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.core = SimpleAgentCore(player_idx, seed_offset=17011)
        self.time_budget_sec = 0.84
        self.depth = 2
        self.branching = 4
        self.samples_per_chance = 6

    def _candidate_cards(self, hand, board):
        return sorted(hand, key=lambda c: self.core.heuristic_card_key(board, c))[
            : self.branching
        ]

    def _sample_current_opp_cards(self, unseen, n_opponents):
        if n_opponents == 0:
            return []
        if len(unseen) >= n_opponents:
            return self.core.rng.sample(unseen, n_opponents)
        if unseen:
            return [self.core.rng.choice(unseen) for _ in range(n_opponents)]
        return [self.core.rng.randint(1, self.core.n_cards) for _ in range(n_opponents)]

    def _next_unseen(self, unseen, opp_cards):
        if not opp_cards:
            return unseen
        opp_set = set(opp_cards)
        return [u for u in unseen if u not in opp_set]

    def _estimate_action_value(
        self,
        board,
        hand,
        my_card,
        unseen,
        n_opponents,
        depth,
        deadline,
        num_samples,
    ):
        total = 0.0
        used = 0
        for _ in range(num_samples):
            if time.perf_counter() >= deadline:
                break
            opp_cards = self._sample_current_opp_cards(unseen, n_opponents)
            next_board, my_penalty = self.core.apply_round(
                board, my_card, opp_cards, self.player_idx
            )
            remaining_hand = [c for c in hand if c != my_card]
            next_unseen = self._next_unseen(unseen, opp_cards)
            future = self._expecti_value(
                next_board,
                remaining_hand,
                next_unseen,
                n_opponents,
                depth,
                deadline,
            )
            total += float(my_penalty) + future
            used += 1

        if used == 0:
            return float("inf")
        return total / used

    def _expecti_value(self, board, hand, unseen, n_opponents, depth, deadline):
        if depth <= 0 or not hand:
            return 0.0

        if time.perf_counter() >= deadline:
            # Fast fallback score when out of time.
            best = min(hand, key=lambda c: self.core.heuristic_card_key(board, c))
            _, immediate = self.core.apply_round(board, best, [], self.player_idx)
            return float(immediate)

        best_value = float("inf")
        candidates = self._candidate_cards(hand, board)

        for my_card in candidates:
            if time.perf_counter() >= deadline:
                break

            value = self._estimate_action_value(
                board,
                hand,
                my_card,
                unseen,
                n_opponents,
                depth - 1,
                deadline,
                self.samples_per_chance,
            )

            if value < best_value:
                best_value = value

        return best_value

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_opponents = max(0, len(history["scores"]) - 1)
        unseen = self.core.build_unseen_pool(hand, history)
        deadline = time.perf_counter() + self.time_budget_sec

        best_card = hand[0]
        best_value = float("inf")

        for my_card in self._candidate_cards(hand, board):
            if time.perf_counter() >= deadline:
                break

            value = self._estimate_action_value(
                board,
                hand,
                my_card,
                unseen,
                n_opponents,
                self.depth - 1,
                deadline,
                self.samples_per_chance + 2,
            )
            if value < best_value:
                best_value = value
                best_card = my_card

        return best_card
