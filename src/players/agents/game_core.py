import random
from typing import Any

class GameCore:
    def __init__(
        self,
        player_idx: int,
        n_cards: int = 104,
        seed_offset: int = 0,
        seed_stride: int = 7919,
    ) -> None:
        self.player_idx = player_idx
        self.n_cards = n_cards
        self.rng = random.Random(seed_offset + seed_stride * (player_idx + 1))

    def card_score(self, card: int) -> int:
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1

    def row_score(self, row: list[int]) -> int:
        return sum(self.card_score(card) for card in row)

    # best fit = largest last card < card
    # return row idx
    # return -1 if all rows' last > card
    def best_fit_row(self, board: list[list[int]], card: int) -> int:
        best_idx = -1
        best_last = -1
        for i, row in enumerate(board):
            # full rows
            if len(row) == 5:
                continue
            last = row[-1]
            if last < card and last > best_last:
                best_last = last
                best_idx = i
        return best_idx

    # the row with the lowest score, then fewest cards, then lowest index
    def forced_take_row(self, board: list[list[int]]) -> int:
        return min(
            range(len(board)),
            key=lambda i: (self.row_score(board[i]), len(board[i]), i)
        )

    # return the score incurred by placing the card
    def place_card(self, board: list[list[int]], card: int) -> int:
        fit_idx = self.best_fit_row(board, card)
        if fit_idx != -1:
            board[fit_idx].append(card)
            return 0

        take_idx = self.forced_take_row(board)
        incurred = self.row_score(board[take_idx])
        board[take_idx] = [card]
        return incurred

    # heuristic: (score incurred, is_take, delta to last card, new row length, card value)
    def heuristic_card_key(
        self,
        board: list[list[int]],
        card: int,
    ) -> tuple[int, int, int, int, int]:
        fit_idx = self.best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self.forced_take_row(board)
            score = self.row_score(board[take_idx])
            return (score, 1, 10**6, 1, card)

        delta = card - board[fit_idx][-1]
        row_len = len(board[fit_idx])
        return (0, 0, delta, row_len + 1, card)

    def greedy_pick(self, hand: list[int], board: list[list[int]]) -> int:
        return min(hand, key=lambda c: self.heuristic_card_key(board, c))

    def build_unseen_pool(self, hand: list[int], history: dict[str, Any]) -> list[int]:
        known = set(hand)

        for row in history["board"]:
            known.update(row)

        for past_board in history["board_history"]:
            for row in past_board:
                known.update(row)

        return [c for c in range(1, self.n_cards + 1) if c not in known]

    def sample_opponent_hands(
        self,
        unseen: list[int],
        n_opponents: int,
        rounds_left: int,
    ) -> list[list[int]]:
        needed = n_opponents * rounds_left
        if needed == 0:
            return []

        if len(unseen) >= needed:
            sampled = self.rng.sample(unseen, needed)
        elif unseen:
            sampled = [self.rng.choice(unseen) for _ in range(needed)]
        else:
            sampled = [self.rng.randint(1, self.n_cards) for _ in range(needed)]

        opp_hands: list[list[int]] = []
        idx = 0
        for _ in range(n_opponents):
            h = sampled[idx: idx + rounds_left]
            h.sort()
            opp_hands.append(h)
            idx += rounds_left
        return opp_hands
