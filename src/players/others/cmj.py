import random
import time

class CMJ:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        try:
            return self._action_impl(hand, history)
        except Exception:
            return hand[0]

    def _action_impl(self, hand, history):
        start_time = time.perf_counter()
        TIME_LIMIT = 0.90

        hand = list(hand)
        if not hand:
            return 1

        board = [list(row) for row in history.get("board", []) if row]
        if len(board) != 4:
            return hand[0]

        unknown = self._get_unknown(hand, board, history)

        best_card = hand[0]
        best_score = float("inf")

        if len(hand) >= 8:
            NUM_SIM = 120
        elif len(hand) >= 5:
            NUM_SIM = 160
        else:
            NUM_SIM = 250

        MIN_SIM = 35
        MARGIN = 0.8

        # 先用 heuristic 排序，讓好牌早點被找到，有利剪枝
        candidates = sorted(
            hand,
            key=lambda c: self._quick_eval(c, hand, board)
        )

        for card in candidates:
            if time.perf_counter() - start_time > TIME_LIMIT:
                return best_card

            total_cost = 0.0
            sims_done = 0

            future = 0.01 * self._future_risk(card, hand)

            for t in range(NUM_SIM):
                if time.perf_counter() - start_time > TIME_LIMIT:
                    return best_card

                cost = self._simulate_depth2(card, hand, board, unknown)
                total_cost += cost
                sims_done += 1

                lower_bound = total_cost / NUM_SIM + future
                if lower_bound > best_score:
                    break

                if sims_done >= MIN_SIM:
                    current_avg = total_cost / sims_done + future
                    if current_avg > best_score + MARGIN:
                        break

            if sims_done == 0:
                continue

            score = total_cost / sims_done + future

            if score < best_score:
                best_score = score
                best_card = card

        return best_card

    def _simulate_depth2(self, first_card, hand, board, unknown):
        """
        depth=2:
        1. 模擬本回合：自己出 first_card，對手隨機出 3 張
        2. 模擬下一回合：自己用剩餘手牌中 heuristic 最佳的一張，對手再隨機出 3 張
        回傳兩回合自己的總扣分。
        """
        sim_board = [row[:] for row in board]
        sim_unknown = unknown[:]

        # ---------- depth 1 ----------
        opp1 = self._sample_without_replacement(sim_unknown, 3)
        for c in opp1:
            self._remove_card(sim_unknown, c)

        played1 = [(-1, c) for c in opp1]
        played1.append((self.player_idx, first_card))
        played1.sort(key=lambda x: x[1])

        total_gain = 0

        for pid, c in played1:
            cost = self._place_card(sim_board, c)
            if pid == self.player_idx:
                total_gain += cost

        # ---------- depth 2 ----------
        remaining_hand = [c for c in hand if c != first_card]
        if not remaining_hand:
            return total_gain

        second_card = min(
            remaining_hand,
            key=lambda c: self._quick_eval(c, remaining_hand, sim_board)
        )

        opp2 = self._sample_without_replacement(sim_unknown, 3)
        for c in opp2:
            self._remove_card(sim_unknown, c)

        played2 = [(-1, c) for c in opp2]
        played2.append((self.player_idx, second_card))
        played2.sort(key=lambda x: x[1])

        for pid, c in played2:
            cost = self._place_card(sim_board, c)
            if pid == self.player_idx:
                total_gain += 0.65 * cost

        return total_gain

    def _quick_eval(self, card, hand, board):
        """
        快速 heuristic，用於：
        1. 排序候選牌
        2. depth=2 時選下一回合牌
        """
        target = self._target_row(board, card)

        if target is None:
            idx = self._choose_row_to_take(board)
            score = self._row_penalty(board[idx])
            if card == min(hand) and len(hand) > 3:
                score += 1.5
            return score

        idx, end = target
        row = board[idx]

        score = 0.0

        if len(row) >= 5:
            score += self._row_penalty(row) + 8.0
        elif len(row) == 4:
            score += 2.5
        elif len(row) == 3:
            score += 0.5

        score += (card - end) * 0.015
        score += 0.01 * self._bullheads(card)

        return score

    def _sample_without_replacement(self, arr, k):
        if not arr:
            return []
        if len(arr) <= k:
            return arr[:]
        return random.sample(arr, k)

    def _remove_card(self, arr, card):
        try:
            arr.remove(card)
        except ValueError:
            pass

    def _target_row(self, board, card):
        best_idx = None
        best_end = -1

        for i, row in enumerate(board):
            if not row:
                continue
            end = row[-1]
            if end < card and end > best_end:
                best_end = end
                best_idx = i

        if best_idx is None:
            return None

        return best_idx, best_end

    def _place_card(self, board, card):
        row_ends = [row[-1] for row in board if row]

        if len(row_ends) != 4:
            return 999

        if card < min(row_ends):
            idx = self._choose_row_to_take(board)
            penalty = self._row_penalty(board[idx])
            board[idx] = [card]
            return penalty

        best_idx = None
        best_end = -1

        for i, row in enumerate(board):
            if not row:
                continue
            end = row[-1]
            if end < card and end > best_end:
                best_end = end
                best_idx = i

        if best_idx is None:
            idx = self._choose_row_to_take(board)
            penalty = self._row_penalty(board[idx])
            board[idx] = [card]
            return penalty

        if len(board[best_idx]) >= 5:
            penalty = self._row_penalty(board[best_idx])
            board[best_idx] = [card]
            return penalty

        board[best_idx].append(card)
        return 0

    def _choose_row_to_take(self, board):
        best_idx = 0
        best_key = None

        for i, row in enumerate(board):
            key = (self._row_penalty(row), len(row), i)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i

        return best_idx

    def _row_penalty(self, row):
        return sum(self._bullheads(c) for c in row if isinstance(c, int))

    def _bullheads(self, card):
        if card == 55:
            return 55
        if card % 11 == 0:
            return 22
        if card % 10 == 0:
            return 10
        if card % 5 == 0:
            return 5
        return 1

    def _future_risk(self, card, hand):
        remaining = [c for c in hand if c != card]

        risk = 0.0

        if card == min(hand) and len(hand) > 3:
            risk += 3.0

        for c in remaining:
            risk += self._bullheads(c) * 0.2

        return risk

    def _get_unknown(self, hand, board, history):
        known = set(hand)

        for row in board:
            for c in row:
                if isinstance(c, int):
                    known.add(c)

        for row in history.get("history_matrix", []):
            if row is None:
                continue
            for c in row:
                if isinstance(c, int) and 1 <= c <= 104:
                    known.add(c)

        return [c for c in range(1, 105) if c not in known]