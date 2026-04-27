import time
import random

class Qwertypig:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.bullheads = [0] * 105
        for i in range(1, 105):
            self.bullheads[i] = self._get_card_score(i)

    def _get_card_score(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def action(self, hand, history):
        start_time = time.perf_counter()
        time_limit = 0.85 

        unseen = set(range(1, 105))
        unseen.difference_update(hand)
        for row in history["board"]: unseen.difference_update(row)
        if history.get("history_matrix"):
            for round_actions in history["history_matrix"]: unseen.difference_update(round_actions)
        unseen_list = list(unseen)

        if len(hand) == 1: return hand[0]

        base_tails = [row[-1] for row in history["board"]]
        base_counts = [len(row) for row in history["board"]]
        base_penalties = [sum(self.bullheads[c] for c in row) for row in history["board"]]

        card_penalties = {c: 0.0 for c in hand}
        card_sims = {c: 0 for c in hand}

        # STAGE 1: Broad Search (0.00s to 0.35s)
        # Evaluate all cards evenly to get a highly robust baseline
        while time.perf_counter() - start_time < 0.35:
            opp_cards = random.sample(unseen_list, 3) if len(unseen_list) >= 3 else unseen_list + [0]*3
            for c in hand:
                tails, counts, penalties = base_tails[:], base_counts[:], base_penalties[:]
                penalty = self._simulate_round(c, opp_cards[:3], tails, counts, penalties)
                card_penalties[c] += penalty
                card_sims[c] += 1

        # Determine the top 3 safest cards from Stage 1
        sorted_cards = sorted(hand, key=lambda c: card_penalties[c] / max(1, card_sims[c]))
        top_candidates = sorted_cards[:min(3, len(sorted_cards))]

        # STAGE 2: Deep Search (0.35s to 0.85s)
        # Focus all remaining CPU time on differentiating the 3 best options
        while time.perf_counter() - start_time < time_limit:
            opp_cards = random.sample(unseen_list, 3) if len(unseen_list) >= 3 else unseen_list + [0]*3
            for c in top_candidates:
                tails, counts, penalties = base_tails[:], base_counts[:], base_penalties[:]
                penalty = self._simulate_round(c, opp_cards[:3], tails, counts, penalties)
                card_penalties[c] += penalty
                card_sims[c] += 1

        return min(top_candidates, key=lambda c: card_penalties[c] / max(1, card_sims[c]))

    def _simulate_round(self, my_card, opp_cards, tails, counts, penalties):
        # We know exactly 4 cards are played. Sort them manually.
        played = [my_card, opp_cards[0], opp_cards[1], opp_cards[2]]
        played.sort()

        my_penalty = 0

        for card in played:
            best_diff = 1000
            best_idx = -1
            
            # Loop Unrolling: Hardcode the 4 row checks to avoid iterator overhead.
            # Variables are re-assigned every loop iteration correctly.
            if tails[0] < card:
                diff = card - tails[0]
                if diff < best_diff: best_diff = diff; best_idx = 0
            if tails[1] < card:
                diff = card - tails[1]
                if diff < best_diff: best_diff = diff; best_idx = 1
            if tails[2] < card:
                diff = card - tails[2]
                if diff < best_diff: best_diff = diff; best_idx = 2
            if tails[3] < card:
                diff = card - tails[3]
                if diff < best_diff: best_diff = diff; best_idx = 3

            is_mine = (card == my_card)

            if best_idx == -1:
                # Automated Selection: Least penalty -> Less cards -> Smallest index
                target_idx = 0
                best_eval = (penalties[0], counts[0], 0)
                
                # Unroll the tuple evaluation
                for i in range(1, 4):
                    ev = (penalties[i], counts[i], i)
                    if ev < best_eval:
                        best_eval = ev
                        target_idx = i
                
                if is_mine: 
                    my_penalty += penalties[target_idx]
                    
                tails[target_idx] = card
                counts[target_idx] = 1
                penalties[target_idx] = self.bullheads[card]
            else:
                if counts[best_idx] == 5:
                    # 6th Card Rule Triggered
                    if is_mine: 
                        my_penalty += penalties[best_idx]
                        
                    tails[best_idx] = card
                    counts[best_idx] = 1
                    penalties[best_idx] = self.bullheads[card]
                else:
                    # Safe Placement
                    tails[best_idx] = card
                    counts[best_idx] += 1
                    penalties[best_idx] += self.bullheads[card]
                    
                    # Targeted Danger Evaluation:
                    # If I safely place the 5th card, I've created a trap. 
                    # Add a minor risk penalty ONLY if my card caused it.
                    if is_mine and counts[best_idx] == 5:
                        my_penalty += penalties[best_idx] * 0.2

        return my_penalty
