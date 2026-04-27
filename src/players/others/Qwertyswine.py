import time
import random
import math

class Qwertyswine:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.bullheads = [0] * 105
        for i in range(1, 105):
            self.bullheads[i] = self._get_card_score(i)
            
        # The alpha multiplier for how much we penalize variance.
        # 0.5 means we care half as much about the spread of risk as we do the average.
        self.risk_factor = 0.5 

    def _get_card_score(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _get_risk_adjusted_score(self, card, penalties_dict, sq_penalties_dict, sims_dict):
        """Calculates EV + (Alpha * Standard Deviation)"""
        sims = max(1, sims_dict[card])
        mean = penalties_dict[card] / sims
        mean_sq = sq_penalties_dict[card] / sims
        
        # Max(0, ...) protects against tiny floating point inaccuracies
        variance = max(0.0, mean_sq - (mean * mean)) 
        std_dev = math.sqrt(variance)
        
        return mean + (self.risk_factor * std_dev)

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
        card_sq_penalties = {c: 0.0 for c in hand} # NEW: Tracking squares for Variance
        card_sims = {c: 0 for c in hand}

        # ---------------------------------------------------------
        # STAGE 1: Broad 1-Ply Search (0.00s to 0.35s)
        # ---------------------------------------------------------
        while time.perf_counter() - start_time < 0.35:
            opp_cards = random.sample(unseen_list, 3) if len(unseen_list) >= 3 else unseen_list + [0]*3
            for c in hand:
                tails, counts, penalties = base_tails[:], base_counts[:], base_penalties[:]
                penalty = self._simulate_round(c, opp_cards[:3], tails, counts, penalties)
                
                card_penalties[c] += penalty
                card_sq_penalties[c] += (penalty * penalty)
                card_sims[c] += 1

        # Sort using the RISK ADJUSTED SCORE, not just the mean
        sorted_cards = sorted(
            hand, 
            key=lambda c: self._get_risk_adjusted_score(c, card_penalties, card_sq_penalties, card_sims)
        )
        top_candidates = sorted_cards[:min(3, len(sorted_cards))]

        # ---------------------------------------------------------
        # STAGE 2: Deep 2-Ply Search (0.35s to 0.85s)
        # ---------------------------------------------------------
        deep_penalties = {c: 0.0 for c in top_candidates}
        deep_sq_penalties = {c: 0.0 for c in top_candidates} # NEW: Tracking squares for Stage 2
        deep_sims = {c: 0 for c in top_candidates}
        
        future_discount = 0.8 

        while time.perf_counter() - start_time < time_limit:
            if len(unseen_list) >= 6:
                opp_pool = random.sample(unseen_list, 6)
                opp1, opp2 = opp_pool[:3], opp_pool[3:]
            else:
                opp1 = random.sample(unseen_list, 3) if len(unseen_list) >= 3 else unseen_list + [0]*3
                opp2 = random.sample(unseen_list, 3) if len(unseen_list) >= 3 else unseen_list + [0]*3

            for c in top_candidates:
                tails, counts, penalties = base_tails[:], base_counts[:], base_penalties[:]
                
                p1 = self._simulate_round(c, opp1, tails, counts, penalties)
                
                p2 = 0
                if len(hand) > 1:
                    remaining_hand = [x for x in hand if x != c]
                    next_card = self._fast_greedy_pick(remaining_hand, tails)
                    p2 = self._simulate_round(next_card, opp2, tails, counts, penalties)

                total_p = p1 + (future_discount * p2)
                
                deep_penalties[c] += total_p
                deep_sq_penalties[c] += (total_p * total_p)
                deep_sims[c] += 1

        # Return the safest 2-Ply card using Risk Adjusted Scoring
        return min(
            top_candidates, 
            key=lambda c: self._get_risk_adjusted_score(c, deep_penalties, deep_sq_penalties, deep_sims)
        )

    def _fast_greedy_pick(self, hand, tails):
        best_c = hand[0] 
        best_diff = 1000
        
        for c in hand:
            d0 = c - tails[0] if c > tails[0] else 1000
            d1 = c - tails[1] if c > tails[1] else 1000
            d2 = c - tails[2] if c > tails[2] else 1000
            d3 = c - tails[3] if c > tails[3] else 1000
            
            min_d = min(d0, d1, d2, d3)
            if min_d < best_diff:
                best_diff = min_d
                best_c = c
                
        return best_c if best_diff != 1000 else hand[0]

    def _simulate_round(self, my_card, opp_cards, tails, counts, penalties):
        played = [my_card, opp_cards[0], opp_cards[1], opp_cards[2]]
        played.sort()

        my_penalty = 0

        for card in played:
            if card == 0: continue 
            
            best_diff = 1000
            best_idx = -1
            
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
                target_idx = 0
                best_eval = (penalties[0], counts[0], 0)
                
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
                    if is_mine: 
                        my_penalty += penalties[best_idx]
                        
                    tails[best_idx] = card
                    counts[best_idx] = 1
                    penalties[best_idx] = self.bullheads[card]
                else:
                    tails[best_idx] = card
                    counts[best_idx] += 1
                    penalties[best_idx] += self.bullheads[card]
                    
                    if is_mine and counts[best_idx] == 5:
                        my_penalty += penalties[best_idx] * 0.2

        return my_penalty
