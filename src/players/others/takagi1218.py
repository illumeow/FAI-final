import time
import random
import math

class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_penalty = 0
        self.untried_moves = []

    def ucb1(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        avg_reward = -(self.total_penalty / self.visits) 
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration


class takagi1218:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self._BULLHEADS = [0] * 105
        for c in range(1, 105):
            if c == 55: self._BULLHEADS[c] = 7
            elif c % 11 == 0: self._BULLHEADS[c] = 5
            elif c % 10 == 0: self._BULLHEADS[c] = 3
            elif c % 5 == 0: self._BULLHEADS[c] = 2
            else: self._BULLHEADS[c] = 1

    def get_unseen_cards(self, hand, history):
        seen = set(hand)
        if history.get('board'):
            for row in history['board']:
                if row: seen.update(row)
        if history.get('history_matrix'):
            for round_moves in history['history_matrix']:
                if round_moves: seen.update(round_moves)
        return list(set(range(1, 105)) - seen)

    def determinize(self, unseen_cards, current_round):
        cards_per_opponent = 10 - current_round + 1
        shuffled = unseen_cards[:]
        random.shuffle(shuffled)

        opponents_hands = []
        for _ in range(3):
            deal_count = min(cards_per_opponent, len(shuffled))
            opp_hand = shuffled[:deal_count]
            shuffled = shuffled[deal_count:]
            opponents_hands.append(opp_hand)
            
        return opponents_hands

    def fast_simulate(self, my_first_card, my_hand, opp_hands, board):
        tails = [row[-1] for row in board]
        lengths = [len(row) for row in board]
        penalties = [sum(self._BULLHEADS[c] for c in row) for row in board]
        
        my_sim_hand = my_hand[:]
        my_sim_hand.remove(my_first_card)
        random.shuffle(my_sim_hand)
        
        opp_sim_hands = [h[:] for h in opp_hands]
        
        my_total_penalty = 0
        turns_left = len(my_sim_hand) + 1

        for turn in range(turns_left):
            played = []
            
            if turn == 0:
                played.append((my_first_card, 0))
            else:
                played.append((my_sim_hand.pop(), 0))
                
            for i in range(3):
                played.append((opp_sim_hands[i].pop(), i + 1))
                
            played.sort(key=lambda x: x[0])
            
            for card, player_id in played:
                target_idx = -1
                min_diff = 1000 
                
                for i in range(4):
                    t = tails[i]
                    if card > t:
                        diff = card - t
                        if diff < min_diff:
                            min_diff = diff
                            target_idx = i
                
                card_penalty = self._BULLHEADS[card]
                
                if target_idx == -1:
                    best_row = 0
                    min_pen = 1000
                    min_len = 10
                    for i in range(4):
                        p, L = penalties[i], lengths[i]
                        if p < min_pen or (p == min_pen and L < min_len):
                            min_pen = p
                            min_len = L
                            best_row = i
                    
                    if player_id == 0: my_total_penalty += min_pen
                    
                    tails[best_row] = card
                    lengths[best_row] = 1
                    penalties[best_row] = card_penalty
                    
                elif lengths[target_idx] == 5:
                    if player_id == 0: my_total_penalty += penalties[target_idx]
                    
                    tails[target_idx] = card
                    lengths[target_idx] = 1
                    penalties[target_idx] = card_penalty
                    
                else:
                    tails[target_idx] = card
                    lengths[target_idx] += 1
                    penalties[target_idx] += card_penalty
                    
        return my_total_penalty


    def action(self, hand, history):
        start_time = time.time()
        time_limit = 0.9 
        
        current_round = history.get('round', 1)
        board = history.get('board', [[], [], [], []])
        unseen_cards = self.get_unseen_cards(hand, history)

        root = Node()
        root.untried_moves = hand[:]

        simulations_run = 0

        while time.time() - start_time < time_limit:
            node = root
            
            opponents_hands = self.determinize(unseen_cards, current_round)
            
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                new_child = Node(move=move, parent=node)
                node.children.append(new_child)
                node = new_child
            else:
                if node.children:
                    node = max(node.children, key=lambda c: c.ucb1())

            simulated_penalty = self.fast_simulate(node.move, hand, opponents_hands, board)
            
            temp_node = node
            while temp_node is not None:
                temp_node.visits += 1
                temp_node.total_penalty += simulated_penalty
                temp_node = temp_node.parent        
            simulations_run += 1

        best_child = min(root.children, key=lambda c: c.total_penalty / c.visits if c.visits > 0 else float('inf'))
        
        return int(best_child.move)