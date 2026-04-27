import random
import time


class penguin5728:
	"""Two-step Monte Carlo agent with controlled lookahead and fixed sampling budgets."""

	def __init__(self, player_idx):
		self.player_idx = player_idx
		self.rng = random.Random(12345 + player_idx)
		self.card_score = [0] + [self._bullheads(card) for card in range(1, 105)]

	@staticmethod
	def _bullheads(card):
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
		return sum(self.card_score[c] for c in row)

	def _best_row(self, board, card):
		best_row_idx = -1
		best_last = -1
		for idx, row in enumerate(board):
			last = row[-1]
			if last < card and last > best_last:
				best_last = last
				best_row_idx = idx
		return best_row_idx

	def _row_choice_for_low_card(self, board):
		best_idx = 0
		best_key = (self._row_score(board[0]), len(board[0]), 0)
		for i in range(1, len(board)):
			key = (self._row_score(board[i]), len(board[i]), i)
			if key < best_key:
				best_key = key
				best_idx = i
		return best_idx

	def _place_card(self, board, card):
		best_row_idx = self._best_row(board, card)
		if best_row_idx != -1:
			row = board[best_row_idx]
			if len(row) >= 5:
				penalty = self._row_score(row)
				board[best_row_idx] = [card]
				return penalty
			row.append(card)
			return 0

		low_idx = self._row_choice_for_low_card(board)
		penalty = self._row_score(board[low_idx])
		board[low_idx] = [card]
		return penalty

	def _play_round(self, board, my_card, opp_cards):
		played = [(my_card, True)]
		for c in opp_cards:
			played.append((c, False))
		played.sort(key=lambda item: item[0])

		my_penalty = 0
		for card, is_me in played:
			p = self._place_card(board, card)
			if is_me:
				my_penalty = p
		return my_penalty

	def _collect_unseen(self, hand, history):
		seen = set(hand)
		for row in history["board"]:
			seen.update(row)
		for round_cards in history.get("history_matrix", []):
			seen.update(round_cards)
		return [card for card in range(1, 105) if card not in seen]

	def _immediate_penalty(self, board, card):
		board_copy = [row[:] for row in board]
		return self._place_card(board_copy, card)

	def _candidate_cards(self, hand, board):
		"""Return top 5-8 candidate cards sorted by immediate penalty."""
		scored = []
		for card in hand:
			immediate = self._immediate_penalty(board, card)
			best_gap = 999
			best_row_idx = self._best_row(board, card)
			if best_row_idx != -1:
				best_gap = card - board[best_row_idx][-1]
			scored.append((card, immediate, best_gap))

		scored.sort(key=lambda x: (x[1], x[2], x[0]))
		limit = 8 if len(hand) >= 8 else len(hand)
		return [item[0] for item in scored[:limit]]

	def _step1_evaluate(self, board, card, unseen, opp_count, samples):
		"""Step 1: Evaluate immediate penalty by sampling current round outcomes."""
		if opp_count <= 0 or not unseen:
			return float(self._immediate_penalty(board, card))

		total = 0.0
		unseen_len = len(unseen)
		for _ in range(samples):
			if unseen_len >= opp_count:
				opp_cards = self.rng.sample(unseen, opp_count)
			else:
				opp_cards = unseen[:]
			sim_board = [row[:] for row in board]
			total += self._play_round(sim_board, card, opp_cards)

		return total / samples

	def _step2_evaluate(self, board, card, unseen, opp_count, samples):
		"""Step 2: Simulate board after current move, sample next-round board penalty."""
		if opp_count <= 0 or not unseen:
			return 0.0

		total = 0.0
		unseen_len = len(unseen)

		for _ in range(samples):
			if unseen_len >= opp_count:
				opp_cards = self.rng.sample(unseen, opp_count)
			else:
				opp_cards = unseen[:]

			# Simulate current round
			sim_board = [row[:] for row in board]
			self._play_round(sim_board, card, opp_cards)

			# Estimate next-round penalty: sample one opponent card and evaluate
			# This gives us a sense of board difficulty for next round
			remaining_unseen = [c for c in unseen if c not in opp_cards and c != card]
			if remaining_unseen and opp_count > 0:
				# Sample a single next-round card to gauge board state
				for _ in range(min(2, len(remaining_unseen))):
					next_card = self.rng.choice(remaining_unseen)
					future_penalty = self._immediate_penalty(sim_board, next_card)
					total += future_penalty * 0.5  # Weight future penalty less than immediate
			else:
				# No future cards available; use immediate board risk as proxy
				total += float(sum(self._row_score(row) for row in sim_board)) * 0.05

		return total / samples if samples > 0 else 0.0

	def action(self, hand, history):
		if len(hand) == 1:
			return hand[0]

		board = history["board"]
		round_no = history["round"]
		opp_count = max(0, len(history["scores"]) - 1)
		unseen = self._collect_unseen(hand, history)
		candidates = self._candidate_cards(hand, board)

		# Fixed budgets: 10 samples for step 1, 3-4 for step 2 lookahead
		step1_samples = 10
		step2_samples = 3 if round_no < 7 else 2

		best_card = candidates[0]
		best_score = float("inf")
		deadline = time.perf_counter() + 0.74

		for card in candidates:
			if time.perf_counter() >= deadline:
				break

			# Step 1: Current round penalty
			penalty1 = self._step1_evaluate(board, card, unseen, opp_count, step1_samples)

			# Step 2: Next-round board state penalty (lookahead)
			penalty2 = self._step2_evaluate(board, card, unseen, opp_count, step2_samples)

			# Combine: weight current round more heavily
			total_score = penalty1 * 0.85 + penalty2 * 0.15

			if total_score < best_score or (total_score == best_score and card > best_card):
				best_score = total_score
				best_card = card

		return best_card
