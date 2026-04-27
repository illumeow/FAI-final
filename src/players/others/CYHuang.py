"""
MCS (Monte Carlo Sampling) Player - Model 0 baseline for 6 nimmt!.

Strategy
--------
For each candidate card c in our hand:
  - Sample many possible "this-round" outcomes:
      * Sample opponents' hands uniformly from the unseen-card pool.
      * Sample one card per opponent uniformly from their sampled hand
        (this is the move they will play this round).
      * Simulate the engine's placement rule for the 4 cards being played
        and compute the points WE incur this round.
  - Q(c) = mean points incurred when we play c.
Pick argmin_c Q(c).

Depth 1 only (this round). Per the ETH paper, raw sample count matters
more than rollout depth for this game.

Safety
------
- Strict wall-clock budget enforced with time.monotonic(); we exit cleanly
  before the engine's SIGALRM fires.
- No multiprocessing / threading.
- Only specific exceptions are caught; never `except:` or BaseException.
"""

import random
import time


# ----------------------------- constants -----------------------------------

N_CARDS = 104
N_ROWS = 4
ROW_CAPACITY = 5  # 6th card takes the row

# Time budget for action(). The engine timeout default is 1.0s with a 0.5s
# buffer, but THE 0.5s BUFFER IS NOT FREE TIME. The engine measures the
# wall-clock between calling player.action() and getting control back; if
# that measurement exceeds 1.0 + 0.5 = 1.5s, we get DISQUALIFIED (it assumes
# we swallowed the TimeoutException). Sources of overhead between our last
# loop iteration and the engine re-taking control include:
#   - SIGALRM handler invocation if we run past 1.0s
#   - Python's exception unwinding, finally block in the engine
#   - subprocess scheduling jitter under joblib parallelism
# So we keep our self-imposed deadline well under 1.0s. Budget 0.65s leaves
# ~350ms of headroom which is enough to absorb any plausible scheduling jitter.
TIME_BUDGET_S = 0.65
SETUP_RESERVE_S = 0.02   # don't even start sampling if less than this remains
PER_ITER_RESERVE_S = 0.003  # exit the rollout loop if this little is left


def _score_of_card(c: int) -> int:
    """Penalty points (bullheads) of a single card."""
    if c % 55 == 0:
        return 7
    if c % 11 == 0:
        return 5
    if c % 10 == 0:
        return 3
    if c % 5 == 0:
        return 2
    return 1


def _row_score(row) -> int:
    s = 0
    for c in row:
        s += _score_of_card(c)
    return s


# ----------------------------- the player ----------------------------------

class CYhuang:
    """
    Monte Carlo Sampling agent. Pure search, no learning, no state across calls.
    """

    # Class-level so __init__ can be argument-free for the framework.
    DEFAULT_TIME_BUDGET = TIME_BUDGET_S

    def __init__(self, player_idx: int, time_budget: float = None,
                 min_samples_per_card: int = 4):
        self.player_idx = player_idx
        # time_budget kept overridable for offline experiments. Final
        # submission will use the default.
        self.time_budget = (time_budget if time_budget is not None
                            else self.DEFAULT_TIME_BUDGET)
        # Kept as a kwarg for backward-compatible JSON configs, but no
        # longer enforced — exceeding the wall-clock deadline to chase a
        # min-sample target risks a DQ. With round-robin sampling, every
        # card gets within 1 sample of every other card before we stop,
        # which is sufficient.
        self.min_samples_per_card = min_samples_per_card
        # Use our own RNG instance. The engine reseeds the global `random`
        # module before each action call, so relying on it would lose any
        # state we tried to keep across turns. A private RNG also keeps
        # the engine's seeding behavior pristine.
        self.rng = random.Random()

    # ------------------------- main entry point ----------------------------

    def action(self, hand, history):
        """
        Return one card from hand to play this round.

        hand: list[int] (sorted ascending)
        history: dict with keys 'board', 'scores', 'round',
                 'history_matrix', 'board_history', 'score_history'
        """
        deadline = time.monotonic() + self.time_budget

        # Defensive: if hand has only one card, no choice.
        if len(hand) == 1:
            return hand[0]

        try:
            # Build current board state from history. The engine passes us the
            # board AT THE START of the round (before placements this round).
            board = history.get("board", [])
            # Make a clean local copy of rows; we'll mutate during sims.
            rows = [list(r) for r in board]

            # Determine the set of cards still unseen (in opponents' hands).
            unseen = self._compute_unseen(hand, history)

            # Number of opponents and their hand size right now.
            n_players = self._infer_n_players(history)
            opp_indices = [p for p in range(n_players) if p != self.player_idx]
            n_opp = len(opp_indices)

            current_round = history.get("round", 0)
            # Each opponent currently holds (n_rounds - current_round) cards.
            # We don't have n_rounds directly; infer from history_matrix length
            # plus our own remaining hand.
            n_rounds_total = self._infer_n_rounds(history, hand)
            opp_hand_size = max(0, n_rounds_total - current_round)

            # Sanity: if our pool is too small for the assumed hand size,
            # cap opp_hand_size. This shouldn't happen in real games but
            # protects us against weird configs.
            if opp_hand_size * n_opp > len(unseen):
                opp_hand_size = len(unseen) // max(1, n_opp)

        except (ValueError, IndexError, KeyError, TypeError):
            # If anything in the history is malformed, fall back to the
            # smallest card (matches the engine's default).
            return hand[0]

        # Per-card score accumulators (in the same order as hand).
        n_cards_in_hand = len(hand)
        totals = [0.0] * n_cards_in_hand
        counts = [0] * n_cards_in_hand

        # If there's basically no time left, return smallest card immediately.
        if time.monotonic() + SETUP_RESERVE_S >= deadline:
            return hand[0]

        # --- Round-robin sampling over candidate cards. ---
        # Round-robin (rather than "do all samples for card 0 then card 1 ...")
        # ensures every card gets equal coverage even if we run out of time.
        rng = self.rng

        # Pre-tuple the unseen pool for fast sampling.
        unseen_list = list(unseen)

        # Edge case: no opponents (shouldn't happen) or no unseen cards.
        if n_opp == 0 or opp_hand_size == 0 or not unseen_list:
            # No uncertainty to integrate over; just pick the card whose
            # immediate placement on the *current* board incurs the least.
            return self._greedy_pick(hand, rows)

        iter_idx = 0
        max_iters = 10_000_000  # hard upper bound, mostly to satisfy linters
        while iter_idx < max_iters:
            iter_idx += 1
            # Check the deadline EVERY iteration. Each iteration costs only
            # ~5-50 microseconds, but a single `_simulate_round_points` can
            # run several milliseconds in the worst case, so the gap between
            # checks (with even a small per-card pass over a 10-card hand)
            # can exceed 50ms — enough to overshoot a tight budget. Cheap to
            # do per-iteration.
            remaining = deadline - time.monotonic()
            if remaining < PER_ITER_RESERVE_S:
                # Out of time. Hard stop. We previously tried to keep going
                # until every card hit min_samples_per_card, but that risks
                # DQ if a long rollout sits between us and the deadline.
                # Better to vote with what we have.
                break

            card_idx = (iter_idx - 1) % n_cards_in_hand
            card = hand[card_idx]

            try:
                # Sample opponents' moves: draw n_opp cards from unseen_list,
                # one per opponent. We sample WITHOUT REPLACEMENT across
                # opponents (they can't share cards). We don't actually need
                # to materialize each opponent's full hand for depth-1; one
                # uniformly random card from a uniformly random hand is
                # equivalent in distribution to drawing n_opp distinct cards
                # from the unseen pool (since each opponent's hand is itself
                # uniform over the pool, and the "card they play" is uniform
                # within that hand).
                opp_cards = rng.sample(unseen_list, n_opp)
                points = self._simulate_round_points(card, opp_cards, rows)
            except (ValueError, IndexError):
                # Sampling could fail if pool < n_opp (already guarded, but
                # be defensive). Skip this iteration.
                continue

            totals[card_idx] += points
            counts[card_idx] += 1

        # --- Pick the card with the lowest mean penalty. ---
        # Tie-break: smallest card (engine-friendly; also matches paper's
        # "play isolated/safe" intuition that bottom of hand is rarely catastrophic).
        best_card = hand[0]
        best_mean = float("inf")
        for i, c in enumerate(hand):
            if counts[i] == 0:
                continue
            mean = totals[i] / counts[i]
            if mean < best_mean - 1e-12:
                best_mean = mean
                best_card = c
            # Strict "<" preserves smallest-card tiebreak because hand is sorted ascending.

        return best_card

    # --------------------------- helpers -----------------------------------

    @staticmethod
    def _infer_n_players(history) -> int:
        scores = history.get("scores")
        if scores is not None:
            return len(scores)
        hm = history.get("history_matrix") or []
        if hm:
            return len(hm[0])
        return 4  # README default

    @staticmethod
    def _infer_n_rounds(history, hand) -> int:
        """Total rounds in the game = rounds played + cards still in our hand."""
        hm = history.get("history_matrix") or []
        return len(hm) + len(hand)

    def _compute_unseen(self, hand, history) -> set:
        """
        All card values that we have NOT seen yet. These are the cards
        currently in opponents' hands. We've seen:
          - our own hand
          - the 4 cards visible on the board RIGHT NOW (the rows passed to us)
          - every card played in completed rounds (history_matrix)
          - the original 4 starter cards (which are the bottom of each row in
            board_history[0], or absorbed into completed rows)

        Easier formulation: start from {1..104}, remove our hand, remove every
        card visible in the current board, remove every card in history_matrix,
        and remove the original starter cards from board_history[0].
        """
        seen = set(hand)

        # Current board (visible row contents).
        board = history.get("board") or []
        for row in board:
            for c in row:
                seen.add(c)

        # Cards played in completed rounds. These may have been collected into
        # someone's pile (no longer on the board) so they're not always in the
        # current board, but they're definitely "out of opponent hands" now.
        hm = history.get("history_matrix") or []
        for round_actions in hm:
            for c in round_actions:
                seen.add(c)

        # Initial board snapshot includes the 4 starter cards. If the very
        # first board snapshot exists, add its contents (covers any card
        # already taken from the very first row).
        bh = history.get("board_history") or []
        if bh:
            for row in bh[0]:
                for c in row:
                    seen.add(c)

        # Build the unseen set.
        unseen = set()
        for c in range(1, N_CARDS + 1):
            if c not in seen:
                unseen.add(c)
        return unseen

    @staticmethod
    def _simulate_round_points(my_card: int, opp_cards, rows) -> int:
        """
        Mirror of engine.process_card_placement, applied in card-ascending
        order to (my_card + opp_cards) on a *copy* of the row state, returning
        the total points WE incur this round.

        rows: list[list[int]] — current board, NOT mutated.
        """
        # Local copy.
        local_rows = [list(r) for r in rows]

        # Build (card, is_me) tuples and sort ascending.
        plays = [(c, False) for c in opp_cards]
        plays.append((my_card, True))
        plays.sort(key=lambda x: x[0])

        my_points = 0

        for card, is_me in plays:
            # Find row whose last card is highest under `card`.
            best_row_idx = -1
            best_last = -1
            for r_idx in range(len(local_rows)):
                last = local_rows[r_idx][-1]
                if last < card and last > best_last:
                    best_last = last
                    best_row_idx = r_idx

            if best_row_idx != -1:
                # Card fits in best_row_idx.
                if len(local_rows[best_row_idx]) >= ROW_CAPACITY:
                    # 6th card: take the row.
                    if is_me:
                        my_points += _row_score(local_rows[best_row_idx])
                    local_rows[best_row_idx] = [card]
                else:
                    local_rows[best_row_idx].append(card)
            else:
                # Lower than all rows: take the row with lowest score
                # (tiebreak: shortest len, then smallest index). Matches engine.
                chosen = 0
                best_key = (_row_score(local_rows[0]), len(local_rows[0]), 0)
                for r_idx in range(1, len(local_rows)):
                    key = (_row_score(local_rows[r_idx]), len(local_rows[r_idx]), r_idx)
                    if key < best_key:
                        best_key = key
                        chosen = r_idx
                if is_me:
                    my_points += _row_score(local_rows[chosen])
                local_rows[chosen] = [card]

        return my_points

    def _greedy_pick(self, hand, rows) -> int:
        """
        Fallback when there's no uncertainty (no opponents / empty pool):
        pick the card whose immediate placement on the current board has
        the lowest score impact for us. Used only in degenerate cases.
        """
        best_card = hand[0]
        best_pts = float("inf")
        for c in hand:
            pts = self._simulate_round_points(c, [], rows)
            if pts < best_pts:
                best_pts = pts
                best_card = c
        return best_card
