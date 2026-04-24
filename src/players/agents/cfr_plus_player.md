# CFRPlusPlayer: Standalone Explanation

`CFRPlusPlayer` is a time-bounded decision policy for 6 Nimmt! that chooses the card with the best estimated future outcome under hidden-information uncertainty.

## Main Components

1. **GameCore**
   - Implements game mechanics helpers (fit row, forced take, place card, score rules, unseen-card pool, sampled opponent hands, greedy card choice).
2. **CFRRolloutEvaluator**
   - Simulates remaining rounds after forcing a chosen first card.
   - Uses epsilon-greedy opponent choices (`opponent_epsilon`) during Monte Carlo CFR iterations.
3. **HandAssignmentUtils**
   - Counts and enumerates exact opponent hand assignments for tiny endgames.

## Decision Pipeline

1. **Build candidate actions**
   - Starts from hand cards.
   - Keeps mandatory cards (extremes, forced-take extremes, risky reset representative).
   - Expands candidates in heuristic order up to an action cap.

2. **Try exact endgame solve (when small enough)**
   - Triggered only if remaining rounds are small (`exact_endgame_rounds`) and unseen-card size matches required opponent cards exactly.
   - Enumerates all opponent hand assignments (up to `exact_endgame_max_assignments`).
   - Computes exact expected loss for each action with deterministic greedy opponents.
   - Returns action with minimum exact expected loss.

3. **Otherwise run CFR+ style iterative search under time budget**
   - Maintains per-action regrets, average strategy mass, and utility stats.
   - Converts positive regrets into a behavioral strategy (Regret Matching Plus).
   - Samples opponent hands from unseen cards each iteration.
   - Evaluates every candidate by rollout loss, then maps to utility as `-loss`.
   - Updates regrets with decay and RM+ clamp:
     - `regret[a] = max(0, decay * regret[a] + utility[a] - node_utility)`
   - Accumulates strategy mass weighted by iteration index.
   - Stops when adaptive time guard predicts insufficient safe time.

4. **Final action choice**
   - If strategy mass exists: choose action with highest averaged strategy probability, then mean utility, then smaller card.
   - Fallback: choose best heuristic card.

## Why This Works

- **Action pruning** keeps search focused under strict per-move time.
- **RM+ updates** stabilize learning signal during short iterative windows.
- **Opponent sampling + rollout** handles hidden cards without full game-tree expansion.
- **Exact endgame mode** removes sampling noise when horizon is tiny and fully constrained.
- **Adaptive time guard** reduces timeout risk by tracking real iteration cost online.
