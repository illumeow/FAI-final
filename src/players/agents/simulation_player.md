# SimulationPlayer: Standalone Explanation

`SimulationPlayer` is a time-bounded Monte Carlo policy for 6 Nimmt! that evaluates each playable card by sampled rollouts of the remaining rounds.

## Main Components

1. **GameCore**
   - Shared game-mechanics helper used by both simulation and CFR agents.
   - Handles row-fit logic, forced-take logic, score calculation, unseen-card pool, opponent-hand sampling, and greedy card selection.
2. **SimulationRolloutEvaluator**
   - Given a fixed first action and sampled opponent hands, simulates the rest of the round sequence.
   - Returns the accumulated score incurred by this player in that rollout.
3. **SimulationStats**
   - Provides aggregation helpers (mean score per candidate card).

## Decision Pipeline

1. **State preparation**
   - Read current board and hand.
   - Infer number of opponents from `history["scores"]`.
   - Build unseen-card pool from known cards.

2. **Phase 1: minimum-coverage sampling**
   - For every candidate card, run at least `min_samples_per_card` rollouts (time permitting).
   - Accumulate total score and sample count per candidate.

3. **Phase 2: adaptive allocation**
   - Repeatedly rank active candidates by:
     - lower mean score,
     - then lower sample count,
     - then card value.
   - Continue sampling under remaining time budget.
   - Periodically prune weaker candidates to focus rollout budget on contenders.

4. **Final action choice**
   - Select the card with minimum `(mean_score, heuristic_card_key)` among original candidates.
   - The rollout mean is primary; heuristic is a tiebreaker.

## Why This Works

- **Determinization** converts hidden-information uncertainty into sampled concrete worlds.
- **Rollout scoring** estimates long-horizon consequences beyond immediate placement.
- **Two-phase allocation** balances fairness (all candidates sampled) and focus (more effort on promising options).
- **Heuristic tie-breaking** adds stability when rollout estimates are close under short time budgets.
