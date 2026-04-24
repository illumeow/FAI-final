# HeuristicPlayer Strategy

`HeuristicPlayer` is a fast risk-aware rule-based policy.

## Core Idea
- Estimate immediate expected penalty from the current board.
- Approximate opponent uncertainty from unseen cards.
- Add a short greedy lookahead over your remaining hand.

## Decision Logic
- For each candidate card, estimate forced-take probability and expected loss.
- Simulate the post-move board and compute a lightweight future cost.
- Choose the card with minimum total estimated risk.

## Strengths
- Very fast and stable under tight time limits.
- Strong when immediate board risk dominates outcomes.

## Weaknesses
- Uses approximations instead of deep rollout search.
- Can miss long-horizon tactical opportunities.
