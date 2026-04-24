# BanditRolloutPlayer Strategy

`BanditRolloutPlayer` uses a UCB multi-armed bandit over candidate cards.

## Core Idea
- Treat each candidate card as a bandit arm.
- Use rollout loss as feedback reward.
- Apply UCB to balance exploration and exploitation under time limits.

## Decision Logic
- Warm-start by sampling each candidate once.
- Repeatedly pull the UCB-best arm while time remains.
- Return the arm with best empirical average loss.

## Strengths
- Adaptive budget allocation to promising actions.
- Usually strong in noisy rollout settings.

## Weaknesses
- Depends on rollout quality and stochastic noise.
- Can under-explore if candidate set is weak.
