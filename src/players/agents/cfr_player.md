# CFRPlayer Strategy

`CFRPlayer` is an online CFR-style regret-matching agent.

## Core Idea
- Maintain regrets over a reduced action set.
- Update strategy using sampled rollout utilities.
- Converge toward stronger mixed actions during the turn.

## Decision Logic
- Build a candidate subset using heuristic prefiltering.
- Repeatedly sample opponent hands and evaluate action utilities.
- Update cumulative regrets and average strategy.
- Play the highest-probability action from average strategy.

## Strengths
- Balances exploration and exploitation in uncertain states.
- Performs well when a mixed policy is beneficial.

## Weaknesses
- Sensitive to candidate filtering quality.
- More compute-heavy than pure heuristics.
