# SimulationPlayer Strategy

`SimulationPlayer` is a time-budgeted Monte Carlo determinization policy.

## Core Idea
- Sample hidden opponent hands from unseen cards.
- Roll out remaining rounds under sampled worlds.
- Allocate most of the 1-second budget to evaluating promising actions.

## Decision Logic
- Guarantee a minimum number of samples per candidate card.
- Continue adaptive sampling and prune weak candidates.
- Pick the card with the lowest estimated expected total penalty.

## Strengths
- Handles uncertainty directly via sampling.
- Usually robust against diverse opponent styles.

## Weaknesses
- Sampling variance can affect decisions.
- Rollout policy quality limits final performance.
