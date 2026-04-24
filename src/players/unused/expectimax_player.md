# ExpectimaxPlayer Strategy

`ExpectimaxPlayer` performs stochastic expectimax search with sampled chance nodes.

## Core Idea
- Model your action as a decision node.
- Model opponents as chance outcomes sampled from unseen cards.
- Evaluate short-depth expected penalty recursively.

## Decision Logic
- Keep a small candidate branch for tractability.
- For each action, sample opponent cards and recurse to limited depth.
- Select the action with minimum expected cumulative loss.

## Strengths
- Clear search-based interpretation.
- Better tactical depth than one-step heuristics.

## Weaknesses
- Depth and branching must remain small under 1 second.
- Performance depends on sampling efficiency.
