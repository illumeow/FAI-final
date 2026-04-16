# BitwiseSearchPlayer Strategy

`BitwiseSearchPlayer` is a fast probabilistic search agent using bitset operations.

## Core Idea
- Encode unseen-card availability as a bit mask.
- Use interval bit counts to estimate probabilities quickly.
- Score actions with expected immediate risk plus short lookahead.

## Decision Logic
- For each card, compute probability that opponents force a row take.
- Use binomial tail estimates for overflow events.
- Add one-step future-risk term and choose minimal score.

## Strengths
- Very efficient probability queries.
- Good speed-to-quality tradeoff.

## Weaknesses
- Relies on model assumptions about opponent distribution.
- Limited deep tactical modeling.
