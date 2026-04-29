# Work Plan — FAI Final Project

Companion to:
- [`tournament_scoring_analysis.md`](tournament_scoring_analysis.md) (what the
  scoring code actually does, with code refs)
- [`strategy_try_list.md`](strategy_try_list.md) (strategy options ranked,
  confirmed eval params, tuning knobs)

This file is the **plan** — sequenced concrete actions, exit criteria, and the
submission strategy. Source documents above contain the reasoning.

## Goal & deadline

- Deadline: **2026-06-14** (≈6.5 weeks from today, 2026-04-30).
- Submit `BestPlayer1` and `BestPlayer2` under `src/players/<sid>/`. Final
  score = better of the two.
- Target: match B55's avg_rank (= 90 pts). Anything beyond that is clamped at
  100 — not worth optimizing past.

## The key insight driving the plan

The grader scores by `avg_rank`, not by penalty. Existing CFR+ and sim agents
minimize per-round penalty — that is the wrong objective. **Switching the
objective to rank is a force multiplier on every approach**, not a separate
strategy. Bake it into all three tracks below.

## Three parallel tracks

### Track A — Rank-rewrites of existing agents (priority: do first)

Cheap, validates the rank-objective hypothesis. Two sub-tasks, can be done
independently:

- **A1: Rank-rewrite of sim+.** Action selection picks
  `argmin E[rank in matchup]` (estimated from rollouts), with optional CVaR
  penalty on rank-4 outcomes. Replaces `argmin E[score]`.
- **A2: Rank-rewrite of CFR+.** Change the regret target from per-round
  penalty to per-game rank.

Exit criterion: each rewritten agent benchmarks at *or above* its score-based
ancestor on a tournament that includes baselines. If yes, the rank-objective
hypothesis is confirmed and these agents become viable submission candidates.

### Track B — RL self-play (priority: start scaffolding immediately)

The actual new strategy. Highest ceiling, biggest engineering risk. Must start
now to fit the timeline.

- Architecture: set/permutation invariant over hand and rows, CPU inference,
  weights ship inside the <2GB zip.
- Reward: `−final_rank` (or concave utility of rank — sweep empirically).
- Training opponent pool: TA baselines + own CFR+/sim+ + noised copies of
  current self. Anneal baseline-heavy → self-play-heavy.
- Inference budget: sub-millisecond per call leaves room for a shallow
  search/lookahead layer on top.

Exit criterion: trained policy benchmarks above the better of the Track A
agents. If not, fall back to Track A agents for submission.

### Track C — Structural improvements to existing agents (opportunistic)

Not pure hyperparameter tuning. Real architectural upgrades that address
known weaknesses:

- **C1: Hand inference for sim+.** Bayesian filtering on opponent hands from
  observed play, replacing uniform-random sampling in rollouts.
- **C2: Smarter rollout opponent model in sim+.** Anything other than uniform
  random play; ideally a heuristic or a fast learned policy.
- **C3: Better terminal eval / abstraction in CFR+** if the current one is
  coarse.
- **C4: Endgame solver.** Trigger when remaining unseen-card state is small
  enough (round 7+) for exact expectimax with rank-based leaf eval. Hybridizes
  cleanly on top of any final agent — strictly additive.
- **C5: Opponent-ID bonus layer.** Classifier infers if a tablemate is an
  identified baseline; if confident, deviate toward best-response. Otherwise
  fall through to the robust agent. Bonus layer only, not standalone.

Out of scope (explicitly *not* doing): chasing 0.05-avg_rank gains via
hyperparameter sweeps, deeper tree abstractions, or knob-tuning on the
existing score-based objectives. Diminishing returns.

## Pre-work (blocks everything; do first this week)

1. **Rank the public baselines.** Spec says they are released unsorted.
   Run a benchmark tournament (≥500 partitions) of all available baselines +
   our current agents. Read off avg_rank for each baseline. Store as a
   reference table.
2. **Estimate where B20 and B55 sit.** Even with only public baselines we can
   get a sense of where the strongest available baseline ranks; this anchors
   "how far are we from 90 pts."
3. **Set up a reproducible benchmark config.** A canonical
   `configs/tournament/benchmark.json`-style config including all baselines +
   all our candidates, used as the single comparison harness for Tracks A/B/C.

## Phased timeline (rough; revise as we learn)

| Phase | Weeks | Tracks active                      | Goal                                            |
|-------|-------|------------------------------------|-------------------------------------------------|
| 0     | 1     | Pre-work                           | Benchmark harness + baseline ranking            |
| 1     | 1     | A (A1+A2), B scaffolding           | Rank-rewrites land; RL training loop runs end-to-end |
| 2     | 2     | B (training), C opportunistic      | RL agent benchmarks vs A; pick C tasks by ROI   |
| 3     | 1.5   | B polish, integration              | Endgame solver layer (C4); pair selection       |
| 4     | 1     | Submission packaging, report       | Final zip + 3-page PDF report                   |

Buffer: ~0 weeks. If RL doesn't land by end of Phase 2, fall back to the two
best Track A agents as the submission pair.

## Decision points

- **End of Phase 1:** Did Track A rank-rewrites improve avg_rank over their
  score-based ancestors? If no, the rank-objective hypothesis is wrong;
  re-examine assumptions before committing more weeks to RL.
- **End of Phase 2:** Has RL surpassed the better of the Track A agents on
  benchmarks? If no, freeze RL, ship Track A pair.
- **Throughout Phase 2-3:** When picking Track C tasks, prioritize by
  *measured benchmark improvement per engineering hour*, not by appeal of the
  idea.

## Submission strategy

Decided at deadline based on benchmark numbers, not in advance:

- **If RL works (above better Track A agent):** `BP1 = RL`,
  `BP2 = better of rank-rewritten sim+ / CFR+` (hedge against RL's
  distribution-shift failure mode).
- **If RL doesn't ship in time or underperforms:**
  `BP1 = better rank-rewritten agent`, `BP2 = the other one`. Two correlated
  strong agents is fine — max-of-two scoring has no diversity penalty;
  diversity is only a hedge.
- The endgame-solver hybrid (C4), if implemented, should be layered onto
  whichever agent ends up as BP1.

## Report (30 pts)

Write alongside the work, not at the end. Sections to cover (per SPEC):
methods tried (including discarded ones), core idea of the two final
algorithms, implementation details + hyperparameters, self-assessment,
strategic profiling (where each agent is strong/weak), agent comparison,
optional future improvements. Capped at 3 A4 pages.

The brainstorming + analysis history (this plan, `tournament_scoring_analysis.md`,
`strategy_try_list.md`) is the raw material for the "methods tried" and
"core idea" sections.

## Novelty bonus (+5 pts)

Multiple things in this plan plausibly qualify:
- Rank-objective rewrite (most agents in the class will optimize score).
- RL self-play with rank reward.
- Endgame solver hybrid.
- Bayesian hand inference for informed rollouts.

Document them clearly in the report — the bonus is up to +5 pts and is awarded
qualitatively + quantitatively, so the writeup matters as much as the work.
