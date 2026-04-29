# Strategy Try List

Companion to [`tournament_scoring_analysis.md`](tournament_scoring_analysis.md).
The analysis establishes that the grader optimizes **rank**, not score. The
strategies below are ranked under that objective. Treat objective design as a
property baked into every approach, not a separate strategy.

## Confirmed eval parameters

From SPEC.md and TA clarification:

- `engine.n_players = 4`, fixed.
- `tournament.type = "random_partition"`, `num_games_per_player >= 500`.
- `duplication_mode = "cycle"` for the midterm; the final eval may change this
  based on midterm results.
- **Eval pool = all students + 55 baselines** (final; midterm is students + 40).
  The opponent mix at any table is heterogeneous — both unknown student agents
  and baselines, in unknown ratio.
- **Final score = linear-in-avg-rank, anchored B55 → 90 pts and B20 → 40 pts,
  clamped to [0, 100].** Strategic gradient lives between B20 and B55. Beating
  B55 is at most +10 pts (clamp at 100); below B20 the score keeps dropping
  until floored at 0.
- Submission contract: config form is fixed `["src.players.<sid>.best_player1",
  "BestPlayer1"]`; agents must run with no args. Two players are submitted
  (`BestPlayer1`, `BestPlayer2`); SPEC explicitly requests they be **diverse**
  ("counter different playing styles").
- Hard limits: 1s/turn, 1GB RAM, single-threaded, no GPU at eval, no network.
- Novelty bonus: up to +5 pts (qualitative + quantitative). Total capped at 100.

## Strategic implications of the confirmed params

1. **The target is "match B55."** Effort spent climbing past B55 is wasted
   (clamp). Any approach whose realistic ceiling is "near B55" is *enough*.
   Stop optimizing once benchmark tournaments reliably put us at or above B55's
   avg_rank.
2. **B20 is the cliff.** Falling below B20 in avg_rank is where most points
   are lost. Robustness against weak/medium opponents matters as much as
   sharpness against the strong.
3. **Two diverse players, not one tuned twice.** SPEC asks for diverse styles
   so that the pair handles different opponent compositions. Pick two
   architecturally distinct approaches.
4. **Mixed pool kills pure opponent-ID.** Other students are unknown; only the
   55 baselines are profile-able. Opponent-ID is at most a bonus layer that
   activates when classifier confidence is high, not a standalone strategy.
5. **500+ partitions is a comfortable variance budget.** No need to
   over-engineer for per-matchup noise — it averages out.
6. **No GPU + no network at eval.** RL training is fine offline (GPU OK), but
   inference must be CPU and trained weights must ship inside the <2GB zip.
7. **Public baselines are unsorted.** Spec says B1=weakest, B55=strongest in
   the full set, but the released subset is unsorted. Ranking the public
   baselines ourselves (by running them in a benchmark tournament) is a
   prerequisite for any opponent-aware approach.

## Ranking

1. **RL self-play with rank-based reward.** Highest ceiling, opponent-agnostic
   by construction if trained on a diverse pool (TA baselines + own CFR/sim +
   noised copies of self). Architecture: set/permutation invariance over hand
   and rows. Reward: `−final_rank` (or a concave utility of rank). Use the
   sub-millisecond inference to free budget for shallow lookahead.

2. **Endgame solver with rank-based leaf eval.** Robust, opponent-agnostic,
   hybridizable, modest engineering. Trigger when the remaining unseen-card
   state is small enough for exact expectimax. Leaf evaluator computes
   `E[rank in matchup]` not `E[score]`.

3. **Consistency-first rewrite of the existing simulation agent.** Same agent,
   but action selection picks `argmin E[rank in matchup]` (estimated from
   rollouts) with a CVaR penalty on rank-4 outcomes. Cheapest concrete win.
   Worth doing first as a baseline that validates the rank-objective
   hypothesis on real tournament results before sinking weeks into RL.

4. **Hand-disposal as stochastic scheduling.** Reframe as assignment of cards
   to rounds. Interesting but opponent-dependent — quality bounded by opponent
   model, so it lands behind RL.

5. **Opponent identification + best-response.** Bonus layer only. Activate
   when classifier confidence is high (clearly a TA baseline at the table);
   otherwise fall through to robust agent. Brittle against unknown student
   agents on its own.

## Recommended sequencing

1. Build the **consistency-first sim rewrite** first (~days). Validate on
   benchmark tournament that rank-based action selection actually improves
   `avg_rank` against the baseline pool. Proves the thesis cheaply before
   committing engineering weeks to harder approaches.

2. If (1) wins, commit to **RL self-play with rank reward** as the main bet.

3. Layer the **endgame solver** on top of whatever final agent emerges — it
   is strictly additive and low-risk.

4. Optionally add the **opponent-ID bonus layer** at the end.

## Picking the two submissions (diversity-aware)

SPEC requires two architecturally diverse players. Reasonable pairings:

- **BestPlayer1 = RL self-play (rank reward).** Highest ceiling, robust to
  unknown student agents.
- **BestPlayer2 = consistency-first sim with endgame-solver layer.** Different
  paradigm (search-based, no learned model), different failure mode (depends
  on rollout sampling rather than on the trained policy distribution). Acts as
  a hedge if RL training underperforms expectations or generalizes badly.

If RL doesn't get over the line in time, fallback pair:

- **BestPlayer1 = consistency-first sim (rank-based, polished).** Solid mid-pack.
- **BestPlayer2 = endgame-solver hybrid (heuristic early game + exact late).**
  Complementary failure modes — one is rollout-quality-bound, the other is
  state-space-bound.

## Tuning hyperparameters (empirical, no TA needed)

- **Concavity of rank utility.** `−rank` is linear; `(N − rank)^2` or
  `−P(rank == N)` punish last-place more. Tune by sweeping on the benchmark
  tournament.
- **CVaR quantile for the consistency-first agent.** 25%? 10%? Search.
- **Training opponent mix for RL.** Ratio of self-play vs baselines vs prior
  CFR/sim. Standard practice: anneal from baseline-heavy → self-play-heavy.

## Remaining open questions

All questions about the eval setup have been answered (see *Confirmed eval
parameters* above). What is left is empirical tuning, not TA clarification:

- **Concavity of the rank utility.** `−rank` is linear; `(N − rank)^2`,
  `−P(rank == N)`, or piecewise schedules punish last-place more. Sweep on
  the benchmark tournament.
- **CVaR quantile for the consistency-first agent.** 25%? 10%? Search.
- **Training opponent mix for RL.** Ratio of self-play vs baselines vs prior
  CFR/sim. Standard practice: anneal baseline-heavy → self-play-heavy.
- **B20 / B55 avg_rank in our local benchmark.** Need to estimate where the
  grading anchors actually sit so we know how far we currently are from 90 pts.
  Run a benchmark tournament including all available baselines + our agents
  and read off B20/B55 avg_ranks from the standings.
