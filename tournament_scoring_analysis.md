# Tournament Scoring Analysis & Strategy Implications

## TL;DR

The grader optimizes **rank**, not score. Every existing agent (CFR+, sim) likely
minimizes expected per-round penalty — that is *not* the objective the tournament
rewards. Fixing the loss function is a force multiplier on every other strategy.

## What the code actually does

Source: `src/tournament_runner.py`.

### Per-game ranking (lines 163-168)

```python
ranks[i] = (2 * better_count + same_count + 1) / 2.0
```

Fractional rank where lower = better. With 4 players: 1.0, 2.0, 3.0, 4.0, with
half-integers for ties (e.g. tied 1st-2nd → 1.5 each).

### Per-tournament ranking (lines 281, 656)

```python
self.player_stats.sort(key=lambda x: (x["avg_rank"], x["avg_score"]))
```

`avg_rank` is the **primary** sort key. `avg_score` is only the tiebreak.

### Calibrated score (lines 377-421)

If `scoring_config` is set, the calibrated score is a clamped linear
interpolation of `avg_rank` between two anchors defined by baseline percentiles:

- `baseline_upper_pct` percentile of baseline ranks → `score_at_upper_pct`
- `baseline_lower_pct` percentile of baseline ranks → `score_at_lower_pct`

Linear in `avg_rank`, clamped to `[0, 100]`.

**Concrete final-eval anchors (from SPEC.md "Baselines Scoring Criteria"):**
- Final: `avg_rank == B55's avg_rank → 90 pts`,
  `avg_rank == B20's avg_rank → 40 pts`. Linear, clamped `[0, 100]`.
- Midterm: `B5 → 60 pts`, `B20 → 90 pts`.
- Baselines B1..B55 are sorted worst → strongest in the *full* internal set;
  the *public* released subset is unsorted (we have to rank them ourselves).

Implication: matching B55 already buys 90/100. Anything beyond B55 is at most
+10 pts (clamp at 100). Below B20 the score keeps dropping linearly until the
floor at 0. **The strategic gradient lives between B20 and B55** — that is the
range where each marginal rank improvement directly converts into points.

### Matchup structure

`RandomPartitionTournamentRunner` (final eval format, confirmed):
- All players are randomly partitioned into tables of 4 each round.
- Your rank is computed *within your table only*.
- `duplication_mode: "cycle"` reruns each table N times with seat rotations on
  the same dealt hands — variance reduction without N! cost.
- Confirmed eval parameters: `engine.n_players = 4`,
  `tournament.num_games_per_player >= 500`, `tournament.type = "random_partition"`,
  `tournament.duplication_mode = "cycle"` (midterm; subject to change for final).
- Eval pool: **all students + 55 baselines** (final). So the opponent at any
  given table is a heterogeneous mix of unknown student agents and baselines.

## Why this changes everything

### 1. Beating the field by 1 point = beating it by 50

Same rank. So aggressive "minimize my expected penalty" play wastes effort on
margin you don't get paid for. The right question is "did I beat the other 3 at
my table" — not "how much did I lose."

### 2. Variance is doubly bad

Alternating 1st/4th gives avg rank 2.5; consistently 2nd gives avg rank 2.0.
The low-variance agent wins even with identical expected score. And the
calibrated-score clamps (≤100 above B55, ≥0 below the floor) make huge wins
worthless and huge losses bounded — both directions favor consistency. With
≥500 partitions in the eval, per-matchup noise averages out, so the policy's
*expected* avg_rank is what matters; we don't need to over-engineer for noise
reduction.

### 3. Existing agents are optimizing the wrong thing

CFR+ and simulation agents almost certainly minimize expected per-round penalty
(that is what is most natural to encode and what training signals provide).
The grader rewards minimum expected *rank* against the other 3 at your table.
These objectives disagree often:

- Take a guaranteed -3 to stay ahead of one specific opponent vs gambling for
  {0, -10}: same expected score, but the safe play preserves rank.
- A row-grab that makes you "lose less than you would have" can still cost you
  rank if it leaves another player at the table also taking a hit.

For strategy implications and the try-list, see
[`strategy_try_list.md`](strategy_try_list.md).
