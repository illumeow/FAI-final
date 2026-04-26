# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project mission

This repo is for the FAI 2026 final project: build strong **6 Nimmt!** agents that minimize bullhead penalties in tournaments. Two best players (`BestPlayer1`, `BestPlayer2`) are submitted; evaluation uses `random_partition` style tournaments against ~55 baselines. Agents must run single-threaded, in <1s/turn, under 1GB RAM.

## Commands

Activate the existing venv first (Python 3.13.11):

```bash
source .venv/bin/activate
```

Run a single game (fastest local validation):

```bash
python run_single_game.py --config configs/game/experimental_agents.json
```

Run a tournament:

```bash
python run_tournament.py --config configs/tournament/benchmark.json
```

Tournament with split config overrides (any of these are optional and merge into the base):

```bash
python run_tournament.py \
  --config configs/tournament/benchmark.json \
  --player-cfg <path> \
  --engine-cfg <path> \
  --tournament-cfg <path>
```

There is no lint command and no unit-test suite in-tree (no pytest/ruff/mypy). Validation is by running games/tournaments and inspecting the JSON output under `results/game/` or `results/tournament/`.

## High-level architecture

- **Entrypoints**: `run_single_game.py` and `run_tournament.py` load JSON config, normalize player config, dynamically import player classes by string path, instantiate players, run engine/tournament, and write timestamped JSON results under `results/`.
- **`src/game_utils.py`** is the config/import bridge:
  - `_preprocess_player_config()` normalizes `players` and optional `baselines` into a single `players` list, tags `is_baseline`, and assigns `player_id`. Baselines are scheduled as normal participants but also serve as score anchors in `random_partition` scoring.
  - `load_players()` imports classes from string module paths.
- **`src/engine.py`** is the game core:
  - Deals cards, runs `n_rounds` (default 10), collects each player's action, processes placements in ascending card order, enforces 6th-card and low-card rules, returns full history.
  - Timeout uses `SIGALRM` + a custom `TimeoutException(BaseException)`. After each player's turn the engine resets `random.seed(None)` so players cannot manipulate the global RNG of others.
  - Invalid action / raised exception / timeout → fallback to the smallest card in hand. **Swallowing `TimeoutException` (still running after timeout + buffer)** → disqualified for the rest of the game.
  - The `history` dict passed to `action()` is a deep copy; mutating it has no effect.
- **`src/tournament_runner.py`** orchestrates evaluations and supports three tournament types:
  - `CombinationTournamentRunner`: all `C(N, n_players)` combinations.
  - `RandomPartitionTournamentRunner`: repeated random partitions; supports isolated subprocess execution with `max_memory_mb_per_matchup` and `matchup_timeout_multiplier`. **This is the format used in final evaluation.**
  - `GroupedRandomPartitionTournamentRunner`: stage-1 global ranking, then stage-2 grouped reruns based on stage-1 rank.
  - `duplication_mode` reuses the same initial hands with seat permutations/cycles to reduce RNG variance: `"permutations"` (N!), `"cycle"` (N), `"none"` (1). Preserve this behavior when changing tournament logic.
- **`src/players/`** layout:
  - `TA/`: provided baselines and reference players (e.g. `random_player`, `human_player`, compiled `public_baselines1.so`).
  - `agents/`: current best agents (`cfr_plus_player.py`, `simulation_player.py`) plus the shared `game_core.py` utility module. `agents/v1/` keeps a frozen earlier version for benchmarking.
  - `unused/`: archived/experimental agents (CFR, expectimax, bitwise search, bandit rollout, heuristic, genetic rollout, etc.) — kept for reference and benchmark configs but not part of the final submission.
- **`configs/`** holds canonical JSON examples: `configs/game/` for single-game runs, `configs/tournament/` for tournaments, `configs/GA/` for genetic-algorithm weight training (`train.json` produces `weights.json` consumed by GA-style players).

## Key conventions

- **Edit scope rule**: only modify files in `configs/` and `src/players/agents/` unless the user explicitly asks otherwise. The engine, tournament runner, and TA player code are course-provided infrastructure.
- **Player class contract**: `__init__(player_idx, **optional_args)` and `action(hand, history) -> int`.
  - `hand` is already sorted; the returned card must be an `int` actually in `hand`, otherwise the smallest-card fallback fires.
  - Treat `history` as read-only — engine passes deep copies.
  - Recompute derived state from `history` rather than caching across turns; cached values can be stale if a previous turn timed out.
- **Hard rules for player code** (course policy, not stylistic):
  - Do **not** use `threading` or `multiprocessing` — penalty up to ×0.5 of final score. Tournament parallelism is already handled by the framework.
  - Do **not** use bare `except:` or `except BaseException:`. `TimeoutException` inherits from `BaseException` specifically so accidental catches are caught — swallowing it triggers disqualification (penalty up to ×0.7).
  - Stay under 1GB RAM per player; the matchup is killed and marked OOM otherwise.
  - Decision time limit is 1s per turn (default `timeout: 1.0`, `timeout_buffer: 0.5`).
- **Player config entries** support three equivalent forms (see `configs/` for examples):
  - `["module.path", "ClassName"]`
  - `["module.path", "ClassName", {args}, "label"]`
  - `{"path": "...", "class": "...", "args": {...}, "label": "..."}`
  - `label` is the column name in tournament standings — useful when multiple entries share the same class. Max 9 characters.
- **Keep `args` empty** in submitted configs; put safe defaults in `__init__` instead. The grader runs with empty args.
- **Fixed evaluation engine config**:
  `{"n_players": 4, "n_rounds": 10, "verbose": false, "timeout": 1.0, "timeout_buffer": 0.5}`.
- **For tournament evaluation use `duplication_mode: "cycle"`** — it's what the TA tournament will use and gives stable variance reduction without the N! cost of `permutations`.
- **Submission**: final players must be `BestPlayer1` in `best_player1.py` and `BestPlayer2` in `best_player2.py`, placed under `src/players/<student_id_lowercase>/`. Baselines may be used to *train* but cannot be imported from the submission.

## Game rules quick reference

- 104 cards, 4 rows seeded with one card each, 4 players, 10 cards per hand, 10 rounds.
- Each round all players reveal simultaneously; cards are placed in **ascending order**.
- Placement: card goes on the row whose last card is the largest value still smaller than it.
- 6th-card rule: a row can only hold 5 cards; the 6th placement takes all 5 (penalty = sum of card scores) and the new card starts the row.
- Low-card rule: if your card is smaller than every row's last card, you must take a row, chosen by `(row_score, len(row), row_index)` — smallest first. Then your card starts that row.
- Card score (engine `_default_score_mapping`): `% 55 == 0 → 7`, `% 11 == 0 → 5`, `% 10 == 0 → 3`, `% 5 == 0 → 2`, else `1`. (Note: differs from the marketing description in `spec.md` — the engine values are the source of truth.)
- `history` keys passed to `action()`: `board`, `scores`, `round`, `history_matrix` (cards each player played per round), `board_history` (board snapshot per round), `score_history` (cumulative scores per round). Your own index is `self.player_idx`.
