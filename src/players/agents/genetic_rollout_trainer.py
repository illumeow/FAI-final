import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .core_utils import SimpleAgentCore
from .genetic_rollout_player import (
    FEATURE_NAMES,
    GeneticFeaturePolicy,
    build_model_payload,
    validate_weights,
)


def _require(mapping, key, where):
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {where}.")
    return mapping[key]


def _require_dict(mapping, key, where):
    value = _require(mapping, key, where)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' in {where} to be a JSON object.")
    return value


class GeneticRolloutTrainer:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.repo_root = Path(__file__).resolve().parents[3]
        self.results_dir = self.repo_root / "results" / "tournament"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        pretrain_cfg = _require_dict(self.cfg, "pretrain", "train config")
        finetune_cfg = _require_dict(self.cfg, "finetune", "train config")

        self.n_cards = int(_require(pretrain_cfg, "n_cards", "train config.pretrain"))
        self.n_rows = int(_require(pretrain_cfg, "n_rows", "train config.pretrain"))
        self.n_opponents = int(_require(pretrain_cfg, "n_opponents", "train config.pretrain"))

        self.population_size = int(_require(pretrain_cfg, "population_size", "train config.pretrain"))
        self.generations = int(_require(pretrain_cfg, "generations", "train config.pretrain"))
        self.elite_size = int(_require(pretrain_cfg, "elite_size", "train config.pretrain"))
        self.state_samples = int(_require(pretrain_cfg, "state_samples", "train config.pretrain"))
        self.eval_rollouts = int(_require(pretrain_cfg, "eval_rollouts", "train config.pretrain"))
        self.tournament_k = int(_require(pretrain_cfg, "tournament_k", "train config.pretrain"))

        self.crossover_rate = float(_require(pretrain_cfg, "crossover_rate", "train config.pretrain"))
        self.mutation_rate = float(_require(pretrain_cfg, "mutation_rate", "train config.pretrain"))
        self.mutation_sigma = float(_require(pretrain_cfg, "mutation_sigma", "train config.pretrain"))
        self.weight_clip = float(_require(pretrain_cfg, "weight_clip", "train config.pretrain"))

        self.rounds_left_min = int(_require(pretrain_cfg, "rounds_left_min", "train config.pretrain"))
        self.rounds_left_max = int(_require(pretrain_cfg, "rounds_left_max", "train config.pretrain"))

        self.random_seed = int(_require(self.cfg, "random_seed", "train config"))
        self.rng = random.Random(self.random_seed)
        self.eval_core = SimpleAgentCore(
            player_idx=0,
            n_cards=self.n_cards,
            seed_offset=39017,
        )

        self.initial_weights = validate_weights(
            _require(self.cfg, "initial_weights", "train config")
        )

        self.finetune_enabled = bool(_require(finetune_cfg, "enabled", "train config.finetune"))
        self.finetune_generations = int(_require(finetune_cfg, "generations", "train config.finetune"))
        self.finetune_population_size = int(_require(finetune_cfg, "population_size", "train config.finetune"))
        self.finetune_elite_size = int(_require(finetune_cfg, "elite_size", "train config.finetune"))
        self.finetune_tournament_k = int(_require(finetune_cfg, "tournament_k", "train config.finetune"))
        self.finetune_crossover_rate = float(_require(finetune_cfg, "crossover_rate", "train config.finetune"))
        self.finetune_mutation_rate = float(_require(finetune_cfg, "mutation_rate", "train config.finetune"))
        self.finetune_mutation_sigma = float(_require(finetune_cfg, "mutation_sigma", "train config.finetune"))
        self.finetune_tournament_repeats = int(_require(finetune_cfg, "tournament_repeats", "train config.finetune"))
        self.finetune_cleanup_results = bool(_require(finetune_cfg, "cleanup_results", "train config.finetune"))
        self.finetune_timeout_sec = int(_require(finetune_cfg, "tournament_timeout_sec", "train config.finetune"))
        self.finetune_min_games_per_player = int(
            _require(finetune_cfg, "min_games_per_player", "train config.finetune")
        )
        raw_budget_scales = _require(
            finetune_cfg,
            "budget_scales",
            "train config.finetune",
        )
        self.finetune_budget_scales = [
            float(scale)
            for scale in raw_budget_scales
            if float(scale) > 0.0
        ]
        if not self.finetune_budget_scales:
            raise ValueError("train config.finetune.budget_scales must include positive values.")
        self.finetune_force_none_duplication_on_fallback = bool(
            _require(
                finetune_cfg,
                "force_none_duplication_on_fallback",
                "train config.finetune",
            )
        )
        self.finetune_candidate_label = str(
            _require(finetune_cfg, "candidate_label", "train config.finetune")
        )
        self.finetune_opponents = _require(
            finetune_cfg,
            "opponents",
            "train config.finetune",
        )
        if not isinstance(self.finetune_opponents, list):
            raise ValueError("train config.finetune.opponents must be a list.")

        self.finetune_engine_cfg = dict(
            _require_dict(
                finetune_cfg,
                "engine",
                "train config.finetune",
            )
        )
        self.finetune_tournament_cfg = dict(
            _require_dict(
                finetune_cfg,
                "tournament",
                "train config.finetune",
            )
        )

    @staticmethod
    def _rank_indices(fitnesses):
        return sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

    @staticmethod
    def _mean(values):
        return sum(values) / float(len(values))

    @staticmethod
    def _log_generation(tag, generation_idx, total_generations, best_fitness, mean_fitness):
        print(
            f"[{tag}] gen={generation_idx:03d}/{total_generations} "
            f"best_fitness={best_fitness:.6f} "
            f"mean_fitness={mean_fitness:.6f}"
        )

    def _clip(self, value):
        if value > self.weight_clip:
            return self.weight_clip
        if value < -self.weight_clip:
            return -self.weight_clip
        return value

    def _sample_opponent_hands(self, unseen, rounds_left):
        needed = self.n_opponents * rounds_left
        if needed <= 0:
            return []

        if len(unseen) >= needed:
            sampled = self.rng.sample(unseen, needed)
        elif unseen:
            sampled = [self.rng.choice(unseen) for _ in range(needed)]
        else:
            sampled = [self.rng.randint(1, self.n_cards) for _ in range(needed)]

        hands = []
        idx = 0
        for _ in range(self.n_opponents):
            h = sampled[idx: idx + rounds_left]
            h.sort()
            hands.append(h)
            idx += rounds_left
        return hands

    def _sample_state(self):
        deck = list(range(1, self.n_cards + 1))
        self.rng.shuffle(deck)

        board = [[deck.pop()] for _ in range(self.n_rows)]
        rounds_left = self.rng.randint(self.rounds_left_min, self.rounds_left_max)
        hand = sorted(deck.pop() for _ in range(rounds_left))
        unseen = deck[:]

        rollout_samples = [
            self._sample_opponent_hands(unseen, rounds_left)
            for _ in range(self.eval_rollouts)
        ]

        return {
            "board": board,
            "hand": hand,
            "unseen": unseen,
            "rollout_samples": rollout_samples,
        }

    def _evaluate_rollout_fitness(self, weights, states):
        policy = GeneticFeaturePolicy(self.eval_core, weights)
        total_loss = 0.0

        for state in states:
            board = state["board"]
            hand = state["hand"]
            unseen = state["unseen"]
            chosen = policy.select_card(hand, board, unseen, self.n_opponents)

            loss = 0.0
            for opp_hands in state["rollout_samples"]:
                loss += float(
                    self.eval_core.rollout_total_penalty(board, hand, chosen, opp_hands)
                )
            total_loss += loss / float(len(state["rollout_samples"]))

        return -(total_loss / float(len(states)))

    def _seed_population(self, center_weights, pop_size, local_sigma, random_fraction):
        center = validate_weights(center_weights)
        population = [center[:]]

        while len(population) < pop_size:
            if self.rng.random() < random_fraction:
                genome = [self.rng.uniform(-1.5, 1.5) for _ in FEATURE_NAMES]
            else:
                genome = [self._clip(w + self.rng.gauss(0.0, local_sigma)) for w in center]
            population.append(genome)
        return population

    def _select_parent(self, population, fitnesses, k):
        kk = min(k, len(population))
        idxs = self.rng.sample(range(len(population)), kk)
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, p1, p2, crossover_rate):
        if self.rng.random() >= crossover_rate:
            return p1[:]

        child = []
        for w1, w2 in zip(p1, p2):
            alpha = self.rng.random()
            child.append(alpha * w1 + (1.0 - alpha) * w2)
        return child

    def _mutate(self, genome, mutation_rate, mutation_sigma):
        child = genome[:]
        for i in range(len(child)):
            if self.rng.random() < mutation_rate:
                child[i] = self._clip(child[i] + self.rng.gauss(0.0, mutation_sigma))
        return child

    def _run_pretrain(self):
        population = self._seed_population(
            self.initial_weights,
            pop_size=self.population_size,
            local_sigma=0.6,
            random_fraction=0.25,
        )

        best_weights = population[0][:]
        best_fitness = float("-inf")

        for gen in range(self.generations):
            states = [self._sample_state() for _ in range(self.state_samples)]
            fitnesses = [self._evaluate_rollout_fitness(g, states) for g in population]

            ranked = self._rank_indices(fitnesses)
            gen_best_fitness = fitnesses[ranked[0]]
            gen_mean_fitness = self._mean(fitnesses)

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = population[ranked[0]][:]

            self._log_generation(
                "GA-PT",
                gen + 1,
                self.generations,
                gen_best_fitness,
                gen_mean_fitness,
            )

            next_population = [population[i][:] for i in ranked[: self.elite_size]]
            while len(next_population) < self.population_size:
                p1 = self._select_parent(population, fitnesses, self.tournament_k)
                p2 = self._select_parent(population, fitnesses, self.tournament_k)
                child = self._crossover(p1, p2, self.crossover_rate)
                child = self._mutate(child, self.mutation_rate, self.mutation_sigma)
                next_population.append(child)

            population = next_population

        return {
            "best_weights": best_weights,
            "best_fitness": best_fitness,
        }

    def _scaled_tournament_cfg(self, budget_scale, is_fallback):
        cfg = dict(self.finetune_tournament_cfg)
        base_games = int(cfg["num_games_per_player"])
        scaled_games = max(
            self.finetune_min_games_per_player,
            int(round(base_games * budget_scale)),
        )
        cfg["num_games_per_player"] = scaled_games
        if is_fallback and self.finetune_force_none_duplication_on_fallback:
            cfg["duplication_mode"] = "none"
        return cfg

    def _build_finetune_tournament_config(self, model_path, tournament_cfg):
        candidate = [
            "src.players.agents.genetic_rollout_player",
            "GeneticRolloutPlayer",
            {"model_path": model_path},
            self.finetune_candidate_label,
        ]
        players = [candidate] + list(self.finetune_opponents)
        return {
            "players": players,
            "engine": dict(self.finetune_engine_cfg),
            "tournament": dict(tournament_cfg),
        }

    def _find_tournament_result(self, stem, min_mtime):
        candidates = [
            p
            for p in self.results_dir.glob(f"*_{stem}.json")
            if p.stat().st_mtime >= min_mtime - 1.0
        ]
        if not candidates:
            raise RuntimeError(
                f"No tournament result file found for config stem '{stem}'."
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _parse_candidate_fitness(self, output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        standings = payload.get("standings", [])
        candidate = next((s for s in standings if int(s.get("id", -1)) == 0), None)
        if candidate is None:
            raise RuntimeError("Could not find candidate player (id=0) in standings.")

        avg_rank = candidate.get("avg_rank")
        if avg_rank is None:
            games = int(candidate.get("games_played", 0))
            if games <= 0:
                return float("-inf")
            avg_rank = float(candidate.get("total_rank", 0.0)) / float(games)

        if not math.isfinite(avg_rank):
            return float("-inf")

        safety_penalty = 0.0
        for key in (
            "dq_count",
            "timeout_count",
            "exception_count",
            "err_oom_count",
            "err_generic_count",
        ):
            safety_penalty += float(candidate.get(key, 0))

        return -float(avg_rank) - 0.05 * safety_penalty

    def _evaluate_tournament_fitness(self, weights):
        repeat_fitness = []
        for rep in range(self.finetune_tournament_repeats):
            with tempfile.TemporaryDirectory(prefix="ga_finetune_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                model_path = tmpdir_path / "weights.json"
                base_stem = (
                    f"ga_ft_{os.getpid()}_{int(time.time() * 1000)}_"
                    f"{self.rng.randint(0, 10**9):09d}_r{rep}"
                )

                model_payload = build_model_payload(weights, metadata={"source": "finetune_eval"})
                with open(model_path, "w", encoding="utf-8") as f:
                    json.dump(model_payload, f, indent=2)

                fitness = None
                last_error = "unknown error"
                for attempt_idx, scale in enumerate(self.finetune_budget_scales):
                    cfg_stem = f"{base_stem}_a{attempt_idx}"
                    cfg_path = tmpdir_path / f"{cfg_stem}.json"
                    scaled_tournament_cfg = self._scaled_tournament_cfg(
                        budget_scale=scale,
                        is_fallback=(attempt_idx > 0),
                    )
                    tournament_cfg = self._build_finetune_tournament_config(
                        str(model_path),
                        scaled_tournament_cfg,
                    )
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(tournament_cfg, f, indent=2)

                    timeout_sec = max(1200, int(self.finetune_timeout_sec * scale))
                    start_mtime = time.time()
                    cmd = [sys.executable, "run_tournament.py", "--config", str(cfg_path)]
                    try:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(self.repo_root),
                            text=True,
                            capture_output=True,
                            timeout=timeout_sec,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        last_error = (
                            f"timeout after {timeout_sec}s "
                            f"(scale={scale}, attempt={attempt_idx + 1})"
                        )
                        print(f"[GA-FT] warning: {last_error}, retrying with smaller budget")
                        continue

                    if proc.returncode != 0:
                        msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
                        last_error = "\n".join(msg.strip().splitlines()[-40:])
                        print(
                            f"[GA-FT] warning: tournament run failed "
                            f"(scale={scale}, attempt={attempt_idx + 1}), retrying"
                        )
                        continue

                    result_path = self._find_tournament_result(cfg_stem, start_mtime)
                    fitness = self._parse_candidate_fitness(result_path)
                    if self.finetune_cleanup_results:
                        result_path.unlink(missing_ok=True)
                    break

                if fitness is None:
                    raise RuntimeError(
                        "All tournament-eval attempts failed during finetune.\n"
                        f"Last error: {last_error}"
                    )
                repeat_fitness.append(fitness)

        return sum(repeat_fitness) / float(len(repeat_fitness))

    def _run_finetune(self, seed_weights):
        population = self._seed_population(
            seed_weights,
            pop_size=self.finetune_population_size,
            local_sigma=max(0.05, self.finetune_mutation_sigma * 1.5),
            random_fraction=0.10,
        )

        best_weights = seed_weights[:]
        best_fitness = float("-inf")

        for gen in range(self.finetune_generations):
            fitnesses = [self._evaluate_tournament_fitness(g) for g in population]
            ranked = self._rank_indices(fitnesses)
            gen_best_fitness = fitnesses[ranked[0]]
            gen_mean_fitness = self._mean(fitnesses)

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = population[ranked[0]][:]

            self._log_generation(
                "GA-FT",
                gen + 1,
                self.finetune_generations,
                gen_best_fitness,
                gen_mean_fitness,
            )

            next_population = [population[i][:] for i in ranked[: self.finetune_elite_size]]
            while len(next_population) < self.finetune_population_size:
                p1 = self._select_parent(population, fitnesses, self.finetune_tournament_k)
                p2 = self._select_parent(population, fitnesses, self.finetune_tournament_k)
                child = self._crossover(p1, p2, self.finetune_crossover_rate)
                child = self._mutate(
                    child,
                    self.finetune_mutation_rate,
                    self.finetune_mutation_sigma,
                )
                next_population.append(child)

            population = next_population

        return {
            "best_weights": best_weights,
            "best_fitness": best_fitness,
        }

    def train(self):
        pretrain_res = self._run_pretrain()
        final_weights = pretrain_res["best_weights"]
        final_fitness = pretrain_res["best_fitness"]
        final_source = "pretrain_rollout"

        finetune_res = None
        if self.finetune_enabled and self.finetune_generations > 0:
            finetune_res = self._run_finetune(final_weights)
            if finetune_res["best_weights"] is not None:
                final_weights = finetune_res["best_weights"]
                final_fitness = finetune_res["best_fitness"]
                final_source = "finetune_tournament"

        payload = build_model_payload(final_weights, best_fitness=final_fitness)
        payload.update(
            {
                "pretrain_best_fitness": pretrain_res["best_fitness"],
                "finetune_best_fitness": (
                    finetune_res["best_fitness"] if finetune_res is not None else None
                ),
                "selected_from": final_source,
                "trainer_config": {
                    "n_cards": self.n_cards,
                    "n_rows": self.n_rows,
                    "n_opponents": self.n_opponents,
                    "pretrain": {
                        "population_size": self.population_size,
                        "generations": self.generations,
                        "elite_size": self.elite_size,
                        "state_samples": self.state_samples,
                        "eval_rollouts": self.eval_rollouts,
                        "tournament_k": self.tournament_k,
                        "crossover_rate": self.crossover_rate,
                        "mutation_rate": self.mutation_rate,
                        "mutation_sigma": self.mutation_sigma,
                        "weight_clip": self.weight_clip,
                        "rounds_left_min": self.rounds_left_min,
                        "rounds_left_max": self.rounds_left_max,
                    },
                    "finetune": {
                        "enabled": self.finetune_enabled,
                        "population_size": self.finetune_population_size,
                        "generations": self.finetune_generations,
                        "elite_size": self.finetune_elite_size,
                        "tournament_k": self.finetune_tournament_k,
                        "crossover_rate": self.finetune_crossover_rate,
                        "mutation_rate": self.finetune_mutation_rate,
                        "mutation_sigma": self.finetune_mutation_sigma,
                        "tournament_repeats": self.finetune_tournament_repeats,
                        "cleanup_results": self.finetune_cleanup_results,
                        "min_games_per_player": self.finetune_min_games_per_player,
                        "budget_scales": self.finetune_budget_scales,
                        "force_none_duplication_on_fallback": self.finetune_force_none_duplication_on_fallback,
                        "candidate_label": self.finetune_candidate_label,
                    },
                    "random_seed": self.random_seed,
                },
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        return payload

    @staticmethod
    def save_model(path, payload):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train GeneticRolloutPlayer with hybrid GA: "
            "rollout pretrain + run_tournament finetune."
        )
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to GA training JSON config.",
    )
    args = parser.parse_args()

    cfg = _load_json(args.train_config)
    output_path = str(_require(cfg, "output_path", "train config"))

    trainer = GeneticRolloutTrainer(cfg)
    payload = trainer.train()
    trainer.save_model(output_path, payload)
    print(f"[GA] saved model to {output_path}")


if __name__ == "__main__":
    main()
