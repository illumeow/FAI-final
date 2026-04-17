import argparse
import json
import os
import random
import time
from math import comb

from .core_utils import SimpleAgentCore


FEATURE_NAMES = [
    "expected_now",
    "forced_take",
    "delta_norm",
    "post_len_norm",
    "target_row_score_norm",
    "card_score_norm",
    "card_value_norm",
    "between_prob",
    "future_proxy_norm",
    "bias",
]

DEFAULT_WEIGHTS = [
    1.35,
    1.00,
    0.22,
    0.28,
    0.44,
    -0.16,
    0.06,
    0.85,
    0.55,
    0.00,
]


def _binom_tail(n, p, threshold):
    if threshold <= 0:
        return 1.0
    if threshold > n:
        return 0.0
    total = 0.0
    for k in range(threshold, n + 1):
        total += comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return total


def _validate_weights(weights):
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} weights, got {len(weights)}."
        )
    return [float(v) for v in weights]


def _load_weights_from_path(model_path):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return _validate_weights(payload)

    if isinstance(payload, dict):
        if "best_weights" in payload:
            return _validate_weights(payload["best_weights"])
        if "weights" in payload:
            return _validate_weights(payload["weights"])

    raise ValueError(
        "Model file must be a list of weights or a dict with "
        "'best_weights'/'weights'."
    )


class GeneticFeaturePolicy:
    def __init__(self, core, weights):
        self.core = core
        self.weights = _validate_weights(weights)
        self.n_cards = core.n_cards

    def _expected_now_components(self, board, card, unseen, n_opponents):
        fit_idx, fit_last = self.core.best_fit_row(board, card)

        if fit_idx == -1:
            take_idx = self.core.forced_take_row(board)
            row_score = float(self.core.row_score(board[take_idx]))
            return {
                "expected_now": row_score,
                "forced_take": 1.0,
                "delta_norm": 1.0,
                "post_len_norm": 0.2,
                "target_row_score_norm": row_score / 30.0,
                "between_prob": 0.0,
            }

        row = board[fit_idx]
        row_len = len(row)
        row_score = float(self.core.row_score(row))
        delta_norm = (card - fit_last) / float(self.n_cards)
        between_prob = 0.0

        if unseen and n_opponents > 0 and card - fit_last > 1:
            between = 0
            for u in unseen:
                if fit_last < u < card:
                    between += 1
            between_prob = between / float(len(unseen))

        if row_len >= 5:
            p_stable = (1.0 - between_prob) ** n_opponents if n_opponents > 0 else 1.0
            expected_now = row_score * p_stable
            post_len_norm = 0.2
        else:
            needed_before_me = 5 - row_len
            p_forced_take = (
                _binom_tail(n_opponents, between_prob, needed_before_me)
                if n_opponents > 0
                else 0.0
            )
            expected_now = row_score * p_forced_take
            post_len_norm = (row_len + 1) / 5.0

        return {
            "expected_now": expected_now,
            "forced_take": 0.0,
            "delta_norm": delta_norm,
            "post_len_norm": post_len_norm,
            "target_row_score_norm": row_score / 30.0,
            "between_prob": between_prob,
        }

    def _future_proxy(self, board, hand, chosen_card, unseen, n_opponents):
        if len(hand) <= 1:
            return 0.0

        next_board = [row.copy() for row in board]
        self.core.place_card(next_board, chosen_card)

        best_next = float("inf")
        for card in hand:
            if card == chosen_card:
                continue
            comp = self._expected_now_components(next_board, card, unseen, n_opponents)
            if comp["expected_now"] < best_next:
                best_next = comp["expected_now"]

        if best_next == float("inf"):
            return 0.0
        return best_next

    def card_features(self, hand, board, card, unseen, n_opponents):
        comp = self._expected_now_components(board, card, unseen, n_opponents)
        future_proxy = self._future_proxy(board, hand, card, unseen, n_opponents)
        return [
            comp["expected_now"] / 20.0,
            comp["forced_take"],
            comp["delta_norm"],
            comp["post_len_norm"],
            comp["target_row_score_norm"],
            self.core.card_score(card) / 7.0,
            card / float(self.n_cards),
            comp["between_prob"],
            future_proxy / 20.0,
            1.0,
        ]

    def score_card(self, hand, board, card, unseen, n_opponents):
        feats = self.card_features(hand, board, card, unseen, n_opponents)
        total = 0.0
        for w, x in zip(self.weights, feats):
            total += w * x
        return total

    def select_card(self, hand, board, unseen, n_opponents):
        best_card = hand[0]
        best_key = None

        for card in hand:
            score = self.score_card(hand, board, card, unseen, n_opponents)
            tie_key = self.core.heuristic_card_key(board, card)
            key = (score, tie_key, card)
            if best_key is None or key < best_key:
                best_key = key
                best_card = card

        return best_card


class GeneticRolloutPlayer:
    """
    Trained linear policy over risk features.

    You can pass a JSON model via model_path. If omitted, it uses default weights.
    """

    def __init__(self, player_idx, model_path=None, weights=None):
        self.player_idx = player_idx
        self.core = SimpleAgentCore(player_idx, seed_offset=25013)
        self.weights = self._resolve_weights(model_path=model_path, weights=weights)
        self.policy = GeneticFeaturePolicy(self.core, self.weights)

    def _resolve_weights(self, model_path=None, weights=None):
        if weights is not None:
            return _validate_weights(weights)
        if model_path is not None:
            return _load_weights_from_path(model_path)
        return DEFAULT_WEIGHTS[:]

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        unseen = self.core.build_unseen_pool(hand, history)
        n_opponents = max(0, len(history["scores"]) - 1)
        return self.policy.select_card(hand, board, unseen, n_opponents)


class GeneticRolloutTrainer:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.n_cards = int(self.cfg.get("n_cards", 104))
        self.n_rows = int(self.cfg.get("n_rows", 4))
        self.n_opponents = int(self.cfg.get("n_opponents", 3))

        self.population_size = int(self.cfg.get("population_size", 40))
        self.generations = int(self.cfg.get("generations", 30))
        self.elite_size = int(self.cfg.get("elite_size", 4))
        self.state_samples = int(self.cfg.get("state_samples", 120))
        self.eval_rollouts = int(self.cfg.get("eval_rollouts", 6))
        self.tournament_k = int(self.cfg.get("tournament_k", 4))

        self.crossover_rate = float(self.cfg.get("crossover_rate", 0.85))
        self.mutation_rate = float(self.cfg.get("mutation_rate", 0.22))
        self.mutation_sigma = float(self.cfg.get("mutation_sigma", 0.18))
        self.weight_clip = float(self.cfg.get("weight_clip", 5.0))

        self.rounds_left_min = int(self.cfg.get("rounds_left_min", 4))
        self.rounds_left_max = int(self.cfg.get("rounds_left_max", 10))

        self.random_seed = int(self.cfg.get("random_seed", 2026))
        self.rng = random.Random(self.random_seed)
        self.eval_core = SimpleAgentCore(player_idx=0, n_cards=self.n_cards, seed_offset=39017)

        initial = self.cfg.get("initial_weights", DEFAULT_WEIGHTS)
        self.initial_weights = _validate_weights(initial)

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

    def _evaluate_genome(self, weights, states):
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

    def _seed_population(self):
        population = [self.initial_weights[:]]
        while len(population) < self.population_size:
            if self.rng.random() < 0.35:
                base = self.initial_weights
                genome = [self._clip(w + self.rng.gauss(0.0, 0.6)) for w in base]
            else:
                genome = [self.rng.uniform(-1.5, 1.5) for _ in FEATURE_NAMES]
            population.append(genome)
        return population

    def _select_parent(self, population, fitnesses):
        k = min(self.tournament_k, len(population))
        idxs = self.rng.sample(range(len(population)), k)
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, p1, p2):
        if self.rng.random() >= self.crossover_rate:
            return p1[:]

        child = []
        for w1, w2 in zip(p1, p2):
            alpha = self.rng.random()
            child.append(alpha * w1 + (1.0 - alpha) * w2)
        return child

    def _mutate(self, genome):
        child = genome[:]
        for i in range(len(child)):
            if self.rng.random() < self.mutation_rate:
                child[i] = self._clip(child[i] + self.rng.gauss(0.0, self.mutation_sigma))
        return child

    def train(self):
        population = self._seed_population()

        best_weights = None
        best_fitness = float("-inf")

        for gen in range(self.generations):
            states = [self._sample_state() for _ in range(self.state_samples)]
            fitnesses = [self._evaluate_genome(g, states) for g in population]

            ranked = sorted(
                range(len(population)),
                key=lambda i: fitnesses[i],
                reverse=True,
            )

            gen_best_fitness = fitnesses[ranked[0]]
            gen_mean_fitness = sum(fitnesses) / float(len(fitnesses))
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = population[ranked[0]][:]

            print(
                f"[GA] gen={gen + 1:03d}/{self.generations} "
                f"best_fitness={gen_best_fitness:.6f} "
                f"mean_fitness={gen_mean_fitness:.6f}"
            )

            next_population = [population[i][:] for i in ranked[: self.elite_size]]
            while len(next_population) < self.population_size:
                p1 = self._select_parent(population, fitnesses)
                p2 = self._select_parent(population, fitnesses)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_population.append(child)

            population = next_population

        return {
            "model_type": "genetic_rollout_linear",
            "feature_names": FEATURE_NAMES,
            "best_weights": best_weights,
            "best_fitness": best_fitness,
            "trainer_config": {
                "n_cards": self.n_cards,
                "n_rows": self.n_rows,
                "n_opponents": self.n_opponents,
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
                "random_seed": self.random_seed,
            },
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

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
        description="Train GeneticRolloutPlayer weights with a genetic algorithm."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to GA training JSON config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional model output path. Overrides output_path in the train config.",
    )
    args = parser.parse_args()

    cfg = _load_json(args.train_config)
    output_path = args.output or cfg.get("output_path")
    if not output_path:
        raise ValueError("Please provide output_path in config or pass --output.")

    trainer = GeneticRolloutTrainer(cfg)
    payload = trainer.train()
    trainer.save_model(output_path, payload)
    print(f"[GA] saved model to {output_path}")


if __name__ == "__main__":
    main()
