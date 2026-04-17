import json
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


def binom_tail(n, p, threshold):
    if threshold <= 0:
        return 1.0
    if threshold > n:
        return 0.0
    total = 0.0
    for k in range(threshold, n + 1):
        total += comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return total


def validate_weights(weights):
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} weights, got {len(weights)}.")
    return [float(v) for v in weights]


def load_weights_from_path(model_path):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return validate_weights(payload)

    if isinstance(payload, dict):
        if "best_weights" in payload:
            return validate_weights(payload["best_weights"])
        if "weights" in payload:
            return validate_weights(payload["weights"])

    raise ValueError(
        "Model file must be a list of weights or a dict with "
        "'best_weights'/'weights'."
    )


def build_model_payload(weights, best_fitness=None, metadata=None):
    payload = {
        "model_type": "genetic_rollout_linear",
        "feature_names": FEATURE_NAMES,
        "best_weights": validate_weights(weights),
        "best_fitness": best_fitness,
    }
    if metadata:
        payload.update(metadata)
    return payload


class GeneticFeaturePolicy:
    def __init__(self, core, weights):
        self.core = core
        self.weights = validate_weights(weights)
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
                binom_tail(n_opponents, between_prob, needed_before_me)
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
            return validate_weights(weights)
        if model_path is not None:
            return load_weights_from_path(model_path)
        return DEFAULT_WEIGHTS[:]

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        unseen = self.core.build_unseen_pool(hand, history)
        n_opponents = max(0, len(history["scores"]) - 1)
        return self.policy.select_card(hand, board, unseen, n_opponents)
