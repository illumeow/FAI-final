# GeneticRolloutPlayer Strategy

`GeneticRolloutPlayer` is a training-based linear policy tuned by a genetic algorithm.

## Core Idea
- Represent each candidate card with risk features (immediate expected risk, forced-take flag, interval density, light future proxy, etc.).
- Score cards with a weighted linear model.
- Evolve those weights offline with GA by minimizing rollout-estimated penalties on sampled game states.

## Training
Use the built-in trainer in the same module:
```bash
python -m src.players.agents.train_genetic_rollout --train-config configs/game/genetic_rollout_train.json
```

The command writes model weights to `output_path` in the training config.

## Inference
- During the game, the player loads trained weights from `model_path` if provided.
- If `model_path` is omitted, it falls back to built-in default weights.
