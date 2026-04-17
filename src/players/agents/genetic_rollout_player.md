# GeneticRolloutPlayer Strategy

`GeneticRolloutPlayer` is a training-based linear policy tuned by a genetic algorithm.

## Core Idea
- Represent each candidate card with risk features (immediate expected risk, forced-take flag, interval density, light future proxy, etc.).
- Score cards with a weighted linear model.
- Use a hybrid GA trainer:
  - Stage 1 (pretrain): rollout-estimated penalties on sampled states.
  - Stage 2 (finetune): real tournament performance from `run_tournament.py`.

## Training
Use the dedicated trainer entrypoint:
```bash
python -m src.players.agents.genetic_rollout_trainer --train-config configs/GA/train.json
```

The command writes model weights to `output_path` in the training config.

## Inference
- During the game, the player loads trained weights from `model_path` if provided.
- If `model_path` is omitted, it falls back to built-in default weights.
