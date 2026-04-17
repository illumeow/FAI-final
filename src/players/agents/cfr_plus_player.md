# CFRPlusPlayer Strategy

`CFRPlusPlayer` is an upgraded CFR-style rollout agent designed to be more robust under the 1-second decision limit.

## Core Improvements
- **RM+ updates**: uses Regret Matching Plus (`regret = max(0, regret + delta)`) for faster practical convergence.
- **Sacrifice-aware pruning**: always keeps low-card reset options in the action subset.
- **Diverse opponent model**: uses epsilon-greedy opponent rollout policy instead of fully greedy opponents.
- **Exact endgame**: when rounds left are small and unseen cards are fully determined, computes exact expected loss by enumerating opponent hand assignments.
- **Adaptive time guard**: estimates per-iteration runtime online and keeps a safety margin to avoid timeouts.

## Intended Usage
Use this as a drop-in player class:
`src.players.agents.cfr_plus_player.CFRPlusPlayer`
