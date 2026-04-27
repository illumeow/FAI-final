
import json
import math
import os
import argparse
import re
from datetime import datetime
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

# Imports moved inside run()

# --- Example Tournament Configuration ---
# TOURNAMENT_CONFIG = {
#     "players": [
#         {
#             "path": "src.players.TA.random_player",
#             "class": "RandomPlayer",
#             "args": {}
#         },
#         ["src.players.TA.random_player", "RandomPlayer"],
#         ["src.players.TA.random_player", "RandomPlayer"],
#         ["src.players.TA.max_player", "MaxPlayer", {}],
#         ["src.players.TA.min_player", "MinPlayer", {}],
#         ["src.players.TA.max_player", "MaxPlayer", {}],
#         ["src.players.TA.min_player", "MinPlayer", {}],
#         ["src.players.TA.max_player", "MaxPlayer", {}],
#         ["src.players.TA.min_player", "MinPlayer", {}]
#     ],
#     "engine": {
#         "n_players": 4,
#         "n_rounds": 10,
#         "verbose": False
#     },
#     "tournament_rounds": 10,
#     "use_permutations": True
# }

def compact_json_dumps(data):
    text = json.dumps(data, indent=4)
    def collapse(match):
        content = match.group(1)
        items = [x.strip() for x in content.split(',')]
        return '[' + ', '.join(items) + ']'
    return re.sub(r'\[\s*([^\[\]\{\}]+?)\s*\]', collapse, text)

def _round_or_none(value, digits):
    if value is None:
        return None
    try:
        if not math.isfinite(value):
            return None
    except TypeError:
        return value
    return round(value, digits)

def build_readable_standings(runner, final_standings):
    has_calibrated = any(
        p.get("calibrated_score") is not None
        and isinstance(p.get("calibrated_score"), (int, float))
        and math.isfinite(p["calibrated_score"])
        for p in final_standings
    )
    has_grouped = any("group_id" in p for p in final_standings)

    if has_grouped:
        sorted_stats = sorted(
            final_standings,
            key=lambda x: (x.get("group_id", -1), x.get("avg_rank_2", float('inf')), x.get("avg_score_2", float('inf'))),
        )
    else:
        sorted_stats = sorted(
            final_standings,
            key=lambda x: (x.get("avg_rank", float('inf')), x.get("avg_score", float('inf'))),
        )

    note_keys = ("dq_count", "timeout_count", "exception_count", "err_oom_count", "err_generic_count")

    out = []
    for i, p in enumerate(sorted_stats):
        cfg = runner.player_configs[p["config_idx"]]
        entry = {
            "rank": i + 1,
            "id": p["id"],
            "label": cfg.get("label", "-"),
            "class": cfg.get("class", "-"),
            "avg_rank": _round_or_none(p.get("avg_rank"), 4),
            "avg_score": _round_or_none(p.get("avg_score"), 4),
            "est_elo": _round_or_none(p.get("est_elo"), 1),
            "games_played": p.get("games_played", 0),
            "is_baseline": p.get("is_baseline", False),
        }
        if has_calibrated:
            entry["calibrated_score"] = _round_or_none(p.get("calibrated_score"), 2)
        if has_grouped:
            entry["group_id"] = p.get("group_id")
            entry["avg_rank_1"] = _round_or_none(p.get("avg_rank_1"), 4)
            entry["avg_rank_2"] = _round_or_none(p.get("avg_rank_2"), 4)
            entry["avg_score_1"] = _round_or_none(p.get("avg_score_1"), 4)
            entry["avg_score_2"] = _round_or_none(p.get("avg_score_2"), 4)
        notes = {k: p[k] for k in note_keys if p.get(k, 0) > 0}
        if notes:
            entry["notes"] = notes
        out.append(entry)
    return out

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def run():
    parser = argparse.ArgumentParser(description="Run a tournament.")
    parser.add_argument("--config", type=str, help="Path to a base JSON configuration file.", default=None)
    parser.add_argument("--player-cfg", type=str, help="Path to a JSON configuration file for players.", default=None)
    parser.add_argument("--engine-cfg", type=str, help="Path to a JSON configuration file for engine.", default=None)
    parser.add_argument("--tournament-cfg", type=str, help="Path to a JSON configuration file for tournament.", default=None)
    args = parser.parse_args()

    # 1. Config Loading
    config = {}
    config_name = "custom"
    
    if args.config:
        print(f"Loading base configuration from {args.config}...")
        config = load_config(args.config)
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        
    if args.player_cfg:
        print(f"Overriding players configuration from {args.player_cfg}...")
        config["players"] = load_config(args.player_cfg)
            
    if args.engine_cfg:
        print(f"Overriding engine configuration from {args.engine_cfg}...")
        config["engine"] = load_config(args.engine_cfg)
             
    if args.tournament_cfg:
        print(f"Overriding tournament configuration from {args.tournament_cfg}...")
        config["tournament"] = load_config(args.tournament_cfg)
             
    if not config:
        print("Error: No configuration provided. Please provide --config or the split config arguments.")
        sys.exit(1)
            
    # 2. Run Tournament
    try:
        t_type = config.get("tournament", {}).get("type", "combination")
        
        if t_type == "combination":
            from src.tournament_runner import CombinationTournamentRunner
            runner = CombinationTournamentRunner(config)
        elif t_type == "random_partition":
            from src.tournament_runner import RandomPartitionTournamentRunner
            runner = RandomPartitionTournamentRunner(config)
        elif t_type == "grouped_random_partition":
            from src.tournament_runner import GroupedRandomPartitionTournamentRunner
            runner = GroupedRandomPartitionTournamentRunner(config)
        else:
            raise ValueError(f"Unknown tournament type: {t_type}")
            
        final_standings, history = runner.run()
        runner.print_standings()
            
    except Exception as e:
        print(f"Tournament failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Save Results
    output_data = {
        "config": config,
        "standings": final_standings,
        "history": history
    }
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{config_name}.json"
    results_dir = os.path.join("results", "tournament")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, filename)
    
    print(f"Saving results to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            f.write(compact_json_dumps(output_data))
        print("Save successful.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # Also save a human-readable standings-only file alongside the full results.
    standings_filename = f"{timestamp}_{config_name}_standings.json"
    standings_file = os.path.join(results_dir, standings_filename)
    print(f"Saving standings summary to {standings_file}...")
    try:
        readable_standings = build_readable_standings(runner, final_standings)
        with open(standings_file, 'w') as f:
            json.dump(readable_standings, f, indent=2)
        print("Standings save successful.")
    except Exception as e:
        print(f"Error saving standings: {e}")

if __name__ == "__main__":
    run()
