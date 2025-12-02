"""
Phase 2: Monte Carlo Evaluation

Evaluates lineup candidates through parallel Monte Carlo simulation.
Uses multiprocessing to evaluate 1000 lineups in ~15-20 minutes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
import time
import sys

from utils.monte_carlo import evaluate_lineups_parallel

# Import config values (add parent dir to path temporarily)
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from config import DEFAULT_SIMS, DEFAULT_PROCESSES


def evaluate_candidates(
    candidates_path: str = 'outputs/lineups_candidates.csv',
    players_path: str = 'players_integrated.csv',
    game_scripts_path: str = 'game_script.csv',
    n_sims: int = DEFAULT_SIMS,
    n_processes: int = DEFAULT_PROCESSES,
    output_path: str = 'outputs/lineup_evaluations.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate lineup candidates via parallel Monte Carlo simulation.

    Args:
        candidates_path: Path to candidates CSV from Phase 1
        players_path: Path to integrated players CSV
        game_scripts_path: Path to game scripts CSV
        n_sims: Number of simulations per lineup
        n_processes: Number of parallel processes (None = auto-detect)
        output_path: Where to save evaluation results
        verbose: Print progress

    Returns:
        DataFrame with evaluation results
    """
    if verbose:
        print("=" * 80)
        print("PHASE 2: MONTE CARLO EVALUATION")
        print("=" * 80)

    # Load data
    if verbose:
        print("\nLoading data...")

    candidates_df = pd.read_csv(candidates_path)
    players_df = pd.read_csv(players_path)
    game_scripts_df = pd.read_csv(game_scripts_path)

    # Normalize column names for multiprocessing compatibility
    if 'id' not in players_df.columns and 'name' in players_df.columns:
        players_df['id'] = players_df['name']
    if 'fdSalary' not in players_df.columns and 'salary' in players_df.columns:
        players_df['fdSalary'] = players_df['salary']

    if verbose:
        print(f"  Candidates: {len(candidates_df)}")
        print(f"  Players: {len(players_df)}")
        print(f"  Games: {len(game_scripts_df)}")

    # Parse player IDs from candidates
    lineups = []
    for _, row in candidates_df.iterrows():
        player_ids = row['player_ids'].split(',')
        lineups.append(player_ids)

    # Run parallel evaluation
    if verbose:
        print(f"\nEvaluating {len(lineups)} lineups with {n_sims} simulations each...")
        print(f"This will take approximately {len(lineups) * n_sims / 1000000:.1f}-{len(lineups) * n_sims / 500000:.1f} minutes with parallelization")

    start_time = time.time()

    results = evaluate_lineups_parallel(
        lineups=lineups,
        players_df=players_df,
        game_scripts_df=game_scripts_df,
        n_sims=n_sims,
        n_processes=n_processes
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nCompleted in {elapsed / 60:.1f} minutes ({elapsed / len(lineups):.2f} sec/lineup)")

    # Combine with candidates info
    evaluations = []
    for i, (candidate_row, result) in enumerate(zip(candidates_df.itertuples(), results)):
        evaluations.append({
            'lineup_id': candidate_row.lineup_id,
            'tier': candidate_row.tier,
            'temperature': candidate_row.temperature,
            'player_ids': candidate_row.player_ids,
            'total_salary': candidate_row.total_salary,
            'milp_projection': candidate_row.total_projection,
            'mean': result['mean'],
            'median': result['median'],
            'p10': result['p10'],
            'p90': result['p90'],
            'std': result['std'],
            'skewness': result['skewness']
        })

    evaluations_df = pd.DataFrame(evaluations)

    # Sort by median (balanced metric)
    evaluations_df = evaluations_df.sort_values('median', ascending=False).reset_index(drop=True)

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    evaluations_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"  Lineups evaluated: {len(evaluations_df)}")
        print(f"  Best median score: {evaluations_df['median'].max():.2f}")
        print(f"  Best P90 score: {evaluations_df['p90'].max():.2f}")
        print(f"  Best mean score: {evaluations_df['mean'].max():.2f}")
        print(f"\nTop 10 lineups by median:")
        for i, row in evaluations_df.head(10).iterrows():
            print(f"    {i+1}. Lineup {row['lineup_id']:4d} (Tier {row['tier']}) - "
                  f"Median: {row['median']:6.2f}, P10: {row['p10']:6.2f}, P90: {row['p90']:6.2f}")
        print(f"\n  Saved to: {output_file}")
        print("=" * 80)

    return evaluations_df


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Evaluate lineup candidates")
    parser.add_argument('--input', default='outputs/lineups_candidates.csv', help='Input candidates CSV')
    parser.add_argument('--players', default='data/intermediate/players_integrated.csv', help='Players CSV')
    parser.add_argument('--game-scripts', default='data/intermediate/game_script.csv', help='Game scripts CSV')
    parser.add_argument('--output', default='outputs/lineup_evaluations.csv', help='Output evaluations CSV')
    parser.add_argument('--sims', type=int, default=DEFAULT_SIMS, help=f'Number of simulations per lineup (default: {DEFAULT_SIMS})')
    parser.add_argument('--processes', type=int, default=DEFAULT_PROCESSES, help='Number of parallel processes (default: auto)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Evaluate
    evaluate_candidates(
        candidates_path=args.input,
        players_path=args.players,
        game_scripts_path=args.game_scripts,
        n_sims=args.sims,
        n_processes=args.processes,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
