"""
Monte Carlo simulation engine with parallel processing.

Evaluates lineups through repeated sampling from player distributions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

from .distribution_fit import sample_shifted_lognormal

# Note: Distribution parameters (mu, sigma, shift) are now pre-computed during
# data integration and stored in the player CSV. No need to fit distributions
# during Monte Carlo simulation.


def sample_game_script(game_probs: Dict[str, float]) -> str:
    """
    Sample a game script based on probabilities.

    Args:
        game_probs: Dict with keys shootout_prob, defensive_prob, blowout_prob, competitive_prob

    Returns:
        Sampled script: 'shootout', 'defensive', 'blowout', or 'competitive'
    """
    scripts = ['shootout', 'defensive', 'blowout', 'competitive']
    probs = [
        game_probs.get('shootout_prob', 0.25),
        game_probs.get('defensive_prob', 0.25),
        game_probs.get('blowout_prob', 0.25),
        game_probs.get('competitive_prob', 0.25)
    ]

    # Normalize probabilities (should sum to 1, but just in case)
    total = sum(probs)
    probs = [p / total for p in probs]

    return np.random.choice(scripts, p=probs)


def determine_script_key(script: str, is_favorite: bool) -> str:
    """
    Determine the floor/ceiling key for a game script.

    Args:
        script: Game script ('shootout', 'defensive', 'blowout', 'competitive')
        is_favorite: True if player's team is favorite

    Returns:
        Key for accessing floor/ceiling columns (e.g., 'blowout_favorite')
    """
    if script == 'blowout':
        return 'blowout_favorite' if is_favorite else 'blowout_underdog'
    else:
        return script


def simulate_player_score(
    player: Dict,
    game_script: str,
    is_favorite: bool
) -> float:
    """
    Simulate a single player's score for a given game script.

    Args:
        player: Player dict with all stats and pre-computed distribution params
        game_script: Sampled game script
        is_favorite: Whether player's team is favorite

    Returns:
        Simulated fantasy points
    """
    # Get scenario-specific distribution parameters (pre-computed in data integration)
    script_key = determine_script_key(game_script, is_favorite)

    # Read pre-computed distribution parameters from player data
    mu = player[f'mu_{script_key}']
    sigma = player[f'sigma_{script_key}']
    shift = player[f'shift_{script_key}']

    # Sample from distribution
    score = sample_shifted_lognormal(mu, sigma, shift, size=1)[0]

    # Floor at 0 (can't have negative points)
    return max(0, score)


def simulate_lineup_once(
    lineup_players: List[Dict],
    game_scripts_df: pd.DataFrame
) -> float:
    """
    Run a single simulation for a lineup.

    Args:
        lineup_players: List of player dicts (9 players)
        game_scripts_df: DataFrame with game script probabilities

    Returns:
        Total lineup score for this simulation
    """
    lineup_total = 0.0

    # Sample game scripts for all games
    game_script_samples = {}
    for _, game in game_scripts_df.iterrows():
        game_id = game['game_id']
        game_script_samples[game_id] = sample_game_script(game.to_dict())

    # Simulate each player
    for player in lineup_players:
        game_id = player['game_id']
        is_favorite = player['is_favorite']

        sampled_script = game_script_samples.get(game_id, 'competitive')
        player_score = simulate_player_score(player, sampled_script, is_favorite)

        lineup_total += player_score

    return lineup_total


def simulate_lineup(
    lineup_players: List[Dict],
    game_scripts_df: pd.DataFrame,
    n_sims: int = 10000
) -> Dict[str, float]:
    """
    Run Monte Carlo simulation for a lineup.

    Args:
        lineup_players: List of player dicts (9 players)
        game_scripts_df: DataFrame with game script probabilities
        n_sims: Number of simulations

    Returns:
        Dict with distribution statistics (mean, median, p10, p90, std)
    """
    scores = []

    for _ in range(n_sims):
        score = simulate_lineup_once(lineup_players, game_scripts_df)
        scores.append(score)

    scores = np.array(scores)

    return {
        'mean': float(np.mean(scores)),
        'median': float(np.percentile(scores, 50)),
        'p10': float(np.percentile(scores, 10)),
        'p90': float(np.percentile(scores, 90)),
        'std': float(np.std(scores)),
        'skewness': float((np.mean(scores) - np.median(scores)) / np.std(scores)) if np.std(scores) > 0 else 0.0
    }


def _evaluate_lineup_worker(args: Tuple[int, List[str], pd.DataFrame, pd.DataFrame, int]) -> Tuple[int, Dict[str, float]]:
    """
    Worker function for parallel lineup evaluation.

    Args:
        args: Tuple of (lineup_idx, player_ids, players_df, game_scripts_df, n_sims)

    Returns:
        Tuple of (lineup_idx, simulation_results)
    """
    lineup_idx, player_ids, players_df, game_scripts_df, n_sims = args

    # Get player data
    lineup_players = []
    for player_id in player_ids:
        # Try by id first (if column exists), then by name
        if 'id' in players_df.columns:
            player_row = players_df[players_df['id'] == player_id]
        else:
            player_row = players_df[players_df['name'] == player_id]

        if len(player_row) == 0:
            # Try by name as fallback
            player_row = players_df[players_df['name'] == player_id]

        if len(player_row) > 0:
            lineup_players.append(player_row.iloc[0].to_dict())

    if len(lineup_players) != len(player_ids):
        warnings.warn(f"Lineup {lineup_idx}: Found {len(lineup_players)} players, expected {len(player_ids)}")

    # Run simulation
    results = simulate_lineup(lineup_players, game_scripts_df, n_sims)

    return lineup_idx, results


def evaluate_lineups_parallel(
    lineups: List[List[str]],
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_sims: int = 10000,
    n_processes: int = None
) -> List[Dict[str, float]]:
    """
    Evaluate multiple lineups in parallel using multiprocessing.

    Args:
        lineups: List of lineups, where each lineup is a list of player IDs
        players_df: DataFrame with all player data
        game_scripts_df: DataFrame with game script probabilities
        n_sims: Number of simulations per lineup
        n_processes: Number of parallel processes (default: CPU count)

    Returns:
        List of simulation results dicts, in same order as lineups
    """
    if n_processes is None:
        n_processes = cpu_count()

    print(f"Evaluating {len(lineups)} lineups with {n_sims} simulations each...")
    print(f"Using {n_processes} parallel processes")

    # Prepare arguments for workers
    args_list = [
        (i, lineup, players_df, game_scripts_df, n_sims)
        for i, lineup in enumerate(lineups)
    ]

    # Run parallel evaluation
    with Pool(processes=n_processes) as pool:
        results = pool.map(_evaluate_lineup_worker, args_list)

    # Sort by lineup index and extract results
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


if __name__ == '__main__':
    # Test simulation with dummy data
    print("Testing Monte Carlo simulation...")

    # Create dummy player
    player = {
        'name': 'Test Player',
        'position': 'RB',
        'fpProjPts': 15.0,
        'floor_shootout': 8.0,
        'ceiling_shootout': 25.0,
        'floor_defensive': 10.0,
        'ceiling_defensive': 20.0,
        'floor_blowout_favorite': 12.0,
        'ceiling_blowout_favorite': 18.0,
        'floor_blowout_underdog': 6.0,
        'ceiling_blowout_underdog': 22.0,
        'floor_competitive': 9.0,
        'ceiling_competitive': 23.0,
        'td_odds_floor_mult': 1.0,
        'td_odds_ceiling_mult': 1.0,
        'game_id': 'TEST@GAME',
        'is_favorite': True
    }

    # Create dummy game scripts
    game_scripts_df = pd.DataFrame([{
        'game_id': 'TEST@GAME',
        'shootout_prob': 0.25,
        'defensive_prob': 0.25,
        'blowout_prob': 0.25,
        'competitive_prob': 0.25
    }])

    # Simulate 100 times
    print("\nRunning 100 simulations...")
    scores = []
    for _ in range(100):
        score = simulate_player_score(player, 'shootout', True)
        scores.append(score)

    print(f"  Mean: {np.mean(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"  P10: {np.percentile(scores, 10):.2f}")
    print(f"  P90: {np.percentile(scores, 90):.2f}")
    print(f"  Std: {np.std(scores):.2f}")
