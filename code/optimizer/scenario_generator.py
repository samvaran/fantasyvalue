"""
Scenario Matrix Generator for CVaR-MILP Optimization

Generates a matrix of simulated player scores across multiple scenarios.
Each scenario samples game scripts and player performances from pre-computed distributions.

The scenario matrix is used by the CVaR optimizer to find lineups that perform
well in the top percentile of outcomes (e.g., p80).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path


def sample_game_scripts_vectorized(
    game_scripts_df: pd.DataFrame,
    n_scenarios: int
) -> Dict[str, np.ndarray]:
    """
    Sample game scripts for all games across all scenarios.

    Args:
        game_scripts_df: DataFrame with game_id and script probabilities
        n_scenarios: Number of scenarios to generate

    Returns:
        Dict mapping game_id -> array of sampled scripts (length n_scenarios)
        Each element is one of: 'shootout', 'defensive', 'blowout', 'competitive'
    """
    game_script_samples = {}
    scripts = ['shootout', 'defensive', 'blowout', 'competitive']

    for _, game in game_scripts_df.iterrows():
        game_id = game['game_id']

        # Get probabilities (normalize to sum to 1)
        probs = np.array([
            game.get('shootout_prob', 0.25),
            game.get('defensive_prob', 0.25),
            game.get('blowout_prob', 0.25),
            game.get('competitive_prob', 0.25)
        ])
        probs = probs / probs.sum()

        # Sample scripts for all scenarios at once
        sampled_indices = np.random.choice(4, size=n_scenarios, p=probs)
        game_script_samples[game_id] = np.array([scripts[i] for i in sampled_indices])

    return game_script_samples


def get_script_key(script: str, is_favorite: bool) -> str:
    """
    Convert game script to the key used for distribution parameters.

    Args:
        script: Game script ('shootout', 'defensive', 'blowout', 'competitive')
        is_favorite: Whether the player's team is the favorite

    Returns:
        Key for distribution params (e.g., 'blowout_favorite', 'shootout')
    """
    if script == 'blowout':
        return 'blowout_favorite' if is_favorite else 'blowout_underdog'
    return script


def generate_scenario_matrix(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_scenarios: int = 1000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate scenario matrix: points[player_idx][scenario_idx]

    Uses vectorized operations for efficiency. For each scenario:
    1. Sample game script for each game
    2. For each player, use the corresponding distribution params
    3. Sample from shifted log-normal distribution

    Args:
        players_df: DataFrame with player data including pre-computed
                   distribution params (mu_*, sigma_*, shift_*)
        game_scripts_df: DataFrame with game script probabilities
        n_scenarios: Number of scenarios to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - scenario_matrix: np.ndarray of shape (n_players, n_scenarios)
        - player_index: DataFrame mapping player index to player info
    """
    if seed is not None:
        np.random.seed(seed)

    n_players = len(players_df)

    # Step 1: Sample game scripts for all games across all scenarios
    game_script_samples = sample_game_scripts_vectorized(game_scripts_df, n_scenarios)

    # Step 2: Build player index for fast lookup
    player_index = players_df[['id', 'name', 'position', 'salary', 'team', 'game_id', 'is_favorite', 'fpProjPts']].copy()
    player_index = player_index.reset_index(drop=True)

    # Step 3: Generate scores for each player across all scenarios
    scenario_matrix = np.zeros((n_players, n_scenarios))

    # Pre-generate standard normal samples for all players and scenarios
    z_samples = np.random.randn(n_players, n_scenarios)

    for player_idx, player in players_df.reset_index(drop=True).iterrows():
        game_id = player.get('game_id')
        is_favorite = player.get('is_favorite', False)

        if game_id not in game_script_samples:
            # Player's game not in game scripts - use competitive as default
            mu = player.get('mu_competitive', np.log(max(1, player.get('fpProjPts', 10))))
            sigma = player.get('sigma_competitive', 0.3)
            shift = player.get('shift_competitive', 0)

            scores = np.exp(mu + sigma * z_samples[player_idx]) + shift
            scenario_matrix[player_idx] = np.maximum(0, scores)
            continue

        # Get the sampled scripts for this game
        scripts_for_game = game_script_samples[game_id]

        # For each scenario, get the appropriate distribution params
        scores = np.zeros(n_scenarios)

        # Group scenarios by script for vectorized sampling
        for script in ['shootout', 'defensive', 'blowout', 'competitive']:
            script_mask = (scripts_for_game == script)
            if not script_mask.any():
                continue

            script_key = get_script_key(script, is_favorite)

            mu = player.get(f'mu_{script_key}')
            sigma = player.get(f'sigma_{script_key}')
            shift = player.get(f'shift_{script_key}')

            # Handle missing distribution params
            if pd.isna(mu) or pd.isna(sigma):
                # Fallback to consensus projection with moderate variance
                consensus = player.get('fpProjPts', 10)
                mu = np.log(max(1, consensus))
                sigma = 0.3
                shift = 0

            # Sample scores for scenarios with this script
            z = z_samples[player_idx][script_mask]
            script_scores = np.exp(mu + sigma * z) + shift
            scores[script_mask] = script_scores

        # Floor at 0 (can't have negative fantasy points)
        scenario_matrix[player_idx] = np.maximum(0, scores)

    return scenario_matrix, player_index


def compute_lineup_stats(
    scenario_matrix: np.ndarray,
    player_indices: np.ndarray,
    alpha: float = 0.20
) -> Dict[str, float]:
    """
    Compute distribution statistics for a lineup from the scenario matrix.

    Args:
        scenario_matrix: Full scenario matrix (n_players, n_scenarios)
        player_indices: Array of player indices in the lineup
        alpha: Percentile for CVaR (0.20 = top 20% = p80)

    Returns:
        Dict with mean, median, p10, p80, std, cvar_score
    """
    # Sum scores across selected players for each scenario
    lineup_scores = scenario_matrix[player_indices].sum(axis=0)

    # Compute percentiles
    p80_idx = int((1 - alpha) * 100)  # For alpha=0.20, this is 80

    stats = {
        'mean': float(np.mean(lineup_scores)),
        'median': float(np.median(lineup_scores)),
        'std': float(np.std(lineup_scores)),
        'p10': float(np.percentile(lineup_scores, 10)),
        'p80': float(np.percentile(lineup_scores, 80)),
        'p90': float(np.percentile(lineup_scores, 90)),
    }

    # CVaR score: average of top alpha% of scenarios
    threshold = np.percentile(lineup_scores, (1 - alpha) * 100)
    top_scores = lineup_scores[lineup_scores >= threshold]
    stats['cvar_score'] = float(np.mean(top_scores)) if len(top_scores) > 0 else stats['p80']

    return stats


if __name__ == '__main__':
    # Test with dummy data
    print("Testing scenario generator...")

    # Create dummy players
    np.random.seed(42)
    players = []
    for pos, count in [('QB', 10), ('RB', 20), ('WR', 30), ('TE', 10), ('D', 10)]:
        for i in range(count):
            consensus = np.random.uniform(5, 25)
            players.append({
                'id': f'{pos}_{i}',
                'name': f'{pos} Player {i}',
                'position': pos,
                'salary': np.random.randint(4500, 9500),
                'team': f'TEAM{i % 5}',
                'opponent': f'OPP{(i + 1) % 5}',
                'game_id': f'TEAM{i % 5}@OPP{(i + 1) % 5}',
                'is_favorite': i % 2 == 0,
                'fpProjPts': consensus,
                # Distribution params for each script
                'mu_shootout': np.log(consensus * 1.1),
                'sigma_shootout': 0.35,
                'shift_shootout': 0,
                'mu_defensive': np.log(consensus * 0.9),
                'sigma_defensive': 0.25,
                'shift_defensive': 0,
                'mu_blowout_favorite': np.log(consensus * 1.05),
                'sigma_blowout_favorite': 0.30,
                'shift_blowout_favorite': 0,
                'mu_blowout_underdog': np.log(consensus * 0.95),
                'sigma_blowout_underdog': 0.40,
                'shift_blowout_underdog': 0,
                'mu_competitive': np.log(consensus),
                'sigma_competitive': 0.30,
                'shift_competitive': 0,
            })

    players_df = pd.DataFrame(players)

    # Create dummy game scripts
    game_scripts = []
    for i in range(5):
        game_scripts.append({
            'game_id': f'TEAM{i}@OPP{(i + 1) % 5}',
            'shootout_prob': 0.3,
            'defensive_prob': 0.2,
            'blowout_prob': 0.25,
            'competitive_prob': 0.25,
        })
    game_scripts_df = pd.DataFrame(game_scripts)

    # Generate scenario matrix
    print(f"\nGenerating scenario matrix for {len(players_df)} players...")
    import time
    start = time.time()
    scenario_matrix, player_index = generate_scenario_matrix(
        players_df, game_scripts_df, n_scenarios=1000, seed=42
    )
    elapsed = time.time() - start

    print(f"  Shape: {scenario_matrix.shape}")
    print(f"  Time: {elapsed*1000:.1f}ms")
    print(f"  Mean score per player: {scenario_matrix.mean(axis=1).mean():.2f}")
    print(f"  Std across scenarios: {scenario_matrix.std(axis=1).mean():.2f}")

    # Test lineup stats
    print("\nTesting lineup stats computation...")
    lineup_indices = np.array([0, 10, 11, 20, 21, 22, 40, 50, 70])  # Random lineup
    stats = compute_lineup_stats(scenario_matrix, lineup_indices, alpha=0.20)
    print(f"  Lineup stats: {stats}")

    print("\nScenario generator test complete!")
