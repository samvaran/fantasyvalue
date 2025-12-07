"""
Scenario Matrix Generator V2 for CVaR-MILP Optimization

Uses game-score-based sampling for more realistic correlations:
1. Sample game outcomes (total, margin) from betting lines
2. Calculate player residual adjustments from game outcomes
3. Sample player scores from skew-normal distributions with adjustments

Key insight:
- Betting lines → player residuals: weak signal (r ≈ 0.03)
- Actual game outcomes → player residuals: strong signal (r ≈ 0.15-0.49)

By sampling game outcomes first, we recover the full signal strength.

Distribution parameters from: scripts/charts/distributions/distribution_analysis_results.md
Game sampling from: scripts/charts/distributions/game_score_distribution_results.md
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .game_score_sampler import (
        sample_games_vectorized,
        calculate_player_adjustments_vectorized,
        GameScoreParams,
    )
except ImportError:
    from game_score_sampler import (
        sample_games_vectorized,
        calculate_player_adjustments_vectorized,
        GameScoreParams,
    )


# ============================================================================
# SKEW-NORMAL PARAMETERS (from distribution_analysis_results.md)
# ============================================================================

SKEW_NORMAL_PARAMS = {
    'QB': {
        'home': {'alpha': 1.124, 'loc': -5.57, 'scale': 10.19},
        'away': {'alpha': 0.918, 'loc': -5.97, 'scale': 9.74},
    },
    'RB': {
        'home': {'alpha': 2.391, 'loc': -8.63, 'scale': 11.39},
        'away': {'alpha': 2.153, 'loc': -9.09, 'scale': 11.35},
    },
    'WR': {
        'home': {'alpha': 2.246, 'loc': -8.38, 'scale': 11.43},
        'away': {'alpha': 2.066, 'loc': -8.51, 'scale': 11.17},
    },
    'TE': {
        'home': {'alpha': 2.344, 'loc': -7.57, 'scale': 9.75},
        'away': {'alpha': 2.262, 'loc': -7.45, 'scale': 9.45},
    },
}

# Sigma scaling: σ = slope × E[pts] + intercept
SIGMA_SCALING = {
    'QB': {'home': {'slope': 0.0683, 'intercept': 6.77}, 'away': {'slope': 0.0800, 'intercept': 6.52}},
    'RB': {'home': {'slope': 0.1228, 'intercept': 6.07}, 'away': {'slope': 0.2072, 'intercept': 5.03}},
    'WR': {'home': {'slope': 0.1439, 'intercept': 5.93}, 'away': {'slope': 0.1902, 'intercept': 5.21}},
    'TE': {'home': {'slope': 0.1243, 'intercept': 5.42}, 'away': {'slope': 0.1394, 'intercept': 4.99}},
}


def sample_skew_normal(alpha: float, loc: float, scale: float, size: int, rng=None) -> np.ndarray:
    """Sample from a skew-normal distribution."""
    if rng is None:
        rng = np.random.default_rng()

    # Use scipy's skewnorm
    return stats.skewnorm.rvs(alpha, loc=loc, scale=scale, size=size, random_state=rng)


def get_scaled_sigma(position: str, is_home: bool, projection: float) -> float:
    """Calculate sigma scaled by projection level."""
    if position not in SIGMA_SCALING:
        return 8.0  # Default for D/ST etc.

    loc = 'home' if is_home else 'away'
    params = SIGMA_SCALING[position][loc]
    return params['slope'] * projection + params['intercept']


def prepare_games_df(game_scripts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game_scripts format to games_df format needed for sampling.

    Converts:
        game_id (AWAY@HOME), spread, total
    To:
        game_id, total_line, spread_line, home_team, away_team

    Note: spread in game_scripts is from favorite perspective (negative = favorite wins)
          spread_line for sampling is home margin (positive = home favored)
    """
    games = []

    for _, row in game_scripts_df.iterrows():
        game_id = row['game_id']

        # Parse AWAY@HOME format
        if '@' in game_id:
            away_team, home_team = game_id.split('@')
        else:
            # Fallback - treat as unknown
            away_team, home_team = 'AWAY', 'HOME'

        total_line = row.get('total', 45.0)

        # Convert spread to home perspective
        # In game_scripts: spread is negative for favorite (e.g., -5.5)
        # favorite column tells us who is favored
        favorite = row.get('favorite', home_team)
        spread = row.get('spread', 0)

        if favorite == home_team:
            # Home is favorite, spread should be positive
            spread_line = abs(spread)
        else:
            # Away is favorite, spread should be negative
            spread_line = -abs(spread)

        games.append({
            'game_id': game_id,
            'total_line': total_line,
            'spread_line': spread_line,
            'home_team': home_team,
            'away_team': away_team,
        })

    return pd.DataFrame(games)


def add_is_home_column(players_df: pd.DataFrame) -> pd.DataFrame:
    """Add is_home column based on game_id string (AWAY@HOME format).

    IMPORTANT: Must use game_id column, not game column!
    - game_id: AWAY@HOME format (e.g., CLE@SF means SF is home)
    - game: TEAM@OPPONENT format (varies by player perspective)
    """
    df = players_df.copy()

    if 'is_home' not in df.columns:
        def get_is_home(row):
            # Use game_id (AWAY@HOME format), NOT game (TEAM@OPPONENT format)
            game_id = row.get('game_id', row.get('game', ''))
            team = row.get('team', '')
            if '@' in game_id:
                away, home = game_id.split('@')
                return team == home
            return False

        df['is_home'] = df.apply(get_is_home, axis=1)

    return df


def generate_scenario_matrix_v2(
    players_df: pd.DataFrame,
    games_df: pd.DataFrame,
    n_scenarios: int = 1000,
    seed: Optional[int] = None,
    use_game_adjustments: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate scenario matrix using game-score-based sampling.

    Pipeline:
    1. Sample game outcomes for all games
    2. For each player:
       a. Get base projection
       b. Sample residual from skew-normal (position × home/away)
       c. Add game-outcome adjustment (if enabled)
       d. Scale variance by projection tier

    Args:
        players_df: DataFrame with columns:
            - id, name, position, salary, team, game_id
            - is_home (bool) - will be derived if not present
            - fpProjPts (projection)
        games_df: DataFrame with columns (from game_scripts or prepared):
            - game_id, total_line (or total), spread_line (or spread), home_team, away_team
            OR game_scripts format that will be converted
        n_scenarios: Number of scenarios
        seed: Random seed
        use_game_adjustments: Whether to apply game outcome adjustments

    Returns:
        Tuple of:
        - scenario_matrix: np.ndarray of shape (n_players, n_scenarios)
        - player_index: DataFrame mapping index to player info
    """
    rng = np.random.default_rng(seed)

    # Ensure players have is_home column
    players_df = add_is_home_column(players_df)

    # Convert games_df if it's in game_scripts format
    if 'total_line' not in games_df.columns and 'total' in games_df.columns:
        games_df = prepare_games_df(games_df)

    n_players = len(players_df)

    # Step 1: Sample game outcomes
    game_samples = sample_games_vectorized(
        games_df, n_scenarios, params=GameScoreParams(), seed=seed
    )

    # Step 2: Calculate game-based adjustments for all players
    if use_game_adjustments:
        game_adjustments = calculate_player_adjustments_vectorized(
            players_df, game_samples, n_scenarios
        )
    else:
        game_adjustments = np.zeros((n_players, n_scenarios))

    # Step 3: Build player index
    player_index = players_df[['id', 'name', 'position', 'salary', 'team', 'game_id', 'is_home', 'fpProjPts']].copy()
    player_index = player_index.reset_index(drop=True)

    # Step 4: Generate player scores
    scenario_matrix = np.zeros((n_players, n_scenarios))

    for idx, player in players_df.reset_index(drop=True).iterrows():
        position = player.get('position', '')
        is_home = player.get('is_home', False)
        projection = player.get('fpProjPts', 10.0)

        # Get distribution parameters
        if position in SKEW_NORMAL_PARAMS:
            loc_key = 'home' if is_home else 'away'
            params = SKEW_NORMAL_PARAMS[position][loc_key]

            # Sample base residual from skew-normal
            base_residual = sample_skew_normal(
                params['alpha'], params['loc'], params['scale'],
                size=n_scenarios, rng=rng
            )

            # Scale variance by projection tier
            base_sigma = get_scaled_sigma(position, is_home, projection)
            # The skew-normal already has its own scale, but we can adjust
            # For now, use the base distribution as-is (it captures the variance structure)

        else:
            # D/ST or unknown position - use normal distribution
            base_residual = rng.normal(0, 5, size=n_scenarios)

        # Add game-based adjustment
        total_residual = base_residual + game_adjustments[idx]

        # Calculate final scores
        scores = projection + total_residual

        # Floor at 0
        scenario_matrix[idx] = np.maximum(0, scores)

    return scenario_matrix, player_index


def compute_lineup_stats(
    scenario_matrix: np.ndarray,
    player_indices: np.ndarray,
    alpha: float = 0.20
) -> Dict[str, float]:
    """
    Compute distribution statistics for a lineup.

    Args:
        scenario_matrix: Full scenario matrix (n_players, n_scenarios)
        player_indices: Array of player indices in the lineup
        alpha: Percentile for CVaR (0.20 = top 20% = p80)

    Returns:
        Dict with mean, median, p10, p80, p90, std, cvar_score
    """
    lineup_scores = scenario_matrix[player_indices].sum(axis=0)

    stats = {
        'mean': float(np.mean(lineup_scores)),
        'median': float(np.median(lineup_scores)),
        'std': float(np.std(lineup_scores)),
        'p10': float(np.percentile(lineup_scores, 10)),
        'p80': float(np.percentile(lineup_scores, 80)),
        'p90': float(np.percentile(lineup_scores, 90)),
    }

    # CVaR: average of top alpha% of scenarios
    threshold = np.percentile(lineup_scores, (1 - alpha) * 100)
    top_scores = lineup_scores[lineup_scores >= threshold]
    stats['cvar_score'] = float(np.mean(top_scores)) if len(top_scores) > 0 else stats['p80']

    return stats


if __name__ == '__main__':
    from pathlib import Path

    print("=" * 70)
    print("SCENARIO GENERATOR V2 TEST")
    print("=" * 70)
    print()

    # Try to load real data
    data_dir = Path(__file__).parent.parent.parent / 'data' / '2025_11_30' / 'intermediate'
    players_path = data_dir / '1_players.csv'
    games_path = data_dir / '2_game_scripts.csv'

    if players_path.exists() and games_path.exists():
        print("Loading real data...")
        players = pd.read_csv(players_path)
        games = pd.read_csv(games_path)
        print(f"  {len(players)} players, {len(games)} games")

        # Test with subset
        test_positions = ['QB', 'RB', 'WR', 'TE']
        players = players[players['position'].isin(test_positions)].head(50)
        print(f"  Testing with {len(players)} players")
    else:
        print("Using synthetic test data...")
        games = pd.DataFrame([
            {'game_id': 'KC@BUF', 'total_line': 51.5, 'spread_line': -1.5,
             'home_team': 'BUF', 'away_team': 'KC'},
            {'game_id': 'DET@MIN', 'total_line': 48.0, 'spread_line': 3.0,
             'home_team': 'MIN', 'away_team': 'DET'},
            {'game_id': 'DAL@PHI', 'total_line': 44.5, 'spread_line': 6.0,
             'home_team': 'PHI', 'away_team': 'DAL'},
        ])

        players = pd.DataFrame([
            # KC@BUF game
            {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'salary': 9200, 'team': 'BUF',
             'game_id': 'KC@BUF', 'is_home': True, 'fpProjPts': 22.5},
            {'id': 2, 'name': 'Patrick Mahomes', 'position': 'QB', 'salary': 9000, 'team': 'KC',
             'game_id': 'KC@BUF', 'is_home': False, 'fpProjPts': 21.0},
            {'id': 3, 'name': 'James Cook', 'position': 'RB', 'salary': 7500, 'team': 'BUF',
             'game_id': 'KC@BUF', 'is_home': True, 'fpProjPts': 14.5},
            {'id': 4, 'name': 'Isiah Pacheco', 'position': 'RB', 'salary': 6800, 'team': 'KC',
             'game_id': 'KC@BUF', 'is_home': False, 'fpProjPts': 12.0},

            # DET@MIN game
            {'id': 5, 'name': 'Sam Darnold', 'position': 'QB', 'salary': 7200, 'team': 'MIN',
             'game_id': 'DET@MIN', 'is_home': True, 'fpProjPts': 18.5},
            {'id': 6, 'name': 'Justin Jefferson', 'position': 'WR', 'salary': 9500, 'team': 'MIN',
             'game_id': 'DET@MIN', 'is_home': True, 'fpProjPts': 19.0},
            {'id': 7, 'name': 'Amon-Ra St. Brown', 'position': 'WR', 'salary': 8800, 'team': 'DET',
             'game_id': 'DET@MIN', 'is_home': False, 'fpProjPts': 17.5},

            # DAL@PHI game
            {'id': 8, 'name': 'AJ Brown', 'position': 'WR', 'salary': 8200, 'team': 'PHI',
             'game_id': 'DAL@PHI', 'is_home': True, 'fpProjPts': 15.0},
            {'id': 9, 'name': 'Dallas Goedert', 'position': 'TE', 'salary': 5500, 'team': 'PHI',
             'game_id': 'DAL@PHI', 'is_home': True, 'fpProjPts': 10.0},
            {'id': 10, 'name': 'CeeDee Lamb', 'position': 'WR', 'salary': 8500, 'team': 'DAL',
             'game_id': 'DAL@PHI', 'is_home': False, 'fpProjPts': 16.0},
        ])

    # Generate scenarios
    print("Generating scenario matrix...")
    import time
    start = time.time()
    scenario_matrix, player_index = generate_scenario_matrix_v2(
        players, games, n_scenarios=1000, seed=42
    )
    elapsed = time.time() - start

    print(f"  Shape: {scenario_matrix.shape}")
    print(f"  Time: {elapsed*1000:.1f}ms")
    print()

    # Print player statistics
    print("PLAYER SCORE DISTRIBUTIONS:")
    print("-" * 70)
    print(f"{'Name':<25} {'Proj':>6} {'Mean':>6} {'Std':>6} {'P10':>6} {'P90':>6}")
    print("-" * 70)

    for idx, row in player_index.iterrows():
        scores = scenario_matrix[idx]
        print(f"{row['name']:<25} {row['fpProjPts']:>6.1f} {scores.mean():>6.1f} {scores.std():>6.1f} "
              f"{np.percentile(scores, 10):>6.1f} {np.percentile(scores, 90):>6.1f}")

    print()

    # Test a lineup
    print("SAMPLE LINEUP STATS:")
    print("-" * 70)
    # Allen, Cook, Jefferson, AJ Brown, Goedert
    lineup_indices = np.array([0, 2, 5, 7, 8])
    lineup_stats = compute_lineup_stats(scenario_matrix, lineup_indices, alpha=0.20)

    print(f"  Lineup: {', '.join(player_index.loc[lineup_indices, 'name'].values)}")
    print(f"  Projections sum: {player_index.loc[lineup_indices, 'fpProjPts'].sum():.1f}")
    print(f"  Simulated mean: {lineup_stats['mean']:.1f}")
    print(f"  Simulated std: {lineup_stats['std']:.1f}")
    print(f"  P10: {lineup_stats['p10']:.1f}")
    print(f"  P80: {lineup_stats['p80']:.1f}")
    print(f"  P90: {lineup_stats['p90']:.1f}")
    print(f"  CVaR (top 20%): {lineup_stats['cvar_score']:.1f}")
    print()

    # Compare with and without game adjustments
    print("EFFECT OF GAME ADJUSTMENTS:")
    print("-" * 70)

    scenario_no_adj, _ = generate_scenario_matrix_v2(
        players, games, n_scenarios=1000, seed=42, use_game_adjustments=False
    )

    lineup_stats_no_adj = compute_lineup_stats(scenario_no_adj, lineup_indices, alpha=0.20)

    print(f"  {'Metric':<15} {'With Adj':>10} {'Without':>10} {'Diff':>10}")
    print("  " + "-" * 45)
    for key in ['mean', 'std', 'p10', 'p90', 'cvar_score']:
        diff = lineup_stats[key] - lineup_stats_no_adj[key]
        print(f"  {key:<15} {lineup_stats[key]:>10.1f} {lineup_stats_no_adj[key]:>10.1f} {diff:>+10.1f}")

    print()
    print("=" * 70)
    print("Scenario generator V2 test complete!")
    print("=" * 70)
