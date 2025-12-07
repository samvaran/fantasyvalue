"""
Game Score Sampler for CVaR Simulations

Samples realistic game outcomes (total points, margin) from betting lines.
Based on historical analysis of residual distributions.

Key insight:
- Betting lines → player residuals: weak signal (r ≈ 0.03)
- Actual game outcomes → player residuals: strong signal (r ≈ 0.15-0.49)

By sampling game outcomes from betting lines, then using actual→residual equations,
we recover the full signal strength with uncertainty naturally baked in.

Distribution Parameters (from historical analysis):
- Total residual: μ=0.31, σ=13.22 (normal)
- Margin residual: μ=0.04, σ=12.75 (normal)
- Correlation: r=0.026 (can sample independently)

Sign Convention:
- spread_line = predicted home margin (positive = home favored)
- margin = home_score - away_score

Player Residual Adjustment:
- residual = intercept + β_score × team_score + β_margin × relative_margin
- relative_margin = team_score - opponent_score (positive = won)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field


# ============================================================================
# REGRESSION COEFFICIENTS (from scripts/game_score_regression.py)
# Model: residual ~ intercept + β_score × team_score + β_margin × relative_margin
# ============================================================================

RESIDUAL_COEFFICIENTS = {
    'QB': {
        'home': {'intercept': -11.148, 'beta_score': 0.4933, 'beta_margin': -0.0882},
        'away': {'intercept': -11.919, 'beta_score': 0.4997, 'beta_margin': -0.0941},
    },
    'RB': {
        'home': {'intercept': -4.278, 'beta_score': 0.1619, 'beta_margin': 0.0314},
        'away': {'intercept': -5.405, 'beta_score': 0.2028, 'beta_margin': 0.0177},
    },
    'WR': {
        'home': {'intercept': -5.300, 'beta_score': 0.2202, 'beta_margin': -0.0536},
        'away': {'intercept': -5.641, 'beta_score': 0.2227, 'beta_margin': -0.0616},
    },
    'TE': {
        'home': {'intercept': -4.917, 'beta_score': 0.1893, 'beta_margin': -0.0741},
        'away': {'intercept': -3.323, 'beta_score': 0.1186, 'beta_margin': -0.0286},
    },
}


def calculate_residual_adjustment(
    position: str,
    is_home: bool,
    team_score: float,
    opponent_score: float
) -> float:
    """
    Calculate expected residual adjustment based on game outcome.

    Args:
        position: Player position (QB, RB, WR, TE)
        is_home: Whether player is on home team
        team_score: Points scored by player's team
        opponent_score: Points scored by opponent

    Returns:
        Expected residual adjustment (add to projection)
    """
    if position not in RESIDUAL_COEFFICIENTS:
        return 0.0

    loc = 'home' if is_home else 'away'
    coef = RESIDUAL_COEFFICIENTS[position][loc]

    relative_margin = team_score - opponent_score

    adjustment = (
        coef['intercept'] +
        coef['beta_score'] * team_score +
        coef['beta_margin'] * relative_margin
    )

    return adjustment


@dataclass
class GameScoreParams:
    """Distribution parameters for game score residuals."""
    # Total residual (actual_total - total_line)
    total_mean: float = 0.31
    total_std: float = 13.22

    # Margin residual (actual_margin - spread_line), home perspective
    margin_mean: float = 0.04
    margin_std: float = 12.75


def sample_game_scores(
    total_line: float,
    spread_line: float,
    n_samples: int = 1,
    params: Optional[GameScoreParams] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample game outcomes from betting lines.

    Args:
        total_line: Vegas over/under (e.g., 47.5)
        spread_line: Predicted home margin (positive = home favored)
        n_samples: Number of samples to generate
        params: Distribution parameters (uses defaults if None)
        rng: Random number generator (creates new if None)

    Returns:
        Tuple of (total_points, margin, home_score, away_score) arrays
    """
    if params is None:
        params = GameScoreParams()
    if rng is None:
        rng = np.random.default_rng()

    # Sample residuals (independent, r=0.026)
    total_residual = rng.normal(params.total_mean, params.total_std, size=n_samples)
    margin_residual = rng.normal(params.margin_mean, params.margin_std, size=n_samples)

    # Calculate game outcomes
    sampled_total = total_line + total_residual
    sampled_margin = spread_line + margin_residual  # home perspective

    # Calculate team scores
    home_score = (sampled_total + sampled_margin) / 2
    away_score = (sampled_total - sampled_margin) / 2

    # Floor at 0 (can't have negative scores)
    home_score = np.maximum(0, home_score)
    away_score = np.maximum(0, away_score)

    return sampled_total, sampled_margin, home_score, away_score


def sample_all_games(
    games_df: pd.DataFrame,
    n_scenarios: int,
    params: Optional[GameScoreParams] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Sample game outcomes for all games across all scenarios.

    Args:
        games_df: DataFrame with columns: game_id, total_line, spread_line, home_team, away_team
        n_scenarios: Number of scenarios to generate
        params: Distribution parameters
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            game_id, scenario, total, margin, home_score, away_score, home_team, away_team
    """
    if params is None:
        params = GameScoreParams()

    rng = np.random.default_rng(seed)

    results = []

    for _, game in games_df.iterrows():
        game_id = game['game_id']
        total_line = game.get('total_line', game.get('total', 45.0))
        spread_line = game.get('spread_line', game.get('spread', 0.0))
        home_team = game.get('home_team', game.get('team', 'HOME'))
        away_team = game.get('away_team', game.get('opponent', 'AWAY'))

        # Sample game outcomes
        total, margin, home, away = sample_game_scores(
            total_line, spread_line, n_scenarios, params, rng
        )

        # Store results for each scenario
        for s in range(n_scenarios):
            results.append({
                'game_id': game_id,
                'scenario': s,
                'total': total[s],
                'margin': margin[s],
                'home_score': home[s],
                'away_score': away[s],
                'home_team': home_team,
                'away_team': away_team,
            })

    return pd.DataFrame(results)


def sample_games_vectorized(
    games_df: pd.DataFrame,
    n_scenarios: int,
    params: Optional[GameScoreParams] = None,
    seed: Optional[int] = None
) -> dict:
    """
    Vectorized game sampling - returns dict of arrays for fast lookup.

    Args:
        games_df: DataFrame with game info
        n_scenarios: Number of scenarios
        params: Distribution parameters
        seed: Random seed

    Returns:
        Dict mapping game_id -> {
            'home_score': array(n_scenarios),
            'away_score': array(n_scenarios),
            'total': array(n_scenarios),
            'margin': array(n_scenarios),
            'home_team': str,
            'away_team': str,
        }
    """
    if params is None:
        params = GameScoreParams()

    rng = np.random.default_rng(seed)

    game_samples = {}

    for _, game in games_df.iterrows():
        game_id = game['game_id']
        total_line = game.get('total_line', game.get('total', 45.0))
        spread_line = game.get('spread_line', game.get('spread', 0.0))
        home_team = game.get('home_team', game.get('team', 'HOME'))
        away_team = game.get('away_team', game.get('opponent', 'AWAY'))

        # Sample game outcomes
        total, margin, home, away = sample_game_scores(
            total_line, spread_line, n_scenarios, params, rng
        )

        game_samples[game_id] = {
            'home_score': home,
            'away_score': away,
            'total': total,
            'margin': margin,
            'home_team': home_team,
            'away_team': away_team,
        }

    return game_samples


def calculate_player_adjustments_vectorized(
    players_df: pd.DataFrame,
    game_samples: Dict,
    n_scenarios: int
) -> np.ndarray:
    """
    Calculate residual adjustments for all players across all scenarios.

    Args:
        players_df: DataFrame with player info (position, team, is_home, game_id)
        game_samples: Dict from sample_games_vectorized()
        n_scenarios: Number of scenarios

    Returns:
        Array of shape (n_players, n_scenarios) with residual adjustments
    """
    n_players = len(players_df)
    adjustments = np.zeros((n_players, n_scenarios))

    for idx, player in players_df.reset_index(drop=True).iterrows():
        position = player.get('position', '')
        game_id = player.get('game_id')
        is_home = player.get('is_home', False)

        if game_id not in game_samples or position not in RESIDUAL_COEFFICIENTS:
            continue

        game = game_samples[game_id]

        if is_home:
            team_scores = game['home_score']
            opp_scores = game['away_score']
        else:
            team_scores = game['away_score']
            opp_scores = game['home_score']

        # Vectorized calculation
        loc = 'home' if is_home else 'away'
        coef = RESIDUAL_COEFFICIENTS[position][loc]

        relative_margin = team_scores - opp_scores

        adjustments[idx] = (
            coef['intercept'] +
            coef['beta_score'] * team_scores +
            coef['beta_margin'] * relative_margin
        )

    return adjustments


if __name__ == '__main__':
    # Test the sampler
    print("=" * 60)
    print("GAME SCORE SAMPLER TEST")
    print("=" * 60)
    print()

    # Single game example
    total_line = 47.5
    spread_line = -3.0  # Home team is 3-point underdog

    total, margin, home, away = sample_game_scores(total_line, spread_line, n_samples=10000)

    print("1. SINGLE GAME SAMPLING")
    print(f"   Game: Total={total_line}, Spread={spread_line} (home is underdog)")
    print(f"   Sampled totals:  mean={total.mean():.1f}, std={total.std():.1f}")
    print(f"   Sampled margins: mean={margin.mean():.1f}, std={margin.std():.1f}")
    print(f"   Home scores:     mean={home.mean():.1f}, std={home.std():.1f}")
    print(f"   Away scores:     mean={away.mean():.1f}, std={away.std():.1f}")
    print()

    # Expected values
    print("   Expected (from betting lines):")
    expected_total = total_line + 0.31
    expected_margin = spread_line + 0.04
    expected_home = (expected_total + expected_margin) / 2
    expected_away = (expected_total - expected_margin) / 2
    print(f"     Total: {expected_total:.1f}, Margin: {expected_margin:.1f}")
    print(f"     Home: {expected_home:.1f}, Away: {expected_away:.1f}")
    print()

    # Test residual adjustment
    print("2. RESIDUAL ADJUSTMENT EXAMPLES")
    print("   Team scores 28, opponent scores 14 (win by 14):")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        home_adj = calculate_residual_adjustment(pos, True, 28, 14)
        away_adj = calculate_residual_adjustment(pos, False, 28, 14)
        print(f"     {pos}: home={home_adj:+.2f}, away={away_adj:+.2f}")
    print()

    print("   Team scores 14, opponent scores 28 (lose by 14):")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        home_adj = calculate_residual_adjustment(pos, True, 14, 28)
        away_adj = calculate_residual_adjustment(pos, False, 14, 28)
        print(f"     {pos}: home={home_adj:+.2f}, away={away_adj:+.2f}")
    print()

    # Test multiple games
    print("3. MULTIPLE GAMES VECTORIZED")
    games = pd.DataFrame([
        {'game_id': 'KC@BUF', 'total_line': 51.5, 'spread_line': -1.5,
         'home_team': 'BUF', 'away_team': 'KC'},
        {'game_id': 'DET@MIN', 'total_line': 48.0, 'spread_line': 3.0,
         'home_team': 'MIN', 'away_team': 'DET'},
    ])

    game_samples = sample_games_vectorized(games, n_scenarios=1000, seed=42)

    for game_id, samples in game_samples.items():
        print(f"   {game_id}:")
        print(f"     Home ({samples['home_team']}): {samples['home_score'].mean():.1f} ± {samples['home_score'].std():.1f}")
        print(f"     Away ({samples['away_team']}): {samples['away_score'].mean():.1f} ± {samples['away_score'].std():.1f}")
    print()

    # Test player adjustments
    print("4. PLAYER ADJUSTMENTS ACROSS SCENARIOS")
    players = pd.DataFrame([
        {'name': 'Josh Allen', 'position': 'QB', 'game_id': 'KC@BUF', 'is_home': True},
        {'name': 'Patrick Mahomes', 'position': 'QB', 'game_id': 'KC@BUF', 'is_home': False},
        {'name': 'James Cook', 'position': 'RB', 'game_id': 'KC@BUF', 'is_home': True},
        {'name': 'Justin Jefferson', 'position': 'WR', 'game_id': 'DET@MIN', 'is_home': True},
        {'name': 'Amon-Ra St. Brown', 'position': 'WR', 'game_id': 'DET@MIN', 'is_home': False},
    ])

    adjustments = calculate_player_adjustments_vectorized(players, game_samples, 1000)

    for idx, player in players.iterrows():
        adj = adjustments[idx]
        loc = "home" if player['is_home'] else "away"
        print(f"   {player['name']} ({player['position']}, {loc}):")
        print(f"     Adjustment: {adj.mean():+.2f} ± {adj.std():.2f} [P10={np.percentile(adj, 10):+.1f}, P90={np.percentile(adj, 90):+.1f}]")
    print()

    print("=" * 60)
    print("Game score sampler test complete!")
    print("=" * 60)
