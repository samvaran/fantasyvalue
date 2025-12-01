"""
Data Integration Pipeline - Phase 2

Integrates game script analysis, TD odds, and team offensive projections
to adjust player floor/ceiling projections.

Pipeline:
1. Load raw data (players, game lines, TD odds)
2. Calculate team offensive totals (projected points, TD odds)
3. Enhanced game script analysis (using team totals as additional signals)
4. Adjust floor/ceiling based on game script (weighted average by probability)
5. Adjust floor/ceiling based on TD odds
6. Pre-compute distribution parameters for all scenarios
7. Output integrated player data

Output: players_integrated.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Import distribution fitting function
from optimizer.utils.distribution_fit import fit_shifted_lognormal


# ============================================================================
# TEAM NAME NORMALIZATION
# ============================================================================

# Map team abbreviations to consistent format (game_lines uses JAX, players use JAC)
TEAM_ABBR_MAP = {
    'JAC': 'JAX',  # Jacksonville
}

def normalize_team_abbr(team: str) -> str:
    """Normalize team abbreviation to match game_lines.csv format."""
    return TEAM_ABBR_MAP.get(team, team)


# ============================================================================
# ADJUSTMENT CONSTANTS (easily tunable)
# ============================================================================

# Game Script Floor Adjustments (multipliers)
GAME_SCRIPT_FLOOR = {
    'shootout': {
        'QB': 0.75,   # Wide range for high variance in shootouts
        'RB': 0.90,   # Reduced floor (less running in shootouts)
        'WR': 0.75,   # Wide range for high variance
        'TE': 0.75,   # Wide range for high variance
        'D': 0.70,    # Low floor (high scoring = fewer points for D)
    },
    'defensive': {
        'QB': 0.75,   # Lower floor but ceiling will compensate
        'RB': 0.95,   # Stable floor (more conservative)
        'WR': 0.75,   # Lower floor
        'TE': 0.75,   # Lower floor
        'D': 1.00,    # Stable floor (low scoring = more points for D)
    },
    'blowout_favorite': {
        'QB': 0.80,   # Lower floor (may sit in 4th quarter)
        'RB': 0.95,   # Stable floor (clock management)
        'WR': 0.80,   # Lower floor
        'TE': 0.80,   # Lower floor
        'D': 1.00,    # Stable floor
    },
    'blowout_underdog': {
        'QB': 0.80,   # Lower floor for variance
        'RB': 0.75,   # Much lower - game script dependent
        'WR': 0.85,   # Decent floor
        'TE': 0.80,   # Lower floor
        'D': 0.75,    # Low floor (garbage time scores)
    },
    'competitive': {
        'QB': 0.85,   # Moderate baseline floor
        'RB': 0.85,   # Moderate baseline floor
        'WR': 0.85,   # Moderate baseline floor
        'TE': 0.85,   # Moderate baseline floor
        'D': 0.85,    # Moderate baseline floor
    }
}

# Game Script Ceiling Adjustments (multipliers)
GAME_SCRIPT_CEILING = {
    'shootout': {
        'QB': 1.40,   # Massive upside (high scoring)
        'RB': 1.05,   # Modest ceiling (less running)
        'WR': 1.45,   # Huge upside
        'TE': 1.40,   # Huge upside
        'D': 1.05,    # Modest ceiling (high scoring = fewer D points)
    },
    'defensive': {
        'QB': 1.15,   # Limited upside (low scoring)
        'RB': 1.15,   # Modest upside
        'WR': 1.10,   # Limited upside
        'TE': 1.15,   # Limited upside
        'D': 1.40,    # High ceiling (sacks, turnovers, TDs)
    },
    'blowout_favorite': {
        'QB': 1.15,   # Limited upside (conservative play)
        'RB': 1.30,   # Good upside
        'WR': 1.10,   # Limited upside
        'TE': 1.15,   # Limited upside
        'D': 1.30,    # Good ceiling (turnovers, sacks)
    },
    'blowout_underdog': {
        'QB': 1.35,   # Big upside when trailing
        'RB': 1.05,   # Modest ceiling
        'WR': 1.35,   # Big upside (passing attack)
        'TE': 1.30,   # Good upside
        'D': 1.05,    # Modest ceiling (garbage time scores)
    },
    'competitive': {
        'QB': 1.25,   # Moderate ceiling
        'RB': 1.20,   # Moderate ceiling
        'WR': 1.25,   # Moderate ceiling
        'TE': 1.25,   # Moderate ceiling
        'D': 1.20,    # Moderate ceiling
    }
}

# TD Odds Adjustments
TD_ODDS_FLOOR_IMPACT = 0.05    # 5% floor boost per 10% TD probability
TD_ODDS_CEILING_IMPACT = 0.15  # 15% ceiling boost per 10% TD probability

# Team offensive totals weight in game script calculation
TEAM_TOTAL_WEIGHT = 0.20  # 20% weight for team offensive projections signal


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_players_raw() -> pd.DataFrame:
    """Load raw player data."""
    csv_path = Path('data/intermediate/players_raw.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Error: {csv_path} not found. Run fetch_data.py first.")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} players from {csv_path}")
    return df


def load_game_lines() -> pd.DataFrame:
    """Load game lines data."""
    csv_path = Path('data/intermediate/game_lines.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Error: {csv_path} not found. Run fetch_data.py first.")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df) // 2} games from game_lines.csv")
    return df


def load_td_odds() -> Dict:
    """Load TD odds data from cache."""
    json_path = Path('cache/dk_td_odds.json')
    if not json_path.exists():
        print(f"  Warning: {json_path} not found. TD odds will not be integrated.")
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded TD odds for {len(data)} players")
    return data


def calculate_team_offensive_totals(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team offensive totals (projected points and TD probability sum).

    Returns DataFrame with columns: team, total_projected_pts, total_td_prob
    """
    # Filter to offensive players only (exclude defenses)
    offensive = players_df[players_df['position'] != 'D'].copy()

    # Use FantasyPros projection (fpProjPts is the main projection in raw data)
    offensive['proj'] = offensive['fpProjPts'].fillna(0)

    # Group by team and sum
    team_totals = offensive.groupby('team').agg({
        'proj': 'sum',
        'tdProbability': lambda x: x.fillna(0).sum()
    }).reset_index()

    team_totals.columns = ['team', 'total_projected_pts', 'total_td_prob']

    print(f"\n  Team offensive totals calculated for {len(team_totals)} teams")
    return team_totals


def enhance_game_script_with_totals(
    game_lines_df: pd.DataFrame,
    team_totals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Enhance game script analysis by incorporating team offensive totals.

    Returns DataFrame with game_id and script probabilities.
    """
    from game_script_continuous import (
        calculate_shootout_score,
        calculate_defensive_score,
        calculate_blowout_score,
        calculate_competitive_score
    )

    games = game_lines_df.to_dict('records')
    results = []

    for i in range(0, len(games), 2):
        game1 = games[i]
        game2 = games[i + 1]

        # Determine favorite
        if game1['spread'] < 0:
            fav = game1
            dog = game2
        else:
            fav = game2
            dog = game1

        # Get team totals
        fav_totals = team_totals_df[team_totals_df['team'] == fav['team_abbr']]
        dog_totals = team_totals_df[team_totals_df['team'] == dog['team_abbr']]

        fav_proj = fav_totals['total_projected_pts'].iloc[0] if len(fav_totals) > 0 else 0
        dog_proj = dog_totals['total_projected_pts'].iloc[0] if len(dog_totals) > 0 else 0
        fav_td_odds = fav_totals['total_td_prob'].iloc[0] if len(fav_totals) > 0 else 0
        dog_td_odds = dog_totals['total_td_prob'].iloc[0] if len(dog_totals) > 0 else 0

        # Calculate base scores (from betting lines)
        shootout_base = calculate_shootout_score(fav['total'], fav['spread'], fav['total_over_odds'])
        defensive_base = calculate_defensive_score(fav['total'], fav['total_under_odds'])
        blowout_base, blowout_conf = calculate_blowout_score(
            fav['spread'], fav['moneyline'], fav['spread_odds'], dog['spread_odds']
        )
        competitive_base = calculate_competitive_score(fav['spread'])

        # Calculate team total signals
        total_proj = fav_proj + dog_proj
        total_td_odds = fav_td_odds + dog_td_odds

        # Team total shootout signal (high combined projections)
        team_shootout_signal = min(total_proj / 55.0, 1.0)  # Normalize to ~55 pts

        # Team total TD odds signal (high combined TD probability)
        team_td_signal = min(total_td_odds / 2.5, 1.0)  # Normalize to ~2.5 TDs

        # Combine signals (weighted)
        shootout_score = (
            shootout_base * (1 - TEAM_TOTAL_WEIGHT) +
            (team_shootout_signal + team_td_signal) / 2 * TEAM_TOTAL_WEIGHT
        )

        # Defensive signal (inverted - low projections)
        defensive_score = (
            defensive_base * (1 - TEAM_TOTAL_WEIGHT) +
            (1 - team_shootout_signal) * TEAM_TOTAL_WEIGHT
        )

        # Blowout and competitive stay the same (less affected by projections)
        blowout_score = blowout_base
        competitive_score = competitive_base

        # Normalize to probability distribution (sum to 1)
        total = shootout_score + defensive_score + blowout_score + competitive_score

        if total > 0:
            shootout_prob = shootout_score / total
            defensive_prob = defensive_score / total
            blowout_prob = blowout_score / total
            competitive_prob = competitive_score / total
        else:
            shootout_prob = defensive_prob = blowout_prob = competitive_prob = 0.25

        # Determine primary script
        scores = {
            'shootout': shootout_prob,
            'defensive': defensive_prob,
            'blowout': blowout_prob,
            'competitive': competitive_prob
        }
        primary_script = max(scores, key=scores.get)

        results.append({
            'game_id': f"{dog['team_abbr']}@{fav['team_abbr']}",
            'favorite': fav['team_abbr'],
            'underdog': dog['team_abbr'],
            'shootout_prob': shootout_prob,
            'defensive_prob': defensive_prob,
            'blowout_prob': blowout_prob,
            'competitive_prob': competitive_prob,
            'primary_script': primary_script,
            'script_strength': max(scores.values()),
            'blowout_confidence': blowout_conf
        })

    return pd.DataFrame(results)


def calculate_floor_ceiling_for_all_game_scripts(
    player: pd.Series,
    game_script: pd.Series
) -> Dict[str, float]:
    """
    Calculate player floor and ceiling for ALL possible game scripts.

    Enforces constraints:
    - Floor must be < consensus (mean projection)
    - Ceiling must be > consensus (mean projection)

    Returns dict with keys:
        floor_shootout, ceiling_shootout,
        floor_defensive, ceiling_defensive,
        floor_blowout_favorite, ceiling_blowout_favorite,
        floor_blowout_underdog, ceiling_blowout_underdog,
        floor_competitive, ceiling_competitive
    """
    position = player['position']
    consensus = player.get('fpProjPts', 0)  # Mean projection
    original_floor = player.get('espnLowScore', consensus * 0.5)
    original_ceiling = player.get('espnHighScore', consensus * 1.5)

    # Calculate for each game script scenario
    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

    results = {}
    for script in scripts:
        floor_mult = GAME_SCRIPT_FLOOR[script][position]
        ceiling_mult = GAME_SCRIPT_CEILING[script][position]

        adjusted_floor = original_floor * floor_mult
        adjusted_ceiling = original_ceiling * ceiling_mult

        # CRITICAL CONSTRAINTS:
        # Floor must never exceed consensus (mean)
        # Ceiling must never fall below consensus (mean)
        adjusted_floor = min(adjusted_floor, consensus * 0.95)  # Cap at 95% of consensus
        adjusted_ceiling = max(adjusted_ceiling, consensus * 1.05)  # Floor at 105% of consensus

        results[f'floor_{script}'] = adjusted_floor
        results[f'ceiling_{script}'] = adjusted_ceiling

    return results


def calculate_distribution_params_for_all_scenarios(
    consensus: float,
    script_floors_ceilings: Dict[str, float],
    td_floor_mult: float,
    td_ceiling_mult: float
) -> Dict[str, float]:
    """
    Pre-compute distribution parameters (mu, sigma, shift) for all game script scenarios.

    This avoids recomputing distributions millions of times during Monte Carlo simulation.

    Args:
        consensus: Player's consensus projection (mean)
        script_floors_ceilings: Dict with floor_X and ceiling_X for each script
        td_floor_mult: TD odds floor multiplier
        td_ceiling_mult: TD odds ceiling multiplier

    Returns:
        Dict with mu_X, sigma_X, shift_X for each script scenario
    """
    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

    results = {}
    for script in scripts:
        # Get base floor/ceiling for this script
        floor = script_floors_ceilings[f'floor_{script}']
        ceiling = script_floors_ceilings[f'ceiling_{script}']

        # Apply TD odds multipliers (same as done in monte_carlo.py)
        floor_adjusted = floor * td_floor_mult
        ceiling_adjusted = ceiling * td_ceiling_mult

        # Fit to floor/ceiling only - let mean vary by scenario
        # This allows boom/bust scenarios to actually affect player scores
        try:
            from optimizer.utils.distribution_fit import fit_lognormal_to_percentiles
            mu, sigma, shift = fit_lognormal_to_percentiles(floor_adjusted, ceiling_adjusted)
            results[f'mu_{script}'] = mu
            results[f'sigma_{script}'] = sigma
            results[f'shift_{script}'] = shift
        except (ValueError, Exception) as e:
            # Fallback to simple parameters if fitting fails
            std = (ceiling_adjusted - floor_adjusted) / 5
            results[f'mu_{script}'] = np.log(consensus) if consensus > 0 else 0
            results[f'sigma_{script}'] = std / consensus if consensus > 0 else 0.2
            results[f'shift_{script}'] = 0.0

    return results


def calculate_td_odds_multipliers(player: pd.Series) -> Tuple[float, float]:
    """
    Calculate TD odds adjustment multipliers (NOT applied, just stored).

    Returns (floor_multiplier, ceiling_multiplier)
    """
    td_prob = player.get('tdProbability', 0)
    if pd.isna(td_prob):
        td_prob = 0

    # Scale TD probability impact (0-100% becomes 0-1.0)
    td_prob_scaled = td_prob / 100.0

    # Floor multiplier: modest boost (e.g., 20% TD prob → 1.01x)
    floor_mult = 1.0 + (td_prob_scaled * TD_ODDS_FLOOR_IMPACT)

    # Ceiling multiplier: stronger boost (e.g., 20% TD prob → 1.03x)
    ceiling_mult = 1.0 + (td_prob_scaled * TD_ODDS_CEILING_IMPACT)

    return floor_mult, ceiling_mult


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("DATA INTEGRATION PIPELINE - PHASE 2")
    print("=" * 80)

    # Step 1: Load raw data
    print("\n=== STEP 1: Load Raw Data ===")
    players_df = load_players_raw()
    game_lines_df = load_game_lines()
    td_odds_dict = load_td_odds()

    # Step 2: Calculate team offensive totals
    print("\n=== STEP 2: Calculate Team Offensive Totals ===")
    team_totals_df = calculate_team_offensive_totals(players_df)

    # Step 3: Enhanced game script analysis
    print("\n=== STEP 3: Enhanced Game Script Analysis ===")
    game_scripts_df = enhance_game_script_with_totals(game_lines_df, team_totals_df)
    print(f"  Calculated game scripts for {len(game_scripts_df)} games")

    # Save enhanced game scripts
    output_dir = Path('data/intermediate')
    output_dir.mkdir(parents=True, exist_ok=True)
    game_scripts_df.to_csv(output_dir / 'game_scripts_enhanced.csv', index=False)
    print(f"  Saved to data/intermediate/game_scripts_enhanced.csv")

    # Step 4: Calculate floor/ceiling for all game scripts + TD odds multipliers
    print("\n=== STEP 4: Calculate Floor/Ceiling for All Game Scripts & TD Odds ===")

    integrated_players = []

    for idx, player in players_df.iterrows():
        # Find player's game script
        player_team = normalize_team_abbr(player['team'])  # Normalize team abbreviation
        game_script = game_scripts_df[
            (game_scripts_df['favorite'] == player_team) |
            (game_scripts_df['underdog'] == player_team)
        ]

        if len(game_script) == 0:
            # Player's team not in game lines (shouldn't happen)
            print(f"  Warning: No game found for {player['name']} ({player_team})")
            integrated_players.append(player)
            continue

        game_script = game_script.iloc[0]

        # Step 4a: Calculate floor/ceiling for ALL game scripts
        script_floors_ceilings = calculate_floor_ceiling_for_all_game_scripts(player, game_script)

        # Step 4b: Calculate TD odds multipliers (not applied yet)
        td_floor_mult, td_ceiling_mult = calculate_td_odds_multipliers(player)

        # Step 4c: Pre-compute distribution parameters for all scenarios
        consensus = player.get('fpProjPts', 0)
        dist_params = calculate_distribution_params_for_all_scenarios(
            consensus,
            script_floors_ceilings,
            td_floor_mult,
            td_ceiling_mult
        )

        # Create integrated player row
        integrated_player = player.copy()

        # Add game script floors/ceilings
        for key, value in script_floors_ceilings.items():
            integrated_player[key] = value

        # Add TD odds multipliers
        integrated_player['td_odds_floor_mult'] = td_floor_mult
        integrated_player['td_odds_ceiling_mult'] = td_ceiling_mult

        # Add pre-computed distribution parameters
        for key, value in dist_params.items():
            integrated_player[key] = value

        # Add game script probabilities
        integrated_player['game_id'] = game_script['game_id']
        integrated_player['shootout_prob'] = game_script['shootout_prob']
        integrated_player['defensive_prob'] = game_script['defensive_prob']
        integrated_player['blowout_prob'] = game_script['blowout_prob']
        integrated_player['competitive_prob'] = game_script['competitive_prob']

        # Determine if player is on favorite or underdog (for simulation phase)
        integrated_player['is_favorite'] = (player_team == game_script['favorite'])

        integrated_players.append(integrated_player)

    # Create integrated DataFrame
    integrated_df = pd.DataFrame(integrated_players)

    # Step 5: Save output
    print("\n=== STEP 5: Save Integrated Data ===")
    output_dir = Path('data/intermediate')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'players_integrated.csv'
    integrated_df.to_csv(output_path, index=False)
    print(f"  Wrote {len(integrated_df)} players to {output_path}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    print(f"  Players processed: {len(integrated_df)}")
    print(f"  Games analyzed: {len(game_scripts_df)}")

    # Show sample floor/ceiling for different game scripts
    print("\n  Sample floor/ceiling by game script (first 3 high-projection players):")
    sample = integrated_df[integrated_df['fpProjPts'] > 10].head(3)
    for _, p in sample.iterrows():
        orig_floor = p.get('espnLowScore', p.get('fpProjPts', 0) * 0.5)
        orig_ceiling = p.get('espnHighScore', p.get('fpProjPts', 0) * 1.5)
        print(f"\n    {p['name']} ({p['position']}, {p['team']}) - Original: {orig_floor:.1f}/{orig_ceiling:.1f}")
        print(f"      Shootout:    {p['floor_shootout']:.1f}/{p['ceiling_shootout']:.1f}")
        print(f"      Defensive:   {p['floor_defensive']:.1f}/{p['ceiling_defensive']:.1f}")
        print(f"      Competitive: {p['floor_competitive']:.1f}/{p['ceiling_competitive']:.1f}")
        print(f"      TD Odds Mult: {p['td_odds_floor_mult']:.3f}x / {p['td_odds_ceiling_mult']:.3f}x")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
