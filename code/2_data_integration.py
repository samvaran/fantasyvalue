"""
Data Integration Pipeline - Step 2

Merges all data sources and calculates game script adjustments.

Inputs (from data/WEEK/inputs/):
- fanduel_salaries.csv
- fantasypros_projections.csv
- game_lines.csv
- td_odds.csv

Outputs (to data/WEEK/intermediate/):
- 1_players.csv - Merged player data with game script adjustments
- 2_game_scripts.csv - Game-level script probabilities

Usage:
    python 2_data_integration.py --week-dir data/2025_12_01
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Import distribution fitting function
from optimizer.utils.distribution_fit import fit_lognormal_to_percentiles


# ============================================================================
# CONFIGURATION
# ============================================================================

# Team name normalization (JAC → JAX, etc.)
TEAM_ABBR_MAP = {
    'JAC': 'JAX',  # Jacksonville
}

# Game Script Floor/Ceiling Adjustments (multipliers)
GAME_SCRIPT_FLOOR = {
    'shootout': {'QB': 0.75, 'RB': 0.90, 'WR': 0.75, 'TE': 0.75, 'D': 0.70},
    'defensive': {'QB': 0.75, 'RB': 0.95, 'WR': 0.75, 'TE': 0.75, 'D': 1.00},
    'blowout_favorite': {'QB': 0.80, 'RB': 0.95, 'WR': 0.80, 'TE': 0.80, 'D': 1.00},
    'blowout_underdog': {'QB': 0.80, 'RB': 0.75, 'WR': 0.85, 'TE': 0.80, 'D': 0.75},
    'competitive': {'QB': 0.85, 'RB': 0.85, 'WR': 0.85, 'TE': 0.85, 'D': 0.85}
}

GAME_SCRIPT_CEILING = {
    'shootout': {'QB': 1.40, 'RB': 1.05, 'WR': 1.45, 'TE': 1.40, 'D': 1.05},
    'defensive': {'QB': 1.15, 'RB': 1.15, 'WR': 1.10, 'TE': 1.15, 'D': 1.40},
    'blowout_favorite': {'QB': 1.15, 'RB': 1.30, 'WR': 1.10, 'TE': 1.15, 'D': 1.30},
    'blowout_underdog': {'QB': 1.35, 'RB': 1.05, 'WR': 1.35, 'TE': 1.30, 'D': 1.05},
    'competitive': {'QB': 1.25, 'RB': 1.20, 'WR': 1.25, 'TE': 1.25, 'D': 1.20}
}

# TD Odds impact on floor/ceiling
TD_ODDS_FLOOR_IMPACT = 0.05    # 5% boost per 10% TD probability
TD_ODDS_CEILING_IMPACT = 0.15  # 15% boost per 10% TD probability


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_team_abbr(team: str) -> str:
    """Normalize team abbreviation."""
    return TEAM_ABBR_MAP.get(team, team)


def normalize_name(name: str, is_defense: bool = False) -> str:
    """
    Normalize player/team name for matching.

    Args:
        name: Player or team name
        is_defense: If True, extract team name from full defense name
    """
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()

    # For defenses, extract just the team name (last word usually)
    # "Seattle Seahawks" -> "seahawks"
    # "Los Angeles Rams" -> "rams"
    # "San Francisco 49ers" -> "49ers"
    if is_defense:
        parts = name.split()
        # Return last word (team name)
        return parts[-1] if parts else ''

    # For players, remove punctuation
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    return ' '.join(name.split())


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fanduel_salaries(inputs_dir: Path) -> pd.DataFrame:
    """Load FanDuel salaries CSV."""
    csv_path = inputs_dir / 'fanduel_salaries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"FanDuel salaries not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} players from FanDuel")

    # Normalize names for matching (handle defenses specially)
    df['name_norm'] = df.apply(
        lambda row: normalize_name(row['Nickname'], is_defense=row['Position'] == 'D'),
        axis=1
    )

    return df


def load_fantasypros_projections(inputs_dir: Path) -> pd.DataFrame:
    """Load FantasyPros projections CSV."""
    csv_path = inputs_dir / 'fantasypros_projections.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"FantasyPros projections not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} players from FantasyPros")

    # Normalize names for matching (handle defenses specially)
    df['name_norm'] = df.apply(
        lambda row: normalize_name(row['name'], is_defense=row['position'] == 'D'),
        axis=1
    )

    return df


def load_game_lines(inputs_dir: Path) -> pd.DataFrame:
    """Load DraftKings game lines CSV."""
    csv_path = inputs_dir / 'game_lines.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Game lines not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df) // 2} games from DraftKings")

    return df


def load_td_odds(inputs_dir: Path) -> pd.DataFrame:
    """Load DraftKings TD odds CSV."""
    csv_path = inputs_dir / 'td_odds.csv'
    if not csv_path.exists():
        print(f"  ⚠️  TD odds not found (optional): {csv_path}")
        return pd.DataFrame(columns=['name', 'tdOdds', 'tdProbability'])

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} players from DraftKings TD odds")

    # Normalize names for matching
    df['name_norm'] = df['name'].apply(normalize_name)

    return df


# ============================================================================
# DATA MERGING
# ============================================================================

def merge_player_data(
    fd_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    td_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all player data sources.

    Primary key: FanDuel players (these are the ones we can actually roster)
    Join with FantasyPros and TD odds using normalized names.
    """
    print("\n📊 Merging player data...")

    merged_players = []

    for _, fd_player in fd_df.iterrows():
        name_norm = fd_player['name_norm']

        # Find matching FantasyPros projection
        fp_match = fp_df[fp_df['name_norm'] == name_norm]

        if len(fp_match) == 0:
            # No FantasyPros projection found - skip this player
            continue

        fp_proj = fp_match.iloc[0]

        # Find matching TD odds (optional)
        td_match = td_df[td_df['name_norm'] == name_norm]
        td_odds = td_match.iloc[0] if len(td_match) > 0 else {}

        # Create merged player
        merged = {
            # IDs and names
            'id': fd_player.get('Id', fd_player.get('id', '')),
            'name': fd_player['Nickname'],
            'position': fd_player['Position'],

            # Team and game info
            'team': fd_player.get('Team', fd_player.get('team', '')),
            'opponent': fd_player.get('Opponent', fd_player.get('opponent', '')),
            'game': fd_player.get('Game', fd_player.get('game', '')),

            # Salary
            'salary': fd_player.get('Salary', fd_player.get('salary', 0)),

            # FanDuel data
            'fppg': fd_player.get('FPPG', fd_player.get('fppg', 0)),
            'injury_status': fd_player.get('Injury Indicator', fd_player.get('injury_status', '')),

            # FantasyPros projections
            'fpProjPts': fp_proj.get('fpProjPts', 0),

            # TD odds
            'tdOdds': td_odds.get('tdOdds', ''),
            'tdProbability': td_odds.get('tdProbability', 0),
        }

        merged_players.append(merged)

    merged_df = pd.DataFrame(merged_players)
    print(f"  ✓ Merged {len(merged_df)} players")

    # Filter out IR players
    if 'injury_status' in merged_df.columns:
        before = len(merged_df)
        merged_df = merged_df[merged_df['injury_status'] != 'IR']
        removed = before - len(merged_df)
        if removed > 0:
            print(f"  ✓ Removed {removed} IR players")

    return merged_df


# ============================================================================
# GAME SCRIPT ANALYSIS
# ============================================================================

def calculate_game_scripts(game_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate game script probabilities for each game.

    Returns DataFrame with columns:
        game_id, favorite, underdog, shootout_prob, defensive_prob,
        blowout_prob, competitive_prob, primary_script
    """
    from game_script import (
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

        # Determine favorite (negative spread = favorite)
        if game1['spread'] < 0:
            fav = game1
            dog = game2
        else:
            fav = game2
            dog = game1

        # Calculate script scores
        shootout_score = calculate_shootout_score(
            fav['total'], fav['spread'], fav.get('total_over_odds', -110)
        )
        defensive_score = calculate_defensive_score(
            fav['total'], fav.get('total_under_odds', -110)
        )
        blowout_score, _ = calculate_blowout_score(
            fav['spread'], fav.get('moneyline', -200),
            fav.get('spread_odds', -110), dog.get('spread_odds', -110)
        )
        competitive_score = calculate_competitive_score(fav['spread'])

        # Normalize to probabilities
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
            'spread': fav['spread'],
            'total': fav['total'],
            'shootout_prob': shootout_prob,
            'defensive_prob': defensive_prob,
            'blowout_prob': blowout_prob,
            'competitive_prob': competitive_prob,
            'primary_script': primary_script,
            'script_strength': max(scores.values())
        })

    df = pd.DataFrame(results)
    print(f"\n📈 Calculated game scripts for {len(df)} games")

    return df


# ============================================================================
# PLAYER ADJUSTMENTS
# ============================================================================

def calculate_floor_ceiling_for_all_scripts(
    position: str,
    consensus: float
) -> Dict[str, float]:
    """
    Calculate floor/ceiling for all game scripts.

    Args:
        position: Player position (QB, RB, WR, TE, D)
        consensus: Player's consensus projection (mean)

    Returns:
        Dict with floor_X and ceiling_X for each script
    """
    # Base floor/ceiling (if no ESPN data)
    base_floor = consensus * 0.5
    base_ceiling = consensus * 1.5

    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
    results = {}

    for script in scripts:
        floor_mult = GAME_SCRIPT_FLOOR[script][position]
        ceiling_mult = GAME_SCRIPT_CEILING[script][position]

        adjusted_floor = base_floor * floor_mult
        adjusted_ceiling = base_ceiling * ceiling_mult

        # Constraints: floor < consensus < ceiling
        adjusted_floor = min(adjusted_floor, consensus * 0.95)
        adjusted_ceiling = max(adjusted_ceiling, consensus * 1.05)

        results[f'floor_{script}'] = adjusted_floor
        results[f'ceiling_{script}'] = adjusted_ceiling

    return results


def calculate_td_odds_multipliers(td_probability: float) -> Tuple[float, float]:
    """Calculate TD odds floor/ceiling multipliers."""
    if pd.isna(td_probability):
        td_probability = 0

    td_prob_scaled = td_probability / 100.0

    floor_mult = 1.0 + (td_prob_scaled * TD_ODDS_FLOOR_IMPACT)
    ceiling_mult = 1.0 + (td_prob_scaled * TD_ODDS_CEILING_IMPACT)

    return floor_mult, ceiling_mult


def calculate_distribution_params(
    script_floors_ceilings: Dict[str, float],
    td_floor_mult: float,
    td_ceiling_mult: float
) -> Dict[str, float]:
    """
    Pre-compute distribution parameters (mu, sigma, shift) for all scenarios.

    This avoids recomputing during Monte Carlo simulation.
    """
    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
    results = {}

    for script in scripts:
        floor = script_floors_ceilings[f'floor_{script}']
        ceiling = script_floors_ceilings[f'ceiling_{script}']

        # Apply TD odds multipliers
        floor_adjusted = floor * td_floor_mult
        ceiling_adjusted = ceiling * td_ceiling_mult

        # Fit log-normal to P10/P90 (let mean vary)
        try:
            mu, sigma, shift = fit_lognormal_to_percentiles(floor_adjusted, ceiling_adjusted)
            results[f'mu_{script}'] = mu
            results[f'sigma_{script}'] = sigma
            results[f'shift_{script}'] = shift
        except (ValueError, Exception):
            # Fallback
            mean_est = (floor_adjusted + ceiling_adjusted) / 2
            std = (ceiling_adjusted - floor_adjusted) / 5
            results[f'mu_{script}'] = np.log(mean_est) if mean_est > 0 else 0
            results[f'sigma_{script}'] = std / mean_est if mean_est > 0 else 0.2
            results[f'shift_{script}'] = 0.0

    return results


def integrate_game_scripts(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add game script adjustments to each player.

    Returns DataFrame with all script floors/ceilings and distribution params.
    """
    print("\n⚙️  Integrating game scripts...")

    integrated_players = []

    for _, player in players_df.iterrows():
        player_team = normalize_team_abbr(player['team'])

        # Find player's game
        game_script = game_scripts_df[
            (game_scripts_df['favorite'] == player_team) |
            (game_scripts_df['underdog'] == player_team)
        ]

        if len(game_script) == 0:
            print(f"  ⚠️  No game found for {player['name']} ({player_team})")
            integrated_players.append(player.to_dict())
            continue

        game_script = game_script.iloc[0]

        # Calculate floor/ceiling for all scripts
        script_floors_ceilings = calculate_floor_ceiling_for_all_scripts(
            player['position'],
            player['fpProjPts']
        )

        # Calculate TD odds multipliers
        td_floor_mult, td_ceiling_mult = calculate_td_odds_multipliers(
            player.get('tdProbability', 0)
        )

        # Pre-compute distribution parameters
        dist_params = calculate_distribution_params(
            script_floors_ceilings,
            td_floor_mult,
            td_ceiling_mult
        )

        # Create integrated player
        integrated = player.to_dict()
        integrated.update(script_floors_ceilings)
        integrated.update(dist_params)
        integrated['td_odds_floor_mult'] = td_floor_mult
        integrated['td_odds_ceiling_mult'] = td_ceiling_mult
        integrated['game_id'] = game_script['game_id']
        integrated['shootout_prob'] = game_script['shootout_prob']
        integrated['defensive_prob'] = game_script['defensive_prob']
        integrated['blowout_prob'] = game_script['blowout_prob']
        integrated['competitive_prob'] = game_script['competitive_prob']
        integrated['is_favorite'] = (player_team == game_script['favorite'])

        integrated_players.append(integrated)

    integrated_df = pd.DataFrame(integrated_players)
    print(f"  ✓ Integrated {len(integrated_df)} players")

    return integrated_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Integrate fantasy football data sources')
    parser.add_argument('--week-dir', required=True, help='Week directory (e.g., data/2025_12_01)')
    args = parser.parse_args()

    week_dir = Path(args.week_dir)
    inputs_dir = week_dir / 'inputs'
    intermediate_dir = week_dir / 'intermediate'
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("DATA INTEGRATION PIPELINE")
    print(f"{'='*80}")
    print(f"\nWeek: {week_dir.name}")
    print(f"Inputs: {inputs_dir}")
    print(f"Outputs: {intermediate_dir}")

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================

    print(f"\n{'='*80}")
    print("STEP 1: LOAD DATA")
    print(f"{'='*80}")

    fd_df = load_fanduel_salaries(inputs_dir)
    fp_df = load_fantasypros_projections(inputs_dir)
    game_lines_df = load_game_lines(inputs_dir)
    td_df = load_td_odds(inputs_dir)

    # ========================================================================
    # STEP 2: MERGE PLAYER DATA
    # ========================================================================

    print(f"\n{'='*80}")
    print("STEP 2: MERGE PLAYER DATA")
    print(f"{'='*80}")

    players_df = merge_player_data(fd_df, fp_df, td_df)

    # ========================================================================
    # STEP 3: CALCULATE GAME SCRIPTS
    # ========================================================================

    print(f"\n{'='*80}")
    print("STEP 3: CALCULATE GAME SCRIPTS")
    print(f"{'='*80}")

    game_scripts_df = calculate_game_scripts(game_lines_df)

    # Save game scripts
    game_scripts_file = intermediate_dir / '2_game_scripts.csv'
    game_scripts_df.to_csv(game_scripts_file, index=False)
    print(f"  ✓ Saved to {game_scripts_file.relative_to(week_dir)}")

    # ========================================================================
    # STEP 4: INTEGRATE GAME SCRIPTS WITH PLAYERS
    # ========================================================================

    print(f"\n{'='*80}")
    print("STEP 4: INTEGRATE GAME SCRIPTS")
    print(f"{'='*80}")

    integrated_df = integrate_game_scripts(players_df, game_scripts_df)

    # Save integrated players
    players_file = intermediate_dir / '1_players.csv'
    integrated_df.to_csv(players_file, index=False)
    print(f"  ✓ Saved to {players_file.relative_to(week_dir)}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n{'='*80}")
    print("✅ INTEGRATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"📊 Summary:")
    print(f"  Players integrated: {len(integrated_df)}")
    print(f"  Games analyzed: {len(game_scripts_df)}")
    print(f"\n📁 Output files:")
    print(f"  {players_file.relative_to(week_dir)} - {len(integrated_df)} players")
    print(f"  {game_scripts_file.relative_to(week_dir)} - {len(game_scripts_df)} games")

    # Show sample
    print(f"\n📈 Sample game scripts:")
    for _, game in game_scripts_df.head(3).iterrows():
        print(f"  {game['game_id']}: {game['primary_script']} ({game['script_strength']:.1%})")

    print(f"\n💡 Next step:")
    print(f"   python 3_run_optimizer.py --week-dir {week_dir}")


if __name__ == '__main__':
    main()
