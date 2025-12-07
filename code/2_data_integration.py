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

# Default fallback floor/ceiling models (learned from historical ESPN data)
# These are overwritten each week with fresh models from that week's ESPN data
# Format: floor/ceiling = intercept + slope * projection
DEFAULT_FLOOR_CEILING_MODELS = {
    'floor': {'intercept': -1.674, 'slope': 0.666, 'r2': 0.862},
    'ceiling': {'intercept': 4.372, 'slope': 1.097, 'r2': 0.882},
}

# Will be populated by build_floor_ceiling_models() each week
FLOOR_CEILING_MODELS = None


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
    name = ' '.join(name.split())

    # Remove common suffixes (Jr, Sr, II, III, IV, V)
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


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


def load_espn_projections(inputs_dir: Path) -> pd.DataFrame:
    """Load ESPN Watson projections CSV (floor/ceiling/boom-bust)."""
    csv_path = inputs_dir / 'espn_projections.csv'
    if not csv_path.exists():
        print(f"  ⚠️  ESPN projections not found (optional): {csv_path}")
        print(f"     Run: python 1_fetch_data.py --espn")
        return pd.DataFrame(columns=['name', 'espnLowScore', 'espnHighScore'])

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} players from ESPN Watson projections")

    # Normalize names for matching
    df['name_norm'] = df['name'].apply(normalize_name)

    return df


# ============================================================================
# DATA MERGING
# ============================================================================

def linear_regression(x_values: list, y_values: list) -> Dict[str, float]:
    """
    Simple linear regression: y = intercept + slope * x

    Returns dict with intercept, slope, r2
    """
    x = np.array(x_values)
    y = np.array(y_values)
    n = len(x)

    if n < 3:
        return {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0}

    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()

    mean_x = sum_x / n
    mean_y = sum_y / n

    # Calculate slope
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return {'intercept': mean_y, 'slope': 0.0, 'r2': 0.0}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = mean_y - slope * mean_x

    # Calculate R²
    ss_total = ((y - mean_y) ** 2).sum()
    predicted = intercept + slope * x
    ss_residual = ((y - predicted) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    return {'intercept': intercept, 'slope': slope, 'r2': r2}


def build_regression_models(merged_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Build linear regression models to convert ESPN projections to FantasyPros scale.

    Uses players with both FP and ESPN data as training set.
    """
    print("\n📈 Building regression models (ESPN → FP scale)...")

    # Filter to players with both FP and ESPN data
    training_data = merged_df[
        (merged_df['fpProjPts'] > 0) &
        (merged_df['espnOutsideProjection'].notna()) &
        (merged_df['espnOutsideProjection'].astype(float) > 0)
    ].copy()

    if len(training_data) < 10:
        print(f"  ⚠️  Only {len(training_data)} players with both FP and ESPN - using identity transform")
        return {
            'espnOutside': {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0},
            'espnScore': {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0},
            'espnSim': {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0},
        }

    print(f"  Training data: {len(training_data)} players with both FP and ESPN projections")

    models = {}

    # Model 1: espnOutsideProjection → fpProjPts
    # This model is also used for espnLowScore and espnHighScore (same Watson source)
    models['espnOutside'] = linear_regression(
        training_data['espnOutsideProjection'].astype(float).tolist(),
        training_data['fpProjPts'].astype(float).tolist()
    )
    print(f"  Model: espnOutside → fpProjPts")
    print(f"    fpProjPts = {models['espnOutside']['intercept']:.3f} + {models['espnOutside']['slope']:.3f} * espnOutside")
    print(f"    R² = {models['espnOutside']['r2']:.3f}")

    # Model 2: espnScoreProjection → fpProjPts
    score_data = training_data[training_data['espnScoreProjection'].notna()]
    if len(score_data) >= 10:
        models['espnScore'] = linear_regression(
            score_data['espnScoreProjection'].astype(float).tolist(),
            score_data['fpProjPts'].astype(float).tolist()
        )
        print(f"  Model: espnScore → fpProjPts")
        print(f"    fpProjPts = {models['espnScore']['intercept']:.3f} + {models['espnScore']['slope']:.3f} * espnScore")
        print(f"    R² = {models['espnScore']['r2']:.3f}")
    else:
        models['espnScore'] = {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0}

    # Model 3: espnSimulationProjection → fpProjPts
    sim_data = training_data[training_data['espnSimulationProjection'].notna()]
    if len(sim_data) >= 10:
        models['espnSim'] = linear_regression(
            sim_data['espnSimulationProjection'].astype(float).tolist(),
            sim_data['fpProjPts'].astype(float).tolist()
        )
        print(f"  Model: espnSim → fpProjPts")
        print(f"    fpProjPts = {models['espnSim']['intercept']:.3f} + {models['espnSim']['slope']:.3f} * espnSim")
        print(f"    R² = {models['espnSim']['r2']:.3f}")
    else:
        models['espnSim'] = {'intercept': 0.0, 'slope': 1.0, 'r2': 0.0}

    return models


def build_floor_ceiling_models(merged_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Build regression models to predict floor/ceiling from projection.

    Uses this week's ESPN data: espnScoreProjection → espnLowScore/espnHighScore
    These models are used as fallback for players without ESPN floor/ceiling.
    """
    global FLOOR_CEILING_MODELS

    print("\n📉 Building floor/ceiling fallback models...")

    # Filter to players with ESPN projection AND floor/ceiling
    training_data = merged_df[
        (merged_df['espnScoreProjection'].notna()) &
        (merged_df['espnScoreProjection'].astype(float) > 0) &
        (merged_df['espnLowScore'].notna()) &
        (merged_df['espnHighScore'].notna())
    ].copy()

    if len(training_data) < 20:
        print(f"  ⚠️  Only {len(training_data)} players with ESPN data - using defaults")
        FLOOR_CEILING_MODELS = DEFAULT_FLOOR_CEILING_MODELS.copy()
        return FLOOR_CEILING_MODELS

    print(f"  Training data: {len(training_data)} players with ESPN floor/ceiling")

    # Model: espnScoreProjection → espnLowScore (floor)
    floor_model = linear_regression(
        training_data['espnScoreProjection'].astype(float).tolist(),
        training_data['espnLowScore'].astype(float).tolist()
    )
    print(f"  Model: projection → floor")
    print(f"    floor = {floor_model['intercept']:.3f} + {floor_model['slope']:.3f} * projection")
    print(f"    R² = {floor_model['r2']:.3f}")

    # Model: espnScoreProjection → espnHighScore (ceiling)
    ceiling_model = linear_regression(
        training_data['espnScoreProjection'].astype(float).tolist(),
        training_data['espnHighScore'].astype(float).tolist()
    )
    print(f"  Model: projection → ceiling")
    print(f"    ceiling = {ceiling_model['intercept']:.3f} + {ceiling_model['slope']:.3f} * projection")
    print(f"    R² = {ceiling_model['r2']:.3f}")

    FLOOR_CEILING_MODELS = {
        'floor': floor_model,
        'ceiling': ceiling_model,
    }

    return FLOOR_CEILING_MODELS


def apply_regression_models(merged_df: pd.DataFrame, models: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Apply regression models to convert all ESPN projections to FP scale.

    Also uses espnOutside model for espnLowScore/espnHighScore (same Watson source).
    """
    print("\n🔄 Converting ESPN projections to FP scale...")

    df = merged_df.copy()

    def transform(value, model):
        if pd.isna(value) or value is None:
            return None
        try:
            v = float(value)
            if v <= 0:
                return None
            return model['intercept'] + model['slope'] * v
        except (ValueError, TypeError):
            return None

    # Convert each ESPN projection type
    df['espnOutsideProjection_fp'] = df['espnOutsideProjection'].apply(
        lambda x: transform(x, models['espnOutside'])
    )
    df['espnScoreProjection_fp'] = df['espnScoreProjection'].apply(
        lambda x: transform(x, models['espnScore'])
    )
    df['espnSimulationProjection_fp'] = df['espnSimulationProjection'].apply(
        lambda x: transform(x, models['espnSim'])
    )

    # Use espnOutside model for Low/High (same Watson source)
    df['espnLowScore_fp'] = df['espnLowScore'].apply(
        lambda x: transform(x, models['espnOutside'])
    )
    df['espnHighScore_fp'] = df['espnHighScore'].apply(
        lambda x: transform(x, models['espnOutside'])
    )

    converted_count = df['espnOutsideProjection_fp'].notna().sum()
    print(f"  ✓ Converted {converted_count} ESPN projections to FP scale")

    return df


def calculate_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted consensus from FP-scaled projections.
    """
    print("\n📊 Calculating weighted consensus...")

    def compute_consensus(row):
        projections = []
        weights = []

        # FantasyPros (primary - already on FP scale)
        fp_pts = row.get('fpProjPts', 0)
        if fp_pts and float(fp_pts) > 0:
            projections.append(float(fp_pts))
            weights.append(2.0)  # Weight FP higher (aggregates multiple sources)

        # ESPN projections (now converted to FP scale)
        espn_score = row.get('espnScoreProjection_fp')
        if espn_score and not pd.isna(espn_score) and float(espn_score) > 0:
            projections.append(float(espn_score))
            weights.append(1.0)

        espn_sim = row.get('espnSimulationProjection_fp')
        if espn_sim and not pd.isna(espn_sim) and float(espn_sim) > 0:
            projections.append(float(espn_sim))
            weights.append(1.0)

        espn_outside = row.get('espnOutsideProjection_fp')
        if espn_outside and not pd.isna(espn_outside) and float(espn_outside) > 0:
            projections.append(float(espn_outside))
            weights.append(0.5)  # Lower weight - boom indicator

        if projections:
            return sum(p * w for p, w in zip(projections, weights)) / sum(weights)
        return fp_pts if fp_pts else 0

    df['consensus'] = df.apply(compute_consensus, axis=1)

    with_espn = (df['espnOutsideProjection_fp'].notna()).sum()
    print(f"  ✓ {with_espn} players with multi-source consensus")

    return df


def merge_player_data(
    fd_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    td_df: pd.DataFrame,
    espn_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Merge all player data sources.

    Primary key: FanDuel players (these are the ones we can actually roster)
    Join with FantasyPros, TD odds, and ESPN projections using normalized names.
    """
    print("\n📊 Merging player data...")

    if espn_df is None:
        espn_df = pd.DataFrame(columns=['name', 'name_norm', 'espnLowScore', 'espnHighScore'])

    merged_players = []
    espn_matches = 0

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

        # Find matching ESPN projection (optional but important!)
        espn_match = espn_df[espn_df['name_norm'] == name_norm] if 'name_norm' in espn_df.columns else pd.DataFrame()
        espn_proj = espn_match.iloc[0] if len(espn_match) > 0 else {}
        if len(espn_match) > 0:
            espn_matches += 1

        # Get individual projection values (raw - will be converted later)
        fp_pts = fp_proj.get('fpProjPts', 0) or 0

        # Create merged player with raw values
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

            # FantasyPros projection (already on FP scale)
            'fpProjPts': fp_pts,

            # Raw ESPN projections (will be converted to FP scale via regression)
            'espnScoreProjection': espn_proj.get('espnScoreProjection', None),
            'espnSimulationProjection': espn_proj.get('espnSimulationProjection', None),
            'espnOutsideProjection': espn_proj.get('espnOutsideProjection', None),
            'espnLowScore': espn_proj.get('espnLowScore', None),
            'espnHighScore': espn_proj.get('espnHighScore', None),

            # TD odds
            'tdOdds': td_odds.get('tdOdds', ''),
            'tdProbability': td_odds.get('tdProbability', 0),
        }

        merged_players.append(merged)

    merged_df = pd.DataFrame(merged_players)
    print(f"  ✓ Merged {len(merged_df)} players")
    print(f"  ✓ ESPN data for {espn_matches} players")

    # Build regression models and convert ESPN projections to FP scale
    models = build_regression_models(merged_df)
    merged_df = apply_regression_models(merged_df, models)

    # Build floor/ceiling fallback models from this week's ESPN data
    build_floor_ceiling_models(merged_df)

    # Calculate weighted consensus using FP-scaled projections
    merged_df = calculate_consensus(merged_df)

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
    consensus: float,
    espn_low: float = None,
    espn_high: float = None
) -> Dict[str, float]:
    """
    Calculate floor/ceiling for all game scripts.

    Uses ESPN Watson floor/ceiling when available (player-specific),
    otherwise falls back to regression-based estimates learned from ESPN data.

    Args:
        position: Player position (QB, RB, WR, TE, D)
        consensus: Player's consensus projection (mean)
        espn_low: ESPN Watson floor (P10 estimate)
        espn_high: ESPN Watson ceiling (P90 estimate)

    Returns:
        Dict with floor_X and ceiling_X for each script
    """
    # Use ESPN floor/ceiling if available (much better - player-specific!)
    if espn_low is not None and espn_high is not None and not pd.isna(espn_low) and not pd.isna(espn_high) and espn_low > 0 and espn_high > 0:
        base_floor = float(espn_low)
        base_ceiling = float(espn_high)
    else:
        # Fallback: use regression models learned from this week's ESPN data
        # Models are built in build_floor_ceiling_models() from players WITH ESPN data
        models = FLOOR_CEILING_MODELS if FLOOR_CEILING_MODELS else DEFAULT_FLOOR_CEILING_MODELS
        floor_model = models['floor']
        ceiling_model = models['ceiling']

        base_floor = max(0, floor_model['intercept'] + floor_model['slope'] * consensus)
        base_ceiling = ceiling_model['intercept'] + ceiling_model['slope'] * consensus

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
        # Use weighted consensus as center, FP-scaled ESPN floor/ceiling for shape
        script_floors_ceilings = calculate_floor_ceiling_for_all_scripts(
            player['position'],
            player.get('consensus', player['fpProjPts']),  # Use consensus if available
            espn_low=player.get('espnLowScore_fp'),  # Use FP-scaled floor
            espn_high=player.get('espnHighScore_fp')  # Use FP-scaled ceiling
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
    espn_df = load_espn_projections(inputs_dir)

    # ========================================================================
    # STEP 2: MERGE PLAYER DATA
    # ========================================================================

    print(f"\n{'='*80}")
    print("STEP 2: MERGE PLAYER DATA")
    print(f"{'='*80}")

    players_df = merge_player_data(fd_df, fp_df, td_df, espn_df)

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
