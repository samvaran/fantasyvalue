"""
Backtest optimizer performance against actual FanDuel results.

This script:
1. Parses actual fantasy points from FanDuel results CSV/JSON
2. Scores generated lineups using actual results
3. Compares actual vs predicted game scripts
4. Analyzes projection accuracy

Usage:
    # New architecture (week-dir + run-dir)
    python 4_backtest.py --week-dir data/2025_12_01 --run-dir run_20251201_143022

    # Manual paths (backward compatible)
    python 4_backtest.py --results data/2025_12_01/inputs/fanduel_results.csv --lineups data/2025_12_01/outputs/run_20251201_143022/3_lineups.csv

Outputs (to run directory):
    - 5_scored_lineups.csv - Lineups with actual scores
    - 6_backtest_games.csv - Game-level accuracy analysis
    - 7_backtest_summary.json - Validation metrics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_player_name(name: str) -> str:
    """Normalize player name for matching."""
    # Convert to lowercase, remove punctuation, handle common variations
    name = name.lower().strip()
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    name = ' '.join(name.split())  # Normalize whitespace

    # Handle defense/team name variations
    # FanDuel uses full city names (e.g., "los angeles chargers")
    # Our data uses just team names (e.g., "chargers")
    # Strip city prefixes to normalize to team name only
    team_mappings = {
        'los angeles chargers': 'chargers',
        'los angeles rams': 'rams',
        'san francisco 49ers': '49ers',
        'new york giants': 'giants',
        'new york jets': 'jets',
        'arizona cardinals': 'cardinals',
        'baltimore ravens': 'ravens',
        'buffalo bills': 'bills',
        'carolina panthers': 'panthers',
        'chicago bears': 'bears',
        'cincinnati bengals': 'bengals',
        'cleveland browns': 'browns',
        'dallas cowboys': 'cowboys',
        'denver broncos': 'broncos',
        'detroit lions': 'lions',
        'green bay packers': 'packers',
        'houston texans': 'texans',
        'indianapolis colts': 'colts',
        'jacksonville jaguars': 'jaguars',
        'kansas city chiefs': 'chiefs',
        'las vegas raiders': 'raiders',
        'miami dolphins': 'dolphins',
        'minnesota vikings': 'vikings',
        'new england patriots': 'patriots',
        'new orleans saints': 'saints',
        'philadelphia eagles': 'eagles',
        'pittsburgh steelers': 'steelers',
        'seattle seahawks': 'seahawks',
        'tampa bay buccaneers': 'buccaneers',
        'tennessee titans': 'titans',
        'washington commanders': 'commanders',
        'atlanta falcons': 'falcons',
    }

    if name in team_mappings:
        name = team_mappings[name]

    return name


# ============================================================================
# PARSE FANDUEL RESULTS
# ============================================================================

def convert_fanduel_json_to_csv(json_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Convert FanDuel results JSON to simplified CSV files.

    Creates:
        - 4_actual_players.csv: name, position, team, salary, actual_points
        - 4_actual_games.csv: game_id, away_team, home_team, away_score, home_score, total_points

    Returns:
        (players_csv_path, games_csv_path)
    """
    print(f"Converting FanDuel JSON to CSV...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    players = []
    games = {}

    if 'data' in data and 'leaders' in data['data']:
        for leader in data['data']['leaders']:
            # Extract player info
            full_name = f"{leader['player']['firstNames']} {leader['player']['lastName']}"
            team = leader['player']['associations'][0]['team']['code'] if leader['player']['associations'] else None

            players.append({
                'name': full_name,
                'position': leader['position'],
                'team': team,
                'salary': leader['salary'],
                'actual_points': leader['stats']['fantasyPoints']
            })

            # Extract game info (deduplicated by game_id)
            if 'fixture' in leader and 'boxscore' in leader:
                fixture = leader['fixture']
                game_id = fixture['id']

                if game_id not in games:
                    boxscore = leader['boxscore']['stats']['stats']
                    away_score = None
                    home_score = None

                    for stat in boxscore:
                        if stat['type'] == 'AWAY_TEAM_TOTAL_SCORE':
                            away_score = int(stat['value'])
                        elif stat['type'] == 'HOME_TEAM_TOTAL_SCORE':
                            home_score = int(stat['value'])

                    if away_score is not None and home_score is not None:
                        games[game_id] = {
                            'game_id': game_id,
                            'away_team': fixture['awayTeam']['code'],
                            'home_team': fixture['homeTeam']['code'],
                            'away_score': away_score,
                            'home_score': home_score,
                            'total_points': away_score + home_score
                        }

    # Save CSVs
    players_df = pd.DataFrame(players)
    games_df = pd.DataFrame(list(games.values()))

    players_csv = output_dir / '4_actual_players.csv'
    games_csv = output_dir / '4_actual_games.csv'

    players_df.to_csv(players_csv, index=False)
    games_df.to_csv(games_csv, index=False)

    print(f"  Created {players_csv} ({len(players_df)} players)")
    print(f"  Created {games_csv} ({len(games_df)} games)")

    return players_csv, games_csv


def load_actual_results(week_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load actual results from CSV files (preferred) or JSON (fallback).

    Supported formats (checked in order):
    1. intermediate/4_actual_players.csv + 4_actual_games.csv (preferred)
    2. inputs/fanduel_results.csv (simple manual format: name,actual_points)
    3. inputs/fanduel_results.json (FanDuel API response)

    Returns:
        (players_df, games_df): DataFrames with actual fantasy points and game scores
    """
    intermediate_dir = week_path / 'intermediate'
    inputs_dir = week_path / 'inputs'

    players_csv = intermediate_dir / '4_actual_players.csv'
    games_csv = intermediate_dir / '4_actual_games.csv'
    simple_csv = inputs_dir / 'fanduel_results.csv'  # Simple manual format
    json_path = inputs_dir / 'fanduel_results.json'

    # Option 1: Check if processed CSVs exist
    if players_csv.exists() and games_csv.exists():
        print(f"Loading actual results from CSV files...")
        players_df = pd.read_csv(players_csv)
        games_df = pd.read_csv(games_csv)
        print(f"  Loaded {len(players_df)} players, {len(games_df)} games")

    # Option 2: Simple manual CSV (name,actual_points)
    elif simple_csv.exists():
        print(f"Loading actual results from simple CSV: {simple_csv}")
        players_df = pd.read_csv(simple_csv)

        # Ensure required columns exist
        if 'name' not in players_df.columns:
            raise ValueError(f"Simple CSV must have 'name' column. Found: {list(players_df.columns)}")
        if 'actual_points' not in players_df.columns:
            # Check for common alternatives
            for alt in ['points', 'fpts', 'fp', 'actual', 'score']:
                if alt in players_df.columns:
                    players_df['actual_points'] = players_df[alt]
                    break
            else:
                raise ValueError(f"Simple CSV must have 'actual_points' column (or 'points'/'fpts'). Found: {list(players_df.columns)}")

        print(f"  Loaded {len(players_df)} players from simple CSV")

        # Create empty games_df since we don't have game-level data
        games_df = pd.DataFrame(columns=['game_id', 'away_team', 'home_team', 'away_score', 'home_score', 'total_points', 'point_differential'])
        print(f"  Note: No game-level data available from simple CSV format")

    # Option 3: FanDuel JSON (convert to CSV)
    elif json_path.exists():
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        players_csv, games_csv = convert_fanduel_json_to_csv(json_path, intermediate_dir)
        players_df = pd.read_csv(players_csv)
        games_df = pd.read_csv(games_csv)

    else:
        raise FileNotFoundError(
            f"No actual results found. Expected one of:\n"
            f"  - {players_csv} and {games_csv} (processed format)\n"
            f"  - {simple_csv} (simple format: name,actual_points)\n"
            f"  - {json_path} (FanDuel API response)"
        )

    # Normalize player names for matching
    players_df['name'] = players_df['name'].apply(normalize_player_name)

    # Add derived columns to games
    games_df['point_differential'] = abs(games_df['away_score'] - games_df['home_score'])

    return players_df, games_df


# ============================================================================
# CLASSIFY ACTUAL GAME SCRIPTS
# ============================================================================

def classify_actual_game_script(total_points: float, point_diff: float) -> str:
    """
    Classify actual game script based on final score.

    Uses same logic as game_script.py for consistency.
    """
    # Shootout: High scoring (total > 50)
    if total_points > 50:
        return 'shootout'

    # Defensive: Low scoring (total < 40)
    elif total_points < 40:
        return 'defensive'

    # Blowout: Large margin (diff > 14)
    elif point_diff > 14:
        # Could be either favorite or underdog, but we'll call it blowout_favorite
        # (We'd need pregame spreads to determine which team was favored)
        return 'blowout_favorite'

    # Competitive: Everything else
    else:
        return 'competitive'


def analyze_game_scripts(games_df: pd.DataFrame, game_scripts_path: str = None) -> pd.DataFrame:
    """
    Analyze actual vs predicted game scripts.

    Args:
        games_df: DataFrame with actual game results
        game_scripts_path: Optional path to predicted game scripts CSV

    Returns:
        DataFrame with actual game script classifications and predictions (if available)
    """
    print("\nAnalyzing game scripts...")

    # Classify actual game scripts
    games_df['actual_script'] = games_df.apply(
        lambda row: classify_actual_game_script(row['total_points'], row['point_differential']),
        axis=1
    )

    # Count actual game scripts
    print("\nActual game script distribution:")
    script_counts = games_df['actual_script'].value_counts()
    for script, count in script_counts.items():
        pct = count / len(games_df) * 100
        print(f"  {script:20s}: {count:2d} ({pct:5.1f}%)")

    # Load predicted game scripts if available
    if game_scripts_path and Path(game_scripts_path).exists():
        print(f"\nLoading predicted game scripts from: {game_scripts_path}")
        predicted_df = pd.read_csv(game_scripts_path)

        # This would require matching games by team names
        # For now, just show what we found
        print(f"  Found {len(predicted_df)} game predictions")

        # TODO: Match predicted to actual and calculate accuracy
        # This requires standardizing team names between sources

    return games_df


# ============================================================================
# SCORE LINEUPS
# ============================================================================

def build_player_lookup(players_df: pd.DataFrame, points_column: str) -> Dict[str, float]:
    """
    Build a dictionary mapping normalized player names to points.

    Args:
        players_df: DataFrame with player data
        points_column: Column name containing points ('actual_points' or 'fpProjPts')

    Returns:
        Dict mapping normalized name -> points
    """
    lookup = {}

    # Determine name column
    if 'name' in players_df.columns:
        name_col = 'name'
    elif 'playerName' in players_df.columns:
        name_col = 'playerName'
    else:
        return lookup

    for _, player in players_df.iterrows():
        name = str(player[name_col])
        norm_name = normalize_player_name(name)
        points = player.get(points_column)
        if points is not None and norm_name:
            lookup[norm_name] = float(points)

    return lookup


def score_lineup(
    lineup_row: pd.Series,
    actual_lookup: Dict[str, float],
    consensus_lookup: Dict[str, float] = None,
    verbose: bool = False
) -> Dict:
    """
    Score a single lineup using actual fantasy points.

    Calculates two totals:
    1. actual_only - Sum of actual points only (0 for missing players)
    2. actual_with_consensus - Sum with consensus fallback for missing players

    Args:
        lineup_row: Row from lineups CSV with position columns
        actual_lookup: Dict mapping normalized name -> actual points
        consensus_lookup: Optional dict for consensus fallback
        verbose: If True, print detailed scoring breakdown

    Returns:
        Dict with both score totals and breakdown
    """
    positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF']

    total_actual_only = 0.0
    total_with_consensus = 0.0
    players_found = 0
    players_consensus = 0
    players_missing = 0
    breakdown = []

    for pos in positions:
        if pos not in lineup_row or pd.isna(lineup_row[pos]):
            continue

        player_name = lineup_row[pos]
        norm_name = normalize_player_name(player_name)

        # O(1) lookup for actual points
        actual_points = actual_lookup.get(norm_name)

        if actual_points is not None:
            total_actual_only += actual_points
            total_with_consensus += actual_points
            players_found += 1
            breakdown.append({
                'position': pos,
                'player': player_name,
                'actual_points': actual_points,
                'consensus_points': actual_points,
                'source': 'actual'
            })
        else:
            # O(1) lookup for consensus fallback
            consensus_points = 0.0
            if consensus_lookup is not None:
                matched_consensus = consensus_lookup.get(norm_name)
                if matched_consensus is not None:
                    consensus_points = matched_consensus
                    players_consensus += 1
                else:
                    players_missing += 1
            else:
                players_missing += 1

            total_with_consensus += consensus_points

            breakdown.append({
                'position': pos,
                'player': player_name,
                'actual_points': 0.0,
                'consensus_points': consensus_points,
                'source': 'consensus' if consensus_points > 0 else 'missing'
            })

    if verbose:
        print(f"\nLineup scoring breakdown:")
        for item in breakdown:
            source_tag = {'actual': '✓', 'consensus': '~', 'missing': '✗'}[item['source']]
            print(f"  {source_tag} {item['position']:5s} {item['player']:25s} actual={item['actual_points']:6.2f} consensus={item['consensus_points']:6.2f} ({item['source']})")
        print(f"  Total (actual only): {total_actual_only:.2f}")
        print(f"  Total (with consensus): {total_with_consensus:.2f}")
        print(f"  Players: {players_found} actual, {players_consensus} consensus, {players_missing} missing")

    return {
        'actual_only': total_actual_only,
        'actual_with_consensus': total_with_consensus,
        'players_actual': players_found,
        'players_consensus': players_consensus,
        'players_missing': players_missing,
        'breakdown': breakdown
    }


def score_all_lineups(
    lineups_df: pd.DataFrame,
    actual_players_df: pd.DataFrame,
    consensus_players_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Score all lineups using actual fantasy points.

    Adds two score columns:
    - actual_only: Sum of actual points (0 for missing players)
    - actual_with_consensus: Sum with consensus fallback for missing players

    Returns:
        (lineups_df, missing_players): DataFrame with scores and list of missing player info
    """
    print(f"\nScoring {len(lineups_df)} lineups using actual results...")

    # Build lookup dictionaries once (O(n) upfront, then O(1) per lookup)
    actual_lookup = build_player_lookup(actual_players_df, 'actual_points')
    consensus_lookup = build_player_lookup(consensus_players_df, 'fpProjPts') if consensus_players_df is not None else None

    print(f"  Built lookups: {len(actual_lookup)} actual players, {len(consensus_lookup) if consensus_lookup else 0} consensus players")

    results = []
    missing_players_set = {}  # Track unique missing players with their positions

    for idx, row in lineups_df.iterrows():
        score_info = score_lineup(row, actual_lookup, consensus_lookup)
        results.append(score_info)

        # Collect missing players (not found in actual results)
        for item in score_info['breakdown']:
            if item['source'] != 'actual':
                player_name = item['player']
                if player_name not in missing_players_set:
                    missing_players_set[player_name] = {
                        'name': player_name,
                        'position': item['position'].rstrip('0123456789'),  # Remove numbers like RB1 -> RB
                        'consensus_points': item['consensus_points'],
                        'source': item['source']
                    }

    # Add results to lineups - both scoring methods
    lineups_df['actual_only'] = [r['actual_only'] for r in results]
    lineups_df['actual_with_consensus'] = [r['actual_with_consensus'] for r in results]
    lineups_df['players_actual'] = [r['players_actual'] for r in results]
    lineups_df['players_consensus'] = [r['players_consensus'] for r in results]
    lineups_df['players_missing'] = [r['players_missing'] for r in results]

    # Calculate projection error using actual_with_consensus (more complete picture)
    if 'mean' in lineups_df.columns:
        lineups_df['error_vs_mean'] = lineups_df['actual_with_consensus'] - lineups_df['mean']
    if 'median' in lineups_df.columns:
        lineups_df['error_vs_median'] = lineups_df['actual_with_consensus'] - lineups_df['median']

    # Check if actual was within projected range (using actual_with_consensus)
    if 'p10' in lineups_df.columns and 'p90' in lineups_df.columns:
        lineups_df['in_range'] = (
            (lineups_df['actual_with_consensus'] >= lineups_df['p10']) &
            (lineups_df['actual_with_consensus'] <= lineups_df['p90'])
        )
        lineups_df['above_p90'] = lineups_df['actual_with_consensus'] > lineups_df['p90']
        lineups_df['below_p10'] = lineups_df['actual_with_consensus'] < lineups_df['p10']

    print(f"\nScoring complete!")
    print(f"  Average actual (only):          {lineups_df['actual_only'].mean():.2f}")
    print(f"  Average actual (with consensus): {lineups_df['actual_with_consensus'].mean():.2f}")
    print(f"  Best actual (with consensus):    {lineups_df['actual_with_consensus'].max():.2f}")
    print(f"  Worst actual (with consensus):   {lineups_df['actual_with_consensus'].min():.2f}")

    # Show player match stats
    avg_actual = lineups_df['players_actual'].mean()
    avg_consensus = lineups_df['players_consensus'].mean()
    avg_missing = lineups_df['players_missing'].mean()
    print(f"  Player matching: {avg_actual:.1f} actual, {avg_consensus:.1f} consensus, {avg_missing:.1f} missing (avg per lineup)")

    if 'mean' in lineups_df.columns:
        avg_error = lineups_df['error_vs_mean'].mean()
        print(f"  Average projection error: {avg_error:+.2f}")

    if 'in_range' in lineups_df.columns:
        in_range_pct = lineups_df['in_range'].sum() / len(lineups_df) * 100
        above_p90_pct = lineups_df['above_p90'].sum() / len(lineups_df) * 100
        below_p10_pct = lineups_df['below_p10'].sum() / len(lineups_df) * 100
        print(f"  Within P10-P90 range: {in_range_pct:.1f}%")
        print(f"  Above P90: {above_p90_pct:.1f}% (target: 10%)")
        print(f"  Below P10: {below_p10_pct:.1f}% (target: 10%)")

    missing_players = list(missing_players_set.values())
    return lineups_df, missing_players


# ============================================================================
# GENERATE REPORTS
# ============================================================================

def generate_summary_report(
    lineups_df: pd.DataFrame,
    games_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """Generate summary statistics and save as JSON."""

    summary = {
        'timestamp': datetime.now().isoformat(),
        'lineups': {
            'total': len(lineups_df),
            'avg_actual_only': float(lineups_df['actual_only'].mean()),
            'avg_actual_with_consensus': float(lineups_df['actual_with_consensus'].mean()),
            'max_actual_with_consensus': float(lineups_df['actual_with_consensus'].max()),
            'min_actual_with_consensus': float(lineups_df['actual_with_consensus'].min()),
            'std_actual_with_consensus': float(lineups_df['actual_with_consensus'].std())
        },
        'player_matching': {
            'avg_players_actual': float(lineups_df['players_actual'].mean()),
            'avg_players_consensus': float(lineups_df['players_consensus'].mean()),
            'avg_players_missing': float(lineups_df['players_missing'].mean())
        },
        'games': {
            'total': len(games_df),
            'avg_total_points': float(games_df['total_points'].mean()),
            'script_distribution': games_df['actual_script'].value_counts().to_dict()
        }
    }

    # Projection accuracy (uses actual_with_consensus for complete picture)
    if 'mean' in lineups_df.columns:
        summary['projections'] = {
            'avg_projected_mean': float(lineups_df['mean'].mean()),
            'avg_error_vs_mean': float(lineups_df['error_vs_mean'].mean()),
            'rmse_vs_mean': float(np.sqrt((lineups_df['error_vs_mean'] ** 2).mean())),
            'correlation_mean': float(lineups_df[['actual_with_consensus', 'mean']].corr().iloc[0, 1])
        }

    # Range calibration
    if 'in_range' in lineups_df.columns:
        summary['calibration'] = {
            'within_p10_p90_pct': float(lineups_df['in_range'].sum() / len(lineups_df) * 100),
            'above_p90_pct': float(lineups_df['above_p90'].sum() / len(lineups_df) * 100),
            'below_p10_pct': float(lineups_df['below_p10'].sum() / len(lineups_df) * 100)
        }

    # Top lineup analysis (by actual_with_consensus)
    top_idx = lineups_df['actual_with_consensus'].idxmax()
    top_lineup = lineups_df.loc[top_idx]

    summary['top_lineup'] = {
        'lineup_id': str(top_lineup.get('lineup_id', 'unknown')),
        'actual_only': float(top_lineup['actual_only']),
        'actual_with_consensus': float(top_lineup['actual_with_consensus']),
        'projected_mean': float(top_lineup.get('mean', 0)),
        'projected_median': float(top_lineup.get('median', 0)),
        'players_actual': int(top_lineup['players_actual']),
        'players_consensus': int(top_lineup['players_consensus']),
        'players_missing': int(top_lineup['players_missing'])
    }

    # Save summary
    summary_path = output_dir / '7_backtest_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return summary


def print_summary(summary: Dict, week_path: Path = None):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\nLineups analyzed: {summary['lineups']['total']}")
    print(f"  Average score (actual only):          {summary['lineups']['avg_actual_only']:.2f}")
    print(f"  Average score (with consensus):       {summary['lineups']['avg_actual_with_consensus']:.2f}")
    print(f"  Best score (with consensus):          {summary['lineups']['max_actual_with_consensus']:.2f}")
    print(f"  Worst score (with consensus):         {summary['lineups']['min_actual_with_consensus']:.2f}")

    if 'player_matching' in summary:
        pm = summary['player_matching']
        print(f"\nPlayer matching (avg per lineup):")
        print(f"  Actual results found:  {pm['avg_players_actual']:.1f}/9")
        print(f"  Consensus fallback:    {pm['avg_players_consensus']:.1f}/9")
        print(f"  Missing:               {pm['avg_players_missing']:.1f}/9")

    if 'projections' in summary:
        print(f"\nProjection accuracy:")
        print(f"  Average projected:  {summary['projections']['avg_projected_mean']:.2f}")
        print(f"  Average error:      {summary['projections']['avg_error_vs_mean']:+.2f}")
        print(f"  RMSE:              {summary['projections']['rmse_vs_mean']:.2f}")
        print(f"  Correlation:       {summary['projections']['correlation_mean']:.3f}")

    if 'calibration' in summary:
        print(f"\nRange calibration:")
        print(f"  Within P10-P90: {summary['calibration']['within_p10_p90_pct']:.1f}%")
        print(f"  Above P90:      {summary['calibration']['above_p90_pct']:.1f}% (target: 10%)")
        print(f"  Below P10:      {summary['calibration']['below_p10_pct']:.1f}% (target: 10%)")

    print(f"\nTop performing lineup:")
    print(f"  ID:                      {summary['top_lineup']['lineup_id']}")
    print(f"  Actual (only):           {summary['top_lineup']['actual_only']:.2f}")
    print(f"  Actual (with consensus): {summary['top_lineup']['actual_with_consensus']:.2f}")
    print(f"  Projected mean:          {summary['top_lineup']['projected_mean']:.2f}")
    print(f"  Projected median:        {summary['top_lineup']['projected_median']:.2f}")
    print(f"  Players: {summary['top_lineup']['players_actual']}/9 actual, "
          f"{summary['top_lineup']['players_consensus']}/9 consensus, "
          f"{summary['top_lineup']['players_missing']}/9 missing")

    print(f"\nGames analyzed: {summary['games']['total']}")
    print(f"  Average total points: {summary['games']['avg_total_points']:.1f}")
    print(f"\n  Game script distribution:")
    for script, count in summary['games']['script_distribution'].items():
        pct = count / summary['games']['total'] * 100
        print(f"    {script:20s}: {count:2d} ({pct:5.1f}%)")

    if 'missing_players' in summary and summary['missing_players']['count'] > 0:
        print(f"\n⚠️  Missing players: {summary['missing_players']['count']}")
        for name in summary['missing_players']['players'][:10]:  # Show first 10
            print(f"    - {name}")
        if summary['missing_players']['count'] > 10:
            print(f"    ... and {summary['missing_players']['count'] - 10} more")

        # Show hint about where to add missing player data
        if week_path:
            actual_players_file = week_path / 'intermediate' / '4_actual_players.csv'
            print(f"\n   💡 To fix: Add missing players to {actual_players_file}")
            print(f"      Format: name,position,team,salary,actual_points")


def plot_backtest_results(lineups_df: pd.DataFrame, output_dir: Path) -> str:
    """
    Create a multi-panel plot showing projected vs actual lineup performance.

    Panels:
    1. Combined scatter: All projections (Mean, Median, P10, P90) vs Actual (large, top)
    2. Histogram: Projection Error distribution (bottom left)
    3. Range plot: Actual vs P10-P90 range calibration (bottom right)

    Args:
        lineups_df: DataFrame with projected and actual scores
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    # Check required columns exist
    required_cols = ['mean', 'median', 'actual_with_consensus']
    if not all(col in lineups_df.columns for col in required_cols):
        print("Warning: Missing required columns for backtest plot")
        return None

    has_percentiles = 'p10' in lineups_df.columns and 'p90' in lineups_df.columns

    # Create figure with GridSpec for custom layout
    # Top panel (scatter) takes up ~60% of height, bottom row takes 40%
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.25)

    # Style settings
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            pass

    colors = {
        'mean': '#3498db',      # Blue
        'median': '#2ecc71',    # Green
        'p10': '#e74c3c',       # Red
        'p90': '#9b59b6',       # Purple
        'neutral': '#7f8c8d',
        'error': '#e74c3c'
    }

    y_actual = lineups_df['actual_with_consensus']

    # =========================================================================
    # Panel 1: Combined Scatter - All Projections vs Actual (spans both top columns)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])  # Top row, spanning both columns

    # Collect all x values to determine axis limits
    all_x_values = list(lineups_df['mean']) + list(lineups_df['median'])
    if has_percentiles:
        all_x_values += list(lineups_df['p10']) + list(lineups_df['p90'])

    # Calculate axis limits with small padding (5%) - independent for x and y
    x_min, x_max = min(all_x_values), max(all_x_values)
    y_min, y_max = y_actual.min(), y_actual.max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # Plot each projection type with larger markers for visibility
    ax1.scatter(lineups_df['mean'], y_actual, alpha=0.5, c=colors['mean'],
                s=40, label='Mean', marker='o')
    ax1.scatter(lineups_df['median'], y_actual, alpha=0.5, c=colors['median'],
                s=40, label='Median', marker='s')

    if has_percentiles:
        ax1.scatter(lineups_df['p10'], y_actual, alpha=0.4, c=colors['p10'],
                    s=30, label='P10', marker='v')
        ax1.scatter(lineups_df['p90'], y_actual, alpha=0.4, c=colors['p90'],
                    s=30, label='P90', marker='^')

    # Add perfect prediction line (y=x) - draw across the overlapping range
    line_min = max(x_min - x_padding, y_min - y_padding)
    line_max = min(x_max + x_padding, y_max + y_padding)
    ax1.plot([line_min, line_max], [line_min, line_max], '--', color=colors['neutral'],
             alpha=0.7, linewidth=2, label='Perfect (y=x)')

    # Calculate correlations
    r_mean = stats.pearsonr(lineups_df['mean'], y_actual)[0]
    r_median = stats.pearsonr(lineups_df['median'], y_actual)[0]

    corr_text = f'r(mean)={r_mean:.3f}, r(median)={r_median:.3f}'
    if has_percentiles:
        r_p10 = stats.pearsonr(lineups_df['p10'], y_actual)[0]
        r_p90 = stats.pearsonr(lineups_df['p90'], y_actual)[0]
        corr_text += f', r(p10)={r_p10:.3f}, r(p90)={r_p90:.3f}'

    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    ax1.set_xlabel('Projected Score', fontsize=11)
    ax1.set_ylabel('Actual Score', fontsize=11)
    ax1.set_title(f'All Projections vs Actual\n({corr_text})', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    # No set_aspect('equal') - allow plot to stretch to full width

    # =========================================================================
    # Panel 2: Error Distribution (bottom left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    errors = lineups_df['actual_with_consensus'] - lineups_df['mean']
    mean_error = errors.mean()
    std_error = errors.std()

    ax3.hist(errors, bins=30, color=colors['mean'], alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color=colors['neutral'], linestyle='--', linewidth=2, label='Zero error')
    ax3.axvline(x=mean_error, color=colors['error'], linestyle='-', linewidth=2,
                label=f'Mean error: {mean_error:+.1f}')

    ax3.set_xlabel('Prediction Error (Actual - Projected Mean)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Prediction Error Distribution\n(μ = {mean_error:+.1f}, σ = {std_error:.1f})',
                  fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    # =========================================================================
    # Panel 3: Calibration - Actual vs P10-P90 Range (bottom right)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    if has_percentiles:
        # Sort by projected median for better visualization
        sorted_df = lineups_df.sort_values('median').reset_index(drop=True)
        x_idx = range(len(sorted_df))

        # Plot P10-P90 range as shaded area
        ax4.fill_between(x_idx, sorted_df['p10'], sorted_df['p90'],
                         alpha=0.3, color=colors['p90'], label='P10-P90 Range')

        # Plot median line
        ax4.plot(x_idx, sorted_df['median'], color=colors['p90'],
                 linewidth=1.5, label='Projected Median')

        # Plot actual points - color based on whether actual was in range
        in_range = (sorted_df['actual_with_consensus'] >= sorted_df['p10']) & \
                   (sorted_df['actual_with_consensus'] <= sorted_df['p90'])
        above_p90 = sorted_df['actual_with_consensus'] > sorted_df['p90']
        below_p10 = sorted_df['actual_with_consensus'] < sorted_df['p10']

        ax4.scatter([i for i, ir in enumerate(in_range) if ir],
                    sorted_df.loc[in_range, 'actual_with_consensus'],
                    c=colors['median'], s=25, alpha=0.7, label='Within range')
        ax4.scatter([i for i, ap in enumerate(above_p90) if ap],
                    sorted_df.loc[above_p90, 'actual_with_consensus'],
                    c=colors['p10'], s=25, alpha=0.7, marker='^', label='Above P90')
        ax4.scatter([i for i, bp in enumerate(below_p10) if bp],
                    sorted_df.loc[below_p10, 'actual_with_consensus'],
                    c=colors['mean'], s=25, alpha=0.7, marker='v', label='Below P10')

        in_range_pct = in_range.sum() / len(in_range) * 100

        ax4.set_xlabel('Lineups (sorted by projected median)')
        ax4.set_ylabel('Fantasy Points')
        ax4.set_title(f'Range Calibration\n({in_range_pct:.1f}% in range, target: 80%)',
                      fontweight='bold')
        ax4.legend(loc='upper left', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'P10/P90 data not available',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Range Calibration', fontweight='bold')

    # Overall title
    fig.suptitle('Backtest Results: Projected vs Actual Performance',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plot_path = output_dir / 'backtest_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    return str(plot_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest optimizer performance against actual FanDuel results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New architecture
  python 4_backtest.py --week-dir data/2025_12_01 --run-dir run_20251201_143022

  # Manual paths (backward compatible)
  python 4_backtest.py --results data/2025_12_01/inputs/fanduel_results.csv --lineups data/2025_12_01/outputs/run_20251201_143022/3_lineups.csv
        """
    )

    # New architecture arguments
    parser.add_argument(
        '--week-dir',
        type=str,
        help='Week directory (e.g., data/2025_12_01)'
    )

    parser.add_argument(
        '--run-dir',
        type=str,
        help='Run directory name (e.g., run_20251201_143022)'
    )

    # Legacy arguments (backward compatible)
    parser.add_argument(
        '--results',
        type=str,
        help='Path to FanDuel results CSV/JSON (e.g., data/2025_12_01/inputs/fanduel_results.csv)'
    )

    parser.add_argument(
        '--run',
        type=str,
        help='Path to run directory (LEGACY - use --week-dir and --run-dir instead)'
    )

    parser.add_argument(
        '--lineups',
        type=str,
        help='Path to lineups CSV (LEGACY - use --week-dir and --run-dir instead)'
    )

    parser.add_argument(
        '--consensus',
        type=str,
        help='Optional path to consensus projections CSV for fallback'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: same as lineups directory)'
    )

    args = parser.parse_args()

    # Determine paths using new or legacy arguments
    week_path = None  # Track if using new architecture

    if args.week_dir and args.run_dir:
        # New architecture - use week_path for loading results
        week_path = Path(args.week_dir)
        run_path = week_path / 'outputs' / args.run_dir

        lineups_path = run_path / '3_lineups.csv'
        output_dir = run_path

        if not args.consensus:
            # Use consensus from intermediate
            consensus_path = week_path / 'intermediate' / '1_players.csv'
            if consensus_path.exists():
                args.consensus = str(consensus_path)

    elif args.lineups:
        # Legacy: manual lineups path
        lineups_path = Path(args.lineups)
        output_dir = lineups_path.parent

        if not args.results:
            print("Error: --results required when using --lineups")
            return
        results_path = Path(args.results)

    elif args.run:
        # Legacy: run directory
        run_path = Path(args.run)
        lineups_path = run_path / 'BEST_LINEUPS.csv'
        if not lineups_path.exists():
            lineups_path = run_path / '3_lineups.csv'  # Try new name
        output_dir = run_path

        if not args.results:
            print("Error: --results required when using --run")
            return
        results_path = Path(args.results)

    else:
        print("Error: Must provide either:")
        print("  1. --week-dir and --run-dir (new architecture)")
        print("  2. --results and --lineups (legacy)")
        print("  3. --results and --run (legacy)")
        return

    # Validate lineups exist
    if not lineups_path.exists():
        print(f"Error: Lineups file not found: {lineups_path}")
        return

    # Override output directory if specified
    if args.output:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FANTASY OPTIMIZER BACKTEST")
    print("=" * 80)
    print(f"\nLineups CSV:   {lineups_path}")
    print(f"Output dir:    {output_dir}")

    # Load actual results (CSV preferred, JSON fallback with auto-conversion)
    if week_path:
        # New architecture: use load_actual_results which handles CSV/JSON
        actual_players_df, games_df = load_actual_results(week_path)
    else:
        # Legacy: results_path was explicitly provided
        if not results_path.exists():
            print(f"Error: Results file not found: {results_path}")
            return

        # For legacy, convert JSON inline if needed
        if str(results_path).endswith('.json'):
            print(f"Loading FanDuel results from: {results_path}")
            with open(results_path, 'r') as f:
                data = json.load(f)

            players = []
            games = {}
            if 'data' in data and 'leaders' in data['data']:
                for leader in data['data']['leaders']:
                    full_name = f"{leader['player']['firstNames']} {leader['player']['lastName']}"
                    team = leader['player']['associations'][0]['team']['code'] if leader['player']['associations'] else None
                    players.append({
                        'name': normalize_player_name(full_name),
                        'position': leader['position'],
                        'team': team,
                        'salary': leader['salary'],
                        'actual_points': leader['stats']['fantasyPoints']
                    })

                    if 'fixture' in leader and 'boxscore' in leader:
                        fixture = leader['fixture']
                        game_id = fixture['id']
                        if game_id not in games:
                            boxscore = leader['boxscore']['stats']['stats']
                            away_score = home_score = None
                            for stat in boxscore:
                                if stat['type'] == 'AWAY_TEAM_TOTAL_SCORE':
                                    away_score = int(stat['value'])
                                elif stat['type'] == 'HOME_TEAM_TOTAL_SCORE':
                                    home_score = int(stat['value'])
                            if away_score is not None and home_score is not None:
                                games[game_id] = {
                                    'game_id': game_id,
                                    'away_team': fixture['awayTeam']['code'],
                                    'home_team': fixture['homeTeam']['code'],
                                    'away_score': away_score,
                                    'home_score': home_score,
                                    'total_points': away_score + home_score,
                                    'point_differential': abs(away_score - home_score)
                                }

            actual_players_df = pd.DataFrame(players)
            games_df = pd.DataFrame(list(games.values()))
            print(f"  Found {len(actual_players_df)} players, {len(games_df)} games")
        else:
            # CSV format
            actual_players_df = pd.read_csv(results_path)
            actual_players_df['name'] = actual_players_df['name'].apply(normalize_player_name)
            # For legacy CSV, games would need separate file - just create empty
            games_df = pd.DataFrame()

    # Analyze game scripts
    games_df = analyze_game_scripts(games_df)

    # Load lineups
    print(f"\nLoading lineups from: {lineups_path}")
    lineups_df = pd.read_csv(lineups_path)
    print(f"  Loaded {len(lineups_df)} lineups")

    # Load consensus projections if provided
    consensus_df = None
    if args.consensus:
        print(f"\nLoading consensus projections from: {args.consensus}")
        consensus_df = pd.read_csv(args.consensus)
        print(f"  Loaded {len(consensus_df)} player projections")

    # Score lineups
    lineups_df, missing_players = score_all_lineups(lineups_df, actual_players_df, consensus_df)

    # Append missing players to 4_actual_players.csv with actual_points=0.0
    if missing_players and week_path:
        actual_csv_path = week_path / 'intermediate' / '4_actual_players.csv'
        if actual_csv_path.exists():
            # Load existing to check for duplicates
            existing_df = pd.read_csv(actual_csv_path)
            existing_names = set(existing_df['name'].apply(normalize_player_name))

            # Load FanDuel salaries for team/salary lookup
            salaries_path = week_path / 'inputs' / 'fanduel_salaries.csv'
            salary_lookup = {}
            if salaries_path.exists():
                salaries_df = pd.read_csv(salaries_path)
                # Build lookup: normalized name -> (team, salary)
                for _, row in salaries_df.iterrows():
                    name = row.get('Nickname', '')
                    if name:
                        norm_name = normalize_player_name(name)
                        salary_lookup[norm_name] = {
                            'team': row.get('Team', ''),
                            'salary': row.get('Salary', 0)
                        }

            # Filter to only truly new players
            new_players = []
            for p in missing_players:
                norm_name = normalize_player_name(p['name'])
                if norm_name not in existing_names:
                    # Look up team and salary from FanDuel salaries
                    lookup_info = salary_lookup.get(norm_name, {})
                    new_players.append({
                        'name': p['name'],
                        'position': p['position'],
                        'team': lookup_info.get('team', ''),
                        'salary': lookup_info.get('salary', 0),
                        'actual_points': 0.0  # Placeholder for manual entry
                    })

            if new_players:
                # Append to CSV
                new_df = pd.DataFrame(new_players)
                new_df.to_csv(actual_csv_path, mode='a', header=False, index=False)
                print(f"\n⚠️  Added {len(new_players)} missing players to: {actual_csv_path}")
                print(f"   Players added with actual_points=0.0:")
                for p in new_players:
                    team_info = f", {p['team']}" if p['team'] else ""
                    salary_info = f", ${p['salary']:,}" if p['salary'] else ""
                    print(f"     - {p['name']} ({p['position']}{team_info}{salary_info})")
                print(f"\n   To complete backtest:")
                print(f"   1. Edit {actual_csv_path}")
                print(f"   2. Fill in actual_points for the 0.0 entries")
                print(f"   3. Re-run backtest")

    # Reorder columns for better readability
    # Position columns first, then scores, then distribution stats, then metadata
    preferred_order = [
        'QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF',
        'cvar_score', 'actual_only', 'actual_with_consensus',
        'mean', 'median', 'std', 'p10', 'p80', 'p90',
        'lineup_id', 'player_ids', 'total_salary', 'strategy', 'anchor_player',
        'players_actual', 'players_consensus', 'players_missing',
        'error_vs_mean', 'error_vs_median', 'in_range', 'above_p90', 'below_p10'
    ]
    # Only include columns that exist in the dataframe
    ordered_cols = [col for col in preferred_order if col in lineups_df.columns]
    # Add any remaining columns not in the preferred order
    remaining_cols = [col for col in lineups_df.columns if col not in ordered_cols]
    lineups_df = lineups_df[ordered_cols + remaining_cols]

    # Round numerical columns to 2 decimal places
    numeric_cols = lineups_df.select_dtypes(include=['float64', 'float32']).columns
    lineups_df[numeric_cols] = lineups_df[numeric_cols].round(2)

    # Save detailed results
    backtest_lineups_path = output_dir / '5_scored_lineups.csv'
    lineups_df.to_csv(backtest_lineups_path, index=False)
    print(f"\nDetailed lineup results saved to: {backtest_lineups_path}")

    backtest_games_path = output_dir / '6_backtest_games.csv'
    games_df.to_csv(backtest_games_path, index=False)
    print(f"Game analysis saved to: {backtest_games_path}")

    # Generate summary
    summary = generate_summary_report(lineups_df, games_df, output_dir)

    # Add missing players info to summary
    if missing_players:
        summary['missing_players'] = {
            'count': len(missing_players),
            'players': [p['name'] for p in missing_players]
        }
        # Re-save with missing players info
        summary_path = output_dir / '7_backtest_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print_summary(summary, week_path=week_path)

    # Generate backtest visualization
    try:
        plot_path = plot_backtest_results(lineups_df, output_dir)
        if plot_path:
            print(f"\nBacktest analysis plot saved to: {plot_path}")

            # Add correlation info to summary interpretation
            if 'projections' in summary and summary['projections'].get('correlation_mean', 0) > 0:
                corr = summary['projections']['correlation_mean']
                if corr > 0.7:
                    print(f"   Strong positive correlation (r={corr:.3f}) - projections are predictive!")
                elif corr > 0.4:
                    print(f"   Moderate positive correlation (r={corr:.3f}) - projections have signal")
                elif corr > 0:
                    print(f"   Weak positive correlation (r={corr:.3f}) - some predictive value")
                else:
                    print(f"   No positive correlation (r={corr:.3f}) - projections need improvement")
    except Exception as e:
        print(f"\nWarning: Could not generate backtest plot: {e}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
