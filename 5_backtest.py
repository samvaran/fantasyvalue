"""
Backtest optimizer performance against actual FanDuel results.

This script:
1. Parses actual fantasy points from FanDuel JSON
2. Scores generated lineups using actual results
3. Compares actual vs predicted game scripts
4. Analyzes projection accuracy

Usage:
    python 5_backtest.py --results data/input/2025_week13.json --run outputs/run_20251130_231349
    python 5_backtest.py --results data/input/2025_week13.json --lineups outputs/run_20251130_231349/BEST_LINEUPS.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from datetime import datetime


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

def parse_fanduel_json(json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse FanDuel results JSON to extract player stats and game info.

    Returns:
        (players_df, games_df): DataFrames with actual fantasy points and game scores
    """
    print(f"Loading FanDuel results from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    players = []
    games = {}

    # Parse leaders (top performers with actual stats)
    if 'data' in data and 'leaders' in data['data']:
        for leader in data['data']['leaders']:
            player_info = {
                'player_id': leader['player']['id'],
                'first_name': leader['player']['firstNames'],
                'last_name': leader['player']['lastName'],
                'position': leader['position'],
                'salary': leader['salary'],
                'actual_points': leader['stats']['fantasyPoints'],
                'team': leader['player']['associations'][0]['team']['code'] if leader['player']['associations'] else None
            }

            # Add full name for matching (normalize to match our other data sources)
            full_name = f"{player_info['first_name']} {player_info['last_name']}"
            player_info['name'] = normalize_player_name(full_name)

            players.append(player_info)

            # Extract game info
            if 'fixture' in leader:
                fixture = leader['fixture']
                game_id = fixture['id']

                if game_id not in games and 'boxscore' in leader:
                    boxscore = leader['boxscore']['stats']['stats']

                    # Extract scores
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
                            'total_points': away_score + home_score,
                            'point_differential': abs(away_score - home_score),
                            'status': fixture['status']
                        }

    players_df = pd.DataFrame(players)
    games_df = pd.DataFrame(list(games.values()))

    print(f"  Found {len(players_df)} players with actual results")
    print(f"  Found {len(games_df)} completed games")

    return players_df, games_df


# ============================================================================
# CLASSIFY ACTUAL GAME SCRIPTS
# ============================================================================

def classify_actual_game_script(total_points: float, point_diff: float) -> str:
    """
    Classify actual game script based on final score.

    Uses same logic as game_script_continuous.py for consistency.
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

def match_player(player_name: str, players_df: pd.DataFrame, points_column: str = 'actual_points') -> float:
    """
    Match player to results and return fantasy points.

    Args:
        player_name: Player name to match
        players_df: DataFrame with player data
        points_column: Column name containing points ('actual_points' or 'fpProjPts')

    Returns:
        Fantasy points if found, None if not found
    """
    norm_name = normalize_player_name(player_name)

    # Determine name columns to check
    if 'name' in players_df.columns:
        name_col = 'name'
    elif 'playerName' in players_df.columns:
        name_col = 'playerName'
    else:
        # Try to match with any column that might be a name
        return None

    # Try exact match on normalized full name
    for _, player in players_df.iterrows():
        if name_col in player.index:
            actual_norm = normalize_player_name(str(player[name_col]))
            if norm_name == actual_norm:
                return player.get(points_column, None)

    # No match found
    return None


def score_lineup(
    lineup_row: pd.Series,
    actual_players_df: pd.DataFrame,
    consensus_players_df: pd.DataFrame = None,
    verbose: bool = False
) -> Dict:
    """
    Score a single lineup using actual fantasy points.

    Args:
        lineup_row: Row from lineups CSV with position columns
        actual_players_df: DataFrame with actual fantasy points
        consensus_players_df: Optional DataFrame with consensus projections for fallback
        verbose: If True, print detailed scoring breakdown

    Returns:
        Dict with actual score and breakdown
    """
    positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF']

    total_actual = 0.0
    players_found = 0
    players_fallback = 0
    breakdown = []

    for pos in positions:
        if pos not in lineup_row or pd.isna(lineup_row[pos]):
            continue

        player_name = lineup_row[pos]

        # Try to find actual points
        actual_points = match_player(player_name, actual_players_df, 'actual_points')

        if actual_points is not None:
            total_actual += actual_points
            players_found += 1
            breakdown.append({
                'position': pos,
                'player': player_name,
                'actual_points': actual_points,
                'source': 'actual'
            })
        else:
            # Fall back to consensus projection
            if consensus_players_df is not None:
                consensus_points = match_player(player_name, consensus_players_df, 'fpProjPts')
                if consensus_points is not None:
                    total_actual += consensus_points
                    players_fallback += 1
                    breakdown.append({
                        'position': pos,
                        'player': player_name,
                        'actual_points': consensus_points,
                        'source': 'consensus'
                    })
                else:
                    breakdown.append({
                        'position': pos,
                        'player': player_name,
                        'actual_points': 0.0,
                        'source': 'missing'
                    })
            else:
                breakdown.append({
                    'position': pos,
                    'player': player_name,
                    'actual_points': 0.0,
                    'source': 'missing'
                })

    if verbose:
        print(f"\nLineup scoring breakdown:")
        for item in breakdown:
            source_tag = {'actual': '✓', 'consensus': '~', 'missing': '✗'}[item['source']]
            print(f"  {source_tag} {item['position']:5s} {item['player']:25s} {item['actual_points']:6.2f} ({item['source']})")
        print(f"  Total: {total_actual:.2f} ({players_found} actual, {players_fallback} consensus)")

    return {
        'actual_total': total_actual,
        'players_found': players_found,
        'players_fallback': players_fallback,
        'breakdown': breakdown
    }


def score_all_lineups(
    lineups_df: pd.DataFrame,
    actual_players_df: pd.DataFrame,
    consensus_players_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Score all lineups using actual fantasy points.

    Returns:
        DataFrame with actual scores added
    """
    print(f"\nScoring {len(lineups_df)} lineups using actual results...")

    results = []
    for idx, row in lineups_df.iterrows():
        score_info = score_lineup(row, actual_players_df, consensus_players_df)
        results.append(score_info)

        if (idx + 1) % 10 == 0:
            print(f"  Scored {idx + 1}/{len(lineups_df)} lineups...")

    # Add results to lineups
    lineups_df['actual_total'] = [r['actual_total'] for r in results]
    lineups_df['players_actual'] = [r['players_found'] for r in results]
    lineups_df['players_consensus'] = [r['players_fallback'] for r in results]

    # Calculate projection error
    if 'mean' in lineups_df.columns:
        lineups_df['error_vs_mean'] = lineups_df['actual_total'] - lineups_df['mean']
    if 'median' in lineups_df.columns:
        lineups_df['error_vs_median'] = lineups_df['actual_total'] - lineups_df['median']

    # Check if actual was within projected range
    if 'p10' in lineups_df.columns and 'p90' in lineups_df.columns:
        lineups_df['in_range'] = (
            (lineups_df['actual_total'] >= lineups_df['p10']) &
            (lineups_df['actual_total'] <= lineups_df['p90'])
        )
        lineups_df['above_p90'] = lineups_df['actual_total'] > lineups_df['p90']
        lineups_df['below_p10'] = lineups_df['actual_total'] < lineups_df['p10']

    print(f"\nScoring complete!")
    print(f"  Average actual score: {lineups_df['actual_total'].mean():.2f}")
    print(f"  Best actual score: {lineups_df['actual_total'].max():.2f}")
    print(f"  Worst actual score: {lineups_df['actual_total'].min():.2f}")

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

    return lineups_df


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
            'avg_actual': float(lineups_df['actual_total'].mean()),
            'max_actual': float(lineups_df['actual_total'].max()),
            'min_actual': float(lineups_df['actual_total'].min()),
            'std_actual': float(lineups_df['actual_total'].std())
        },
        'games': {
            'total': len(games_df),
            'avg_total_points': float(games_df['total_points'].mean()),
            'script_distribution': games_df['actual_script'].value_counts().to_dict()
        }
    }

    # Projection accuracy
    if 'mean' in lineups_df.columns:
        summary['projections'] = {
            'avg_projected_mean': float(lineups_df['mean'].mean()),
            'avg_error_vs_mean': float(lineups_df['error_vs_mean'].mean()),
            'rmse_vs_mean': float(np.sqrt((lineups_df['error_vs_mean'] ** 2).mean())),
            'correlation_mean': float(lineups_df[['actual_total', 'mean']].corr().iloc[0, 1])
        }

    # Range calibration
    if 'in_range' in lineups_df.columns:
        summary['calibration'] = {
            'within_p10_p90_pct': float(lineups_df['in_range'].sum() / len(lineups_df) * 100),
            'above_p90_pct': float(lineups_df['above_p90'].sum() / len(lineups_df) * 100),
            'below_p10_pct': float(lineups_df['below_p10'].sum() / len(lineups_df) * 100)
        }

    # Top lineup analysis
    top_idx = lineups_df['actual_total'].idxmax()
    top_lineup = lineups_df.loc[top_idx]

    summary['top_lineup'] = {
        'lineup_id': str(top_lineup.get('lineup_id', 'unknown')),
        'actual_total': float(top_lineup['actual_total']),
        'projected_mean': float(top_lineup.get('mean', 0)),
        'projected_median': float(top_lineup.get('median', 0)),
        'players_actual': int(top_lineup['players_actual']),
        'players_consensus': int(top_lineup['players_consensus'])
    }

    # Save summary
    summary_path = output_dir / 'backtest_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return summary


def print_summary(summary: Dict):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\nLineups analyzed: {summary['lineups']['total']}")
    print(f"  Average actual score: {summary['lineups']['avg_actual']:.2f}")
    print(f"  Best actual score:    {summary['lineups']['max_actual']:.2f}")
    print(f"  Worst actual score:   {summary['lineups']['min_actual']:.2f}")

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
    print(f"  ID:               {summary['top_lineup']['lineup_id']}")
    print(f"  Actual score:     {summary['top_lineup']['actual_total']:.2f}")
    print(f"  Projected mean:   {summary['top_lineup']['projected_mean']:.2f}")
    print(f"  Projected median: {summary['top_lineup']['projected_median']:.2f}")
    print(f"  Players matched:  {summary['top_lineup']['players_actual']}/9 actual, "
          f"{summary['top_lineup']['players_consensus']}/9 consensus")

    print(f"\nGames analyzed: {summary['games']['total']}")
    print(f"  Average total points: {summary['games']['avg_total_points']:.1f}")
    print(f"\n  Game script distribution:")
    for script, count in summary['games']['script_distribution'].items():
        pct = count / summary['games']['total'] * 100
        print(f"    {script:20s}: {count:2d} ({pct:5.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest optimizer performance against actual FanDuel results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to FanDuel results JSON (e.g., data/input/2025_week13.json)'
    )

    parser.add_argument(
        '--run',
        type=str,
        help='Path to run directory (e.g., outputs/run_20251130_231349)'
    )

    parser.add_argument(
        '--lineups',
        type=str,
        help='Path to lineups CSV (e.g., outputs/run_20251130_231349/BEST_LINEUPS.csv)'
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

    # Determine lineups path
    if args.lineups:
        lineups_path = Path(args.lineups)
    elif args.run:
        lineups_path = Path(args.run) / 'BEST_LINEUPS.csv'
    else:
        print("Error: Must provide either --run or --lineups")
        return

    if not lineups_path.exists():
        print(f"Error: Lineups file not found: {lineups_path}")
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = lineups_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FANTASY OPTIMIZER BACKTEST")
    print("=" * 80)
    print(f"\nResults JSON:  {args.results}")
    print(f"Lineups CSV:   {lineups_path}")
    print(f"Output dir:    {output_dir}")

    # Parse FanDuel results
    actual_players_df, games_df = parse_fanduel_json(args.results)

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
    lineups_df = score_all_lineups(lineups_df, actual_players_df, consensus_df)

    # Save detailed results
    backtest_lineups_path = output_dir / 'backtest_lineups.csv'
    lineups_df.to_csv(backtest_lineups_path, index=False)
    print(f"\nDetailed lineup results saved to: {backtest_lineups_path}")

    backtest_games_path = output_dir / 'backtest_games.csv'
    games_df.to_csv(backtest_games_path, index=False)
    print(f"Game analysis saved to: {backtest_games_path}")

    # Generate summary
    summary = generate_summary_report(lineups_df, games_df, output_dir)
    print_summary(summary)

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
