"""
Fetch current week fantasy football data from all sources.

This script ONLY downloads raw data and converts to CSV format.
All data processing and merging happens in 2_data_integration.py.

Features:
- Smart caching: skips re-download if files exist (unless --force)
- CSV-first: converts all JSON downloads to CSV immediately
- Current week only: no --week argument (use run_week.py for historical data)

Usage:
    # Fetch all data sources to default location (current week)
    python 1_fetch_data.py

    # Fetch to specific week directory
    python 1_fetch_data.py --week-dir data/2025_12_01

    # Force re-fetch even if files exist
    python 1_fetch_data.py --force

    # Fetch specific sources only
    python 1_fetch_data.py --fp --dk
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from scrapers import (
    FantasyProsScraper,
    FanDuelLoader,
    DraftKingsGameLinesScraper,
    DraftKingsTdOddsScraper,
    EspnPlayerListScraper,
    EspnProjectionsScraper,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_week_dir() -> Path:
    """Get default week directory for current week (YYYY_MM_DD format)."""
    from run_week import get_current_week
    return Path('data') / get_current_week()


def json_to_csv(json_data: dict | list, output_path: Path, mode: str = 'dict'):
    """
    Convert JSON data to CSV.

    Args:
        json_data: JSON data (dict or list)
        output_path: Output CSV path
        mode: 'dict' (dict of lists) or 'list' (list of dicts)
    """
    if mode == 'dict':
        # Dict of lists (e.g., {'QB': [...], 'RB': [...]})
        all_rows = []
        for key, items in json_data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        item['_category'] = key
                        all_rows.append(item)
            elif isinstance(items, dict):
                items['_category'] = key
                all_rows.append(items)

        df = pd.DataFrame(all_rows)
    else:
        # List of dicts
        df = pd.DataFrame(json_data)

    df.to_csv(output_path, index=False)
    return len(df)


# ============================================================================
# FETCH FUNCTIONS
# ============================================================================

def fetch_fanduel_salaries(week_dir: Path, force: bool = False) -> Path:
    """
    Load FanDuel salaries CSV (user must provide this file).

    Args:
        week_dir: Week directory
        force: Ignored (user must manually provide file)

    Returns:
        Path to fanduel_salaries.csv
    """
    print("📄 FanDuel Salaries")

    inputs_dir = week_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = inputs_dir / 'fanduel_salaries.csv'

    # Look for FanDuel CSV in inputs directory OR week directory
    # User might drop it in either place
    fanduel_files = list(inputs_dir.glob('FanDuel-NFL-*.csv'))
    fanduel_files += list(week_dir.glob('FanDuel-NFL-*.csv'))
    # Also check for players-list pattern (e.g., FanDuel-NFL-2025 PST-12 PST-07 PST-123865-players-list.csv)
    fanduel_files += list(inputs_dir.glob('*-players-list.csv'))
    fanduel_files += list(week_dir.glob('*-players-list.csv'))

    if fanduel_files and not force:
        # Found existing FanDuel file
        latest = max(fanduel_files, key=lambda p: p.stat().st_mtime)

        # If not already named correctly, copy it
        if latest != output_file:
            import shutil
            shutil.copy2(latest, output_file)
            print(f"  ✓ Copied: {latest.name} → fanduel_salaries.csv")
        else:
            print(f"  ✓ Using existing: {output_file.name}")

        return output_file

    # No FanDuel file found
    print(f"  ⚠️  No FanDuel CSV found in {inputs_dir}")
    print(f"  📥 Please download from FanDuel and save as:")
    print(f"     {output_file}")
    print(f"  ℹ️  Or place any FanDuel-NFL-*.csv in {inputs_dir}")

    return output_file


def fetch_fantasypros(week_dir: Path, force: bool = False) -> Path:
    """
    Fetch FantasyPros projections and convert to CSV.

    Args:
        week_dir: Week directory
        force: Force re-fetch even if file exists

    Returns:
        Path to fantasypros_projections.csv
    """
    print("\n🏈 FantasyPros Projections")

    inputs_dir = week_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = inputs_dir / 'fantasypros_projections.csv'

    # Check if already exists (smart caching)
    if output_file.exists() and not force:
        print(f"  ✓ Using cached: {output_file.name}")
        return output_file

    # Fetch fresh data
    print("  ⏳ Fetching from FantasyPros API...")
    scraper = FantasyProsScraper()
    data = scraper.get_data(use_cache=False)

    # Convert to CSV
    # data is a dict like {'QB': [...], 'RB': [...], ...}
    rows = []
    for position, players in data.items():
        for player in players:
            player['position'] = position
            rows.append(player)

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(df)} players to {output_file.name}")

    return output_file


def fetch_game_lines(week_dir: Path, force: bool = False) -> Path:
    """
    Fetch DraftKings game lines and convert to CSV.

    Args:
        week_dir: Week directory
        force: Force re-fetch even if file exists

    Returns:
        Path to game_lines.csv
    """
    print("\n📊 DraftKings Game Lines")

    inputs_dir = week_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = inputs_dir / 'game_lines.csv'

    # Check if already exists (smart caching)
    if output_file.exists() and not force:
        print(f"  ✓ Using cached: {output_file.name}")
        return output_file

    # Fetch fresh data
    print("  ⏳ Fetching from DraftKings...")
    scraper = DraftKingsGameLinesScraper()
    data = scraper.get_data(use_cache=False)

    # Convert to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(df)} games to {output_file.name}")

    return output_file


def fetch_td_odds(week_dir: Path, force: bool = False) -> Path:
    """
    Fetch DraftKings TD odds and convert to CSV.

    Args:
        week_dir: Week directory
        force: Force re-fetch even if file exists

    Returns:
        Path to td_odds.csv
    """
    print("\n🎯 DraftKings TD Odds")

    inputs_dir = week_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = inputs_dir / 'td_odds.csv'

    # Check if already exists (smart caching)
    if output_file.exists() and not force:
        print(f"  ✓ Using cached: {output_file.name}")
        return output_file

    # Fetch fresh data
    print("  ⏳ Fetching from DraftKings...")
    scraper = DraftKingsTdOddsScraper()
    data = scraper.get_data(use_cache=False)

    # Convert to CSV (data is a dict with player names as keys)
    rows = []
    for player_name, odds_data in data.items():
        row = {'name': player_name}
        row.update(odds_data)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(df)} players to {output_file.name}")

    return output_file


def fetch_espn_projections(week_dir: Path, force: bool = False) -> Path:
    """
    Fetch ESPN Watson projections (floor/ceiling/boom-bust).

    This is a two-step process:
    1. Fetch ESPN player IDs
    2. Fetch individual projections for relevant players

    Args:
        week_dir: Week directory
        force: Force re-fetch even if file exists

    Returns:
        Path to espn_projections.csv
    """
    print("\n📈 ESPN Watson Projections")

    inputs_dir = week_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = inputs_dir / 'espn_projections.csv'

    # Check if already exists (smart caching)
    if output_file.exists() and not force:
        print(f"  ✓ Using cached: {output_file.name}")
        return output_file

    # Step 1: Get ESPN player IDs
    print("  ⏳ Step 1: Fetching ESPN player IDs...")
    player_list_scraper = EspnPlayerListScraper()
    espn_ids = player_list_scraper.get_data(use_cache=not force)
    print(f"  ✓ Got {len(espn_ids)} ESPN player IDs")

    # Step 2: Load FantasyPros projections to know which players to fetch
    fp_file = inputs_dir / 'fantasypros_projections.csv'
    if not fp_file.exists():
        print(f"  ⚠️  FantasyPros projections not found. Run with --fp first.")
        return output_file

    fp_df = pd.read_csv(fp_file)
    print(f"  📋 Found {len(fp_df)} players in FantasyPros")

    # Normalize names and match with ESPN IDs
    from scrapers import normalize_name
    players_with_ids = []
    matched = 0
    for _, row in fp_df.iterrows():
        name_norm = normalize_name(row['name'])
        espn_id = espn_ids.get(name_norm)
        if espn_id:
            matched += 1
            players_with_ids.append({
                'name': name_norm,
                'espnId': espn_id,
                'fpProjPts': row.get('fpts', row.get('projection', 0)),
                'position': row.get('position', '')
            })

    print(f"  ✓ Matched {matched}/{len(fp_df)} players with ESPN IDs")

    # Step 3: Fetch projections for each player
    print("  ⏳ Step 2: Fetching ESPN projections (this may take a few minutes)...")
    proj_scraper = EspnProjectionsScraper()
    projections = proj_scraper.fetch_all(players_with_ids, min_projection=2.5)

    # Convert to CSV
    rows = []
    for name, proj_data in projections.items():
        row = {'name': name}
        row.update(proj_data)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(df)} projections to {output_file.name}")

    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch current week fantasy football data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all data to default location (current week)
  python 1_fetch_data.py

  # Fetch to specific week directory
  python 1_fetch_data.py --week-dir data/2025_12_01

  # Force re-fetch even if files exist
  python 1_fetch_data.py --force

  # Fetch specific sources only
  python 1_fetch_data.py --fp --dk
        """
    )

    parser.add_argument(
        '--week-dir',
        type=str,
        help='Week directory to save data (default: current week)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch even if files exist'
    )

    # Source selection (default: all)
    parser.add_argument('--fd', action='store_true', help='Fetch FanDuel salaries only')
    parser.add_argument('--fp', action='store_true', help='Fetch FantasyPros only')
    parser.add_argument('--lines', action='store_true', help='Fetch DK game lines only')
    parser.add_argument('--dk', action='store_true', help='Fetch DK TD odds only')
    parser.add_argument('--espn', action='store_true', help='Fetch ESPN projections only')

    args = parser.parse_args()

    # ========================================================================
    # DETERMINE WEEK DIRECTORY
    # ========================================================================

    if args.week_dir:
        week_dir = Path(args.week_dir)
    else:
        week_dir = get_current_week_dir()

    print(f"\n{'='*80}")
    print("FETCH CURRENT WEEK DATA")
    print(f"{'='*80}")
    print(f"\nWeek directory: {week_dir}")
    print(f"Force re-fetch: {args.force}")

    # Determine what to fetch (default: all)
    fetch_all = not any([args.fd, args.fp, args.lines, args.dk, args.espn])

    fetch_fd = args.fd or fetch_all
    fetch_fp = args.fp or fetch_all
    fetch_lines = args.lines or fetch_all
    fetch_dk_odds = args.dk or fetch_all
    fetch_espn = args.espn or fetch_all

    # ========================================================================
    # FETCH DATA
    # ========================================================================

    files_created = []

    # FanDuel salaries (user must provide)
    if fetch_fd:
        fd_file = fetch_fanduel_salaries(week_dir, args.force)
        if fd_file.exists():
            files_created.append(fd_file)

    # FantasyPros projections
    if fetch_fp:
        fp_file = fetch_fantasypros(week_dir, args.force)
        if fp_file.exists():
            files_created.append(fp_file)

    # DraftKings game lines
    if fetch_lines:
        lines_file = fetch_game_lines(week_dir, args.force)
        if lines_file.exists():
            files_created.append(lines_file)

    # DraftKings TD odds
    if fetch_dk_odds:
        odds_file = fetch_td_odds(week_dir, args.force)
        if odds_file.exists():
            files_created.append(odds_file)

    # ESPN Watson projections (must run after FantasyPros)
    if fetch_espn:
        espn_file = fetch_espn_projections(week_dir, args.force)
        if espn_file.exists():
            files_created.append(espn_file)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n{'='*80}")
    print("✅ FETCH COMPLETE")
    print(f"{'='*80}\n")

    print(f"📁 Output directory: {week_dir / 'inputs'}\n")
    print("Files created:")
    for file in files_created:
        print(f"  ✓ {file.relative_to(week_dir)}")

    # Check for missing files
    inputs_dir = week_dir / 'inputs'
    required_files = [
        'fanduel_salaries.csv',
        'fantasypros_projections.csv',
        'game_lines.csv',
        'td_odds.csv',
        'espn_projections.csv'
    ]

    missing = [f for f in required_files if not (inputs_dir / f).exists()]

    if missing:
        print("\n⚠️  Missing files:")
        for f in missing:
            print(f"  ✗ {f}")
        print("\nRun again with --force or add files manually.")
    else:
        print("\n✅ All required files present!")
        print("\n💡 Next step:")
        print(f"   python 2_data_integration.py --week-dir {week_dir}")


if __name__ == '__main__':
    main()
