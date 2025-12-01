"""
Archive weekly data for historical backtesting and analysis.

This script captures all time-sensitive data (projections, lines, odds) that
disappears after the week completes, allowing you to replay the full pipeline
later for backtesting and model tuning.

Usage:
    # Archive current week's data (before games are played)
    python 6_archive_week.py --week 13 --year 2025 --season-type REG

    # Archive with custom name
    python 6_archive_week.py --name "2025_week13_thanksgiving"

    # Restore archived data for replay
    python 6_archive_week.py --restore data/archive/2025_week13_REG
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd


# ============================================================================
# ARCHIVE STRUCTURE
# ============================================================================

def create_archive_structure(archive_dir: Path) -> dict:
    """
    Create directory structure for archived week.

    Structure:
        archive/YYYY_weekNN_TYPE/
        ├── metadata.json          # Week info, timestamp, data sources
        ├── input/
        │   ├── fanduel_salaries.csv      # FanDuel contest CSV
        │   └── actual_results.json       # FanDuel results (add after week)
        ├── scraped/
        │   ├── fantasypros.json          # Raw FantasyPros response
        │   ├── draftkings_lines.json     # Raw DK game lines
        │   └── draftkings_odds.json      # Raw DK TD odds
        └── intermediate/
            ├── players_raw.csv           # Merged player data
            ├── players_integrated.csv    # Final integrated data
            ├── game_lines.csv            # Processed game lines
            └── game_script_continuous.csv # Game script analysis

    Returns:
        Dict with paths to all directories
    """
    archive_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        'root': archive_dir,
        'input': archive_dir / 'input',
        'scraped': archive_dir / 'scraped',
        'intermediate': archive_dir / 'intermediate',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    return dirs


# ============================================================================
# ARCHIVE WEEK DATA
# ============================================================================

def archive_week(
    week: int,
    year: int,
    season_type: str = 'REG',
    custom_name: str = None,
    archive_root: str = 'data/archive'
) -> Path:
    """
    Archive all data for a specific week.

    Args:
        week: Week number (1-18)
        year: Year (e.g., 2025)
        season_type: 'REG' or 'POST'
        custom_name: Optional custom name instead of auto-generated
        archive_root: Root directory for archives

    Returns:
        Path to created archive directory
    """
    # Create archive directory name
    if custom_name:
        archive_name = custom_name
    else:
        archive_name = f"{year}_week{week:02d}_{season_type}"

    archive_dir = Path(archive_root) / archive_name
    print(f"\n{'='*80}")
    print(f"ARCHIVING WEEK {week}, {year} ({season_type})")
    print(f"{'='*80}\n")
    print(f"Archive directory: {archive_dir}")

    # Create directory structure
    dirs = create_archive_structure(archive_dir)

    # ========================================================================
    # 1. METADATA
    # ========================================================================

    metadata = {
        'week': week,
        'year': year,
        'season_type': season_type,
        'archived_at': datetime.now().isoformat(),
        'archive_version': '1.0',
        'data_sources': {
            'fanduel_salaries': 'local CSV',
            'fantasypros': 'scraped',
            'draftkings_lines': 'scraped',
            'draftkings_odds': 'scraped',
        },
        'files': {}
    }

    # ========================================================================
    # 2. INPUT DATA (FanDuel salaries)
    # ========================================================================

    print("📁 Archiving input data...")

    # Find FanDuel CSV in data/input/
    input_dir = Path('data/input')
    fanduel_files = list(input_dir.glob('FanDuel-NFL-*.csv'))

    if fanduel_files:
        latest_fanduel = max(fanduel_files, key=lambda p: p.stat().st_mtime)
        dest = dirs['input'] / 'fanduel_salaries.csv'
        shutil.copy2(latest_fanduel, dest)
        print(f"  ✓ Copied: {latest_fanduel.name} -> {dest.name}")
        metadata['files']['fanduel_salaries'] = str(dest.relative_to(archive_dir))
    else:
        print(f"  ⚠️  No FanDuel CSV found in {input_dir}")

    # Note: actual_results.json added later after week completes
    print(f"  ℹ️  Add actual_results.json after week completes")

    # ========================================================================
    # 3. SCRAPED DATA (raw API responses)
    # ========================================================================

    print("\n🌐 Archiving scraped data...")

    # Check cache directory for scraped data
    cache_dir = Path('cache')

    if cache_dir.exists():
        # FantasyPros projections
        fp_files = list(cache_dir.glob('fantasypros_*.json'))
        if fp_files:
            latest_fp = max(fp_files, key=lambda p: p.stat().st_mtime)
            dest = dirs['scraped'] / 'fantasypros.json'
            shutil.copy2(latest_fp, dest)
            print(f"  ✓ Copied: {latest_fp.name}")
            metadata['files']['fantasypros'] = str(dest.relative_to(archive_dir))

        # DraftKings game lines
        dk_lines = list(cache_dir.glob('draftkings_lines_*.json'))
        if dk_lines:
            latest_dk = max(dk_lines, key=lambda p: p.stat().st_mtime)
            dest = dirs['scraped'] / 'draftkings_lines.json'
            shutil.copy2(latest_dk, dest)
            print(f"  ✓ Copied: {latest_dk.name}")
            metadata['files']['draftkings_lines'] = str(dest.relative_to(archive_dir))

        # DraftKings TD odds
        dk_odds = list(cache_dir.glob('draftkings_odds_*.json'))
        if dk_odds:
            latest_odds = max(dk_odds, key=lambda p: p.stat().st_mtime)
            dest = dirs['scraped'] / 'draftkings_odds.json'
            shutil.copy2(latest_odds, dest)
            print(f"  ✓ Copied: {latest_odds.name}")
            metadata['files']['draftkings_odds'] = str(dest.relative_to(archive_dir))
    else:
        print(f"  ⚠️  Cache directory not found: {cache_dir}")

    # ========================================================================
    # 4. INTERMEDIATE DATA (processed files)
    # ========================================================================

    print("\n⚙️  Archiving intermediate data...")

    intermediate_source = Path('data/intermediate')

    if intermediate_source.exists():
        files_to_copy = [
            'players_raw.csv',
            'players_integrated.csv',
            'game_lines.csv',
            'game_script_continuous.csv',
        ]

        for filename in files_to_copy:
            source = intermediate_source / filename
            if source.exists():
                dest = dirs['intermediate'] / filename
                shutil.copy2(source, dest)
                print(f"  ✓ Copied: {filename}")
                metadata['files'][filename.replace('.csv', '')] = str(dest.relative_to(archive_dir))
            else:
                print(f"  ⚠️  Not found: {filename}")
    else:
        print(f"  ⚠️  Intermediate directory not found: {intermediate_source}")

    # ========================================================================
    # 5. SAVE METADATA
    # ========================================================================

    metadata_path = dirs['root'] / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Archive complete: {archive_dir}")
    print(f"   Files archived: {len(metadata['files'])}")
    print(f"\n💡 To add actual results after week completes:")
    print(f"   cp data/input/YYYY_weekNN.json {dirs['input']}/actual_results.json")

    return archive_dir


# ============================================================================
# RESTORE ARCHIVED DATA
# ============================================================================

def restore_week(archive_dir: str, target_dir: str = 'data') -> dict:
    """
    Restore archived week data to working directories for replay.

    This copies archived data back to data/input and data/intermediate
    so you can run the pipeline as if it were that week.

    Args:
        archive_dir: Path to archived week directory
        target_dir: Root target directory (default: 'data')

    Returns:
        Metadata dict with week info
    """
    archive_path = Path(archive_dir)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Load metadata
    metadata_path = archive_path / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {archive_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"\n{'='*80}")
    print(f"RESTORING WEEK {metadata['week']}, {metadata['year']} ({metadata['season_type']})")
    print(f"{'='*80}\n")
    print(f"Archived: {metadata['archived_at']}")

    target_path = Path(target_dir)

    # Create target directories
    (target_path / 'input').mkdir(parents=True, exist_ok=True)
    (target_path / 'intermediate').mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # RESTORE INPUT FILES
    # ========================================================================

    print("\n📁 Restoring input data...")

    input_archive = archive_path / 'input'
    if input_archive.exists():
        for file in input_archive.glob('*'):
            dest = target_path / 'input' / file.name
            shutil.copy2(file, dest)
            print(f"  ✓ Restored: {file.name}")

    # ========================================================================
    # RESTORE INTERMEDIATE FILES
    # ========================================================================

    print("\n⚙️  Restoring intermediate data...")

    intermediate_archive = archive_path / 'intermediate'
    if intermediate_archive.exists():
        for file in intermediate_archive.glob('*'):
            dest = target_path / 'intermediate' / file.name
            shutil.copy2(file, dest)
            print(f"  ✓ Restored: {file.name}")

    # ========================================================================
    # NOTE ABOUT SCRAPED DATA
    # ========================================================================

    print("\n💡 Note: Raw scraped data is in archive/scraped/ for reference,")
    print("   but restored intermediate files are ready to use directly.")

    print(f"\n✅ Restore complete!")
    print(f"\n🚀 You can now run the optimizer with this historical data:")
    print(f"   python 3_run_optimizer.py")

    return metadata


# ============================================================================
# LIST ARCHIVES
# ============================================================================

def list_archives(archive_root: str = 'data/archive'):
    """List all available archives with summary info."""
    archive_path = Path(archive_root)

    if not archive_path.exists():
        print(f"No archives found in {archive_root}")
        return

    archives = sorted(archive_path.iterdir())

    if not archives:
        print(f"No archives found in {archive_root}")
        return

    print(f"\n{'='*80}")
    print(f"AVAILABLE ARCHIVES ({len(archives)})")
    print(f"{'='*80}\n")

    for archive_dir in archives:
        if not archive_dir.is_dir():
            continue

        metadata_file = archive_dir / 'metadata.json'

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            week = metadata.get('week', '?')
            year = metadata.get('year', '?')
            season_type = metadata.get('season_type', '?')
            archived_at = metadata.get('archived_at', '?')
            num_files = len(metadata.get('files', {}))

            # Check if has actual results
            has_results = (archive_dir / 'input' / 'actual_results.json').exists()
            results_marker = "✓ Results" if has_results else "✗ No results"

            print(f"{archive_dir.name:40s}  Week {week:2d} {year}  {season_type:4s}  {num_files:2d} files  {results_marker}")
        else:
            print(f"{archive_dir.name:40s}  (no metadata)")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Archive or restore weekly DFS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Archive current week
  python 6_archive_week.py --week 13 --year 2025

  # Archive with custom name
  python 6_archive_week.py --name "2025_week13_thanksgiving"

  # Restore archived week
  python 6_archive_week.py --restore data/archive/2025_week13_REG

  # List all archives
  python 6_archive_week.py --list
        """
    )

    parser.add_argument('--week', type=int, help='Week number (1-18)')
    parser.add_argument('--year', type=int, help='Year (e.g., 2025)')
    parser.add_argument('--season-type', default='REG', choices=['REG', 'POST'], help='Regular or postseason')
    parser.add_argument('--name', help='Custom archive name')
    parser.add_argument('--archive-root', default='data/archive', help='Archive root directory')

    parser.add_argument('--restore', metavar='ARCHIVE_DIR', help='Restore from archive directory')
    parser.add_argument('--list', action='store_true', help='List all available archives')

    args = parser.parse_args()

    # List archives
    if args.list:
        list_archives(args.archive_root)
        return

    # Restore archive
    if args.restore:
        restore_week(args.restore)
        return

    # Archive current week
    if not args.week or not args.year:
        parser.error("--week and --year required (or use --restore/--list)")

    archive_week(
        week=args.week,
        year=args.year,
        season_type=args.season_type,
        custom_name=args.name,
        archive_root=args.archive_root
    )


if __name__ == '__main__':
    main()
