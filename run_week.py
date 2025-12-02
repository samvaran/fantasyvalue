"""
Orchestration script to run the full DFS optimizer pipeline.

This script coordinates all 4 pipeline steps for current or historical weeks:
1. 1_fetch_data.py - Download current week data (skipped if data exists or for historical weeks)
2. 2_data_integration.py - Merge and normalize data sources (skipped if already done)
3. 3_run_optimizer.py - Generate and optimize lineups (always runs)
4. 4_backtest.py - Score lineups against actual results (if fanduel_results.json exists)

Current Week Behavior:
    - If no data exists: Fetches all data sources from scratch
    - If data exists: Uses existing data (use --fetch to re-download)
    - Skips Step 2 if intermediate/1_players.csv already exists

Historical Week Behavior:
    - Always uses existing data (cannot re-fetch historical data)
    - Skips Step 2 if intermediate/1_players.csv already exists
    - Runs optimizer and backtest (if results available)

Usage:
    # Analyze current week (auto-fetches if needed)
    python run_week.py

    # Current week with fresh data
    python run_week.py --fetch

    # Analyze historical week (uses existing data)
    python run_week.py --week 2025-12-01

    # Run with optimizer options
    python run_week.py --candidates 1000 --sims 10000 --fitness balanced

    # Quick test mode
    python run_week.py --quick-test
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================================
# WEEK DETECTION
# ============================================================================

def get_current_week() -> str:
    """
    Get current NFL week directory name (YYYY_MM_DD format).

    NFL weeks typically start on Tuesday and games are played on Sunday.
    This returns the Sunday date for the current week.

    Logic:
    - Monday-Saturday: Returns upcoming Sunday
    - Sunday before 10am PT: Returns today (current Sunday)
    - Sunday after 10am PT: Returns next Sunday

    Returns:
        Week directory name (e.g., "2025_12_01")
    """
    from datetime import timezone

    # Get current time in Pacific timezone
    # Pacific is UTC-8 (PST) or UTC-7 (PDT)
    # We'll use a simple UTC offset approach
    now_utc = datetime.now(timezone.utc)
    # Pacific time is UTC-8 hours (approximation - doesn't handle DST perfectly)
    pacific_offset = timedelta(hours=-8)
    now_pacific = now_utc + pacific_offset

    # Find next Sunday (or today if it's Sunday before 10am PT)
    days_until_sunday = (6 - now_pacific.weekday()) % 7
    if days_until_sunday == 0 and now_pacific.hour < 10:
        # If it's Sunday before 10am PT, use today
        sunday = now_pacific
    else:
        # Otherwise use next Sunday
        sunday = now_pacific + timedelta(days=days_until_sunday)

    return sunday.strftime('%Y_%m_%d')


def validate_week_format(week: str) -> bool:
    """
    Validate week directory name format (YYYY_MM_DD).

    Args:
        week: Week directory name

    Returns:
        True if valid format, False otherwise
    """
    try:
        datetime.strptime(week, '%Y_%m_%d')
        return True
    except ValueError:
        return False


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def print_step_header(step_num: int, step_name: str):
    """Print a formatted step header."""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*80}\n")


def run_step(script_name: str, args: list, step_name: str, optional: bool = False) -> bool:
    """
    Run a pipeline step and handle errors.

    Args:
        script_name: Name of script to run
        args: Command line arguments
        step_name: Human-readable step name
        optional: If True, don't fail if script errors

    Returns:
        True if successful, False if failed
    """
    try:
        result = subprocess.run([sys.executable, script_name] + args, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"\n⚠️  {step_name} failed (non-fatal): {e}")
            return False
        else:
            print(f"\n❌ {step_name} failed!")
            sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run DFS optimizer pipeline for current or historical weeks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current week (fetch fresh data)
  python run_week.py

  # Analyze historical week (use existing data)
  python run_week.py --week 2025-12-01

  # Re-fetch data for existing week
  python run_week.py --week 2025-12-01 --fetch

  # Quick test mode
  python run_week.py --quick-test

  # Full production run with custom settings
  python run_week.py --candidates 1000 --sims 10000 --fitness balanced
        """
    )

    # Week selection
    parser.add_argument(
        '--week',
        help='Week directory name (YYYY_MM_DD). If not provided, uses current week.'
    )
    parser.add_argument(
        '--fetch',
        action='store_true',
        help='Force re-fetch data for current week (ignored for historical weeks)'
    )

    # Optimizer options (passed through to 3_run_optimizer.py)
    parser.add_argument(
        '--candidates',
        type=int,
        help='Number of candidate lineups (default: 1000, quick-test: 50)'
    )
    parser.add_argument(
        '--sims',
        type=int,
        help='Simulations per lineup (default: 10000, quick-test: 1000)'
    )
    parser.add_argument(
        '--fitness',
        default='balanced',
        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
        help='Fitness function for optimization (default: balanced)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Max genetic refinement iterations (default: 5, quick-test: 2)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (50 candidates, 1000 sims, 2 iterations)'
    )
    parser.add_argument(
        '--processes',
        type=int,
        help='Number of parallel processes for simulations'
    )

    # Run selection (for resuming)
    parser.add_argument(
        '--run-name',
        help='Resume specific run (e.g., run_20251201_143022)'
    )

    args = parser.parse_args()

    # ========================================================================
    # DETERMINE WEEK
    # ========================================================================

    if args.week:
        if not validate_week_format(args.week):
            print(f"❌ Invalid week format: {args.week}")
            print("   Expected format: YYYY_MM_DD (e.g., 2025_12_01)")
            sys.exit(1)
        week = args.week
        is_current = False
    else:
        week = get_current_week()
        is_current = True

    week_dir = Path('data') / week
    inputs_dir = week_dir / 'inputs'

    print(f"\n{'='*80}")
    print(f"NFL DFS OPTIMIZER PIPELINE")
    print(f"{'='*80}")
    print(f"\nWeek: {week}")
    print(f"Type: {'Current week (will fetch fresh data)' if is_current else 'Historical week (using existing data)'}")
    print(f"Directory: {week_dir}")

    if args.quick_test:
        print(f"Mode: Quick test (50 candidates, 1000 sims, 2 iterations)")
    else:
        print(f"Mode: Production")

    # ========================================================================
    # STEP 1: FETCH DATA (conditional)
    # ========================================================================

    should_fetch = False
    should_skip_integration = False

    # CURRENT WEEK logic
    if is_current:
        if not inputs_dir.exists():
            # No data exists - fetch from scratch
            print(f"\n📥 Current week ({week}) - No existing data found")
            print(f"   Fetching all data sources from scratch...")
            should_fetch = True
        elif args.fetch:
            # User explicitly wants fresh data
            print(f"\n📥 Current week ({week}) - --force flag provided")
            print(f"   Re-fetching all data sources...")
            should_fetch = True
        else:
            # Data exists - use it
            print(f"\n📂 Current week ({week}) - Using existing data from {inputs_dir}")
            print(f"   ℹ️  To fetch fresh data, use: python run_week.py --fetch")
            print(f"   ℹ️  For granular control: python code/1_fetch_data.py --week-dir {week}")

            # Check if integration has already been done
            if (intermediate_dir / '1_players.csv').exists():
                print(f"\n✓ Data integration already complete - will skip Step 2")
                should_skip_integration = True

    # HISTORICAL WEEK logic
    else:
        if args.fetch:
            print(f"\n⚠️  --fetch flag ignored for historical week ({week})")
            print(f"   Historical data cannot be re-fetched (no longer available online)")

        if not inputs_dir.exists():
            print(f"\n❌ No input data found for historical week: {inputs_dir}")
            print(f"   Historical weeks require existing data (cannot fetch from APIs)")
            print(f"   Please ensure data exists in {inputs_dir}")
            sys.exit(1)

        print(f"\n📂 Historical week ({week}) - Using existing data from {inputs_dir}")

        # Check if integration has already been done
        if (intermediate_dir / '1_players.csv').exists():
            print(f"✓ Data integration already complete - will skip Step 2")
            should_skip_integration = True

    # Execute fetch if needed
    if should_fetch:
        print_step_header(1, "FETCH DATA")
        fetch_args = ['--week-dir', str(week)]
        if args.fetch:
            fetch_args.append('--force')
        run_step('code/1_fetch_data.py', fetch_args, 'Data fetch')
    else:
        print(f"\n⏭️  Skipping Step 1 (using existing data)")

    # Verify inputs exist
    if not inputs_dir.exists():
        print(f"\n❌ No input data found in {inputs_dir}")
        print("   Run with --fetch to download data, or add data manually.")
        sys.exit(1)

    # ========================================================================
    # STEP 2: DATA INTEGRATION
    # ========================================================================

    if should_skip_integration:
        print(f"\n⏭️  Skipping Step 2 (data already integrated)")
    else:
        print_step_header(2, "DATA INTEGRATION")
        integration_args = ['--week-dir', str(week)]
        run_step('code/2_data_integration.py', integration_args, 'Data integration')

    # Verify intermediate files exist
    intermediate_dir = week_dir / 'intermediate'
    players_file = intermediate_dir / '1_players.csv'

    if not players_file.exists():
        print(f"\n❌ Data integration did not produce expected output: {players_file}")
        sys.exit(1)

    # ========================================================================
    # STEP 3: RUN OPTIMIZER
    # ========================================================================

    print_step_header(3, "RUN OPTIMIZER")

    optimizer_args = ['--week-dir', str(week)]

    # Add optimizer options
    if args.quick_test:
        optimizer_args.append('--quick-test')
    else:
        if args.candidates:
            optimizer_args.extend(['--candidates', str(args.candidates)])
        if args.sims:
            optimizer_args.extend(['--sims', str(args.sims)])
        if args.iterations:
            optimizer_args.extend(['--iterations', str(args.iterations)])

    optimizer_args.extend(['--fitness', args.fitness])

    if args.processes:
        optimizer_args.extend(['--processes', str(args.processes)])

    if args.run_name:
        optimizer_args.extend(['--run-name', args.run_name])

    run_step('code/3_run_optimizer.py', optimizer_args, 'Optimizer')

    # Find latest run directory
    outputs_dir = week_dir / 'outputs'
    if not outputs_dir.exists():
        print(f"\n❌ No outputs directory found: {outputs_dir}")
        sys.exit(1)

    runs = sorted(outputs_dir.glob('run_*'))
    if not runs:
        print(f"\n❌ No run directories found in {outputs_dir}")
        sys.exit(1)

    latest_run = runs[-1]
    lineups_file = latest_run / '3_lineups.csv'

    if not lineups_file.exists():
        print(f"\n❌ Optimizer did not produce expected output: {lineups_file}")
        sys.exit(1)

    # ========================================================================
    # STEP 4: BACKTEST (optional, only if results available)
    # ========================================================================

    results_file = inputs_dir / 'fanduel_results.json'

    if results_file.exists():
        print_step_header(4, "BACKTEST")

        backtest_args = [
            '--week-dir', str(week),
            '--run-dir', latest_run.name
        ]

        run_step('code/4_backtest.py', backtest_args, 'Backtest', optional=True)
    else:
        print(f"\n⏭️  Skipping Step 4 (no actual results available)")
        print(f"\n📋 To run backtest after games complete:")
        print(f"   1. Go to https://www.fanduel.com/live")
        print(f"   2. Open Chrome DevTools (F12) → Network tab")
        print(f"   3. Filter by 'graphql'")
        print(f"   4. Scroll down the player list to load all players")
        print(f"   5. Find the latest 'graphql' request in the Network tab")
        print(f"   6. Right-click → Copy → Copy Response")
        print(f"   7. Save response to: {results_file}")
        print(f"   8. Run: python code/4_backtest.py --week-dir {week} --run-dir {latest_run.name}")

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*80}\n")

    print(f"📊 Results location:")
    print(f"   {lineups_file}\n")

    if results_file.exists():
        backtest_summary = latest_run / '7_backtest_summary.json'
        if backtest_summary.exists():
            print(f"📈 Backtest results:")
            print(f"   {backtest_summary}\n")

    print(f"💡 Next steps:")
    if is_current:
        if not results_file.exists():
            print(f"   1. Review optimized lineups in {lineups_file}")
            print(f"   2. Submit lineups to FanDuel")
            print(f"   3. After games complete, download results (see instructions above)")
            print(f"   4. Run backtest to evaluate performance")
        else:
            print(f"   1. Review backtest results in {latest_run / '7_backtest_summary.json'}")
            print(f"   2. Compare projected vs actual scores")
            print(f"   3. Use insights to refine future lineups")
    else:
        # Historical week
        if not results_file.exists():
            print(f"   1. Add FanDuel results to: {results_file}")
            print(f"   2. Run: python code/4_backtest.py --week-dir {week} --run-dir {latest_run.name}")
        else:
            print(f"   1. Review backtest analysis in {latest_run / '7_backtest_summary.json'}")
            print(f"   2. Compare projection accuracy and calibration")
            print(f"   3. Use historical insights for future optimizations")


if __name__ == '__main__':
    main()
