"""
Orchestration script to run the full DFS optimizer pipeline.

This script coordinates all 4 pipeline steps for current or historical weeks:
1. 1_fetch_data.py - Download current week data (skipped if data exists or for historical weeks)
2. 2_data_integration.py - Merge and normalize data sources (skipped if already done)
3. 3_run_optimizer.py - Generate and optimize lineups (skipped if results already exist)
4. 4_backtest.py - Score lineups against actual results (if fanduel_results.json exists)

Current Week Behavior:
    - If no data exists: Fetches all data sources from scratch
    - If data exists: Uses existing data (use --fetch to re-download)
    - Skips Step 2 if intermediate/1_players.csv already exists
    - Skips Step 3 if fanduel_results.json exists (games already complete)

Historical Week Behavior:
    - Always uses existing data (cannot re-fetch historical data)
    - Skips Step 2 if intermediate/1_players.csv already exists
    - Skips Step 3 if fanduel_results.json exists
    - Runs backtest if results available

Usage:
    # Analyze current week (auto-fetches if needed)
    python run_week.py

    # Current week with fresh data
    python run_week.py --fetch

    # Analyze historical week (uses existing data)
    python run_week.py --week 2025-12-01

    # Re-run integration and optimizer (even if outputs exist)
    python run_week.py --week 2025-12-01 --rerun

    # Run with optimizer options
    python run_week.py --candidates 1000 --sims 10000 --fitness balanced

    # Re-run backtest only (skip steps 1-3)
    python run_week.py --week 2025-12-01 --backtest-only

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


def normalize_week_format(week: str) -> str:
    """
    Normalize week format to YYYY_MM_DD.

    Accepts both underscore (2025_12_01) and hyphen (2025-12-01) formats.

    Args:
        week: Week directory name in either format

    Returns:
        Normalized week string (YYYY_MM_DD) or None if invalid
    """
    # Normalize hyphens to underscores
    normalized = week.replace('-', '_')

    try:
        datetime.strptime(normalized, '%Y_%m_%d')
        return normalized
    except ValueError:
        return None


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
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Force re-run steps 2 (integration) and 3 (optimizer) even if outputs exist'
    )

    # Optimizer options (passed through to 3_run_optimizer.py)
    parser.add_argument(
        '--candidates',
        type=int,
        help='Number of candidate lineups (uses config.py default)'
    )
    parser.add_argument(
        '--sims',
        type=int,
        help='Simulations per lineup (uses config.py default)'
    )
    parser.add_argument(
        '--fitness',
        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
        help='Fitness function for optimization (uses config.py default)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Max genetic refinement iterations (uses config.py default)'
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

    # Backtest only mode
    parser.add_argument(
        '--backtest-only',
        action='store_true',
        help='Skip steps 1-3 and only run backtest (step 4)'
    )

    args = parser.parse_args()

    # ========================================================================
    # DETERMINE WEEK
    # ========================================================================

    if args.week:
        week = normalize_week_format(args.week)
        if week is None:
            print(f"❌ Invalid week format: {args.week}")
            print("   Expected format: YYYY_MM_DD or YYYY-MM-DD (e.g., 2025_12_01 or 2025-12-01)")
            sys.exit(1)
        is_current = False
    else:
        week = get_current_week()
        is_current = True

    week_dir = Path('data') / week
    inputs_dir = week_dir / 'inputs'
    intermediate_dir = week_dir / 'intermediate'

    print(f"\n{'='*80}")
    print(f"NFL DFS OPTIMIZER PIPELINE")
    print(f"{'='*80}")
    print(f"\nWeek: {week}")
    print(f"Type: {'Current week (will fetch fresh data)' if is_current else 'Historical week (using existing data)'}")
    print(f"Directory: {week_dir}")

    print(f"Mode: Production")

    if args.rerun:
        print(f"Rerun: Yes (forcing steps 2 and 3 to re-run)")

    if args.backtest_only:
        print(f"Mode: Backtest only (skipping steps 1-3)")

    # ========================================================================
    # CHECK FOR ACTUAL RESULTS (needed for both modes)
    # ========================================================================

    actual_players_csv = intermediate_dir / '4_actual_players.csv'
    results_csv = inputs_dir / 'fanduel_results.csv'  # Simple manual format
    results_json = inputs_dir / 'fanduel_results.json'
    has_actual_results = actual_players_csv.exists() or results_csv.exists() or results_json.exists()
    outputs_dir = week_dir / 'outputs'

    # ========================================================================
    # BACKTEST ONLY MODE - Skip to step 4
    # ========================================================================

    if args.backtest_only:
        if not has_actual_results:
            print(f"\n❌ Cannot run backtest: no actual results found")
            print(f"   Expected one of:")
            print(f"     - {actual_players_csv}")
            print(f"     - {results_csv} (simple: name,actual_points)")
            print(f"     - {results_json}")
            sys.exit(1)

        # Find run directory
        if not outputs_dir.exists() or not list(outputs_dir.glob('run_*')):
            print(f"\n❌ No previous optimizer runs found in {outputs_dir}")
            print(f"   Cannot run backtest without lineups.")
            sys.exit(1)

        runs = sorted(outputs_dir.glob('run_*'))
        latest_run = runs[-1]
        lineups_file = latest_run / '3_lineups.csv'

        if not lineups_file.exists():
            print(f"\n❌ No lineups found in {latest_run}")
            sys.exit(1)

        print(f"\n📂 Using run: {latest_run.name}")

        # Jump directly to backtest
        print_step_header(4, "BACKTEST")

        backtest_args = [
            '--week-dir', str(week_dir),
            '--run-dir', latest_run.name
        ]

        run_step('code/4_backtest.py', backtest_args, 'Backtest', optional=True)

        # Quick completion message
        print(f"\n{'='*80}")
        print("✅ BACKTEST COMPLETE!")
        print(f"{'='*80}\n")

        backtest_summary = latest_run / '7_backtest_summary.json'
        if backtest_summary.exists():
            print(f"📈 Results: {backtest_summary}")

        sys.exit(0)

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
            print(f"   ℹ️  For granular control: python code/1_fetch_data.py --week-dir {week_dir}")

            # Check if integration has already been done (skip unless --rerun)
            if (intermediate_dir / '1_players.csv').exists() and not args.rerun:
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

        # Check if integration has already been done (skip unless --rerun)
        if (intermediate_dir / '1_players.csv').exists() and not args.rerun:
            print(f"✓ Data integration already complete - will skip Step 2")
            should_skip_integration = True

    # Execute fetch if needed
    if should_fetch:
        print_step_header(1, "FETCH DATA")
        fetch_args = ['--week-dir', str(week_dir)]
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
        integration_args = ['--week-dir', str(week_dir)]
        run_step('code/2_data_integration.py', integration_args, 'Data integration')

    # Verify intermediate files exist
    players_file = intermediate_dir / '1_players.csv'

    if not players_file.exists():
        print(f"\n❌ Data integration did not produce expected output: {players_file}")
        sys.exit(1)

    # ========================================================================
    # STEP 3: RUN OPTIMIZER (skip if results already exist)
    # ========================================================================

    # Note: actual_players_csv, results_json, has_actual_results, outputs_dir
    # are already defined above in the "CHECK FOR ACTUAL RESULTS" section

    if has_actual_results and not args.rerun:
        # Results exist - skip optimizer and just find existing run for backtest
        print(f"\n⏭️  Skipping Step 3 (actual results already exist)")
        print(f"   Games are complete - no need to generate new lineups")
        print(f"   To re-run optimizer anyway: python run_week.py --week {week} --rerun")

        # Find latest run directory for backtest
        if not outputs_dir.exists() or not list(outputs_dir.glob('run_*')):
            print(f"\n❌ No previous optimizer runs found in {outputs_dir}")
            print(f"   Cannot run backtest without lineups.")
            print(f"   Run optimizer first: python code/3_run_optimizer.py --week-dir {week_dir}")
            sys.exit(1)

        runs = sorted(outputs_dir.glob('run_*'))
        latest_run = runs[-1]
        lineups_file = latest_run / '3_lineups.csv'

        if not lineups_file.exists():
            print(f"\n❌ No lineups found in {latest_run}")
            sys.exit(1)

        print(f"   Using existing run: {latest_run.name}")
    else:
        # No results yet - run optimizer
        print_step_header(3, "RUN OPTIMIZER")

        optimizer_args = ['--week-dir', str(week_dir)]

        # Add optimizer options (only if explicitly provided, otherwise use config.py defaults)
        if args.candidates:
            optimizer_args.extend(['--candidates', str(args.candidates)])
        if args.sims:
            optimizer_args.extend(['--sims', str(args.sims)])
        if args.iterations:
            optimizer_args.extend(['--iterations', str(args.iterations)])
        if args.fitness:
            optimizer_args.extend(['--fitness', args.fitness])
        if args.processes:
            optimizer_args.extend(['--processes', str(args.processes)])

        if args.run_name:
            optimizer_args.extend(['--run-name', args.run_name])

        run_step('code/3_run_optimizer.py', optimizer_args, 'Optimizer')

        # Find latest run directory
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
    # STEP 4: BACKTEST (only if results available)
    # ========================================================================

    if has_actual_results:
        print_step_header(4, "BACKTEST")

        backtest_args = [
            '--week-dir', str(week_dir),
            '--run-dir', latest_run.name
        ]

        run_step('code/4_backtest.py', backtest_args, 'Backtest', optional=True)
    else:
        print(f"\n⏭️  Skipping Step 4 (no actual results available)")
        print(f"\n📋 To run backtest after games complete, use ONE of these methods:")
        print(f"\n   Option A: Simple CSV (easiest for manual entry)")
        print(f"   1. Create: {results_csv}")
        print(f"   2. Format: name,actual_points")
        print(f"   3. Example rows:")
        print(f"      Josh Allen,25.5")
        print(f"      Derrick Henry,18.2")
        print(f"   4. Run: python run_week.py --week {week} --backtest-only")
        print(f"\n   Option B: FanDuel JSON (full data)")
        print(f"   1. Go to https://www.fanduel.com/live")
        print(f"   2. Open Chrome DevTools (F12) → Network tab")
        print(f"   3. Filter by 'graphql'")
        print(f"   4. Scroll down the player list to load all players")
        print(f"   5. Find the latest 'graphql' request in the Network tab")
        print(f"   6. Right-click → Copy → Copy Response")
        print(f"   7. Save response to: {results_json}")
        print(f"   8. Run: python run_week.py --week {week} --backtest-only")

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*80}\n")

    print(f"📊 Results location:")
    print(f"   {lineups_file}\n")

    if has_actual_results:
        backtest_summary = latest_run / '7_backtest_summary.json'
        if backtest_summary.exists():
            print(f"📈 Backtest results:")
            print(f"   {backtest_summary}\n")

    print(f"💡 Next steps:")
    if is_current:
        if not has_actual_results:
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
        if not has_actual_results:
            print(f"   1. Add FanDuel results to: {results_json}")
            print(f"   2. Run: python code/4_backtest.py --week-dir {week_dir} --run-dir {latest_run.name}")
        else:
            print(f"   1. Review backtest analysis in {latest_run / '7_backtest_summary.json'}")
            print(f"   2. Compare projection accuracy and calibration")
            print(f"   3. Use historical insights for future optimizations")


if __name__ == '__main__':
    main()
