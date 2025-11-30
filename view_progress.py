"""
View real-time progress of optimizer run.

Displays current state, iteration history, and best lineups found so far.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def view_progress(run_dir: Path):
    """Display progress of optimizer run."""
    state_file = run_dir / 'optimizer_state.json'

    if not state_file.exists():
        print(f"No state file found in {run_dir}")
        return

    # Load state
    with open(state_file, 'r') as f:
        state = json.load(f)

    print("=" * 80)
    print(f"OPTIMIZER PROGRESS: {run_dir.name}".center(80))
    print("=" * 80)

    # Phase completion status
    print("\nPhase Status:")
    print(f"  Phase 1 (Candidates):   {'✓ Complete' if state['phase_1_complete'] else '✗ Pending'}")
    if state['phase_1_complete']:
        print(f"    - Candidates: {state.get('n_candidates', 'N/A')}")
        print(f"    - Time: {format_time(state.get('phase_1_time', 0))}")

    print(f"  Phase 2 (Evaluation):   {'✓ Complete' if state['phase_2_complete'] else '✗ Pending'}")
    if state['phase_2_complete']:
        print(f"    - Simulations: {state.get('n_simulations', 'N/A')}")
        print(f"    - Time: {format_time(state.get('phase_2_time', 0))}")

    # Iteration history
    if len(state['iterations']) > 0:
        print(f"  Phase 3 (Refinement):   {len(state['iterations'])} iterations complete")

        print("\n  Iteration History:")
        print("  " + "-" * 76)
        print(f"  {'Iter':<6} {'Time':<8} {'Best Fitness':<14} {'Median':<10} {'P90':<10} {'Gens':<6}")
        print("  " + "-" * 76)

        for it in state['iterations']:
            print(f"  {it['iteration']:<6} "
                  f"{format_time(it['time']):<8} "
                  f"{it['best_fitness']:<14.2f} "
                  f"{it['best_median']:<10.2f} "
                  f"{it['best_p90']:<10.2f} "
                  f"{it['n_generations']:<6}")

        # Fitness progression
        print("\n  Fitness Progression:")
        history = state['best_fitness_history']
        if len(history) > 1:
            improvements = [
                ((history[i] - history[i-1]) / history[i-1] * 100) if history[i-1] != 0 else 0
                for i in range(1, len(history))
            ]
            print(f"    Initial:  {history[0]:.2f}")
            print(f"    Current:  {history[-1]:.2f}")
            print(f"    Best:     {max(history):.2f}")
            print(f"    Avg improvement: {sum(improvements) / len(improvements):.2f}%")

            # Check if converging
            if len(improvements) >= 3:
                recent_improvements = improvements[-3:]
                avg_recent = sum(recent_improvements) / len(recent_improvements)
                if avg_recent < 1.0:
                    print(f"    Status: Converging (recent avg: {avg_recent:.2f}%)")
                else:
                    print(f"    Status: Still improving (recent avg: {avg_recent:.2f}%)")

    # Best lineups found
    best_lineups_file = run_dir / 'BEST_LINEUPS.csv'
    if best_lineups_file.exists():
        print("\n  Best Lineups Found:")
        print("  " + "-" * 76)
        lineups = pd.read_csv(best_lineups_file)
        for i, row in lineups.head(5).iterrows():
            fitness_col = 'fitness' if 'fitness' in row else 'median'
            print(f"  {i+1}. Fitness: {row.get(fitness_col, 0):.2f}, "
                  f"Median: {row['median']:.2f}, "
                  f"P90: {row['p90']:.2f}, "
                  f"P10: {row['p10']:.2f}")
        print(f"\n  Full lineups: {best_lineups_file}")

    # Total time
    if state.get('total_time', 0) > 0:
        print(f"\nTotal Time: {format_time(state['total_time'])}")

    # Files created
    print(f"\nOutput Directory: {run_dir}")
    print("  Files created:")
    for file in sorted(run_dir.glob('*.csv')):
        size_kb = file.stat().st_size / 1024
        print(f"    - {file.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 80)


def list_runs(output_dir: Path):
    """List all optimizer runs."""
    runs = sorted(output_dir.glob('run_*'), key=lambda x: x.name, reverse=True)

    if not runs:
        print("No optimizer runs found.")
        return

    print("=" * 80)
    print("AVAILABLE OPTIMIZER RUNS")
    print("=" * 80)

    for run_dir in runs[:10]:  # Show last 10 runs
        state_file = run_dir / 'optimizer_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)

            status = []
            if state['phase_1_complete']:
                status.append(f"Phase1✓")
            if state['phase_2_complete']:
                status.append(f"Phase2✓")
            if len(state['iterations']) > 0:
                status.append(f"Phase3({len(state['iterations'])}it)")

            best_fitness = max(state['best_fitness_history']) if state['best_fitness_history'] else 0

            print(f"\n  {run_dir.name}")
            print(f"    Status: {' | '.join(status) if status else 'In progress'}")
            if best_fitness > 0:
                print(f"    Best fitness: {best_fitness:.2f}")
            print(f"    Location: {run_dir}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="View optimizer progress")
    parser.add_argument('--run-name', type=str, default=None,
                        help='Specific run to view (default: latest)')
    parser.add_argument('--list', action='store_true',
                        help='List all available runs')
    parser.add_argument('--output-dir', default='outputs',
                        help='Output directory (default: outputs)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.list:
        list_runs(output_dir)
        return

    # Find run directory
    if args.run_name:
        if args.run_name.startswith('run_'):
            run_dir = output_dir / args.run_name
        else:
            run_dir = output_dir / f'run_{args.run_name}'
    else:
        # Get latest run
        runs = sorted(output_dir.glob('run_*'), key=lambda x: x.name, reverse=True)
        if not runs:
            print("No optimizer runs found.")
            print("Run the optimizer first: python run_optimizer.py")
            return
        run_dir = runs[0]

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    view_progress(run_dir)


if __name__ == '__main__':
    main()
