"""
Complete DFS Optimizer Orchestration Script

Runs all 3 phases with iterative refinement until convergence:
- Phase 1: Generate diverse candidates via MILP
- Phase 2: Evaluate via Monte Carlo simulation
- Phase 3: Refine via genetic algorithm (multiple iterations)

Saves progress after each step for resumability and real-time monitoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import sys

# Import configuration
try:
    from config import *
except ImportError:
    # Try 0_config.py if config.py doesn't exist
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "0_config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        # Import all config variables into global namespace
        for name in dir(config_module):
            if not name.startswith('_'):
                globals()[name] = getattr(config_module, name)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Using fallback defaults...")
        # Fallback defaults
        PLAYERS_INTEGRATED = 'data/intermediate/players_integrated.csv'
        GAME_SCRIPTS = 'data/intermediate/game_script_continuous.csv'
        OUTPUTS_DIR = 'outputs'
        DEFAULT_CANDIDATES = 1000
        QUICK_TEST_CANDIDATES = 50
        DEFAULT_SIMS = 10000
        QUICK_TEST_SIMS = 1000
        DEFAULT_MAX_GENERATIONS = 50
        QUICK_TEST_MAX_GENERATIONS = 30
        DEFAULT_CONVERGENCE_PATIENCE = 5
        DEFAULT_CONVERGENCE_THRESHOLD = 0.01
        DEFAULT_FITNESS = 'balanced'
        DEFAULT_PROCESSES = None

# Add optimizer to path
sys.path.insert(0, str(Path(__file__).parent / 'optimizer'))

from optimizer.generate_candidates import generate_candidates
from optimizer.evaluate_lineups import evaluate_candidates
from optimizer.optimize_genetic import optimize_genetic


class OptimizerOrchestrator:
    """Orchestrates the complete optimization pipeline with checkpointing."""

    def __init__(
        self,
        week_dir: str = None,
        run_name: str = None,
        config_params: dict = None
    ):
        # Week directory setup
        if week_dir:
            week_path = Path(week_dir)
            self.players_path = week_path / 'intermediate' / '1_players.csv'
            self.game_scripts_path = week_path / 'intermediate' / '2_game_scripts.csv'
            self.output_dir = week_path / 'outputs'
        else:
            # Fallback to old paths for backward compatibility
            self.players_path = Path(PLAYERS_INTEGRATED)
            self.game_scripts_path = Path(GAME_SCRIPTS)
            self.output_dir = Path(OUTPUTS_DIR)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create run-specific directory
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'run_{run_name}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Store config parameters for saving to config.json
        self.config_params = config_params or {}

        # State tracking
        self.state_file = self.run_dir / 'optimizer_state.json'
        self.state = self.load_state()

        print(f"Run directory: {self.run_dir}")

    def load_state(self):
        """Load optimizer state from JSON."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'phase_1_complete': False,
            'phase_2_complete': False,
            'iterations': [],
            'best_fitness_history': [],
            'convergence_count': 0,
            'total_time': 0
        }

    def save_state(self):
        """Save optimizer state to JSON."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def print_header(self, title):
        """Print formatted section header."""
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80 + "\n")

    def run_phase_1(self, n_candidates: int = 1000, force: bool = False):
        """Phase 1: Generate diverse candidate lineups."""
        candidates_file = self.run_dir / '1_candidates.csv'

        if self.state['phase_1_complete'] and not force and candidates_file.exists():
            print(f"Phase 1 already complete. Loading from {candidates_file}")
            return pd.read_csv(candidates_file)

        self.print_header("PHASE 1: CANDIDATE GENERATION")

        start_time = time.time()

        # Load data
        players_df = pd.read_csv(self.players_path)
        game_scripts_df = pd.read_csv(self.game_scripts_path)

        # Generate candidates (saves to temporary location first)
        temp_candidates_file = self.run_dir / 'candidates_temp.csv'
        candidates_df = generate_candidates(
            players_df=players_df,
            game_scripts_df=game_scripts_df,
            n_lineups=n_candidates,
            output_path=str(temp_candidates_file),
            verbose=True
        )

        # Format with position-based columns and save
        candidates_formatted = self.format_lineups_by_position(candidates_df)
        candidates_formatted.to_csv(candidates_file, index=False)

        # Clean up temp file
        if temp_candidates_file.exists():
            temp_candidates_file.unlink()

        elapsed = time.time() - start_time
        self.state['phase_1_complete'] = True
        self.state['phase_1_time'] = elapsed
        self.state['n_candidates'] = len(candidates_df)
        self.save_state()

        print(f"\nPhase 1 complete in {elapsed:.1f} seconds")
        return candidates_df

    def run_phase_2(self, n_sims: int = 10000, n_processes: int = None, force: bool = False):
        """Phase 2: Evaluate candidates via Monte Carlo."""
        candidates_file = self.run_dir / '1_candidates.csv'
        evaluations_file = self.run_dir / '2_simulations.csv'

        if self.state['phase_2_complete'] and not force and evaluations_file.exists():
            print(f"Phase 2 already complete. Loading from {evaluations_file}")
            return pd.read_csv(evaluations_file)

        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

        self.print_header("PHASE 2: MONTE CARLO EVALUATION")

        start_time = time.time()

        # Evaluate candidates (saves to temporary location first)
        temp_evaluations_file = self.run_dir / 'evaluations_temp.csv'
        evaluations_df = evaluate_candidates(
            candidates_path=str(candidates_file),
            players_path=self.players_path,
            game_scripts_path=self.game_scripts_path,
            n_sims=n_sims,
            n_processes=n_processes,
            output_path=str(temp_evaluations_file),
            verbose=True
        )

        # Format with position-based columns and save
        evaluations_formatted = self.format_lineups_by_position(evaluations_df)
        evaluations_formatted.to_csv(evaluations_file, index=False)

        # Clean up temp file
        if temp_evaluations_file.exists():
            temp_evaluations_file.unlink()

        elapsed = time.time() - start_time
        self.state['phase_2_complete'] = True
        self.state['phase_2_time'] = elapsed
        self.state['n_simulations'] = n_sims
        self.save_state()

        # Also save to lineups_after_phase2.csv for backwards compatibility
        evaluations_formatted.to_csv(self.run_dir / 'lineups_after_phase2.csv', index=False)

        print(f"\nPhase 2 complete in {elapsed / 60:.1f} minutes")
        print(f"All {len(evaluations_df)} lineups saved to: {evaluations_file}")

        return evaluations_df

    def run_phase_3_iteration(
        self,
        iteration: int,
        evaluations_df: pd.DataFrame,
        fitness_function: str = 'balanced',
        n_offspring: int = 100,
        n_generations: int = 20,
        n_sims: int = 10000,
        n_processes: int = None
    ):
        """Phase 3: Single iteration of genetic optimization."""
        self.print_header(f"PHASE 3: GENETIC OPTIMIZATION - ITERATION {iteration}")

        start_time = time.time()

        # Save current evaluations as input for this iteration
        iteration_dir = self.run_dir / f'iteration_{iteration}'
        iteration_dir.mkdir(parents=True, exist_ok=True)

        input_file = iteration_dir / 'input_evaluations.csv'
        evaluations_df.to_csv(input_file, index=False)

        # Run genetic optimization
        optimal_df = optimize_genetic(
            evaluations_path=str(input_file),
            players_path=self.players_path,
            game_scripts_path=self.game_scripts_path,
            fitness_function=fitness_function,
            population_size=100,
            n_offspring=n_offspring,
            max_generations=n_generations,
            n_sims=n_sims,
            n_processes=n_processes,
            output_path=str(iteration_dir / 'optimal_lineups.csv'),
            verbose=True
        )

        elapsed = time.time() - start_time

        # Track iteration results
        iteration_result = {
            'iteration': iteration,
            'time': elapsed,
            'best_fitness': float(optimal_df.iloc[0]['fitness']),
            'best_median': float(optimal_df.iloc[0]['median']),
            'best_p90': float(optimal_df.iloc[0]['p90']),
            'n_generations': n_generations,
            'timestamp': datetime.now().isoformat()
        }

        self.state['iterations'].append(iteration_result)
        self.state['best_fitness_history'].append(iteration_result['best_fitness'])
        self.save_state()

        # Save all lineups from this iteration with position-based formatting
        all_lineups_formatted = self.format_lineups_by_position(optimal_df)
        all_lineups_formatted.to_csv(iteration_dir / f'lineups_iteration_{iteration}.csv', index=False)

        print(f"\nIteration {iteration} complete in {elapsed / 60:.1f} minutes")
        print(f"  Best fitness: {iteration_result['best_fitness']:.2f}")
        print(f"  Best median: {iteration_result['best_median']:.2f}")
        print(f"  All {len(optimal_df)} lineups saved to: {iteration_dir / f'lineups_iteration_{iteration}.csv'}")

        return optimal_df, iteration_result

    def check_convergence(self, patience: int = 3, threshold: float = 0.01):
        """Check if optimization has converged."""
        if len(self.state['best_fitness_history']) < patience + 1:
            return False

        recent = self.state['best_fitness_history'][-patience-1:]
        improvements = [
            abs(recent[i] - recent[i-1]) / abs(recent[i-1]) if recent[i-1] != 0 else 0
            for i in range(1, len(recent))
        ]

        # Converged if all recent improvements are below threshold
        converged = all(imp < threshold for imp in improvements)

        if converged:
            self.state['convergence_count'] += 1
        else:
            self.state['convergence_count'] = 0

        return self.state['convergence_count'] >= patience

    def format_lineups_by_position(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        """Reformat lineups to have separate columns for each position.

        Order: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
        """
        # Load player data to get positions
        players_df = pd.read_csv(self.players_path)
        player_positions = dict(zip(players_df['name'].str.lower(), players_df['position']))

        formatted_rows = []

        for _, row in lineups_df.iterrows():
            # Parse player_ids (comma-separated)
            player_names = [p.strip() for p in row['player_ids'].split(',')]

            # Group players by position
            qbs = []
            rbs = []
            wrs = []
            tes = []
            defs = []

            for player in player_names:
                pos = player_positions.get(player.lower(), 'UNKNOWN')
                if pos == 'QB':
                    qbs.append(player)
                elif pos == 'RB':
                    rbs.append(player)
                elif pos == 'WR':
                    wrs.append(player)
                elif pos == 'TE':
                    tes.append(player)
                elif pos in ('DEF', 'D'):  # Handle both 'DEF' and 'D'
                    defs.append(player)

            # Build row in specified order: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
            formatted_row = {
                'QB': qbs[0] if len(qbs) > 0 else '',
                'RB1': rbs[0] if len(rbs) > 0 else '',
                'RB2': rbs[1] if len(rbs) > 1 else '',
                'WR1': wrs[0] if len(wrs) > 0 else '',
                'WR2': wrs[1] if len(wrs) > 1 else '',
                'WR3': wrs[2] if len(wrs) > 2 else '',
                'TE': tes[0] if len(tes) > 0 else '',
                'DEF': defs[0] if len(defs) > 0 else '',
            }

            # Determine FLEX (extra RB, WR, or TE)
            flex = ''
            if len(rbs) > 2:
                flex = rbs[2]
            elif len(wrs) > 3:
                flex = wrs[3]
            elif len(tes) > 1:
                flex = tes[1]
            formatted_row['FLEX'] = flex

            # Add ALL other columns from the original row (preserves everything)
            for col in lineups_df.columns:
                if col not in ['player_ids'] and col not in formatted_row:
                    formatted_row[col] = row[col]

            # Also keep player_ids for backwards compatibility
            if 'player_ids' in row:
                formatted_row['player_ids'] = row['player_ids']

            formatted_rows.append(formatted_row)

        # Create new dataframe with position columns first, then stats
        position_cols = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF']
        priority_stats = ['lineup_id', 'player_ids', 'tier', 'temperature', 'total_projection',
                         'milp_projection', 'total_salary',
                         'mean', 'median', 'p10', 'p90', 'std', 'skewness', 'fitness']

        formatted_df = pd.DataFrame(formatted_rows)

        # Build column order: positions first, then priority stats, then any remaining columns
        column_order = position_cols.copy()
        for col in priority_stats:
            if col in formatted_df.columns and col not in column_order:
                column_order.append(col)

        # Add any remaining columns not in the priority list
        for col in formatted_df.columns:
            if col not in column_order:
                column_order.append(col)

        return formatted_df[column_order]

    def run_full_pipeline(
        self,
        n_candidates: int = 1000,
        n_sims_phase2: int = 10000,
        n_sims_phase3: int = 10000,
        fitness_function: str = 'balanced',
        max_generations: int = 50,
        convergence_patience: int = 5,
        convergence_threshold: float = 0.01,
        n_processes: int = None,
        force_restart: bool = False
    ):
        """Run complete optimization pipeline with single genetic algorithm run."""
        self.print_header("DFS LINEUP OPTIMIZER - FULL PIPELINE")

        # Save configuration to 0_config.json
        config = {
            'timestamp': datetime.now().isoformat(),
            'week': str(self.output_dir.parent.name) if self.output_dir.parent.name.startswith('20') else 'unknown',
            'optimizer': {
                'candidates': n_candidates,
                'simulations': n_sims_phase2,
                'fitness': fitness_function,
                'max_generations': max_generations,
                'convergence_patience': convergence_patience,
                'convergence_threshold': convergence_threshold,
                'processes': n_processes or 'auto'
            },
            'data_sources': {
                'players': str(self.players_path),
                'game_scripts': str(self.game_scripts_path)
            },
            'completed': False,
            'phases_completed': []
        }
        config.update(self.config_params)  # Add any extra params

        config_file = self.run_dir / '0_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration:")
        print(f"  Candidates: {n_candidates}")
        print(f"  Phase 2 simulations: {n_sims_phase2}")
        print(f"  Phase 3 simulations: {n_sims_phase3}")
        print(f"  Fitness function: {fitness_function}")
        print(f"  Max generations: {max_generations}")
        print(f"  Convergence patience: {convergence_patience}")
        print(f"  Convergence threshold: {convergence_threshold * 100}%")
        print(f"  Parallel processes: {n_processes or 'auto'}")

        pipeline_start = time.time()

        # Phase 1: Generate candidates
        candidates_df = self.run_phase_1(
            n_candidates=n_candidates,
            force=force_restart
        )

        # Phase 2: Initial evaluation
        evaluations_df = self.run_phase_2(
            n_sims=n_sims_phase2,
            n_processes=n_processes,
            force=force_restart
        )

        # Phase 3: Genetic optimization (single run with early stopping)
        optimal_df, _ = self.run_phase_3_iteration(
            iteration=1,
            evaluations_df=evaluations_df,
            fitness_function=fitness_function,
            n_sims=n_sims_phase3,
            n_processes=n_processes,
            n_generations=max_generations
        )

        # Results already stored by run_phase_3_iteration()

        # Final summary
        pipeline_elapsed = time.time() - pipeline_start
        self.state['total_time'] = pipeline_elapsed
        self.save_state()

        self.print_header("OPTIMIZATION COMPLETE")

        print(f"Total time: {pipeline_elapsed / 60:.1f} minutes")
        print(f"Total iterations: {len(self.state['iterations'])}")
        print(f"Best fitness achieved: {max(self.state['best_fitness_history']):.2f}")

        # Save final results
        final_results = {
            'run_name': self.run_dir.name,
            'total_time_minutes': pipeline_elapsed / 60,
            'total_iterations': len(self.state['iterations']),
            'best_fitness': max(self.state['best_fitness_history']),
            'fitness_history': self.state['best_fitness_history'],
            'iterations': self.state['iterations']
        }

        with open(self.run_dir / '4_summary.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        # Update config.json with completion status
        config_file = self.run_dir / '0_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            config['completed'] = True
            config['phases_completed'] = ['candidates', 'simulations'] + [f'genetic_{i+1}' for i in range(len(self.state['iterations']))]
            config['total_time_minutes'] = pipeline_elapsed / 60
            config['best_fitness'] = max(self.state['best_fitness_history'])
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

        # Copy best lineups to main outputs
        best_iteration = max(self.state['iterations'], key=lambda x: x['best_fitness'])
        best_iteration_dir = self.run_dir / f"iteration_{best_iteration['iteration']}"
        best_lineups_file = best_iteration_dir / f"lineups_iteration_{best_iteration['iteration']}.csv"

        if best_lineups_file.exists():
            best_lineups = pd.read_csv(best_lineups_file)

            # File is already formatted, just copy it to BEST_LINEUPS.csv
            best_lineups.to_csv(self.run_dir / '3_lineups.csv', index=False)
            print(f"\nBest lineups saved to: {self.run_dir / '3_lineups.csv'}")

            print(f"\nTop 5 Best Lineups:")
            for i, row in best_lineups.head(5).iterrows():
                print(f"  {i+1}. Fitness: {row.get('fitness', row.get('median')):.2f}, "
                      f"Median: {row['median']:.2f}, P90: {row['p90']:.2f}")

        print(f"\nAll results saved in: {self.run_dir}")
        print(f"  - 0_config.json: Run configuration")
        print(f"  - 1_candidates.csv: All generated candidates")
        print(f"  - 2_simulations.csv: Monte Carlo evaluations")
        print(f"  - 3_lineups.csv: Best lineups (TOP N)")
        print(f"  - 4_summary.json: Complete run statistics")
        print(f"  - optimizer_state.json: State for resumption")
        print(f"  - iteration_N/: Results from each iteration")


def main():
    parser = argparse.ArgumentParser(
        description="DFS Lineup Optimizer - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (50 candidates, 1000 sims, 30 generations)
  python run_optimizer.py --quick-test

  # Full production run (1000 candidates, 10k sims, 50 generations)
  python run_optimizer.py --candidates 1000 --sims 10000

  # Custom generations with early stopping
  python run_optimizer.py --max-generations 100 --patience 5

  # Aggressive fitness (high upside)
  python run_optimizer.py --fitness tournament
        """
    )

    # Pipeline configuration
    parser.add_argument('--candidates', type=int, default=1000,
                        help='Number of candidate lineups to generate (default: 1000)')
    parser.add_argument('--sims', type=int, default=10000,
                        help='Number of Monte Carlo simulations (default: 10000)')
    parser.add_argument('--max-generations', type=int, default=DEFAULT_MAX_GENERATIONS,
                        help=f'Maximum generations for genetic algorithm (default: {DEFAULT_MAX_GENERATIONS})')
    parser.add_argument('--fitness', default='balanced',
                        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                        help='Fitness function (default: balanced)')

    # Convergence settings
    parser.add_argument('--patience', type=int, default=DEFAULT_CONVERGENCE_PATIENCE,
                        help=f'Convergence patience for early stopping (default: {DEFAULT_CONVERGENCE_PATIENCE})')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Convergence threshold in %% (default: 0.01 = 1%%)')

    # Performance
    parser.add_argument('--processes', type=int, default=DEFAULT_PROCESSES,
                        help=f'Number of parallel processes (default: auto-detect)')

    # Data location
    parser.add_argument('--week-dir', type=str, default=None,
                        help='Week directory (e.g., data/2025_12_01). If not provided, uses legacy paths.')

    # Run management
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--force-restart', action='store_true',
                        help='Force restart from Phase 1 even if state exists')

    # Convenience flags
    parser.add_argument('--quick-test', action='store_true',
                        help=f'Quick test: {QUICK_TEST_CANDIDATES} candidates, {QUICK_TEST_SIMS} sims, {QUICK_TEST_MAX_GENERATIONS} generations')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.candidates = QUICK_TEST_CANDIDATES
        args.sims = QUICK_TEST_SIMS
        args.max_generations = QUICK_TEST_MAX_GENERATIONS
        print("QUICK TEST MODE")

    # Create orchestrator
    orchestrator = OptimizerOrchestrator(
        week_dir=args.week_dir,
        run_name=args.run_name
    )

    # Run pipeline
    orchestrator.run_full_pipeline(
        n_candidates=args.candidates,
        n_sims_phase2=args.sims,
        n_sims_phase3=args.sims,
        fitness_function=args.fitness,
        max_generations=args.max_generations,
        convergence_patience=args.patience,
        convergence_threshold=args.threshold,
        n_processes=args.processes,
        force_restart=args.force_restart
    )


if __name__ == '__main__':
    main()
