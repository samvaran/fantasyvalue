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
from config import *

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

        # State tracking (consolidated into results.json)
        self.state_file = self.run_dir / 'results.json'
        self.state = self.load_state()

        print(f"Run directory: {self.run_dir}")

    def load_state(self):
        """Load optimizer state from results.json."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                # Extract state from results.json structure
                return data.get('state', {
                    'phase_1_complete': False,
                    'phase_2_complete': False,
                    'iterations': [],
                    'best_fitness_history': [],
                    'convergence_count': 0,
                    'total_time': 0
                })
        return {
            'phase_1_complete': False,
            'phase_2_complete': False,
            'iterations': [],
            'best_fitness_history': [],
            'convergence_count': 0,
            'total_time': 0
        }

    def save_state(self):
        """Save optimizer state to results.json."""
        # Build results structure with state nested inside
        results = {
            'run_name': self.run_dir.name,
            'state': self.state,
            'summary': {
                'total_iterations': len(self.state['iterations']),
                'best_fitness': max(self.state['best_fitness_history']) if self.state['best_fitness_history'] else 0,
                'total_time_minutes': self.state.get('total_time', 0) / 60
            }
        }
        with open(self.state_file, 'w') as f:
            json.dump(results, f, indent=2)

    def print_header(self, title):
        """Print formatted section header."""
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80 + "\n")

    def run_phase_1(self, n_candidates: int = DEFAULT_CANDIDATES, fitness_name: str = DEFAULT_FITNESS, force: bool = False):
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
            fitness_name=fitness_name,
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

    def run_phase_2(self, n_sims: int = DEFAULT_SIMS, n_processes: int = None,
                    n_select: int = None, fitness_function: str = DEFAULT_FITNESS, force: bool = False):
        """Phase 2: Evaluate candidates via Monte Carlo and select top N by fitness.

        Args:
            n_sims: Number of Monte Carlo simulations
            n_processes: Number of parallel processes
            n_select: Number of top lineups to select (None = keep all)
            fitness_function: Fitness function for selection
            force: Force re-evaluation even if complete
        """
        candidates_file = self.run_dir / '1_candidates.csv'
        all_evaluations_file = self.run_dir / '2a_simulations.csv'  # All before selection
        selected_evaluations_file = self.run_dir / '2b_simulations.csv'  # After selection

        if self.state['phase_2_complete'] and not force and selected_evaluations_file.exists():
            print(f"Phase 2 already complete. Loading from {selected_evaluations_file}")
            return pd.read_csv(selected_evaluations_file)

        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

        self.print_header("PHASE 2: MONTE CARLO EVALUATION")

        start_time = time.time()

        # Evaluate ALL candidates (saves to temporary location first)
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

        # Clean up temp file
        if temp_evaluations_file.exists():
            temp_evaluations_file.unlink()

        # Calculate fitness for ALL lineups first (needed for both saving and selection)
        from optimizer.utils.genetic_operators import FITNESS_FUNCTIONS
        fitness_func = FITNESS_FUNCTIONS.get(fitness_function, FITNESS_FUNCTIONS['tournament'])

        evaluations_df['fitness'] = evaluations_df.apply(
            lambda row: fitness_func(row.to_dict()),
            axis=1
        )

        # Save ALL evaluations BEFORE selection to 2a_simulations.csv
        all_evaluations_formatted = self.format_lineups_by_position(evaluations_df)
        all_evaluations_formatted.to_csv(all_evaluations_file, index=False)
        print(f"\nAll {len(evaluations_df)} evaluations saved to: {all_evaluations_file}")

        # Select top N by fitness after MC evaluation
        if n_select is not None and n_select < len(evaluations_df):
            print(f"\n=== SELECTING TOP {n_select} BY FITNESS ({fitness_function}) ===")
            print(f"  Total evaluated: {len(evaluations_df)}")

            # Sort by fitness and select top N (fitness already calculated above)
            evaluations_df = evaluations_df.sort_values('fitness', ascending=False).head(n_select)
            evaluations_df = evaluations_df.reset_index(drop=True)
            evaluations_df['lineup_id'] = range(len(evaluations_df))

            # Show strategy breakdown in selected
            print(f"  Selected top {len(evaluations_df)} by {fitness_function} fitness")

            if 'strategy' in evaluations_df.columns:
                strategy_counts = evaluations_df['strategy'].value_counts()
                print(f"\n  Strategy breakdown in top {len(evaluations_df)}:")
                for strat, count in strategy_counts.items():
                    print(f"    {strat}: {count}")

            if 'tier' in evaluations_df.columns:
                tier_counts = evaluations_df['tier'].value_counts().sort_index()
                print(f"\n  Tier breakdown in top {len(evaluations_df)}:")
                for tier, count in tier_counts.items():
                    tier_name = {1: 'chalk', 2: 'moderate', 3: 'contrarian'}.get(tier, str(tier))
                    print(f"    Tier {tier} ({tier_name}): {count}")

        # Format with position-based columns and save SELECTED lineups to 2b_simulations.csv
        evaluations_formatted = self.format_lineups_by_position(evaluations_df)
        evaluations_formatted.to_csv(selected_evaluations_file, index=False)

        elapsed = time.time() - start_time
        self.state['phase_2_complete'] = True
        self.state['phase_2_time'] = elapsed
        self.state['n_simulations'] = n_sims
        self.state['n_selected'] = len(evaluations_df)
        self.save_state()

        print(f"\nPhase 2 complete in {elapsed / 60:.1f} minutes")
        print(f"Selected {len(evaluations_df)} lineups saved to: {selected_evaluations_file}")

        return evaluations_df

    def run_phase_3_iteration(
        self,
        iteration: int,
        evaluations_df: pd.DataFrame,
        fitness_function: str = DEFAULT_FITNESS,
        n_generations: int = DEFAULT_MAX_GENERATIONS,
        n_sims: int = DEFAULT_SIMS,
        n_processes: int = None
    ):
        """Phase 3: Single iteration of genetic optimization."""
        self.print_header(f"PHASE 3: GENETIC OPTIMIZATION - ITERATION {iteration}")

        start_time = time.time()

        # Save current evaluations as input for this iteration
        iteration_dir = self.run_dir / f'iteration_{iteration}'
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # Save input snapshot (matches 2_simulations.csv naming)
        input_file = iteration_dir / f'2_simulations_iter{iteration}.csv'
        evaluations_df.to_csv(input_file, index=False)

        # Run genetic optimization
        # New GA design: 50 parents, 4 pairing strategies × 25 offspring = 100 offspring
        output_file = iteration_dir / f'3_lineups_iter{iteration}.csv'
        optimal_df = optimize_genetic(
            evaluations_path=str(input_file),
            players_path=str(self.players_path),
            game_scripts_path=str(self.game_scripts_path),
            fitness_function=fitness_function,
            population_size=100,
            max_generations=n_generations,
            n_sims=n_sims,
            n_processes=n_processes,
            output_path=str(output_file),
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

        # Format lineups with position columns (overwrites raw output from optimize_genetic)
        all_lineups_formatted = self.format_lineups_by_position(optimal_df)
        all_lineups_formatted.to_csv(output_file, index=False)

        print(f"\nIteration {iteration} complete in {elapsed / 60:.1f} minutes")
        print(f"  Best fitness: {iteration_result['best_fitness']:.2f}")
        print(f"  Best median: {iteration_result['best_median']:.2f}")
        print(f"  All {len(optimal_df)} lineups saved to: {output_file}")

        return optimal_df, iteration_result

    def check_convergence(self, patience: int = DEFAULT_CONVERGENCE_PATIENCE, threshold: float = DEFAULT_CONVERGENCE_THRESHOLD):
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

        # Create mappings: support both ID-based and name-based lookups
        # ID -> position (for FanDuel IDs like "123617-138820")
        id_to_position = dict(zip(players_df['id'].astype(str), players_df['position']))
        # ID -> name (to convert IDs to readable names)
        id_to_name = dict(zip(players_df['id'].astype(str), players_df['name']))
        # name -> position (fallback for name-based lookups)
        name_to_position = dict(zip(players_df['name'].str.lower(), players_df['position']))

        formatted_rows = []

        for _, row in lineups_df.iterrows():
            # Parse player_ids (comma-separated)
            player_ids_raw = [p.strip() for p in row['player_ids'].split(',')]

            # Group players by position
            qbs = []
            rbs = []
            wrs = []
            tes = []
            defs = []

            for player_id in player_ids_raw:
                # Try ID lookup first, then name lookup
                pos = id_to_position.get(player_id)
                if pos is None:
                    pos = name_to_position.get(player_id.lower(), 'UNKNOWN')

                # Get player name (convert ID to name if possible)
                player_name = id_to_name.get(player_id, player_id)
                if pos == 'QB':
                    qbs.append(player_name)
                elif pos == 'RB':
                    rbs.append(player_name)
                elif pos == 'WR':
                    wrs.append(player_name)
                elif pos == 'TE':
                    tes.append(player_name)
                elif pos in ('DEF', 'D'):  # Handle both 'DEF' and 'D'
                    defs.append(player_name)

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
        n_candidates: int = DEFAULT_CANDIDATES,
        n_sims_phase2: int = DEFAULT_SIMS,
        n_sims_phase3: int = DEFAULT_SIMS,
        fitness_function: str = DEFAULT_FITNESS,
        max_generations: int = DEFAULT_MAX_GENERATIONS,
        convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        n_processes: int = None,
        force_restart: bool = False
    ):
        """Run complete optimization pipeline with single genetic algorithm run."""
        self.print_header("DFS LINEUP OPTIMIZER - FULL PIPELINE")

        # Save configuration to config.json (input parameters only)
        config = {
            'timestamp': datetime.now().isoformat(),
            'week': str(self.output_dir.parent.name) if self.output_dir.parent.name.startswith('20') else 'unknown',
            'candidates': n_candidates,
            'simulations': n_sims_phase2,
            'fitness': fitness_function,
            'max_generations': max_generations,
            'convergence_patience': convergence_patience,
            'convergence_threshold': convergence_threshold,
            'processes': n_processes or 'auto',
            'data_sources': {
                'players': str(self.players_path),
                'game_scripts': str(self.game_scripts_path)
            }
        }
        config.update(self.config_params)  # Add any extra params

        config_file = self.run_dir / 'config.json'
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
            fitness_name=fitness_function,
            force=force_restart
        )

        # Phase 2: Evaluate ALL candidates via MC, then select top N by fitness
        evaluations_df = self.run_phase_2(
            n_sims=n_sims_phase2,
            n_processes=n_processes,
            n_select=n_candidates,  # Select top N AFTER MC evaluation
            fitness_function=fitness_function,
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

        # Final summary - save_state() handles all results
        pipeline_elapsed = time.time() - pipeline_start
        self.state['total_time'] = pipeline_elapsed
        self.state['completed'] = True
        self.state['phases_completed'] = ['candidates', 'simulations'] + [f'genetic_{i+1}' for i in range(len(self.state['iterations']))]
        self.save_state()  # This writes to results.json

        self.print_header("OPTIMIZATION COMPLETE")

        print(f"Total time: {pipeline_elapsed / 60:.1f} minutes")
        print(f"Total iterations: {len(self.state['iterations'])}")
        print(f"Best fitness achieved: {max(self.state['best_fitness_history']):.2f}")

        # Copy best lineups to main outputs
        best_iteration = max(self.state['iterations'], key=lambda x: x['best_fitness'])
        best_iter_num = best_iteration['iteration']
        best_iteration_dir = self.run_dir / f"iteration_{best_iter_num}"
        best_lineups_file = best_iteration_dir / f"3_lineups_iter{best_iter_num}.csv"

        if best_lineups_file.exists():
            best_lineups = pd.read_csv(best_lineups_file)

            # Copy to final 3_lineups.csv
            best_lineups.to_csv(self.run_dir / '3_lineups.csv', index=False)
            print(f"\nBest lineups saved to: {self.run_dir / '3_lineups.csv'}")

            print(f"\nTop 5 Best Lineups:")
            for i, row in best_lineups.head(5).iterrows():
                print(f"  {i+1}. Fitness: {row.get('fitness', row.get('median')):.2f}, "
                      f"Median: {row['median']:.2f}, P90: {row['p90']:.2f}")

        # Copy training metrics plot to main run directory for easy access
        training_plot = best_iteration_dir / 'training_metrics.png'
        if training_plot.exists():
            import shutil
            shutil.copy(training_plot, self.run_dir / 'training_metrics.png')
            print(f"Training metrics plot: {self.run_dir / 'training_metrics.png'}")

        print(f"\nAll results saved in: {self.run_dir}")
        print(f"  - config.json: Run configuration (input params)")
        print(f"  - results.json: Run results & state for resumption")
        print(f"  - 1_candidates.csv: All generated candidates")
        print(f"  - 2a_simulations.csv: All Monte Carlo evaluations (before selection)")
        print(f"  - 2b_simulations.csv: Selected lineups (after tournament selection)")
        print(f"  - 3_lineups.csv: Final optimized lineups")
        print(f"  - training_metrics.png: GA training visualization")
        print(f"  - iteration_N/: Snapshots from each GA iteration")


def main():
    parser = argparse.ArgumentParser(
        description="DFS Lineup Optimizer - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings from config.py
  python run_optimizer.py

  # Override specific settings
  python run_optimizer.py --candidates 500 --sims 10000

  # Custom generations with early stopping
  python run_optimizer.py --max-generations 100 --patience 5

  # Aggressive fitness (high upside)
  python run_optimizer.py --fitness tournament
        """
    )

    # Pipeline configuration
    parser.add_argument('--candidates', type=int, default=DEFAULT_CANDIDATES,
                        help=f'Number of candidate lineups to generate (default: {DEFAULT_CANDIDATES})')
    parser.add_argument('--sims', type=int, default=DEFAULT_SIMS,
                        help=f'Number of Monte Carlo simulations (default: {DEFAULT_SIMS})')
    parser.add_argument('--max-generations', type=int, default=DEFAULT_MAX_GENERATIONS,
                        help=f'Maximum generations for genetic algorithm (default: {DEFAULT_MAX_GENERATIONS})')
    parser.add_argument('--fitness', default=DEFAULT_FITNESS,
                        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                        help=f'Fitness function (default: {DEFAULT_FITNESS})')

    # Convergence settings
    parser.add_argument('--patience', type=int, default=DEFAULT_CONVERGENCE_PATIENCE,
                        help=f'Convergence patience for early stopping (default: {DEFAULT_CONVERGENCE_PATIENCE})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_CONVERGENCE_THRESHOLD,
                        help=f'Convergence threshold (default: {DEFAULT_CONVERGENCE_THRESHOLD})')

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

    args = parser.parse_args()

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
