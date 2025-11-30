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

# Add optimizer to path
sys.path.insert(0, str(Path(__file__).parent / 'optimizer'))

from optimizer.generate_candidates import generate_candidates
from optimizer.evaluate_lineups import evaluate_candidates
from optimizer.optimize_genetic import optimize_genetic


class OptimizerOrchestrator:
    """Orchestrates the complete optimization pipeline with checkpointing."""

    def __init__(
        self,
        players_path: str = 'players_integrated.csv',
        game_scripts_path: str = 'game_script_continuous.csv',
        output_dir: str = 'outputs',
        run_name: str = None
    ):
        self.players_path = players_path
        self.game_scripts_path = game_scripts_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create run-specific directory
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'run_{run_name}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
        candidates_file = self.run_dir / 'candidates.csv'

        if self.state['phase_1_complete'] and not force and candidates_file.exists():
            print(f"Phase 1 already complete. Loading from {candidates_file}")
            return pd.read_csv(candidates_file)

        self.print_header("PHASE 1: CANDIDATE GENERATION")

        start_time = time.time()

        # Load data
        players_df = pd.read_csv(self.players_path)
        game_scripts_df = pd.read_csv(self.game_scripts_path)

        # Generate candidates
        candidates_df = generate_candidates(
            players_df=players_df,
            game_scripts_df=game_scripts_df,
            n_lineups=n_candidates,
            output_path=str(candidates_file),
            verbose=True
        )

        elapsed = time.time() - start_time
        self.state['phase_1_complete'] = True
        self.state['phase_1_time'] = elapsed
        self.state['n_candidates'] = len(candidates_df)
        self.save_state()

        print(f"\nPhase 1 complete in {elapsed:.1f} seconds")
        return candidates_df

    def run_phase_2(self, n_sims: int = 10000, n_processes: int = None, force: bool = False):
        """Phase 2: Evaluate candidates via Monte Carlo."""
        candidates_file = self.run_dir / 'candidates.csv'
        evaluations_file = self.run_dir / 'evaluations.csv'

        if self.state['phase_2_complete'] and not force and evaluations_file.exists():
            print(f"Phase 2 already complete. Loading from {evaluations_file}")
            return pd.read_csv(evaluations_file)

        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

        self.print_header("PHASE 2: MONTE CARLO EVALUATION")

        start_time = time.time()

        # Evaluate candidates
        evaluations_df = evaluate_candidates(
            candidates_path=str(candidates_file),
            players_path=self.players_path,
            game_scripts_path=self.game_scripts_path,
            n_sims=n_sims,
            n_processes=n_processes,
            output_path=str(evaluations_file),
            verbose=True
        )

        elapsed = time.time() - start_time
        self.state['phase_2_complete'] = True
        self.state['phase_2_time'] = elapsed
        self.state['n_simulations'] = n_sims
        self.save_state()

        # Save top 10 snapshot
        top_10 = evaluations_df.head(10)
        top_10.to_csv(self.run_dir / 'top_10_after_phase2.csv', index=False)

        print(f"\nPhase 2 complete in {elapsed / 60:.1f} minutes")
        print(f"Top 10 saved to: {self.run_dir / 'top_10_after_phase2.csv'}")

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

        # Save top 10 from this iteration
        top_10 = optimal_df.head(10)
        top_10.to_csv(iteration_dir / f'top_10_iteration_{iteration}.csv', index=False)

        print(f"\nIteration {iteration} complete in {elapsed / 60:.1f} minutes")
        print(f"  Best fitness: {iteration_result['best_fitness']:.2f}")
        print(f"  Best median: {iteration_result['best_median']:.2f}")
        print(f"  Top 10 saved to: {iteration_dir / f'top_10_iteration_{iteration}.csv'}")

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

    def run_full_pipeline(
        self,
        n_candidates: int = 1000,
        n_sims_phase2: int = 10000,
        n_sims_phase3: int = 10000,
        fitness_function: str = 'balanced',
        max_iterations: int = 10,
        convergence_patience: int = 3,
        convergence_threshold: float = 0.01,
        n_processes: int = None,
        force_restart: bool = False
    ):
        """Run complete optimization pipeline with iterative refinement."""
        self.print_header("DFS LINEUP OPTIMIZER - FULL PIPELINE")

        print(f"Configuration:")
        print(f"  Candidates: {n_candidates}")
        print(f"  Phase 2 simulations: {n_sims_phase2}")
        print(f"  Phase 3 simulations: {n_sims_phase3}")
        print(f"  Fitness function: {fitness_function}")
        print(f"  Max iterations: {max_iterations}")
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

        # Phase 3: Iterative refinement
        iteration = len(self.state['iterations']) + 1

        while iteration <= max_iterations:
            # Run iteration
            optimal_df, iteration_result = self.run_phase_3_iteration(
                iteration=iteration,
                evaluations_df=evaluations_df,
                fitness_function=fitness_function,
                n_sims=n_sims_phase3,
                n_processes=n_processes
            )

            # Update evaluations with new lineups
            # Merge optimal lineups back into evaluation pool for next iteration
            evaluations_df = pd.concat([
                evaluations_df.head(900),  # Keep top 900 from previous
                optimal_df.head(100)  # Add top 100 from this iteration
            ]).sort_values('median', ascending=False).reset_index(drop=True)

            # Check convergence
            converged = self.check_convergence(
                patience=convergence_patience,
                threshold=convergence_threshold
            )

            if converged:
                print(f"\n{'='*80}")
                print("CONVERGENCE REACHED!".center(80))
                print(f"{'='*80}")
                print(f"No significant improvement in last {convergence_patience} iterations")
                print(f"Stopping optimization.")
                break

            iteration += 1

            # Progress update
            print(f"\n{'='*80}")
            print(f"Progress: Iteration {iteration - 1}/{max_iterations}")
            print(f"Best fitness so far: {max(self.state['best_fitness_history']):.2f}")
            print(f"{'='*80}")

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

        with open(self.run_dir / 'final_summary.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        # Copy best lineups to main outputs
        best_iteration = max(self.state['iterations'], key=lambda x: x['best_fitness'])
        best_iteration_dir = self.run_dir / f"iteration_{best_iteration['iteration']}"
        best_lineups_file = best_iteration_dir / f"top_10_iteration_{best_iteration['iteration']}.csv"

        if best_lineups_file.exists():
            best_lineups = pd.read_csv(best_lineups_file)
            best_lineups.to_csv(self.run_dir / 'BEST_LINEUPS.csv', index=False)
            print(f"\nBest lineups saved to: {self.run_dir / 'BEST_LINEUPS.csv'}")

            print(f"\nTop 5 Best Lineups:")
            for i, row in best_lineups.head(5).iterrows():
                print(f"  {i+1}. Fitness: {row.get('fitness', row.get('median')):.2f}, "
                      f"Median: {row['median']:.2f}, P90: {row['p90']:.2f}")

        print(f"\nAll results saved in: {self.run_dir}")
        print(f"  - candidates.csv: All generated candidates")
        print(f"  - evaluations.csv: Monte Carlo evaluations")
        print(f"  - iteration_N/: Results from each iteration")
        print(f"  - BEST_LINEUPS.csv: Top 10 best lineups found")
        print(f"  - final_summary.json: Complete run statistics")
        print(f"  - optimizer_state.json: State for resumption")


def main():
    parser = argparse.ArgumentParser(
        description="DFS Lineup Optimizer - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (100 candidates, 1000 sims)
  python run_optimizer.py --quick-test

  # Full production run (1000 candidates, 10k sims)
  python run_optimizer.py --candidates 1000 --sims 10000

  # Resume from previous run
  python run_optimizer.py --run-name 20241130_143022

  # Aggressive fitness (high upside)
  python run_optimizer.py --fitness tournament
        """
    )

    # Pipeline configuration
    parser.add_argument('--candidates', type=int, default=1000,
                        help='Number of candidate lineups to generate (default: 1000)')
    parser.add_argument('--sims', type=int, default=10000,
                        help='Number of Monte Carlo simulations (default: 10000)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Maximum number of refinement iterations (default: 10)')
    parser.add_argument('--fitness', default='balanced',
                        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                        help='Fitness function (default: balanced)')

    # Convergence settings
    parser.add_argument('--patience', type=int, default=3,
                        help='Convergence patience (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Convergence threshold in %% (default: 0.01 = 1%%)')

    # Performance
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes (default: auto)')

    # Run management
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--force-restart', action='store_true',
                        help='Force restart from Phase 1 even if state exists')

    # Convenience flags
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test: 50 candidates, 1000 sims, 2 iterations')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.candidates = 50
        args.sims = 1000
        args.iterations = 2
        print("QUICK TEST MODE")

    # Create orchestrator
    orchestrator = OptimizerOrchestrator(
        run_name=args.run_name
    )

    # Run pipeline
    orchestrator.run_full_pipeline(
        n_candidates=args.candidates,
        n_sims_phase2=args.sims,
        n_sims_phase3=args.sims,
        fitness_function=args.fitness,
        max_iterations=args.iterations,
        convergence_patience=args.patience,
        convergence_threshold=args.threshold,
        n_processes=args.processes,
        force_restart=args.force_restart
    )


if __name__ == '__main__':
    main()
