"""
Phase 3: Genetic Algorithm Refinement

Uses evolutionary optimization to discover better lineups through:
- Tournament selection
- Position-aware crossover
- Salary-preserving mutation
- Parallel Monte Carlo evaluation
- Checkpointing for resumption
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import time
from typing import List, Dict, Optional

from utils.genetic_operators import (
    tournament_select, crossover, mutate,
    FITNESS_FUNCTIONS
)
from utils.monte_carlo import evaluate_lineups_parallel


def save_checkpoint(
    generation: int,
    population: List[Dict],
    best_lineup: Dict,
    output_dir: str = 'outputs/checkpoints'
):
    """Save generation checkpoint to JSON."""
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'generation': generation,
        'population': population,
        'best_lineup': best_lineup,
        'timestamp': time.time()
    }

    checkpoint_file = checkpoint_dir / f'generation_{generation}.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint from JSON."""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def optimize_genetic(
    evaluations_path: str = 'outputs/lineup_evaluations.csv',
    players_path: str = 'players_integrated.csv',
    game_scripts_path: str = 'game_script_continuous.csv',
    fitness_function: str = 'balanced',
    population_size: int = 100,
    n_offspring: int = 100,
    max_generations: int = 20,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.8,
    n_sims: int = 10000,
    n_processes: int = None,
    convergence_patience: int = 5,
    output_path: str = 'outputs/lineups_optimal.csv',
    resume_from: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Optimize lineups using genetic algorithm with parallel evaluation.

    Args:
        evaluations_path: Path to Phase 2 evaluation results
        players_path: Path to integrated players CSV
        game_scripts_path: Path to game scripts CSV
        fitness_function: Fitness function name (conservative, balanced, aggressive, tournament)
        population_size: Size of population each generation
        n_offspring: Number of offspring to generate per generation
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        n_sims: Number of Monte Carlo simulations for new lineups
        n_processes: Number of parallel processes
        convergence_patience: Stop if no improvement for this many generations
        output_path: Where to save optimal lineups
        resume_from: Path to checkpoint to resume from
        verbose: Print progress

    Returns:
        DataFrame with top lineups
    """
    if verbose:
        print("=" * 80)
        print("PHASE 3: GENETIC ALGORITHM REFINEMENT")
        print("=" * 80)

    # Load data
    if verbose:
        print("\nLoading data...")

    evaluations_df = pd.read_csv(evaluations_path)
    players_df = pd.read_csv(players_path)
    game_scripts_df = pd.read_csv(game_scripts_path)

    # Get fitness function
    if fitness_function not in FITNESS_FUNCTIONS:
        raise ValueError(f"Unknown fitness function: {fitness_function}. Choose from: {list(FITNESS_FUNCTIONS.keys())}")

    fitness_func = FITNESS_FUNCTIONS[fitness_function]

    if verbose:
        print(f"  Evaluations loaded: {len(evaluations_df)}")
        print(f"  Fitness function: {fitness_function}")
        print(f"  Population size: {population_size}")
        print(f"  Offspring per generation: {n_offspring}")

    # Initialize population
    if resume_from:
        if verbose:
            print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        population = checkpoint['population']
        start_generation = checkpoint['generation'] + 1
        best_fitness_history = []  # Could load from checkpoint if saved
    else:
        # Top 100 from Phase 2
        population = evaluations_df.head(population_size).to_dict('records')
        start_generation = 0
        best_fitness_history = []

    # Add fitness scores to population
    for lineup in population:
        lineup['fitness'] = fitness_func(lineup)

    best_overall = max(population, key=lambda x: x['fitness'])

    if verbose:
        print(f"\nStarting evolution from generation {start_generation}")
        print(f"  Initial best fitness: {best_overall['fitness']:.2f}")

    # Evolution loop
    for generation in range(start_generation, max_generations):
        if verbose:
            print(f"\n=== Generation {generation + 1}/{max_generations} ===")

        gen_start_time = time.time()

        # 1. Selection
        parents = tournament_select(
            population=population,
            fitness_func=fitness_func,
            tournament_size=5,
            n_parents=n_offspring // 2  # Need pairs for crossover
        )

        if verbose:
            print(f"  Selected {len(parents)} parents")

        # 2. Crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(
                parents[i],
                parents[i + 1],
                players_df,
                crossover_rate=crossover_rate
            )
            offspring.extend([child1, child2])

        if verbose:
            print(f"  Created {len(offspring)} offspring via crossover")

        # 3. Mutation
        mutated_count = 0
        for i, child in enumerate(offspring):
            mutated = mutate(
                child,
                players_df,
                mutation_rate=mutation_rate
            )
            if mutated['player_ids'] != child['player_ids']:
                mutated_count += 1
                offspring[i] = mutated

        if verbose:
            print(f"  Mutated {mutated_count} offspring")

        # 4. Evaluate offspring (PARALLEL)
        if verbose:
            print(f"  Evaluating {len(offspring)} offspring with {n_sims} simulations each...")

        # Convert offspring to lineup format for evaluation
        offspring_lineups = []
        for child in offspring:
            player_ids = child['player_ids'].split(',') if isinstance(child['player_ids'], str) else child['player_ids']
            offspring_lineups.append(player_ids)

        eval_start = time.time()
        offspring_results = evaluate_lineups_parallel(
            lineups=offspring_lineups,
            players_df=players_df,
            game_scripts_df=game_scripts_df,
            n_sims=n_sims,
            n_processes=n_processes
        )
        eval_time = time.time() - eval_start

        if verbose:
            print(f"  Evaluation complete in {eval_time:.1f} seconds")

        # Add simulation results and fitness to offspring
        for child, result in zip(offspring, offspring_results):
            child.update(result)
            child['fitness'] = fitness_func(child)

        # 5. Replacement (elitism + best offspring)
        # Keep top 20 from current population
        population_sorted = sorted(population, key=lambda x: x['fitness'], reverse=True)
        elite = population_sorted[:20]

        # Add best offspring
        offspring_sorted = sorted(offspring, key=lambda x: x['fitness'], reverse=True)
        new_population = elite + offspring_sorted[:population_size - 20]

        population = new_population

        # Track best
        generation_best = max(population, key=lambda x: x['fitness'])
        best_fitness_history.append(generation_best['fitness'])

        if generation_best['fitness'] > best_overall['fitness']:
            best_overall = generation_best
            improvement = True
        else:
            improvement = False

        gen_time = time.time() - gen_start_time

        if verbose:
            print(f"  Generation best fitness: {generation_best['fitness']:.2f} "
                  f"({'NEW BEST!' if improvement else 'no improvement'})")
            print(f"  Overall best fitness: {best_overall['fitness']:.2f}")
            print(f"  Generation time: {gen_time / 60:.1f} minutes")

        # 6. Save checkpoint
        save_checkpoint(generation, population, best_overall)

        if verbose:
            print(f"  Checkpoint saved: generation_{generation}.json")

        # 7. Check convergence
        if len(best_fitness_history) >= convergence_patience:
            recent_improvements = [
                best_fitness_history[i] > best_fitness_history[i - 1]
                for i in range(-convergence_patience, 0)
            ]
            if not any(recent_improvements):
                if verbose:
                    print(f"\n  Converged! No improvement in last {convergence_patience} generations.")
                break

    # Final results
    if verbose:
        print("\n=== OPTIMIZATION COMPLETE ===")
        print(f"  Total generations: {generation + 1}")
        print(f"  Best fitness: {best_overall['fitness']:.2f}")

    # Get top 10 lineups
    final_population_sorted = sorted(population, key=lambda x: x['fitness'], reverse=True)
    top_lineups = final_population_sorted[:10]

    # Convert to DataFrame
    top_lineups_df = pd.DataFrame(top_lineups)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top_lineups_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nTop 10 lineups:")
        for i, lineup in enumerate(top_lineups):
            print(f"  {i + 1}. Fitness: {lineup['fitness']:6.2f}, "
                  f"Median: {lineup['median']:6.2f}, "
                  f"P90: {lineup['p90']:6.2f}, "
                  f"P10: {lineup['p10']:6.2f}")
        print(f"\n  Saved to: {output_file}")
        print("=" * 80)

    return top_lineups_df


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Genetic algorithm optimization")
    parser.add_argument('--input', default='outputs/lineup_evaluations.csv', help='Input evaluations CSV')
    parser.add_argument('--players', default='players_integrated.csv', help='Players CSV')
    parser.add_argument('--game-scripts', default='game_script_continuous.csv', help='Game scripts CSV')
    parser.add_argument('--output', default='outputs/lineups_optimal.csv', help='Output optimal lineups CSV')
    parser.add_argument('--fitness', default='balanced',
                        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                        help='Fitness function')
    parser.add_argument('--population', type=int, default=100, help='Population size')
    parser.add_argument('--offspring', type=int, default=100, help='Offspring per generation')
    parser.add_argument('--generations', type=int, default=20, help='Max generations')
    parser.add_argument('--mutation-rate', type=float, default=0.3, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('--sims', type=int, default=10000, help='Simulations per lineup')
    parser.add_argument('--processes', type=int, default=None, help='Parallel processes')
    parser.add_argument('--patience', type=int, default=5, help='Convergence patience')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Optimize
    optimize_genetic(
        evaluations_path=args.input,
        players_path=args.players,
        game_scripts_path=args.game_scripts,
        fitness_function=args.fitness,
        population_size=args.population,
        n_offspring=args.offspring,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        n_sims=args.sims,
        n_processes=args.processes,
        convergence_patience=args.patience,
        output_path=args.output,
        resume_from=args.resume_from,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
