"""
Phase 3: Genetic Algorithm Refinement

Uses evolutionary optimization to discover better lineups through:
- Tournament selection without replacement (k=2)
- Four pairing strategies: strength-matched, inverse-matched, random, dissimilar
- Position-aware crossover
- Salary-preserving mutation
- Elite archive to preserve best lineups ever seen
- Parallel Monte Carlo evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import time
import sys
from typing import List, Dict, Optional, Set

from utils.genetic_operators import (
    tournament_select_without_replacement,
    crossover_with_strategy,
    mutate,
    FITNESS_FUNCTIONS,
    calculate_lineup_salary,
    get_lineup_key,
    get_player_ids
)
from utils.monte_carlo import evaluate_lineups_parallel

# Import config values (add parent dir to path temporarily)
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from config import (
    DEFAULT_FITNESS, DEFAULT_SIMS, DEFAULT_PROCESSES,
    DEFAULT_MAX_GENERATIONS, DEFAULT_CONVERGENCE_PATIENCE,
    GENETIC_NUM_PARENTS, GENETIC_MUTATION_RATE,
    ELITE_ARCHIVE_SIZE, SALARY_CAP
)


def update_elite_archive(
    elite_archive: List[Dict],
    candidates: List[Dict],
    players_df: pd.DataFrame,
    fitness_func,
    max_size: int = ELITE_ARCHIVE_SIZE,
    salary_cap: float = SALARY_CAP
) -> List[Dict]:
    """
    Update elite archive with qualifying candidates.

    Candidates must:
    1. Be under salary cap
    2. Not be a duplicate of existing archive member

    Args:
        elite_archive: Current elite archive
        candidates: New candidates to consider
        players_df: DataFrame with player data (for salary calculation)
        fitness_func: Fitness function to rank lineups
        max_size: Maximum archive size
        salary_cap: Maximum allowed salary

    Returns:
        Updated elite archive (sorted by fitness descending)
    """
    # Get existing lineup keys for duplicate checking
    existing_keys = {get_lineup_key(lineup) for lineup in elite_archive}

    # Filter and add qualifying candidates
    for candidate in candidates:
        # Check for duplicate
        key = get_lineup_key(candidate)
        if key in existing_keys:
            continue

        # Calculate actual salary (don't trust stored value after crossover)
        actual_salary = calculate_lineup_salary(candidate, players_df)
        candidate['total_salary'] = actual_salary

        # Check salary cap
        if actual_salary > salary_cap:
            continue

        # Add to archive
        elite_archive.append(candidate)
        existing_keys.add(key)

    # Sort by fitness and trim to max size
    elite_archive.sort(key=fitness_func, reverse=True)
    return elite_archive[:max_size]


def optimize_genetic(
    evaluations_path: str = 'outputs/lineup_evaluations.csv',
    players_path: str = 'players_integrated.csv',
    game_scripts_path: str = 'game_script.csv',
    fitness_function: str = DEFAULT_FITNESS,
    population_size: int = 100,
    n_parents: int = GENETIC_NUM_PARENTS,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    mutation_rate: float = GENETIC_MUTATION_RATE,
    n_sims: int = DEFAULT_SIMS,
    n_processes: int = DEFAULT_PROCESSES,
    convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
    elite_archive_size: int = ELITE_ARCHIVE_SIZE,
    output_path: str = 'outputs/lineups_optimal.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Optimize lineups using genetic algorithm with elite archive.

    GA Design:
    - Population size: 100 (from Phase 2 evaluations)
    - Select 50 parents via tournament (k=2) without replacement
    - Remaining 50 are non-parents
    - Generate 100 offspring via 4 pairing strategies (25 each):
      1. Strength-matched: pair by rank (1st+2nd, 3rd+4th, ...)
      2. Inverse-matched: pair extremes (1st+50th, 2nd+49th, ...)
      3. Random: random pairing
      4. Dissimilar: maximize player difference
    - Replacement: Keep top 25 parents + top 15 non-parents + top 15 from each offspring group
    - Elite archive: Track best 100 valid lineups ever seen (under cap, unique)

    Args:
        evaluations_path: Path to Phase 2 evaluation results
        players_path: Path to integrated players CSV
        game_scripts_path: Path to game scripts CSV
        fitness_function: Fitness function name
        population_size: Size of population each generation (default: 100)
        n_parents: Number of parents to select (default: 50)
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation
        n_sims: Number of Monte Carlo simulations for offspring
        n_processes: Number of parallel processes
        convergence_patience: Stop if no improvement for this many generations
        elite_archive_size: Size of elite archive
        output_path: Where to save optimal lineups
        verbose: Print progress

    Returns:
        DataFrame with elite archive lineups
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

    # Normalize column names
    if 'id' not in players_df.columns and 'name' in players_df.columns:
        players_df['id'] = players_df['name']
    if 'fdSalary' not in players_df.columns and 'salary' in players_df.columns:
        players_df['fdSalary'] = players_df['salary']

    # Get fitness function
    if fitness_function not in FITNESS_FUNCTIONS:
        raise ValueError(f"Unknown fitness function: {fitness_function}. Choose from: {list(FITNESS_FUNCTIONS.keys())}")

    fitness_func = FITNESS_FUNCTIONS[fitness_function]

    if verbose:
        print(f"  Evaluations loaded: {len(evaluations_df)}")
        print(f"  Fitness function: {fitness_function}")
        print(f"  Population size: {population_size}")
        print(f"  Parents per generation: {n_parents}")
        print(f"  Elite archive size: {elite_archive_size}")

    # Initialize population from Phase 2 (top candidates)
    population = evaluations_df.head(population_size).to_dict('records')

    # Add fitness scores to population
    for lineup in population:
        lineup['fitness'] = fitness_func(lineup)

    # Initialize elite archive with valid lineups from initial population
    elite_archive = []
    elite_archive = update_elite_archive(
        elite_archive, population, players_df, fitness_func,
        max_size=elite_archive_size, salary_cap=SALARY_CAP
    )

    best_archive_fitness = elite_archive[0]['fitness'] if elite_archive else 0
    best_fitness_history = [best_archive_fitness]

    if verbose:
        print(f"\nInitial elite archive: {len(elite_archive)} lineups")
        print(f"  Best fitness: {best_archive_fitness:.2f}")
        print(f"\nStarting evolution...")

    # Evolution loop
    for generation in range(max_generations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{max_generations}")
            print(f"{'='*60}")

        gen_start_time = time.time()

        # =================================================================
        # 1. SELECTION: Tournament (k=2) without replacement
        # =================================================================
        parents, non_parents = tournament_select_without_replacement(
            population=population,
            fitness_func=fitness_func,
            tournament_size=2,
            n_parents=n_parents
        )

        if verbose:
            print(f"\nSelection:")
            print(f"  Parents: {len(parents)} (fitness range: {fitness_func(parents[-1]):.2f} - {fitness_func(parents[0]):.2f})")
            print(f"  Non-parents: {len(non_parents)}")

        # =================================================================
        # 2. CROSSOVER: Four pairing strategies, 25 offspring each
        # =================================================================
        offspring_groups = {}

        for strategy in ['strength', 'inverse', 'random', 'dissimilar']:
            offspring_groups[strategy] = crossover_with_strategy(
                parents=parents,
                players_df=players_df,
                strategy=strategy,
                n_offspring=25,
                crossover_rate=1.0  # Always crossover
            )

        if verbose:
            print(f"\nCrossover (4 strategies x 25 offspring = 100 total):")
            for strategy, offspring in offspring_groups.items():
                print(f"  {strategy}: {len(offspring)} offspring")

        # =================================================================
        # 3. MUTATION: Apply to all offspring
        # =================================================================
        mutated_counts = {strategy: 0 for strategy in offspring_groups}

        for strategy, offspring_list in offspring_groups.items():
            for i, child in enumerate(offspring_list):
                original_ids = child.get('player_ids', '')
                mutated = mutate(
                    child,
                    players_df,
                    mutation_rate=mutation_rate,
                    aggressive=False
                )
                if mutated.get('player_ids', '') != original_ids:
                    mutated_counts[strategy] += 1
                    offspring_groups[strategy][i] = mutated

        if verbose:
            total_mutated = sum(mutated_counts.values())
            print(f"\nMutation ({mutation_rate:.0%} rate):")
            print(f"  Total mutated: {total_mutated}/100 offspring")

        # =================================================================
        # 4. EVALUATE: All offspring via parallel Monte Carlo
        # =================================================================
        all_offspring = []
        offspring_strategy_map = []  # Track which strategy each offspring came from

        for strategy, offspring_list in offspring_groups.items():
            for child in offspring_list:
                all_offspring.append(child)
                offspring_strategy_map.append(strategy)

        if verbose:
            print(f"\nEvaluation:")
            print(f"  Evaluating {len(all_offspring)} offspring with {n_sims} simulations each...")

        # Convert to lineup format for evaluation
        offspring_lineups = []
        for child in all_offspring:
            player_ids = get_player_ids(child)
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
            print(f"  Completed in {eval_time:.1f} seconds")

        # Add simulation results and fitness to offspring
        for child, result in zip(all_offspring, offspring_results):
            child.update(result)
            child['fitness'] = fitness_func(child)

        # Reorganize offspring back into strategy groups with results
        evaluated_groups = {strategy: [] for strategy in offspring_groups}
        for child, strategy in zip(all_offspring, offspring_strategy_map):
            evaluated_groups[strategy].append(child)

        # Sort each group by fitness
        for strategy in evaluated_groups:
            evaluated_groups[strategy].sort(key=fitness_func, reverse=True)

        # =================================================================
        # 5. REPLACEMENT: Structured selection from each group
        # =================================================================
        # Keep: 25 best parents + 15 best non-parents + 15 from each offspring group (60)
        # Total: 25 + 15 + 60 = 100

        new_population = []

        # Top 25 parents
        new_population.extend(parents[:25])

        # Top 15 non-parents
        new_population.extend(non_parents[:15])

        # Top 15 from each offspring group
        for strategy in ['strength', 'inverse', 'random', 'dissimilar']:
            new_population.extend(evaluated_groups[strategy][:15])

        population = new_population

        if verbose:
            print(f"\nReplacement:")
            print(f"  25 parents + 15 non-parents + 60 offspring (15 per strategy) = {len(population)}")

        # =================================================================
        # 6. UPDATE ELITE ARCHIVE
        # =================================================================
        # Consider all evaluated lineups (parents, non-parents, offspring)
        all_evaluated = parents + non_parents + all_offspring

        archive_size_before = len(elite_archive)
        elite_archive = update_elite_archive(
            elite_archive, all_evaluated, players_df, fitness_func,
            max_size=elite_archive_size, salary_cap=SALARY_CAP
        )

        new_in_archive = len(elite_archive) - archive_size_before
        current_best = elite_archive[0]['fitness'] if elite_archive else 0
        improvement = current_best > best_archive_fitness

        if improvement:
            best_archive_fitness = current_best

        best_fitness_history.append(best_archive_fitness)

        gen_time = time.time() - gen_start_time

        if verbose:
            print(f"\nElite Archive:")
            print(f"  Size: {len(elite_archive)}/{elite_archive_size}")
            print(f"  Best fitness: {best_archive_fitness:.2f} {'(NEW BEST!)' if improvement else ''}")
            print(f"\nGeneration time: {gen_time:.1f} seconds")

        # =================================================================
        # 7. CHECK CONVERGENCE
        # =================================================================
        if len(best_fitness_history) > convergence_patience:
            recent = best_fitness_history[-convergence_patience:]
            if all(recent[i] == recent[0] for i in range(len(recent))):
                if verbose:
                    print(f"\nConverged! No improvement in last {convergence_patience} generations.")
                break

    # =================================================================
    # FINAL RESULTS: Output elite archive
    # =================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal generations: {generation + 1}")
        print(f"Elite archive size: {len(elite_archive)}")
        print(f"Best fitness: {best_archive_fitness:.2f}")

    # Convert elite archive to DataFrame
    elite_df = pd.DataFrame(elite_archive)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    elite_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nTop 10 lineups from elite archive:")
        for i, lineup in enumerate(elite_archive[:10]):
            salary = lineup.get('total_salary', 0)
            print(f"  {i + 1}. Fitness: {lineup['fitness']:6.2f}, "
                  f"Median: {lineup.get('median', 0):6.2f}, "
                  f"P90: {lineup.get('p90', 0):6.2f}, "
                  f"Salary: ${salary:,.0f}")
        print(f"\nSaved {len(elite_archive)} lineups to: {output_file}")
        print("=" * 80)

    return elite_df


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Genetic algorithm optimization")
    parser.add_argument('--input', default='outputs/lineup_evaluations.csv', help='Input evaluations CSV')
    parser.add_argument('--players', default='players_integrated.csv', help='Players CSV')
    parser.add_argument('--game-scripts', default='game_script.csv', help='Game scripts CSV')
    parser.add_argument('--output', default='outputs/lineups_optimal.csv', help='Output optimal lineups CSV')
    parser.add_argument('--fitness', default=DEFAULT_FITNESS,
                        choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                        help=f'Fitness function (default: {DEFAULT_FITNESS})')
    parser.add_argument('--population', type=int, default=100, help='Population size (default: 100)')
    parser.add_argument('--parents', type=int, default=GENETIC_NUM_PARENTS, help=f'Parents per generation (default: {GENETIC_NUM_PARENTS})')
    parser.add_argument('--generations', type=int, default=DEFAULT_MAX_GENERATIONS, help=f'Max generations (default: {DEFAULT_MAX_GENERATIONS})')
    parser.add_argument('--mutation-rate', type=float, default=GENETIC_MUTATION_RATE, help=f'Mutation rate (default: {GENETIC_MUTATION_RATE})')
    parser.add_argument('--sims', type=int, default=DEFAULT_SIMS, help=f'Simulations per lineup (default: {DEFAULT_SIMS})')
    parser.add_argument('--processes', type=int, default=DEFAULT_PROCESSES, help='Parallel processes (default: auto)')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONVERGENCE_PATIENCE, help=f'Convergence patience (default: {DEFAULT_CONVERGENCE_PATIENCE})')
    parser.add_argument('--archive-size', type=int, default=ELITE_ARCHIVE_SIZE, help=f'Elite archive size (default: {ELITE_ARCHIVE_SIZE})')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Optimize
    optimize_genetic(
        evaluations_path=args.input,
        players_path=args.players,
        game_scripts_path=args.game_scripts,
        fitness_function=args.fitness,
        population_size=args.population,
        n_parents=args.parents,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        n_sims=args.sims,
        n_processes=args.processes,
        convergence_patience=args.patience,
        elite_archive_size=args.archive_size,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
