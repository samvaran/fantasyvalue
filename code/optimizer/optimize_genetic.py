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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.genetic_operators import (
    tournament_select_without_replacement,
    crossover_with_strategy,
    mutate,
    FITNESS_FUNCTIONS,
    calculate_lineup_salary,
    get_lineup_key,
    get_player_ids,
    calculate_hamming_distance
)
from utils.monte_carlo import evaluate_lineups_parallel


def calculate_population_diversity(population: List[Dict], fitness_func) -> Dict[str, float]:
    """
    Calculate diversity metrics for a population.

    Returns:
        Dict with:
        - fitness_std: Standard deviation of fitness scores
        - fitness_range: Max - min fitness
        - avg_hamming: Average pairwise Hamming distance (player differences)
        - unique_players: Number of unique players across all lineups
        - unique_lineups: Number of completely unique lineups
    """
    if not population:
        return {'fitness_std': 0, 'fitness_range': 0, 'avg_hamming': 0,
                'unique_players': 0, 'unique_lineups': 0}

    # Fitness diversity
    fitnesses = [fitness_func(lineup) for lineup in population]
    fitness_std = float(np.std(fitnesses))
    fitness_range = float(max(fitnesses) - min(fitnesses))

    # Player diversity - unique players across population
    all_players = set()
    lineup_keys = set()
    for lineup in population:
        player_ids = get_player_ids(lineup)
        all_players.update(player_ids)
        lineup_keys.add(get_lineup_key(lineup))

    unique_players = len(all_players)
    unique_lineups = len(lineup_keys)

    # Average Hamming distance (sample if population is large)
    n = len(population)
    if n <= 20:
        # Full pairwise comparison for small populations
        hamming_sum = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                hamming_sum += calculate_hamming_distance(population[i], population[j])
                count += 1
        avg_hamming = hamming_sum / count if count > 0 else 0
    else:
        # Sample for larger populations
        import random
        sample_size = 100
        hamming_sum = 0
        for _ in range(sample_size):
            i, j = random.sample(range(n), 2)
            hamming_sum += calculate_hamming_distance(population[i], population[j])
        avg_hamming = hamming_sum / sample_size

    return {
        'fitness_std': round(fitness_std, 2),
        'fitness_range': round(fitness_range, 2),
        'avg_hamming': round(avg_hamming, 2),
        'unique_players': unique_players,
        'unique_lineups': unique_lineups
    }


def plot_training_metrics(
    best_fitness_history: List[float],
    diversity_history: List[Dict[str, float]],
    output_path: str,
    final_squeeze_generation: Optional[int] = None
) -> str:
    """
    Create a multi-panel plot of training metrics over generations.

    Args:
        best_fitness_history: List of best fitness values per generation
        diversity_history: List of diversity metric dicts per generation
        output_path: Directory to save the plot
        final_squeeze_generation: Generation where final squeeze occurred (if any)

    Returns:
        Path to saved plot file
    """
    generations = list(range(len(best_fitness_history)))

    # Extract diversity metrics
    fitness_std = [d['fitness_std'] for d in diversity_history]
    fitness_range = [d['fitness_range'] for d in diversity_history]
    avg_hamming = [d['avg_hamming'] for d in diversity_history]
    unique_players = [d['unique_players'] for d in diversity_history]
    unique_lineups = [d['unique_lineups'] for d in diversity_history]

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Style settings - try seaborn styles, fall back gracefully
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            pass  # Use default style

    colors = {
        'fitness': '#2ecc71',
        'hamming': '#3498db',
        'unique_players': '#9b59b6',
        'unique_lineups': '#e74c3c',
        'fitness_std': '#f39c12',
        'fitness_range': '#1abc9c'
    }

    # Helper to add final squeeze marker
    def add_squeeze_marker(ax, gen):
        if gen is not None:
            ax.axvline(x=gen, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(gen + 0.2, ax.get_ylim()[1] * 0.95, 'Final\nSqueeze',
                   fontsize=8, color='red', alpha=0.8, va='top')

    # 1. Best Fitness (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(generations, best_fitness_history, color=colors['fitness'],
             linewidth=2, marker='o', markersize=4, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Best Fitness Over Generations', fontweight='bold')
    ax1.legend(loc='lower right')
    add_squeeze_marker(ax1, final_squeeze_generation)

    # 2. Avg Hamming Distance (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(generations, avg_hamming, color=colors['hamming'],
             linewidth=2, marker='s', markersize=4, label='Avg Hamming Distance')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Avg Hamming Distance')
    ax2.set_title('Population Diversity (Hamming)', fontweight='bold')
    ax2.legend(loc='upper right')
    add_squeeze_marker(ax2, final_squeeze_generation)

    # 3. Unique Players (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(generations, unique_players, color=colors['unique_players'],
             linewidth=2, marker='^', markersize=4, label='Unique Players')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Count')
    ax3.set_title('Unique Players in Population', fontweight='bold')
    ax3.legend(loc='upper right')
    add_squeeze_marker(ax3, final_squeeze_generation)

    # 4. Unique Lineups (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(generations, unique_lineups, color=colors['unique_lineups'],
             linewidth=2, marker='d', markersize=4, label='Unique Lineups')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Count')
    ax4.set_title('Unique Lineups in Population', fontweight='bold')
    ax4.legend(loc='upper right')
    add_squeeze_marker(ax4, final_squeeze_generation)

    # 5. Fitness Spread - σ and Range (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(generations, fitness_std, color=colors['fitness_std'],
             linewidth=2, marker='o', markersize=4, label='Fitness σ')
    ax5.plot(generations, fitness_range, color=colors['fitness_range'],
             linewidth=2, marker='s', markersize=4, label='Fitness Range', alpha=0.7)
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Value')
    ax5.set_title('Fitness Distribution Spread', fontweight='bold')
    ax5.legend(loc='upper right')
    add_squeeze_marker(ax5, final_squeeze_generation)

    # 6. Combined normalized metrics (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])

    # Normalize each metric to [0, 1] for comparison
    def normalize(data):
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [0.5] * len(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    ax6.plot(generations, normalize(best_fitness_history), color=colors['fitness'],
             linewidth=2, label='Fitness', alpha=0.8)
    ax6.plot(generations, normalize(avg_hamming), color=colors['hamming'],
             linewidth=2, label='Hamming', alpha=0.8)
    ax6.plot(generations, normalize(unique_players), color=colors['unique_players'],
             linewidth=2, label='Unique Players', alpha=0.8)
    ax6.set_xlabel('Generation')
    ax6.set_ylabel('Normalized Value (0-1)')
    ax6.set_title('Combined Metrics (Normalized)', fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    add_squeeze_marker(ax6, final_squeeze_generation)

    # Overall title
    fig.suptitle('Genetic Algorithm Training Metrics', fontsize=14, fontweight='bold', y=0.98)

    # Save plot
    output_dir = Path(output_path).parent
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    return str(plot_path)


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
    diversity_history = []  # Track population diversity over generations
    archive_diversity_history = []  # Track elite archive diversity
    final_squeeze_done = False  # Track if we've done the final high-sim generation
    final_squeeze_generation = None  # Track which generation the squeeze happened

    # Calculate initial diversity for BOTH population and archive
    initial_pop_diversity = calculate_population_diversity(population, fitness_func)
    initial_archive_diversity = calculate_population_diversity(elite_archive, fitness_func)
    diversity_history.append(initial_pop_diversity)
    archive_diversity_history.append(initial_archive_diversity)

    if verbose:
        print(f"\n{'='*60}")
        print(f"INITIAL STATE (Generation 0)")
        print(f"{'='*60}")
        print(f"\nStarting Population ({len(population)} lineups):")
        print(f"  Unique lineups:      {initial_pop_diversity['unique_lineups']}/{len(population)}")
        print(f"  Unique players:      {initial_pop_diversity['unique_players']}")
        print(f"  Avg Hamming dist:    {initial_pop_diversity['avg_hamming']:.1f}")
        print(f"  Fitness range:       {initial_pop_diversity['fitness_range']:.1f} (σ={initial_pop_diversity['fitness_std']:.1f})")

        print(f"\nElite Archive ({len(elite_archive)} lineups):")
        print(f"  Best fitness:        {best_archive_fitness:.2f}")
        print(f"  Unique players:      {initial_archive_diversity['unique_players']}")
        print(f"  Avg Hamming dist:    {initial_archive_diversity['avg_hamming']:.1f}")

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

        # Calculate and track diversity for BOTH population and archive
        gen_pop_diversity = calculate_population_diversity(population, fitness_func)
        gen_archive_diversity = calculate_population_diversity(elite_archive, fitness_func)
        diversity_history.append(gen_pop_diversity)
        archive_diversity_history.append(gen_archive_diversity)

        gen_time = time.time() - gen_start_time

        if verbose:
            print(f"\nElite Archive:")
            print(f"  Size: {len(elite_archive)}/{elite_archive_size}")
            print(f"  Best fitness: {best_archive_fitness:.2f} {'(NEW BEST!)' if improvement else ''}")

            # Show population diversity trend
            prev_diversity = diversity_history[-2] if len(diversity_history) > 1 else gen_pop_diversity
            hamming_delta = gen_pop_diversity['avg_hamming'] - prev_diversity['avg_hamming']
            players_delta = gen_pop_diversity['unique_players'] - prev_diversity['unique_players']
            print(f"\nPopulation Diversity:")
            print(f"  Unique lineups: {gen_pop_diversity['unique_lineups']}/{len(population)}")
            print(f"  Unique players: {gen_pop_diversity['unique_players']} ({players_delta:+d})")
            print(f"  Avg Hamming distance: {gen_pop_diversity['avg_hamming']:.1f} ({hamming_delta:+.1f})")
            print(f"  Fitness range: {gen_pop_diversity['fitness_range']:.1f} (σ={gen_pop_diversity['fitness_std']:.1f})")

            print(f"\nGeneration time: {gen_time:.1f} seconds")

        # =================================================================
        # 7. CHECK CONVERGENCE (with final squeeze)
        # =================================================================
        if len(best_fitness_history) > convergence_patience:
            recent = best_fitness_history[-convergence_patience:]
            no_improvement = all(recent[i] == recent[0] for i in range(len(recent)))

            if no_improvement and not final_squeeze_done:
                # About to converge - do one final generation with 2x simulations
                final_squeeze_done = True
                final_squeeze_generation = generation + 1  # Next generation will be the squeeze
                n_sims = n_sims * 2  # Double simulation count for remaining generations

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"FINAL SQUEEZE: No improvement in {convergence_patience} generations")
                    print(f"Doubling simulations to {n_sims} for final push...")
                    print(f"{'='*60}")
                # Continue to next generation with more sims
                continue
            elif no_improvement and final_squeeze_done:
                # Already did final squeeze, now truly converged
                if verbose:
                    print(f"\nConverged! No improvement after final squeeze.")
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
        if final_squeeze_done:
            print(f"Final squeeze: Yes (doubled sims on last generation)")

        # Diversity summary - Population
        if len(diversity_history) >= 2:
            first_div = diversity_history[0]
            last_div = diversity_history[-1]
            print(f"\nPopulation diversity evolution:")
            print(f"  Unique players: {first_div['unique_players']} → {last_div['unique_players']} "
                  f"({last_div['unique_players'] - first_div['unique_players']:+d})")
            print(f"  Avg Hamming:    {first_div['avg_hamming']:.1f} → {last_div['avg_hamming']:.1f} "
                  f"({last_div['avg_hamming'] - first_div['avg_hamming']:+.1f})")
            print(f"  Fitness σ:      {first_div['fitness_std']:.1f} → {last_div['fitness_std']:.1f} "
                  f"({last_div['fitness_std'] - first_div['fitness_std']:+.1f})")

        # Diversity summary - Elite Archive
        if len(archive_diversity_history) >= 2:
            first_arch = archive_diversity_history[0]
            last_arch = archive_diversity_history[-1]
            print(f"\nElite archive diversity evolution:")
            print(f"  Unique players: {first_arch['unique_players']} → {last_arch['unique_players']} "
                  f"({last_arch['unique_players'] - first_arch['unique_players']:+d})")
            print(f"  Avg Hamming:    {first_arch['avg_hamming']:.1f} → {last_arch['avg_hamming']:.1f} "
                  f"({last_arch['avg_hamming'] - first_arch['avg_hamming']:+.1f})")

    # Generate training metrics plot
    if len(best_fitness_history) > 1 and len(diversity_history) > 1:
        try:
            plot_path = plot_training_metrics(
                best_fitness_history=best_fitness_history,
                diversity_history=diversity_history,
                output_path=output_path,
                final_squeeze_generation=final_squeeze_generation
            )
            if verbose:
                print(f"\nTraining metrics plot saved to: {plot_path}")
        except Exception as e:
            if verbose:
                import traceback
                print(f"\nWarning: Could not generate training plot: {e}")
                traceback.print_exc()

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
