"""
Genetic algorithm operators: selection, crossover, mutation.

Used in Phase 3 for iterative lineup refinement.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Set
import random
import sys
from pathlib import Path

# Import config values (add parent dir to path temporarily)
_parent_dir = str(Path(__file__).parent.parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from config import GENETIC_TOURNAMENT_SIZE, GENETIC_NUM_PARENTS, SALARY_CAP


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_player_id_column(players_df: pd.DataFrame) -> str:
    """
    Determine which column to use as player identifier.
    Prefers 'id' if it exists, otherwise uses 'name'.
    """
    if 'id' in players_df.columns:
        return 'id'
    elif 'name' in players_df.columns:
        return 'name'
    else:
        raise ValueError("Players dataframe must have either 'id' or 'name' column")


def get_player_ids(lineup: Dict) -> List[str]:
    """Extract player IDs as a list from lineup dict."""
    player_ids = lineup.get('player_ids', '')
    if isinstance(player_ids, str):
        return player_ids.split(',') if player_ids else []
    return list(player_ids)


def calculate_lineup_salary(lineup: Dict, players_df: pd.DataFrame) -> float:
    """
    Calculate total salary for a lineup.

    Args:
        lineup: Lineup dict with 'player_ids' key
        players_df: DataFrame with player data including salary

    Returns:
        Total salary of all players in lineup
    """
    player_ids = get_player_ids(lineup)
    id_col = get_player_id_column(players_df)
    salary_col = 'salary' if 'salary' in players_df.columns else 'fdSalary'

    total_salary = 0
    for pid in player_ids:
        player = players_df[players_df[id_col] == pid]
        if len(player) > 0:
            total_salary += player.iloc[0][salary_col]

    return total_salary


def calculate_hamming_distance(lineup1: Dict, lineup2: Dict) -> int:
    """
    Calculate Hamming distance (number of different players) between two lineups.
    """
    ids1 = set(get_player_ids(lineup1))
    ids2 = set(get_player_ids(lineup2))
    return len(ids1.symmetric_difference(ids2))


def get_lineup_key(lineup: Dict) -> str:
    """Get a unique key for a lineup based on sorted player IDs."""
    player_ids = get_player_ids(lineup)
    return ','.join(sorted(player_ids))


# ============================================================================
# SELECTION
# ============================================================================

def tournament_select_without_replacement(
    population: List[Dict],
    fitness_func: Callable[[Dict], float],
    tournament_size: int = GENETIC_TOURNAMENT_SIZE,
    n_parents: int = GENETIC_NUM_PARENTS
) -> Tuple[List[Dict], List[Dict]]:
    """
    Tournament selection WITHOUT replacement.

    Selects n_parents from population, returns both selected parents
    and the remaining non-parents.

    Args:
        population: List of lineup dicts with simulation results
        fitness_func: Function to calculate fitness from lineup dict
        tournament_size: Number of individuals per tournament (k)
        n_parents: Number of parents to select

    Returns:
        Tuple of (parents, non_parents) - both sorted by fitness descending
    """
    # Work with indices to track what's been selected
    available_indices = list(range(len(population)))
    parent_indices = []

    for _ in range(min(n_parents, len(population))):
        if len(available_indices) < tournament_size:
            # Not enough left for full tournament, just take best remaining
            if available_indices:
                best_idx = max(available_indices, key=lambda i: fitness_func(population[i]))
                parent_indices.append(best_idx)
                available_indices.remove(best_idx)
        else:
            # Run tournament
            tournament_indices = random.sample(available_indices, tournament_size)
            best_idx = max(tournament_indices, key=lambda i: fitness_func(population[i]))
            parent_indices.append(best_idx)
            available_indices.remove(best_idx)

    # Build parent and non-parent lists
    parents = [population[i] for i in parent_indices]
    non_parents = [population[i] for i in available_indices]

    # Sort both by fitness descending
    parents.sort(key=fitness_func, reverse=True)
    non_parents.sort(key=fitness_func, reverse=True)

    return parents, non_parents


def tournament_select(
    population: List[Dict],
    fitness_func: Callable[[Dict], float],
    tournament_size: int = GENETIC_TOURNAMENT_SIZE,
    n_parents: int = GENETIC_NUM_PARENTS
) -> List[Dict]:
    """
    Tournament selection WITH replacement (legacy behavior).

    Args:
        population: List of lineup dicts with simulation results
        fitness_func: Function to calculate fitness from lineup dict
        tournament_size: Number of individuals per tournament
        n_parents: Number of parents to select

    Returns:
        List of selected parent lineups
    """
    parents = []

    for _ in range(n_parents):
        # Randomly select tournament_size individuals
        tournament = random.sample(population, min(tournament_size, len(population)))

        # Pick the best one based on fitness
        best = max(tournament, key=fitness_func)
        parents.append(best)

    return parents


# ============================================================================
# CROSSOVER
# ============================================================================

def crossover(
    parent1: Dict,
    parent2: Dict,
    players_df: pd.DataFrame,
    crossover_rate: float = 0.8
) -> Tuple[Dict, Dict]:
    """
    Position-aware crossover: swap players by position to maintain valid lineups.

    Args:
        parent1: First parent lineup dict with 'player_ids' key
        parent2: Second parent lineup dict with 'player_ids' key
        players_df: DataFrame with player data (to get positions)
        crossover_rate: Probability of crossover (vs just returning parents)

    Returns:
        Tuple of (child1, child2) lineup dicts
    """
    if random.random() > crossover_rate:
        # No crossover, return copies of parents
        return parent1.copy(), parent2.copy()

    # Parse player IDs
    p1_ids = parent1['player_ids'].split(',') if isinstance(parent1['player_ids'], str) else parent1['player_ids']
    p2_ids = parent2['player_ids'].split(',') if isinstance(parent2['player_ids'], str) else parent2['player_ids']

    # Build position mappings
    def get_player_positions(player_ids):
        positions = {}
        for pid in player_ids:
            # Try to find player by id if id column exists, otherwise by name
            if 'id' in players_df.columns:
                player = players_df[players_df['id'] == pid]
            else:
                player = players_df[players_df['name'] == pid]

            # Fallback to name if id lookup failed
            if len(player) == 0 and 'name' in players_df.columns:
                player = players_df[players_df['name'] == pid]

            if len(player) > 0:
                pos = player.iloc[0]['position']
                if pos not in positions:
                    positions[pos] = []
                positions[pos].append(pid)
        return positions

    p1_positions = get_player_positions(p1_ids)
    p2_positions = get_player_positions(p2_ids)

    # Crossover: for each position, randomly choose parent to inherit from
    child1_ids = []
    child2_ids = []

    all_positions = set(p1_positions.keys()) | set(p2_positions.keys())

    for pos in all_positions:
        p1_players = p1_positions.get(pos, [])
        p2_players = p2_positions.get(pos, [])

        if random.random() < 0.5:
            # Child1 inherits from parent1, child2 from parent2
            child1_ids.extend(p1_players)
            child2_ids.extend(p2_players)
        else:
            # Child1 inherits from parent2, child2 from parent1
            child1_ids.extend(p2_players)
            child2_ids.extend(p1_players)

    # Create child dicts - preserve metadata from parents where applicable
    child1 = {
        'player_ids': ','.join(child1_ids) if isinstance(parent1['player_ids'], str) else child1_ids,
        'lineup_id': f"child_{random.randint(100000, 999999)}"
    }

    # Preserve metadata fields if they exist in parents (use parent1 as default)
    for field in ['tier', 'temperature', 'milp_projection', 'total_salary']:
        if field in parent1:
            child1[field] = parent1[field]

    child2 = {
        'player_ids': ','.join(child2_ids) if isinstance(parent2['player_ids'], str) else child2_ids,
        'lineup_id': f"child_{random.randint(100000, 999999)}"
    }

    # Preserve metadata fields if they exist in parents (use parent2 as default)
    for field in ['tier', 'temperature', 'milp_projection', 'total_salary']:
        if field in parent2:
            child2[field] = parent2[field]

    return child1, child2


# ============================================================================
# PAIRING STRATEGIES
# ============================================================================

def pair_strength_matched(parents: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Pair parents by rank: 1st with 2nd, 3rd with 4th, etc.
    Assumes parents are already sorted by fitness descending.
    """
    pairs = []
    for i in range(0, len(parents) - 1, 2):
        pairs.append((parents[i], parents[i + 1]))
    return pairs


def pair_inverse_matched(parents: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Pair parents by inverse rank: 1st with last, 2nd with 2nd-last, etc.
    Assumes parents are already sorted by fitness descending.
    """
    pairs = []
    n = len(parents)
    for i in range(n // 2):
        pairs.append((parents[i], parents[n - 1 - i]))
    return pairs


def pair_random(parents: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Pair parents randomly.
    """
    shuffled = parents.copy()
    random.shuffle(shuffled)
    pairs = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    return pairs


def pair_dissimilar(parents: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Pair parents to maximize player difference (Hamming distance).
    Uses greedy matching: for each unpaired parent, find the most dissimilar
    unpaired partner.
    """
    if len(parents) < 2:
        return []

    pairs = []
    remaining = list(range(len(parents)))

    while len(remaining) >= 2:
        # Take the first remaining parent
        idx1 = remaining.pop(0)
        parent1 = parents[idx1]

        # Find the most dissimilar remaining parent
        best_idx = None
        best_distance = -1

        for idx2 in remaining:
            parent2 = parents[idx2]
            distance = calculate_hamming_distance(parent1, parent2)
            if distance > best_distance:
                best_distance = distance
                best_idx = idx2

        if best_idx is not None:
            remaining.remove(best_idx)
            pairs.append((parent1, parents[best_idx]))

    return pairs


def crossover_with_strategy(
    parents: List[Dict],
    players_df: pd.DataFrame,
    strategy: str = 'random',
    n_offspring: int = 25,
    crossover_rate: float = 1.0
) -> List[Dict]:
    """
    Produce offspring using a specific pairing strategy.

    Args:
        parents: List of parent lineups (sorted by fitness descending)
        players_df: DataFrame with player data
        strategy: One of 'strength', 'inverse', 'random', 'dissimilar'
        n_offspring: Number of offspring to produce
        crossover_rate: Probability of crossover per pair

    Returns:
        List of offspring lineups
    """
    # Get pairs based on strategy
    if strategy == 'strength':
        pairs = pair_strength_matched(parents)
    elif strategy == 'inverse':
        pairs = pair_inverse_matched(parents)
    elif strategy == 'random':
        pairs = pair_random(parents)
    elif strategy == 'dissimilar':
        pairs = pair_dissimilar(parents)
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")

    # Generate offspring from pairs
    offspring = []
    pair_idx = 0

    while len(offspring) < n_offspring and pairs:
        parent1, parent2 = pairs[pair_idx % len(pairs)]
        child1, child2 = crossover(parent1, parent2, players_df, crossover_rate)
        offspring.append(child1)
        if len(offspring) < n_offspring:
            offspring.append(child2)
        pair_idx += 1

        # If we've used all pairs once and still need more, cycle through again
        if pair_idx >= len(pairs) and len(offspring) < n_offspring:
            pair_idx = 0

    return offspring[:n_offspring]


# ============================================================================
# MUTATION
# ============================================================================

def mutate(
    lineup: Dict,
    players_df: pd.DataFrame,
    mutation_rate: float = 0.3,
    salary_tolerance: float = 500,
    aggressive: bool = False
) -> Dict:
    """
    Mutate a lineup by swapping 1-2 players with similar salary.

    Args:
        lineup: Lineup dict with 'player_ids' key
        players_df: DataFrame with player data
        mutation_rate: Probability of mutation
        salary_tolerance: Max salary difference for swaps (default $500)
        aggressive: If True, swap 2-4 players with wider salary tolerance for more diversity

    Returns:
        Mutated lineup dict
    """
    if random.random() > mutation_rate:
        # No mutation
        return lineup.copy()

    # Parse player IDs
    player_ids = lineup['player_ids'].split(',') if isinstance(lineup['player_ids'], str) else lineup['player_ids']

    # Choose 1-2 players to swap (or 2-4 if aggressive)
    if aggressive:
        n_swaps = random.randint(2, 4)
        # Wider salary tolerance for more diversity
        salary_tolerance = salary_tolerance * 2
    else:
        n_swaps = random.randint(1, 2)

    mutated_ids = player_ids.copy()

    for _ in range(n_swaps):
        if len(mutated_ids) == 0:
            break

        # Pick a random player to replace
        idx_to_replace = random.randint(0, len(mutated_ids) - 1)
        old_player_id = mutated_ids[idx_to_replace]

        # Get player ID column name
        id_col = get_player_id_column(players_df)

        # Get old player info
        old_player = players_df[players_df[id_col] == old_player_id]

        if len(old_player) == 0:
            continue

        old_player = old_player.iloc[0]

        # Get salary column (try 'salary' first, fallback to 'fdSalary')
        salary_col = 'salary' if 'salary' in players_df.columns else 'fdSalary'
        old_salary = old_player[salary_col]
        old_position = old_player['position']

        # Find candidates: same position, similar salary, not already in lineup
        candidates = players_df[
            (players_df['position'] == old_position) &
            (players_df[salary_col] >= old_salary - salary_tolerance) &
            (players_df[salary_col] <= old_salary + salary_tolerance) &
            (~players_df[id_col].isin(mutated_ids))
        ]

        if len(candidates) == 0:
            continue

        # Pick a random candidate
        new_player = candidates.sample(1).iloc[0]
        mutated_ids[idx_to_replace] = new_player[id_col]

    # Create mutated dict - preserve metadata from original lineup
    mutated_lineup = {
        'player_ids': ','.join(mutated_ids) if isinstance(lineup['player_ids'], str) else mutated_ids,
        'lineup_id': f"mutated_{random.randint(100000, 999999)}"
    }

    # Preserve metadata fields if they exist in original lineup
    for field in ['tier', 'temperature', 'milp_projection', 'total_salary']:
        if field in lineup:
            mutated_lineup[field] = lineup[field]

    return mutated_lineup


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

def fitness_conservative(lineup: Dict) -> float:
    """Conservative: median - 0.5 * std (high floor, low risk)"""
    return lineup.get('median', 0) - 0.5 * lineup.get('std', 0)


def fitness_balanced(lineup: Dict) -> float:
    """Balanced: median (solid expected outcome)"""
    return lineup.get('median', 0)


def fitness_aggressive(lineup: Dict) -> float:
    """Aggressive: p90 - p10 (boom potential vs bust risk)"""
    return lineup.get('p90', 0) - lineup.get('p10', 0)


def fitness_tournament(lineup: Dict) -> float:
    """Tournament: p90 (pure upside, swing for fences)"""
    return lineup.get('p90', 0)


# Fitness function lookup
FITNESS_FUNCTIONS = {
    'conservative': fitness_conservative,
    'balanced': fitness_balanced,
    'aggressive': fitness_aggressive,
    'tournament': fitness_tournament
}


if __name__ == '__main__':
    # Test genetic operators
    print("Testing genetic operators...")

    # Create dummy lineups
    lineup1 = {
        'player_ids': 'p1,p2,p3,p4,p5,p6,p7,p8,p9',
        'lineup_id': 1,
        'median': 150,
        'p10': 120,
        'p90': 180,
        'std': 15
    }

    lineup2 = {
        'player_ids': 'p10,p11,p12,p13,p14,p15,p16,p17,p18',
        'lineup_id': 2,
        'median': 155,
        'p10': 125,
        'p90': 185,
        'std': 12
    }

    # Test fitness functions
    print("\nFitness scores for lineup 1:")
    print(f"  Conservative: {fitness_conservative(lineup1):.2f}")
    print(f"  Balanced: {fitness_balanced(lineup1):.2f}")
    print(f"  Aggressive: {fitness_aggressive(lineup1):.2f}")
    print(f"  Tournament: {fitness_tournament(lineup1):.2f}")

    # Test tournament selection
    population = [lineup1, lineup2]
    parents = tournament_select(population, fitness_balanced, tournament_size=2, n_parents=5)
    print(f"\nTournament selection picked {len(parents)} parents")

    print("\nGenetic operators test complete!")
