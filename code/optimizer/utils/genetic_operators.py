"""
Genetic algorithm operators: selection, crossover, mutation.

Used in Phase 3 for iterative lineup refinement.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
import random


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


# ============================================================================
# SELECTION
# ============================================================================

def tournament_select(
    population: List[Dict],
    fitness_func: Callable[[Dict], float],
    tournament_size: int = 5,
    n_parents: int = 50
) -> List[Dict]:
    """
    Tournament selection: randomly select k individuals, pick the best.

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
