"""
Tools for measuring and maintaining diversity in genetic algorithm populations.
"""

import numpy as np
from typing import List, Dict, Set
from collections import Counter


def calculate_uniqueness(population: List[Dict]) -> float:
    """
    Calculate what percentage of the population is truly unique.

    Returns:
        Float between 0 and 1 (1 = all unique, 0 = all duplicates)
    """
    player_id_sets = []
    for lineup in population:
        player_ids = lineup['player_ids']
        if isinstance(player_ids, str):
            player_ids = player_ids.split(',')
        # Sort to handle same players in different order
        player_id_sets.append(tuple(sorted(player_ids)))

    unique_count = len(set(player_id_sets))
    total_count = len(population)

    return unique_count / total_count if total_count > 0 else 0


def calculate_player_diversity(population: List[Dict]) -> Dict[str, float]:
    """
    Calculate how diverse the player pool is across the population.

    Returns:
        Dict with:
        - player_count: Number of unique players used
        - avg_player_usage: Average times each player appears
        - max_player_usage: Most common player usage count
    """
    all_players = []
    for lineup in population:
        player_ids = lineup['player_ids']
        if isinstance(player_ids, str):
            player_ids = player_ids.split(',')
        all_players.extend(player_ids)

    player_counts = Counter(all_players)

    return {
        'player_count': len(player_counts),
        'avg_player_usage': np.mean(list(player_counts.values())),
        'max_player_usage': max(player_counts.values()) if player_counts else 0,
        'player_counts': player_counts  # For analysis
    }


def hamming_distance(lineup1: Dict, lineup2: Dict) -> int:
    """
    Calculate Hamming distance between two lineups (number of different players).
    """
    ids1 = set(lineup1['player_ids'].split(',') if isinstance(lineup1['player_ids'], str) else lineup1['player_ids'])
    ids2 = set(lineup2['player_ids'].split(',') if isinstance(lineup2['player_ids'], str) else lineup2['player_ids'])

    # Number of players that differ
    return len(ids1.symmetric_difference(ids2))


def calculate_avg_distance(population: List[Dict]) -> float:
    """
    Calculate average Hamming distance between all pairs in population.
    Higher = more diverse.
    """
    if len(population) < 2:
        return 0

    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distances.append(hamming_distance(population[i], population[j]))

    return np.mean(distances)


def is_duplicate(lineup: Dict, population: List[Dict], threshold: int = 2) -> bool:
    """
    Check if a lineup is too similar to existing population members.

    Args:
        lineup: Lineup to check
        population: Existing population
        threshold: Max number of different players to be considered unique

    Returns:
        True if lineup is a duplicate (Hamming distance < threshold for any member)
    """
    for member in population:
        if hamming_distance(lineup, member) < threshold:
            return True
    return False


def deduplicate_population(population: List[Dict], min_distance: int = 2) -> List[Dict]:
    """
    Remove near-duplicate lineups from population.
    Keeps the first occurrence (typically the one with better fitness).

    Args:
        population: Population (assumed sorted by fitness)
        min_distance: Minimum Hamming distance required

    Returns:
        Deduplicated population
    """
    unique_population = []

    for lineup in population:
        if not is_duplicate(lineup, unique_population, threshold=min_distance):
            unique_population.append(lineup)

    return unique_population


def get_diversity_report(population: List[Dict]) -> str:
    """
    Generate a human-readable diversity report for debugging.
    """
    uniqueness = calculate_uniqueness(population)
    player_div = calculate_player_diversity(population)
    avg_dist = calculate_avg_distance(population)

    report = f"""
Diversity Metrics:
  Population size: {len(population)}
  Unique lineups: {int(uniqueness * len(population))} ({uniqueness:.1%})
  Unique players used: {player_div['player_count']}
  Avg player usage: {player_div['avg_player_usage']:.1f}x
  Max player usage: {player_div['max_player_usage']}x
  Avg Hamming distance: {avg_dist:.1f}
"""
    return report
