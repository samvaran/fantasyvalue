"""
MILP (Mixed Integer Linear Programming) solver for lineup optimization.

Uses PuLP library to formulate and solve the knapsack problem with constraints.

Performance optimizations:
- Pre-compute player data as lists/dicts (avoid iterrows)
- Pre-group players by position
- Cache solver instance
"""

import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import warnings

# Global cache for prepared player data
_player_cache = {}


def _prepare_player_data(players_df: pd.DataFrame) -> Dict:
    """
    Pre-compute player data structures for fast MILP construction.
    Returns cached data if DataFrame hasn't changed.
    """
    # Create a cache key based on DataFrame shape and column sum (fast hash proxy)
    cache_key = (len(players_df), players_df['projection'].sum() if 'projection' in players_df.columns else 0)

    if cache_key in _player_cache:
        return _player_cache[cache_key]

    # Normalize column names
    df = players_df.copy()
    if 'id' not in df.columns and 'name' in df.columns:
        df['id'] = df['name']
    if 'fdSalary' not in df.columns and 'salary' in df.columns:
        df['fdSalary'] = df['salary']

    # Drop players with missing data
    required_cols = ['id', 'name', 'position', 'fdSalary', 'projection']
    df = df.dropna(subset=required_cols)

    # Pre-compute as lists (much faster than iterrows)
    player_ids = df['id'].tolist()
    player_projections = df['projection'].tolist()
    player_salaries = df['fdSalary'].tolist()
    player_positions = df['position'].tolist()

    # Build lookup dicts
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    id_to_projection = dict(zip(player_ids, player_projections))
    id_to_salary = dict(zip(player_ids, player_salaries))

    # Pre-group by position
    position_ids = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'D': []}
    for pid, pos in zip(player_ids, player_positions):
        if pos in position_ids:
            position_ids[pos].append(pid)

    # Store player data for solution extraction
    player_data = {pid: df[df['id'] == pid].iloc[0].to_dict() for pid in player_ids}

    result = {
        'player_ids': player_ids,
        'id_to_projection': id_to_projection,
        'id_to_salary': id_to_salary,
        'position_ids': position_ids,
        'player_data': player_data,
        'df': df
    }

    # Cache (keep only last 3 to avoid memory bloat)
    if len(_player_cache) > 3:
        _player_cache.clear()
    _player_cache[cache_key] = result

    return result


def create_lineup_milp(
    players_df: pd.DataFrame,
    salary_cap: float = 60000,
    exclude_players: Optional[Set[str]] = None,
    max_overlap_with: Optional[List[List[str]]] = None,
    max_overlap: int = 7
) -> Optional[Dict]:
    """
    Create and solve MILP problem for optimal lineup.

    Args:
        players_df: DataFrame with columns: id, name, position, fdSalary, projection
        salary_cap: Maximum total salary (default: 60000)
        exclude_players: Set of player IDs to exclude
        max_overlap_with: List of previous lineups to enforce diversity
        max_overlap: Maximum number of overlapping players with previous lineups

    Returns:
        Dict with lineup info, or None if infeasible
    """
    # Get pre-computed player data
    data = _prepare_player_data(players_df)

    player_ids = data['player_ids']
    id_to_projection = data['id_to_projection']
    id_to_salary = data['id_to_salary']
    position_ids = data['position_ids']
    player_data = data['player_data']

    # Filter out excluded players
    if exclude_players:
        player_ids = [pid for pid in player_ids if pid not in exclude_players]

    if len(player_ids) == 0:
        warnings.warn("No valid players after filtering")
        return None

    # Create MILP problem
    prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)

    # Decision variables: binary (0 or 1) for each player
    player_vars = pulp.LpVariable.dicts("player", player_ids, cat=pulp.LpBinary)

    # Objective: Maximize total projected points
    prob += pulp.lpSum([
        player_vars[pid] * id_to_projection[pid]
        for pid in player_ids
    ]), "Total_Projection"

    # Constraint 1: Salary cap
    prob += pulp.lpSum([
        player_vars[pid] * id_to_salary[pid]
        for pid in player_ids
    ]) <= salary_cap, "Salary_Cap"

    # Constraint 2: Exactly 9 players
    prob += pulp.lpSum([player_vars[pid] for pid in player_ids]) == 9, "Total_Players"

    # Constraint 3: Position requirements (using pre-grouped positions)
    # Filter position lists by available players
    available_set = set(player_ids)

    qb_ids = [pid for pid in position_ids['QB'] if pid in available_set]
    rb_ids = [pid for pid in position_ids['RB'] if pid in available_set]
    wr_ids = [pid for pid in position_ids['WR'] if pid in available_set]
    te_ids = [pid for pid in position_ids['TE'] if pid in available_set]
    def_ids = [pid for pid in position_ids['D'] if pid in available_set]

    # QB: exactly 1
    if qb_ids:
        prob += pulp.lpSum([player_vars[pid] for pid in qb_ids]) == 1, "QB_Count"
    else:
        warnings.warn("No QBs available")
        return None

    # RB: at least 2 (can be 3 with FLEX)
    if len(rb_ids) >= 2:
        prob += pulp.lpSum([player_vars[pid] for pid in rb_ids]) >= 2, "RB_Min"
        prob += pulp.lpSum([player_vars[pid] for pid in rb_ids]) <= 3, "RB_Max"
    else:
        warnings.warn("Not enough RBs available")
        return None

    # WR: at least 3 (can be 4 with FLEX)
    if len(wr_ids) >= 3:
        prob += pulp.lpSum([player_vars[pid] for pid in wr_ids]) >= 3, "WR_Min"
        prob += pulp.lpSum([player_vars[pid] for pid in wr_ids]) <= 4, "WR_Max"
    else:
        warnings.warn("Not enough WRs available")
        return None

    # TE: at least 1 (can be 2 with FLEX)
    if te_ids:
        prob += pulp.lpSum([player_vars[pid] for pid in te_ids]) >= 1, "TE_Min"
        prob += pulp.lpSum([player_vars[pid] for pid in te_ids]) <= 2, "TE_Max"
    else:
        warnings.warn("Not enough TEs available")
        return None

    # DEF: exactly 1
    if def_ids:
        prob += pulp.lpSum([player_vars[pid] for pid in def_ids]) == 1, "DEF_Count"
    else:
        warnings.warn("No DEFs available")
        return None

    # FLEX constraint: RB + WR + TE = 6 or 7 total
    flex_ids = rb_ids + wr_ids + te_ids
    prob += pulp.lpSum([player_vars[pid] for pid in flex_ids]) >= 6, "FLEX_Min"
    prob += pulp.lpSum([player_vars[pid] for pid in flex_ids]) <= 7, "FLEX_Max"

    # Constraint 4: Diversity (max overlap with previous lineups)
    if max_overlap_with:
        for i, prev_lineup in enumerate(max_overlap_with):
            overlap_ids = [pid for pid in prev_lineup if pid in available_set]
            if overlap_ids:
                prob += pulp.lpSum([player_vars[pid] for pid in overlap_ids]) <= max_overlap, f"Diversity_{i}"

    # Solve with optimized settings
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30, gapRel=0.02)
    prob.solve(solver)

    # Check status
    if prob.status != pulp.LpStatusOptimal:
        warnings.warn(f"MILP solver failed with status: {pulp.LpStatus[prob.status]}")
        return None

    # Extract solution (using pre-computed player data)
    selected_players = []
    total_salary = 0
    total_projection = 0

    for pid in player_ids:
        if player_vars[pid].varValue == 1:
            selected_players.append(player_data[pid])
            total_salary += id_to_salary[pid]
            total_projection += id_to_projection[pid]

    return {
        'players': selected_players,
        'total_salary': total_salary,
        'total_projection': total_projection,
        'player_ids': [p['id'] for p in selected_players]
    }


def clear_player_cache():
    """Clear the player data cache. Call when switching to a different player pool."""
    global _player_cache
    _player_cache.clear()


def generate_diverse_lineups(
    players_df: pd.DataFrame,
    n_lineups: int,
    max_overlap: int = 7,
    salary_cap: float = 60000,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate multiple diverse lineups using MILP.

    Args:
        players_df: DataFrame with player data
        n_lineups: Number of lineups to generate
        max_overlap: Maximum overlapping players between lineups
        salary_cap: Salary cap
        verbose: Print progress

    Returns:
        List of lineup dicts
    """
    lineups = []
    previous_lineups = []

    for i in range(n_lineups):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_lineups} lineups...")

        lineup = create_lineup_milp(
            players_df,
            salary_cap=salary_cap,
            max_overlap_with=previous_lineups if i > 0 else None,
            max_overlap=max_overlap
        )

        if lineup is None:
            if verbose:
                print(f"  Warning: Could not generate lineup {i + 1}. Stopping early.")
            break

        lineups.append(lineup)
        previous_lineups.append(lineup['player_ids'])

    if verbose:
        print(f"  Successfully generated {len(lineups)} lineups")

    return lineups


if __name__ == '__main__':
    import time

    # Test MILP solver with dummy data
    print("Testing MILP solver...")

    # Create realistic player pool (~150 players like real NFL slate)
    np.random.seed(42)
    players = []

    # More realistic position distribution
    position_counts = {'QB': 32, 'RB': 50, 'WR': 60, 'TE': 25, 'D': 32}

    for pos, count in position_counts.items():
        for i in range(count):
            players.append({
                'id': f'{pos}_{i}',
                'name': f'{pos} Player {i}',
                'position': pos,
                'fdSalary': np.random.randint(4500, 9500),
                'projection': np.random.uniform(5, 25)
            })

    players_df = pd.DataFrame(players)

    print(f"\nPlayer pool: {len(players_df)} players")
    print(f"  QBs: {len(players_df[players_df['position'] == 'QB'])}")
    print(f"  RBs: {len(players_df[players_df['position'] == 'RB'])}")
    print(f"  WRs: {len(players_df[players_df['position'] == 'WR'])}")
    print(f"  TEs: {len(players_df[players_df['position'] == 'TE'])}")
    print(f"  DEFs: {len(players_df[players_df['position'] == 'D'])}")

    # Benchmark: Generate single lineup
    print("\n--- Single Lineup Generation ---")
    start = time.time()
    lineup = create_lineup_milp(players_df)
    elapsed = time.time() - start
    print(f"Time: {elapsed*1000:.1f}ms")

    if lineup:
        print(f"Projection: {lineup['total_projection']:.2f}, Salary: ${lineup['total_salary']:,}")

    # Benchmark: Generate 50 diverse lineups
    print("\n--- Generating 50 Diverse Lineups ---")
    clear_player_cache()  # Clear cache to measure cold start
    start = time.time()
    lineups = generate_diverse_lineups(players_df, n_lineups=50, max_overlap=6, verbose=False)
    elapsed = time.time() - start

    print(f"Generated {len(lineups)} lineups in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(lineups)*1000:.1f}ms per lineup")

    # Benchmark: Generate 200 lineups (typical production run)
    print("\n--- Generating 200 Diverse Lineups ---")
    clear_player_cache()
    start = time.time()
    lineups = generate_diverse_lineups(players_df, n_lineups=200, max_overlap=7, verbose=False)
    elapsed = time.time() - start

    print(f"Generated {len(lineups)} lineups in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(lineups)*1000:.1f}ms per lineup")

    print("\nMILP solver test complete!")
