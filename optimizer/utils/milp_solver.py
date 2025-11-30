"""
MILP (Mixed Integer Linear Programming) solver for lineup optimization.

Uses PuLP library to formulate and solve the knapsack problem with constraints.
"""

import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
import warnings


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
    # Normalize column names (handle both 'id' and 'name', 'fdSalary' and 'salary')
    if 'id' not in players_df.columns and 'name' in players_df.columns:
        players_df['id'] = players_df['name']
    if 'fdSalary' not in players_df.columns and 'salary' in players_df.columns:
        players_df['fdSalary'] = players_df['salary']

    # Filter out excluded players
    if exclude_players:
        players_df = players_df[~players_df['id'].isin(exclude_players)].copy()

    # Drop players with missing data
    required_cols = ['id', 'name', 'position', 'fdSalary', 'projection']
    players_df = players_df.dropna(subset=required_cols)

    if len(players_df) == 0:
        warnings.warn("No valid players after filtering")
        return None

    # Create MILP problem
    prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)

    # Decision variables: binary (0 or 1) for each player
    player_vars = {}
    for _, player in players_df.iterrows():
        player_id = player['id']
        player_vars[player_id] = pulp.LpVariable(
            f"player_{player_id}",
            cat=pulp.LpBinary
        )

    # Objective: Maximize total projected points
    prob += pulp.lpSum([
        player_vars[player['id']] * player['projection']
        for _, player in players_df.iterrows()
    ]), "Total_Projection"

    # Constraint 1: Salary cap
    prob += pulp.lpSum([
        player_vars[player['id']] * player['fdSalary']
        for _, player in players_df.iterrows()
    ]) <= salary_cap, "Salary_Cap"

    # Constraint 2: Exactly 9 players
    prob += pulp.lpSum([
        player_vars[player_id] for player_id in player_vars
    ]) == 9, "Total_Players"

    # Constraint 3: Position requirements
    # FanDuel positions: QB(1), RB(2), WR(3), TE(1), FLEX(1 - RB/WR/TE), DEF(1)

    # Helper function to filter players by position
    def get_position_vars(pos: str) -> List:
        return [
            player_vars[player['id']]
            for _, player in players_df.iterrows()
            if player['position'] == pos
        ]

    # QB: exactly 1
    qb_vars = get_position_vars('QB')
    if qb_vars:
        prob += pulp.lpSum(qb_vars) == 1, "QB_Count"
    else:
        warnings.warn("No QBs available")
        return None

    # RB: at least 2 (can be 3 with FLEX)
    rb_vars = get_position_vars('RB')
    if rb_vars:
        prob += pulp.lpSum(rb_vars) >= 2, "RB_Min"
        prob += pulp.lpSum(rb_vars) <= 3, "RB_Max"
    else:
        warnings.warn("Not enough RBs available")
        return None

    # WR: at least 3 (can be 4 with FLEX)
    wr_vars = get_position_vars('WR')
    if wr_vars:
        prob += pulp.lpSum(wr_vars) >= 3, "WR_Min"
        prob += pulp.lpSum(wr_vars) <= 4, "WR_Max"
    else:
        warnings.warn("Not enough WRs available")
        return None

    # TE: at least 1 (can be 2 with FLEX)
    te_vars = get_position_vars('TE')
    if te_vars:
        prob += pulp.lpSum(te_vars) >= 1, "TE_Min"
        prob += pulp.lpSum(te_vars) <= 2, "TE_Max"
    else:
        warnings.warn("Not enough TEs available")
        return None

    # DEF: exactly 1
    def_vars = get_position_vars('D')
    if def_vars:
        prob += pulp.lpSum(def_vars) == 1, "DEF_Count"
    else:
        warnings.warn("No DEFs available")
        return None

    # FLEX constraint: RB + WR + TE = 6 or 7 total
    # (2-3 RB + 3-4 WR + 1-2 TE = 6-7)
    flex_vars = rb_vars + wr_vars + te_vars
    prob += pulp.lpSum(flex_vars) >= 6, "FLEX_Min"
    prob += pulp.lpSum(flex_vars) <= 7, "FLEX_Max"

    # Constraint 4: Diversity (max overlap with previous lineups)
    if max_overlap_with:
        for i, prev_lineup in enumerate(max_overlap_with):
            overlap_vars = [
                player_vars[player_id]
                for player_id in prev_lineup
                if player_id in player_vars
            ]
            if overlap_vars:
                prob += pulp.lpSum(overlap_vars) <= max_overlap, f"Diversity_{i}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Suppress solver output

    # Check status
    if prob.status != pulp.LpStatusOptimal:
        warnings.warn(f"MILP solver failed with status: {pulp.LpStatus[prob.status]}")
        return None

    # Extract solution
    selected_players = []
    total_salary = 0
    total_projection = 0

    for _, player in players_df.iterrows():
        player_id = player['id']
        if player_vars[player_id].varValue == 1:
            selected_players.append(player.to_dict())
            total_salary += player['fdSalary']
            total_projection += player['projection']

    return {
        'players': selected_players,
        'total_salary': total_salary,
        'total_projection': total_projection,
        'player_ids': [p['id'] for p in selected_players]
    }


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
    # Test MILP solver with dummy data
    print("Testing MILP solver...")

    # Create dummy players
    players = []
    for i in range(50):
        pos = np.random.choice(['QB', 'RB', 'WR', 'TE', 'D'], p=[0.1, 0.25, 0.35, 0.15, 0.15])
        players.append({
            'id': f'player_{i}',
            'name': f'Player {i}',
            'position': pos,
            'fdSalary': np.random.randint(4000, 9000),
            'projection': np.random.uniform(5, 25)
        })

    players_df = pd.DataFrame(players)

    print(f"\nPlayer pool: {len(players_df)} players")
    print(f"  QBs: {len(players_df[players_df['position'] == 'QB'])}")
    print(f"  RBs: {len(players_df[players_df['position'] == 'RB'])}")
    print(f"  WRs: {len(players_df[players_df['position'] == 'WR'])}")
    print(f"  TEs: {len(players_df[players_df['position'] == 'TE'])}")
    print(f"  DEFs: {len(players_df[players_df['position'] == 'D'])}")

    # Generate single lineup
    print("\nGenerating optimal lineup...")
    lineup = create_lineup_milp(players_df)

    if lineup:
        print(f"\nLineup generated:")
        print(f"  Total projection: {lineup['total_projection']:.2f}")
        print(f"  Total salary: ${lineup['total_salary']:,}")
        print(f"  Players: {len(lineup['players'])}")
        for p in lineup['players']:
            print(f"    {p['position']:3} {p['name']:15} ${p['fdSalary']:5,} → {p['projection']:.2f} pts")

    # Generate 5 diverse lineups
    print("\n\nGenerating 5 diverse lineups...")
    lineups = generate_diverse_lineups(players_df, n_lineups=5, max_overlap=6)

    print(f"\nGenerated {len(lineups)} lineups")
    for i, lineup in enumerate(lineups):
        print(f"  Lineup {i+1}: {lineup['total_projection']:.2f} pts, ${lineup['total_salary']:,}")
