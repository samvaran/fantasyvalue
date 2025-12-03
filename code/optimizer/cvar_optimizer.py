"""
CVaR-MILP Optimizer for DFS Lineup Ceiling Optimization

Uses Conditional Value at Risk (CVaR) formulation to optimize for the
top percentile of lineup outcomes (e.g., p80) rather than expected value.

Mathematical Formulation (Upper-tail CVaR maximization):
    Maximize: t - (1 / (S × α)) × Σ z[s]

    Subject to:
        z[s] ≥ t - Σ(x[i] × points[i][s])    ∀s ∈ {1..S}
        z[s] ≥ 0
        Σ(x[i] × salary[i]) ≤ SALARY_CAP
        Position constraints (QB=1, RB=2-3, WR=3-4, TE=1-2, DEF=1)
        Σ x[i] = 9

    Where:
        x[i] = binary, 1 if player i selected
        t = continuous, threshold approximating the (1-α) percentile
        z[s] = continuous, shortfall below threshold in scenario s
        points[i][s] = pre-simulated score for player i in scenario s

    The objective maximizes the average of outcomes above the threshold,
    penalizing scenarios that fall below it.
"""

import pulp
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set
import warnings
import sys
from pathlib import Path

# Import config
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config import SALARY_CAP, FORCE_INCLUDE, EXCLUDE_PLAYERS


def optimize_lineup_cvar(
    players_df: pd.DataFrame,
    scenario_matrix: np.ndarray,
    player_index: pd.DataFrame,
    alpha: float = 0.20,
    salary_cap: float = SALARY_CAP,
    force_include: List[str] = None,
    exclude_players: List[str] = None,
    exclude_lineups: List[List[str]] = None,
    time_limit: int = 60,
    gap_tolerance: float = 0.02,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Solve CVaR-MILP for optimal ceiling lineup.

    Args:
        players_df: Full player DataFrame with id, name, position, salary
        scenario_matrix: Pre-computed points[player_idx][scenario_idx]
        player_index: DataFrame mapping indices to player info (from scenario generator)
        alpha: CVaR tail probability (0.20 = optimize for p80)
        salary_cap: Maximum total salary
        force_include: List of player IDs to force into lineup
        exclude_players: List of player IDs to exclude
        exclude_lineups: List of previous lineups to exclude (exact match prevention)
        time_limit: Solver time limit in seconds
        gap_tolerance: MIP gap tolerance (0.005 = 0.5%)
        verbose: Print solver output

    Returns:
        Dict with lineup info or None if infeasible
    """
    force_include = force_include or list(FORCE_INCLUDE)
    exclude_players = exclude_players or list(EXCLUDE_PLAYERS)
    exclude_lineups = exclude_lineups or []

    n_players, n_scenarios = scenario_matrix.shape

    # Build player ID to index mapping
    id_to_idx = {str(row['id']): idx for idx, row in player_index.iterrows()}

    # Get valid player indices (not excluded)
    exclude_set = set(str(p).lower() for p in exclude_players)
    valid_indices = []
    for idx, row in player_index.iterrows():
        player_id = str(row['id'])
        player_name = str(row.get('name', '')).lower()
        if player_id.lower() not in exclude_set and player_name not in exclude_set:
            valid_indices.append(idx)

    if len(valid_indices) < 9:
        warnings.warn(f"Not enough valid players ({len(valid_indices)}) after exclusions")
        return None

    # Group players by position
    position_indices = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'D': []}
    for idx in valid_indices:
        pos = player_index.loc[idx, 'position']
        if pos in position_indices:
            position_indices[pos].append(idx)

    # Validate we have enough players at each position
    if len(position_indices['QB']) < 1:
        warnings.warn("Not enough QBs")
        return None
    if len(position_indices['RB']) < 2:
        warnings.warn("Not enough RBs")
        return None
    if len(position_indices['WR']) < 3:
        warnings.warn("Not enough WRs")
        return None
    if len(position_indices['TE']) < 1:
        warnings.warn("Not enough TEs")
        return None
    if len(position_indices['D']) < 1:
        warnings.warn("Not enough DEFs")
        return None

    # Create MILP problem
    prob = pulp.LpProblem("CVaR_Lineup_Optimization", pulp.LpMaximize)

    # Decision variables
    # x[i] = binary, 1 if player i selected
    x = pulp.LpVariable.dicts("x", valid_indices, cat=pulp.LpBinary)

    # t = continuous, threshold for CVaR (the VaR threshold at 1-alpha percentile)
    t = pulp.LpVariable("t", cat=pulp.LpContinuous)

    # z[s] = continuous, shortfall below threshold in scenario s (z >= 0)
    z = pulp.LpVariable.dicts("z", range(n_scenarios), lowBound=0, cat=pulp.LpContinuous)

    # Objective: Maximize upper-tail CVaR = t - (1 / (S × α)) × Σ z[s]
    # This maximizes the expected value of the top alpha% of outcomes
    cvar_weight = 1.0 / (n_scenarios * alpha)
    prob += t - cvar_weight * pulp.lpSum([z[s] for s in range(n_scenarios)]), "CVaR_Objective"

    # CVaR constraints: z[s] >= t - lineup_score[s]
    # z[s] captures how much scenario s falls below threshold t
    for s in range(n_scenarios):
        prob += z[s] >= t - pulp.lpSum([
            x[i] * scenario_matrix[i, s] for i in valid_indices
        ]), f"CVaR_{s}"

    # Salary constraint
    prob += pulp.lpSum([
        x[i] * player_index.loc[i, 'salary'] for i in valid_indices
    ]) <= salary_cap, "Salary_Cap"

    # Total players = 9
    prob += pulp.lpSum([x[i] for i in valid_indices]) == 9, "Total_Players"

    # Position constraints
    # QB: exactly 1
    prob += pulp.lpSum([x[i] for i in position_indices['QB']]) == 1, "QB_Count"

    # RB: 2-3
    prob += pulp.lpSum([x[i] for i in position_indices['RB']]) >= 2, "RB_Min"
    prob += pulp.lpSum([x[i] for i in position_indices['RB']]) <= 3, "RB_Max"

    # WR: 3-4
    prob += pulp.lpSum([x[i] for i in position_indices['WR']]) >= 3, "WR_Min"
    prob += pulp.lpSum([x[i] for i in position_indices['WR']]) <= 4, "WR_Max"

    # TE: 1-2
    prob += pulp.lpSum([x[i] for i in position_indices['TE']]) >= 1, "TE_Min"
    prob += pulp.lpSum([x[i] for i in position_indices['TE']]) <= 2, "TE_Max"

    # DEF: exactly 1
    prob += pulp.lpSum([x[i] for i in position_indices['D']]) == 1, "DEF_Count"

    # FLEX constraint: RB + WR + TE must equal 7 (since QB=1, DEF=1, total=9)
    flex_indices = position_indices['RB'] + position_indices['WR'] + position_indices['TE']
    prob += pulp.lpSum([x[i] for i in flex_indices]) == 7, "FLEX_Total"

    # Force include constraints
    for player_id in force_include:
        player_id_str = str(player_id)
        if player_id_str in id_to_idx:
            idx = id_to_idx[player_id_str]
            if idx in valid_indices:
                prob += x[idx] == 1, f"Force_{player_id_str}"
        else:
            # Try matching by name
            for idx in valid_indices:
                if str(player_index.loc[idx, 'name']).lower() == player_id_str.lower():
                    prob += x[idx] == 1, f"Force_{player_id_str}"
                    break

    # Exclude previous lineups (exact match prevention)
    for lineup_idx, prev_lineup in enumerate(exclude_lineups):
        # Convert player IDs to indices
        prev_indices = []
        for player_id in prev_lineup:
            player_id_str = str(player_id)
            if player_id_str in id_to_idx:
                idx = id_to_idx[player_id_str]
                if idx in valid_indices:
                    prev_indices.append(idx)

        if len(prev_indices) >= 9:
            # Can't have all 9 same players
            prob += pulp.lpSum([x[i] for i in prev_indices]) <= 8, f"Exclude_Lineup_{lineup_idx}"

    # Solve with HiGHS (much faster than CBC)
    solver = pulp.HiGHS(
        msg=1 if verbose else 0,
        timeLimit=time_limit,
        gapRel=gap_tolerance
    )
    prob.solve(solver)

    # Check status
    if prob.status != pulp.LpStatusOptimal:
        if verbose:
            warnings.warn(f"CVaR solver status: {pulp.LpStatus[prob.status]}")
        return None

    # Extract solution
    selected_indices = [i for i in valid_indices if x[i].varValue > 0.5]

    if len(selected_indices) != 9:
        warnings.warn(f"Solver returned {len(selected_indices)} players instead of 9")
        return None

    # Get player info
    selected_players = []
    player_ids = []
    total_salary = 0

    for idx in selected_indices:
        player_info = player_index.loc[idx].to_dict()
        selected_players.append(player_info)
        player_ids.append(str(player_info['id']))
        total_salary += player_info['salary']

    # Compute lineup statistics from scenario matrix
    lineup_scores = scenario_matrix[selected_indices].sum(axis=0)

    # CVaR score from objective
    cvar_score = pulp.value(prob.objective)

    # Additional statistics
    stats = {
        'mean': float(np.mean(lineup_scores)),
        'median': float(np.median(lineup_scores)),
        'std': float(np.std(lineup_scores)),
        'p10': float(np.percentile(lineup_scores, 10)),
        'p80': float(np.percentile(lineup_scores, 80)),
        'p90': float(np.percentile(lineup_scores, 90)),
        'cvar_score': float(cvar_score),
    }

    # Compute actual CVaR for verification
    threshold = np.percentile(lineup_scores, (1 - alpha) * 100)
    top_scores = lineup_scores[lineup_scores >= threshold]
    stats['cvar_actual'] = float(np.mean(top_scores)) if len(top_scores) > 0 else stats['p80']

    return {
        'players': selected_players,
        'player_ids': player_ids,
        'player_indices': selected_indices,
        'total_salary': total_salary,
        **stats
    }


def optimize_multiple_lineups(
    players_df: pd.DataFrame,
    scenario_matrix: np.ndarray,
    player_index: pd.DataFrame,
    n_lineups: int = 100,
    alpha: float = 0.20,
    salary_cap: float = SALARY_CAP,
    force_include: List[str] = None,
    exclude_players: List[str] = None,
    time_limit: int = 60,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate multiple diverse lineups using CVaR-MILP with exact-lineup exclusion.

    Args:
        players_df: Full player DataFrame
        scenario_matrix: Pre-computed scenario matrix
        player_index: Player index DataFrame
        n_lineups: Number of lineups to generate
        alpha: CVaR tail probability
        salary_cap: Maximum salary
        force_include: Players to force in all lineups
        exclude_players: Players to exclude from all lineups
        time_limit: Solver time limit per lineup
        verbose: Print progress

    Returns:
        List of lineup dicts
    """
    from tqdm import tqdm

    lineups = []
    excluded_lineups = []

    iterator = tqdm(range(n_lineups), desc="Generating lineups", disable=not verbose)

    for i in iterator:
        lineup = optimize_lineup_cvar(
            players_df=players_df,
            scenario_matrix=scenario_matrix,
            player_index=player_index,
            alpha=alpha,
            salary_cap=salary_cap,
            force_include=force_include,
            exclude_players=exclude_players,
            exclude_lineups=excluded_lineups,
            time_limit=time_limit,
            verbose=False
        )

        if lineup is None:
            if verbose:
                tqdm.write(f"  Could not generate lineup {i + 1}. Stopping.")
            break

        lineup['lineup_id'] = i
        lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])

        if verbose and (i + 1) % 10 == 0:
            tqdm.write(f"  Generated {i + 1} lineups. Best CVaR: {max(l['cvar_score'] for l in lineups):.2f}")

    return lineups


if __name__ == '__main__':
    # Test with dummy data
    print("Testing CVaR optimizer...")
    import time

    # Create dummy players
    np.random.seed(42)
    players = []
    for pos, count in [('QB', 10), ('RB', 20), ('WR', 30), ('TE', 10), ('D', 10)]:
        for i in range(count):
            consensus = np.random.uniform(8, 22)
            players.append({
                'id': f'{pos}_{i}',
                'name': f'{pos} Player {i}',
                'position': pos,
                'salary': np.random.randint(4500, 9000),
                'team': f'TEAM{i % 5}',
                'fpProjPts': consensus,
            })

    players_df = pd.DataFrame(players)

    # Create dummy scenario matrix (simpler for testing)
    n_players = len(players_df)
    n_scenarios = 500  # Fewer scenarios for faster testing

    # Generate scenarios with some correlation structure
    base_scores = players_df['fpProjPts'].values
    scenario_matrix = np.zeros((n_players, n_scenarios))

    for s in range(n_scenarios):
        # Add random variation
        noise = np.random.randn(n_players) * 5
        scenario_matrix[:, s] = np.maximum(0, base_scores + noise)

    # Create player index
    player_index = players_df[['id', 'name', 'position', 'salary', 'team', 'fpProjPts']].copy()

    print(f"\nPlayers: {n_players}, Scenarios: {n_scenarios}")

    # Test single lineup optimization
    print("\n--- Single Lineup Optimization ---")
    start = time.time()
    lineup = optimize_lineup_cvar(
        players_df=players_df,
        scenario_matrix=scenario_matrix,
        player_index=player_index,
        alpha=0.20,
        verbose=True
    )
    elapsed = time.time() - start

    if lineup:
        print(f"Time: {elapsed:.2f}s")
        print(f"CVaR Score: {lineup['cvar_score']:.2f}")
        print(f"Mean: {lineup['mean']:.2f}, Median: {lineup['median']:.2f}")
        print(f"P10: {lineup['p10']:.2f}, P80: {lineup['p80']:.2f}, P90: {lineup['p90']:.2f}")
        print(f"Salary: ${lineup['total_salary']:,}")
        print(f"Players: {lineup['player_ids']}")
    else:
        print("Failed to generate lineup")

    # Test multiple lineup generation
    print("\n--- Multiple Lineup Generation (10 lineups) ---")
    start = time.time()
    lineups = optimize_multiple_lineups(
        players_df=players_df,
        scenario_matrix=scenario_matrix,
        player_index=player_index,
        n_lineups=10,
        alpha=0.20,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nGenerated {len(lineups)} lineups in {elapsed:.2f}s")
    print(f"Average time per lineup: {elapsed/len(lineups):.2f}s")

    if lineups:
        cvar_scores = [l['cvar_score'] for l in lineups]
        print(f"CVaR range: {min(cvar_scores):.2f} - {max(cvar_scores):.2f}")

    print("\nCVaR optimizer test complete!")
