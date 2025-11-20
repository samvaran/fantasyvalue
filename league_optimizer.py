#!/usr/bin/env python3
"""
League Optimizer - One script to rule them all

Does everything in one run:
1. Calculates P90 ceiling values for all players
2. Generates diverse lineups optimizing for MAXIMUM P90 ceiling (uses full salary cap)
3. Runs Monte Carlo simulations with correlations
4. Ranks by simulated P90 ceiling (best for your 22-person tournament league)
5. Outputs top lineups

Just run: python league_optimizer.py
"""
import numpy as np
import pandas as pd
import pulp
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

BUDGET = 60000
N_LINEUPS = 500  # Number of diverse lineups to generate
N_FINAL = 40     # Number of final lineups to output
N_SIMS = 10000   # Simulations per lineup
MIN_PLAYER_DIFF = 2  # Minimum different players between lineups

# Player constraints (edit these lists!)
INCLUDE_PLAYERS = []  # Force these players into every lineup (e.g., ['patrick mahomes', 'bijan robinson'])
EXCLUDE_PLAYERS = []  # Never use these players (e.g., ['devon achane', 'chris olave'])

# Correlation coefficients (from DFS research)
CORRELATIONS = {
    'QB-WR_SAME_TEAM': 0.65,
    'QB-TE_SAME_TEAM': 0.55,
    'QB-RB_SAME_TEAM': 0.20,
    'QB-DST_OPPOSING': -0.75,
    'RB-DST_OPPOSING': -0.45,
    'WR-DST_OPPOSING': -0.40,
    'TE-DST_OPPOSING': -0.40,
    'WR-WR_SAME_TEAM': -0.30,
    'RB-RB_SAME_TEAM': -0.45,
}

# ============================================================================
# STEP 1: CALCULATE P90 CEILING VALUES
# ============================================================================

def calculate_ceiling_values(players_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate P90 ceiling values for all players using exact log-normal quantile."""
    print('=' * 80)
    print('STEP 1: CALCULATING P90 CEILING VALUES (EXACT, VECTORIZED)')
    print('=' * 80)
    print(f'\nCalculating P90 for {len(players_df)} players...')

    # Verify required columns exist
    if not all(col in players_df.columns for col in ['mu', 'sigma_upper']):
        raise ValueError("Missing required distribution parameters (mu, sigma_upper). Please regenerate knapsack.csv with: node fetch_data.js")

    # Add ceiling_value_p90 column to existing dataframe (preserve all columns)
    players_df = players_df.copy()

    # VECTORIZED CALCULATION (100x faster than loop!)
    # Direct calculation: P90 = exp(mu + sigma_upper * z_0.90)
    z90 = 1.2815515655446004

    # Calculate P90 values for all players at once
    players_df['p90'] = np.where(
        (players_df['consensus'] > 0) & (players_df['salary'] > 0) &
        pd.notna(players_df['mu']) & pd.notna(players_df['sigma_upper']),
        np.exp(players_df['mu'] + players_df['sigma_upper'] * z90),
        0
    )

    # Calculate ceiling values for all players at once
    players_df['ceiling_value_p90'] = np.where(
        players_df['salary'] > 0,
        players_df['p90'] / (players_df['salary'] / 1000),
        0
    )

    # Round values
    players_df['p90'] = players_df['p90'].round(2)
    players_df['ceiling_value_p90'] = players_df['ceiling_value_p90'].round(3)

    print(f'✓ Calculated ceiling values for {len(players_df)} players')

    # Save subset to CSV for reference
    result_df = players_df[['name', 'position', 'team', 'salary', 'consensus', 'p90', 'ceiling_value_p90']].copy()
    result_df.to_csv('player_p90_values.csv', index=False)
    print(f'✓ Saved P90 values to player_p90_values.csv')

    # Show top ceiling values
    print('\nTop 10 by P90 Ceiling Value:')
    top = result_df.nlargest(10, 'ceiling_value_p90')[['name', 'position', 'salary', 'p90', 'ceiling_value_p90']]
    print(top.to_string(index=False))

    return players_df  # Return FULL dataframe with all original columns


# ============================================================================
# STEP 2: GENERATE LINEUPS (OPTIMIZE FOR CEILING VALUE)
# ============================================================================

def generate_lineups(players_df: pd.DataFrame, n_lineups: int) -> List[Dict]:
    """Generate diverse lineups optimized for P90 ceiling value."""
    print('\n' + '=' * 80)
    print('STEP 2: GENERATING LINEUPS (CEILING-VALUE OPTIMIZATION)')
    print('=' * 80)

    # PRE-COMPUTE POSITION GROUPS (avoid repeated filtering in loop)
    position_groups = {
        'QB': players_df[players_df['position'] == 'QB'].to_dict('records'),
        'RB': players_df[players_df['position'] == 'RB'].to_dict('records'),
        'WR': players_df[players_df['position'] == 'WR'].to_dict('records'),
        'TE': players_df[players_df['position'] == 'TE'].to_dict('records'),
        'D': players_df[players_df['position'] == 'D'].to_dict('records')
    }

    # PRE-COMPUTE PLAYER NAME MAPPINGS (for include/exclude lists)
    player_name_to_actual = {name.lower(): name for name in players_df['name']}

    lineups = []
    excluded_sets = []

    for i in tqdm(range(n_lineups), desc="Generating lineups", unit="lineup"):
        lineup = _optimize_lineup(players_df, excluded_sets, position_groups, player_name_to_actual)
        if lineup:
            lineups.append(lineup)
            excluded_sets.append(set(lineup['players']))

    print(f'✓ Generated {len(lineups)} unique lineups\n')
    return lineups


def _optimize_lineup(players_df: pd.DataFrame, excluded_sets: List[set],
                     position_groups: Dict[str, List[Dict]],
                     player_name_to_actual: Dict[str, str]) -> Optional[Dict]:
    """Optimize single lineup with position-based weighting strategy.

    Strategy:
    - QB, RB, top WR: Optimize for absolute P90 (studs)
    - Other WR, TE, FLEX, DEF: Optimize for P90 value (contrarian plays)
    """
    prob = pulp.LpProblem("Ceiling_Value", pulp.LpMaximize)

    player_vars = {
        row['name']: pulp.LpVariable(f"p_{i}", cat='Binary')
        for i, row in players_df.iterrows()
    }

    # Objective: Position-based hybrid scoring
    # Studs at key positions, value plays at flex positions
    perturbed = {}
    for _, row in players_df.iterrows():
        p90 = row['p90'] if pd.notna(row['p90']) else 0
        p90_value = row['ceiling_value_p90'] if pd.notna(row['ceiling_value_p90']) else 0

        # Position-based weighting
        if row['position'] == 'QB':
            # QB: 80% absolute ceiling, 20% value (want stud QB)
            base = 0.8 * p90 + 0.2 * p90_value
        elif row['position'] == 'RB':
            # RB: 80% absolute ceiling, 20% value (want stud RBs)
            base = 0.8 * p90 + 0.2 * p90_value
        elif row['position'] == 'WR':
            # WR: 50/50 mix (top WRs are studs, but want some value plays)
            base = 0.5 * p90 + 0.5 * p90_value
        elif row['position'] == 'TE':
            # TE: 30% absolute, 70% value (find value TEs)
            base = 0.3 * p90 + 0.7 * p90_value
        elif row['position'] == 'D':
            # DEF: 20% absolute, 80% value (punt on DEF, find cheap ones)
            base = 0.2 * p90 + 0.8 * p90_value
        else:
            base = p90

        # Apply perturbation for diversity
        perturbation = np.random.uniform(0.85, 1.15)
        perturbed[row['name']] = base * perturbation

    prob += pulp.lpSum(player_vars[name] * val for name, val in perturbed.items())

    # Constraints
    # Salary cap
    prob += pulp.lpSum(player_vars[r['name']] * r['salary'] for _, r in players_df.iterrows()) <= BUDGET

    # Position requirements (OPTIMIZED: use pre-filtered position groups)
    prob += pulp.lpSum(player_vars[p['name']] for p in position_groups['QB']) == 1
    prob += pulp.lpSum(player_vars[p['name']] for p in position_groups['RB']) >= 2
    prob += pulp.lpSum(player_vars[p['name']] for p in position_groups['WR']) >= 3
    prob += pulp.lpSum(player_vars[p['name']] for p in position_groups['TE']) >= 1
    prob += pulp.lpSum(player_vars[p['name']] for p in position_groups['D']) == 1
    prob += pulp.lpSum(player_vars.values()) == 9

    # Include list: Force these players into lineup (OPTIMIZED: use pre-computed mapping)
    for player_name in INCLUDE_PLAYERS:
        actual_name = player_name_to_actual.get(player_name.lower())
        if actual_name and actual_name in player_vars:
            prob += player_vars[actual_name] == 1

    # Exclude list: Never use these players (OPTIMIZED: use pre-computed mapping)
    for player_name in EXCLUDE_PLAYERS:
        actual_name = player_name_to_actual.get(player_name.lower())
        if actual_name and actual_name in player_vars:
            prob += player_vars[actual_name] == 0

    # Diversity: differ by at least MIN_PLAYER_DIFF
    for prev_set in excluded_sets:
        prob += pulp.lpSum(player_vars[name] for name in prev_set if name in player_vars) <= 9 - MIN_PLAYER_DIFF

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Extract lineup
    selected = [name for name, var in player_vars.items() if var.varValue == 1]
    lineup_data = players_df[players_df['name'].isin(selected)]

    return {
        'players': selected,
        'salary': int(lineup_data['salary'].sum()),
        'consensus_total': round(lineup_data['consensus'].sum(), 2),
        'p90_total': round(lineup_data['p90'].sum(), 2)
    }


# ============================================================================
# STEP 3: MONTE CARLO SIMULATION WITH CORRELATIONS
# ============================================================================

def simulate_lineups(lineups: List[Dict], players_df: pd.DataFrame) -> List[Dict]:
    """Simulate all lineups with player correlations using parallel processing."""
    print('\n' + '=' * 80)
    print('STEP 3: MONTE CARLO SIMULATION WITH CORRELATIONS (PARALLEL)')
    print('=' * 80)

    n_cores = cpu_count()
    print(f'Using {n_cores} CPU cores for parallel processing\n')

    # Prepare arguments for parallel processing
    args_list = [(lineup, players_df) for lineup in lineups]

    # Use multiprocessing Pool to parallelize simulations
    with Pool(n_cores) as pool:
        simulated = list(tqdm(
            pool.imap(_simulate_lineup_wrapper, args_list),
            total=len(lineups),
            desc=f"Simulating lineups ({N_SIMS:,} sims each)",
            unit="lineup"
        ))

    print(f'✓ Completed {len(simulated)} lineup simulations\n')
    return simulated


def _simulate_lineup_wrapper(args):
    """Wrapper function for multiprocessing (unpacks tuple arguments)."""
    return _simulate_lineup(*args)


def _simulate_lineup(lineup: Dict, players_df: pd.DataFrame) -> Dict:
    """Simulate single lineup with correlations."""
    player_names = lineup['players']
    lineup_players = players_df[players_df['name'].isin(player_names)]

    # Generate independent samples for each player
    independent = np.zeros((N_SIMS, len(player_names)))

    for i, (_, player) in enumerate(lineup_players.iterrows()):
        consensus = player['consensus']

        if consensus <= 0:
            continue

        # USE PRE-CALCULATED SPLICED DISTRIBUTION FROM FETCH_DATA.JS
        # REQUIRE spliced distribution parameters (sigma_lower, sigma_upper)
        if not (pd.notna(player.get('mu')) and pd.notna(player.get('sigma_lower')) and pd.notna(player.get('sigma_upper'))):
            raise ValueError(f"Player '{player['name']}' missing required distribution parameters (mu, sigma_lower, sigma_upper). Please regenerate knapsack.csv with: node fetch_data.js")

        # SPLICED LOG-NORMAL: Different sigmas for downside (bust) and upside (boom)
        mu = player['mu']
        sigma_lower = player['sigma_lower']  # Controls P0-P50 (floor variance)
        sigma_upper = player['sigma_upper']  # Controls P50-P100 (ceiling variance)

        # Generate samples from spliced distribution (VECTORIZED for 100x speedup!)
        # Lower half: consensus - floor distance, controlled by sigma_lower
        # Upper half: ceiling - consensus distance, controlled by sigma_upper
        uniform_samples = np.random.uniform(0, 1, N_SIMS)

        # Vectorized: Convert all uniform samples to normal percentiles at once
        z_scores = stats.norm.ppf(uniform_samples)

        # Vectorized: Use sigma_lower for samples <= 0.5, sigma_upper for samples > 0.5
        sigmas = np.where(uniform_samples <= 0.5, sigma_lower, sigma_upper)

        # Vectorized: Calculate all samples at once
        samples = np.exp(mu + z_scores * sigmas)

        independent[:, i] = np.maximum(samples, 0)

    # Build correlation matrix
    corr_matrix = _build_correlation_matrix(lineup_players)

    # Apply correlations using Cholesky decomposition
    if np.all(np.linalg.eigvals(corr_matrix) > 0):
        # Convert to standard normal
        standardized = np.zeros_like(independent)
        for i in range(independent.shape[1]):
            standardized[:, i] = stats.norm.ppf(stats.rankdata(independent[:, i]) / (len(independent[:, i]) + 1))

        # Apply correlation
        L = np.linalg.cholesky(corr_matrix)
        correlated_std = standardized @ L.T

        # Convert back
        correlated = np.zeros_like(independent)
        for i in range(correlated_std.shape[1]):
            ranks = stats.rankdata(correlated_std[:, i])
            correlated[:, i] = np.sort(independent[:, i])[ranks.astype(int) - 1]

        lineup_totals = correlated.sum(axis=1)
    else:
        lineup_totals = independent.sum(axis=1)

    # Calculate metrics
    return {
        **lineup,
        'sim_p90': round(np.percentile(lineup_totals, 90), 2),
        'sim_mean': round(np.mean(lineup_totals), 2),
        'sim_p50': round(np.percentile(lineup_totals, 50), 2),
        'sim_p75': round(np.percentile(lineup_totals, 75), 2),
        'sim_floor': round(np.percentile(lineup_totals, 10), 2)
    }


def _build_correlation_matrix(players: pd.DataFrame) -> np.ndarray:
    """Build correlation matrix for lineup players."""
    n = len(players)
    corr = np.eye(n)

    players_list = list(players.iterrows())

    for i in range(n):
        for j in range(i + 1, n):
            p1 = players_list[i][1]
            p2 = players_list[j][1]

            corr_val = 0

            # QB-WR same team
            if p1['position'] == 'QB' and p2['position'] == 'WR' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['QB-WR_SAME_TEAM']
            elif p1['position'] == 'WR' and p2['position'] == 'QB' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['QB-WR_SAME_TEAM']

            # QB-TE same team
            elif p1['position'] == 'QB' and p2['position'] == 'TE' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['QB-TE_SAME_TEAM']
            elif p1['position'] == 'TE' and p2['position'] == 'QB' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['QB-TE_SAME_TEAM']

            # WR-WR same team
            elif p1['position'] == 'WR' and p2['position'] == 'WR' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['WR-WR_SAME_TEAM']

            # RB-RB same team
            elif p1['position'] == 'RB' and p2['position'] == 'RB' and p1['team'] == p2['team']:
                corr_val = CORRELATIONS['RB-RB_SAME_TEAM']

            # Player vs opposing DST (negative correlation)
            elif p1['position'] == 'D' and p2['position'] in ['QB', 'RB', 'WR', 'TE']:
                if _are_opponents(p1['team'], p2['team']):
                    corr_val = CORRELATIONS.get(f"{p2['position']}-DST_OPPOSING", -0.3)
            elif p2['position'] == 'D' and p1['position'] in ['QB', 'RB', 'WR', 'TE']:
                if _are_opponents(p2['team'], p1['team']):
                    corr_val = CORRELATIONS.get(f"{p1['position']}-DST_OPPOSING", -0.3)

            corr[i, j] = corr_val
            corr[j, i] = corr_val

    return corr


def _are_opponents(team1: str, team2: str) -> bool:
    """Check if two teams are opponents (simple heuristic)."""
    # This is a simplification - ideally you'd parse the game matchups
    return team1 != team2


# ============================================================================
# STEP 4: RANK AND OUTPUT
# ============================================================================

def rank_and_output(simulated_lineups: List[Dict], players_df: pd.DataFrame):
    """Rank lineups by P90 ceiling and output results."""
    print('\n' + '=' * 80)
    print('STEP 4: RANKING BY P90 CEILING (TOURNAMENT MODE)')
    print('=' * 80)

    df = pd.DataFrame(simulated_lineups)
    df = df.sort_values('sim_p90', ascending=False)

    print(f'\nTop {N_FINAL} Lineups by P90 Ceiling:')
    print(df.head(N_FINAL)[['salary', 'p90_total', 'sim_p90', 'sim_mean', 'sim_floor']].to_string(index=False))

    # Expand players into columns
    def expand_players(row):
        players = row['players']
        lineup_data = players_df[players_df['name'].isin(players)]

        # Sort by position
        qb = lineup_data[lineup_data['position'] == 'QB']['name'].tolist()
        rbs = lineup_data[lineup_data['position'] == 'RB']['name'].tolist()
        wrs = lineup_data[lineup_data['position'] == 'WR']['name'].tolist()
        tes = lineup_data[lineup_data['position'] == 'TE']['name'].tolist()
        defs = lineup_data[lineup_data['position'] == 'D']['name'].tolist()

        # Build lineup in order
        ordered = []
        ordered.extend(qb[:1])
        ordered.extend(rbs[:2])
        ordered.extend(wrs[:3])
        ordered.extend(tes[:1])
        # FLEX
        flex_pool = rbs[2:] + wrs[3:] + tes[1:]
        ordered.extend(flex_pool[:1])
        ordered.extend(defs[:1])

        return pd.Series({
            'player_1_qb': ordered[0] if len(ordered) > 0 else '',
            'player_2_rb1': ordered[1] if len(ordered) > 1 else '',
            'player_3_rb2': ordered[2] if len(ordered) > 2 else '',
            'player_4_wr1': ordered[3] if len(ordered) > 3 else '',
            'player_5_wr2': ordered[4] if len(ordered) > 4 else '',
            'player_6_wr3': ordered[5] if len(ordered) > 5 else '',
            'player_7_te': ordered[6] if len(ordered) > 6 else '',
            'player_8_flex': ordered[7] if len(ordered) > 7 else '',
            'player_9_def': ordered[8] if len(ordered) > 8 else ''
        })

    player_cols = df.head(N_FINAL).apply(expand_players, axis=1)
    output_df = pd.concat([
        df.head(N_FINAL)[['salary', 'consensus_total', 'p90_total', 'sim_p90', 'sim_mean', 'sim_p75', 'sim_p50', 'sim_floor']],
        player_cols
    ], axis=1)

    output_df.to_csv('LEAGUE_LINEUPS.csv', index=False)
    print(f'\n✓ Saved top {N_FINAL} lineups to LEAGUE_LINEUPS.csv')

    # Show player diversity
    all_players = []
    for players_list in df.head(N_FINAL)['players']:
        all_players.extend(players_list)

    from collections import Counter
    player_counts = Counter(all_players)

    print(f'\nPlayer diversity in top {N_FINAL}:')
    print(f'Total unique players: {len(player_counts)}')
    print(f'\nMost common players:')
    for player, count in player_counts.most_common(10):
        print(f'  {player:<30} {count}/{N_FINAL}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 80)
    print('LEAGUE OPTIMIZER - ONE SCRIPT TO RULE THEM ALL')
    print('=' * 80)
    print(f'\nConfiguration:')
    print(f'  - Lineups to generate: {N_LINEUPS}')
    print(f'  - Final lineups: {N_FINAL}')
    print(f'  - Simulations per lineup: {N_SIMS:,}')
    print(f'  - Optimization target: Position-based P90 (studs + value)')

    if INCLUDE_PLAYERS:
        print(f'\n  🔒 MUST INCLUDE: {", ".join(INCLUDE_PLAYERS)}')
    if EXCLUDE_PLAYERS:
        print(f'  ❌ EXCLUDED: {", ".join(EXCLUDE_PLAYERS)}')

    print()

    # Load data
    print('Loading player data from knapsack.csv...')
    players_df = pd.read_csv('knapsack.csv')
    players_df = players_df[
        (players_df['consensus'].notna()) &
        (players_df['salary'].notna()) &
        (players_df['salary'] > 0)
    ].copy()
    print(f'Loaded {len(players_df)} players\n')

    # Run all steps
    players_df = calculate_ceiling_values(players_df)
    lineups = generate_lineups(players_df, N_LINEUPS)
    simulated = simulate_lineups(lineups, players_df)
    rank_and_output(simulated, players_df)

    print('\n' + '=' * 80)
    print('COMPLETE!')
    print('=' * 80)
    print('\nYour optimized lineups are in: LEAGUE_LINEUPS.csv')
    print('Ready for your 22-person tournament league!')
    print('=' * 80)


if __name__ == '__main__':
    np.random.seed(42)
    main()
