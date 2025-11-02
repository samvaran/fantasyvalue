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
EXCLUDE_PLAYERS = ['stefon diggs']  # Never use these players (e.g., ['devon achane', 'chris olave'])

# Correlation coefficients (from DFS research)
CORRELATIONS = {
    'QB-WR_SAME_TEAM': 0.65,
    'QB-TE_SAME_TEAM': 0.55,
    'QB-RB_SAME_TEAM': 0.20,
    'QB-DST_OPPOSING': -0.45,
    'RB-DST_OPPOSING': -0.35,
    'WR-DST_OPPOSING': -0.30,
    'TE-DST_OPPOSING': -0.30,
    'WR-WR_SAME_TEAM': -0.15,
    'RB-RB_SAME_TEAM': -0.40,
}

# ============================================================================
# STEP 1: CALCULATE P90 CEILING VALUES
# ============================================================================

def calculate_ceiling_values(players_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate P90 ceiling values for all players using log-normal simulation."""
    print('=' * 80)
    print('STEP 1: CALCULATING P90 CEILING VALUES')
    print('=' * 80)
    print(f'\nSimulating {len(players_df)} players...')

    results = []

    for _, row in players_df.iterrows():
        consensus = row['consensus']
        uncertainty = row['uncertainty'] if pd.notna(row['uncertainty']) else consensus * 0.3
        salary = row['salary']

        if consensus <= 0 or salary <= 0:
            p90 = 0
            ceiling_value = 0
        else:
            # Log-normal parameters
            mean = consensus
            std = uncertainty
            variance = std ** 2
            sigma_squared = np.log(1 + variance / (mean ** 2))
            mu = np.log(mean) - sigma_squared / 2
            sigma = np.sqrt(sigma_squared)

            # Simulate
            samples = np.random.lognormal(mu, sigma, 1000)  # Quick 1k sims per player
            samples = np.maximum(samples, 0)
            p90 = np.percentile(samples, 90)
            ceiling_value = p90 / (salary / 1000)

        results.append({
            'name': row['name'],
            'position': row['position'],
            'team': row['team'],
            'salary': salary,
            'consensus': consensus,
            'uncertainty': uncertainty,
            'p90': round(p90, 2),
            'ceiling_value_p90': round(ceiling_value, 3)
        })

    result_df = pd.DataFrame(results)
    print(f'✓ Calculated ceiling values for {len(result_df)} players')

    # Save to CSV
    result_df.to_csv('player_p90_values.csv', index=False)
    print(f'✓ Saved P90 values to player_p90_values.csv')

    # Show top ceiling values
    print('\nTop 10 by P90 Ceiling Value:')
    top = result_df.nlargest(10, 'ceiling_value_p90')[['name', 'position', 'salary', 'p90', 'ceiling_value_p90']]
    print(top.to_string(index=False))

    return result_df


# ============================================================================
# STEP 2: GENERATE LINEUPS (OPTIMIZE FOR CEILING VALUE)
# ============================================================================

def generate_lineups(players_df: pd.DataFrame, n_lineups: int) -> List[Dict]:
    """Generate diverse lineups optimized for P90 ceiling value."""
    print('\n' + '=' * 80)
    print('STEP 2: GENERATING LINEUPS (CEILING-VALUE OPTIMIZATION)')
    print('=' * 80)
    print(f'\nGenerating {n_lineups} diverse lineups...')

    lineups = []
    excluded_sets = []

    for i in range(n_lineups):
        lineup = _optimize_lineup(players_df, excluded_sets)
        if lineup:
            lineups.append(lineup)
            excluded_sets.append(set(lineup['players']))

        if (i + 1) % 100 == 0:
            print(f'  Generated {i + 1}/{n_lineups}...')

    print(f'✓ Generated {len(lineups)} unique lineups')
    return lineups


def _optimize_lineup(players_df: pd.DataFrame, excluded_sets: List[set]) -> Optional[Dict]:
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

    # Position requirements
    prob += pulp.lpSum(player_vars[r['name']] for _, r in players_df.iterrows() if r['position'] == 'QB') == 1
    prob += pulp.lpSum(player_vars[r['name']] for _, r in players_df.iterrows() if r['position'] == 'RB') >= 2
    prob += pulp.lpSum(player_vars[r['name']] for _, r in players_df.iterrows() if r['position'] == 'WR') >= 3
    prob += pulp.lpSum(player_vars[r['name']] for _, r in players_df.iterrows() if r['position'] == 'TE') >= 1
    prob += pulp.lpSum(player_vars[r['name']] for _, r in players_df.iterrows() if r['position'] == 'D') == 1
    prob += pulp.lpSum(player_vars.values()) == 9

    # Include list: Force these players into lineup
    for player_name in INCLUDE_PLAYERS:
        if player_name.lower() in [p.lower() for p in player_vars.keys()]:
            matching = [name for name in player_vars.keys() if name.lower() == player_name.lower()]
            if matching:
                prob += player_vars[matching[0]] == 1

    # Exclude list: Never use these players
    for player_name in EXCLUDE_PLAYERS:
        if player_name.lower() in [p.lower() for p in player_vars.keys()]:
            matching = [name for name in player_vars.keys() if name.lower() == player_name.lower()]
            if matching:
                prob += player_vars[matching[0]] == 0

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
    """Simulate all lineups with player correlations."""
    print('\n' + '=' * 80)
    print('STEP 3: MONTE CARLO SIMULATION WITH CORRELATIONS')
    print('=' * 80)
    print(f'\nRunning {N_SIMS:,} simulations per lineup...')

    simulated = []

    for i, lineup in enumerate(lineups):
        if (i + 1) % 100 == 0 or i == 0:
            print(f'  Simulating lineup {i + 1}/{len(lineups)}...')

        sim_results = _simulate_lineup(lineup, players_df)
        simulated.append(sim_results)

    print(f'✓ Completed {len(simulated)} lineup simulations')
    return simulated


def _simulate_lineup(lineup: Dict, players_df: pd.DataFrame) -> Dict:
    """Simulate single lineup with correlations."""
    player_names = lineup['players']
    lineup_players = players_df[players_df['name'].isin(player_names)]

    # Generate independent samples for each player
    independent = np.zeros((N_SIMS, len(player_names)))

    for i, (_, player) in enumerate(lineup_players.iterrows()):
        consensus = player['consensus']
        uncertainty = player['uncertainty']

        if consensus <= 0:
            continue

        # Log-normal parameters
        mean = consensus
        std = uncertainty
        variance = std ** 2
        sigma_squared = np.log(1 + variance / (mean ** 2))
        mu = np.log(mean) - sigma_squared / 2
        sigma = np.sqrt(sigma_squared)

        samples = np.random.lognormal(mu, sigma, N_SIMS)
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
    # Import scipy for correlations
    from scipy import stats
    np.random.seed(42)
    main()
