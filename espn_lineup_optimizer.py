#!/usr/bin/env python3
"""
ESPN League Lineup Optimizer

For traditional fantasy leagues with fixed rosters.
Evaluates ALL possible lineups from your roster using Monte Carlo simulation.
Ranks by P90 ceiling to maximize your weekly upside.

Lineup requirements:
- 1 QB
- 2 RB
- 2 WR
- 1 TE
- 1 FLEX (RB/WR/TE)
- 1 D/ST
- (K excluded from analysis)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SIMS = 10000  # Monte Carlo simulations per lineup

# Correlation coefficients (same as DFS optimizer)
CORRELATIONS = {
    'QB-WR_SAME_TEAM': 0.65,
    'QB-TE_SAME_TEAM': 0.55,
    'QB-RB_SAME_TEAM': 0.20,
    'QB-DST_OPPOSING': -0.45,
    'RB-DST_OPPOSING': -0.35,
    'WR-DST_OPPOSING': -0.30,
    'TE-DST_OPPOSING': -0.25,
    'RB-RB_SAME_TEAM': -0.40,
}

# ============================================================================
# YOUR ROSTER - EDIT THIS!
# ============================================================================

# Enter your players' names EXACTLY as they appear in knapsack.csv
# Use lowercase for consistency
MY_ROSTER = [
    # QBs
    'sam darnold',
    'jordan love',

    # RBs
    'jahmyr gibbs',
    'jaylen warren',
    'kareem hunt',
    'tyler allgeier',

    # WRs
    'amonra st brown',
    'drake london',
    'george pickens',
    'keenan allen',

    # TEs
    'juwan johnson',

    # D/ST
    'packers',
    'chiefs',
]

# ram's roster
# MY_ROSTER = [
#     # QBs
#     'josh allen',

#     # RBs
#     'ashton jeanty',
#     'james cook',
#     'kyle monangai',

#     # WRs
#     'brian thomas jr',
#     'marvin harrison jr',
#     'kayshon boutte',
#     'tez johnson',
#     'khalil shakir',

#     # TEs
#     'oronde gadsden',
#     'hunter henry',

#     # D/ST
#     'eagles',
#     'rams',
# ]

# ============================================================================
# LINEUP CONSTRAINTS - EDIT THESE TO FORCE/EXCLUDE PLAYERS
# ============================================================================

# MUST_INCLUDE: Players that MUST be in every generated lineup
# Leave empty [] if you don't want to force any players
MUST_INCLUDE = [
    # Example: 'jahmyr gibbs',
]

# MUST_EXCLUDE: Players that must NOT be in any generated lineup
# Leave empty [] if you don't want to exclude anyone
MUST_EXCLUDE = [
    # Example: 'keenan allen',
]

# ============================================================================
# LOAD PLAYER DATA
# ============================================================================

def load_roster_data() -> pd.DataFrame:
    """Load player data from espn_players_full.csv for your roster."""
    print('=' * 80)
    print('ESPN LEAGUE LINEUP OPTIMIZER')
    print('=' * 80)
    print('\nLoading player data from espn_players_full.csv...')

    # Load all ESPN players (not limited to FanDuel slate)
    df = pd.read_csv('espn_players_full.csv')

    # Normalize names for matching
    df['name_lower'] = df['name'].str.lower().str.strip()

    # Filter to only your roster
    roster_lower = [name.lower().strip() for name in MY_ROSTER]
    roster_df = df[df['name_lower'].isin(roster_lower)].copy()

    # Check for missing players
    found_names = set(roster_df['name_lower'].tolist())
    missing = set(roster_lower) - found_names

    if missing:
        print(f'\n⚠️  WARNING: Could not find {len(missing)} players in espn_players_full.csv:')
        for name in sorted(missing):
            print(f'  - {name}')
        print('\nMake sure player names match exactly (check spelling, punctuation, etc.)')
        print('Check espn_players_full.csv for correct names.')
        print('Run: node fetch_espn_data.js to regenerate ESPN player data\n')

    print(f'\n✓ Loaded {len(roster_df)} players from your roster')
    print(f'\nRoster breakdown:')
    for pos in ['QB', 'RB', 'WR', 'TE', 'D']:
        count = len(roster_df[roster_df['position'] == pos])
        print(f'  {pos}: {count}')

    return roster_df

# ============================================================================
# LINEUP GENERATION
# ============================================================================

def generate_all_lineups(roster_df: pd.DataFrame) -> List[Dict]:
    """Generate all valid lineup combinations from roster."""
    print('\n' + '=' * 80)
    print('STEP 1: GENERATING ALL POSSIBLE LINEUPS')
    print('=' * 80)

    # Normalize constraint lists
    must_include_lower = [name.lower().strip() for name in MUST_INCLUDE]
    must_exclude_lower = [name.lower().strip() for name in MUST_EXCLUDE]

    # Print constraints if any
    if must_include_lower:
        print(f'\n✓ MUST INCLUDE in every lineup: {", ".join(MUST_INCLUDE)}')
    if must_exclude_lower:
        print(f'✓ MUST EXCLUDE from all lineups: {", ".join(MUST_EXCLUDE)}')

    # Separate by position
    qbs = roster_df[roster_df['position'] == 'QB'].to_dict('records')
    rbs = roster_df[roster_df['position'] == 'RB'].to_dict('records')
    wrs = roster_df[roster_df['position'] == 'WR'].to_dict('records')
    tes = roster_df[roster_df['position'] == 'TE'].to_dict('records')
    dsts = roster_df[roster_df['position'] == 'D'].to_dict('records')

    # Apply exclusions
    if must_exclude_lower:
        qbs = [p for p in qbs if p['name'].lower() not in must_exclude_lower]
        rbs = [p for p in rbs if p['name'].lower() not in must_exclude_lower]
        wrs = [p for p in wrs if p['name'].lower() not in must_exclude_lower]
        tes = [p for p in tes if p['name'].lower() not in must_exclude_lower]
        dsts = [p for p in dsts if p['name'].lower() not in must_exclude_lower]

    # Flex eligible (RB/WR/TE)
    flex_eligible = rbs + wrs + tes

    lineups = []

    # Generate all combinations
    for qb in qbs:
        for rb_pair in combinations(rbs, 2):
            for wr_pair in combinations(wrs, 2):
                for te in tes:
                    # Flex must be someone NOT already in lineup
                    used_players = set([qb['name'], rb_pair[0]['name'], rb_pair[1]['name'],
                                       wr_pair[0]['name'], wr_pair[1]['name'], te['name']])

                    available_flex = [p for p in flex_eligible if p['name'] not in used_players]

                    for flex in available_flex:
                        for dst in dsts:
                            lineup = {
                                'QB': qb,
                                'RB1': rb_pair[0],
                                'RB2': rb_pair[1],
                                'WR1': wr_pair[0],
                                'WR2': wr_pair[1],
                                'TE': te,
                                'FLEX': flex,
                                'DST': dst,
                            }

                            # Check MUST_INCLUDE constraint
                            if must_include_lower:
                                lineup_players_lower = [p['name'].lower() for p in lineup.values()]
                                if not all(inc in lineup_players_lower for inc in must_include_lower):
                                    continue  # Skip this lineup

                            lineups.append(lineup)

    print(f'\n✓ Generated {len(lineups)} valid lineups')

    if len(lineups) == 0:
        print('\n⚠️  ERROR: No valid lineups generated!')
        print('Check that you have enough players at each position:')
        print(f'  QBs: {len(qbs)} (need 1+)')
        print(f'  RBs: {len(rbs)} (need 2+)')
        print(f'  WRs: {len(wrs)} (need 2+)')
        print(f'  TEs: {len(tes)} (need 1+)')
        print(f'  D/STs: {len(dsts)} (need 1+)')
        print(f'  Flex eligible: {len(flex_eligible)} (need 3+ after starters)')

    return lineups

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def build_correlation_matrix(lineup: Dict) -> np.ndarray:
    """Build correlation matrix for lineup players."""
    players = [lineup['QB'], lineup['RB1'], lineup['RB2'], lineup['WR1'],
               lineup['WR2'], lineup['TE'], lineup['FLEX'], lineup['DST']]
    n = len(players)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = players[i], players[j]
            corr = 0.0

            # QB correlations
            if p1['position'] == 'QB':
                if p2['position'] == 'WR' and p1['team'] == p2['team']:
                    corr = CORRELATIONS['QB-WR_SAME_TEAM']
                elif p2['position'] == 'TE' and p1['team'] == p2['team']:
                    corr = CORRELATIONS['QB-TE_SAME_TEAM']
                elif p2['position'] == 'RB' and p1['team'] == p2['team']:
                    corr = CORRELATIONS['QB-RB_SAME_TEAM']
                elif p2['position'] == 'D' and p1['team'] != p2['team']:
                    # QB vs opposing DST (check if DST is opponent)
                    if p1.get('game') and p2.get('team') in p1.get('game', ''):
                        corr = CORRELATIONS['QB-DST_OPPOSING']

            # RB correlations
            elif p1['position'] == 'RB':
                if p2['position'] == 'D' and p1['team'] != p2['team']:
                    if p1.get('game') and p2.get('team') in p1.get('game', ''):
                        corr = CORRELATIONS['RB-DST_OPPOSING']
                elif p2['position'] == 'RB' and p1['team'] == p2['team']:
                    corr = CORRELATIONS['RB-RB_SAME_TEAM']

            # WR correlations
            elif p1['position'] == 'WR':
                if p2['position'] == 'D' and p1['team'] != p2['team']:
                    if p1.get('game') and p2.get('team') in p1.get('game', ''):
                        corr = CORRELATIONS['WR-DST_OPPOSING']

            # TE correlations
            elif p1['position'] == 'TE':
                if p2['position'] == 'D' and p1['team'] != p2['team']:
                    if p1.get('game') and p2.get('team') in p1.get('game', ''):
                        corr = CORRELATIONS['TE-DST_OPPOSING']

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix

def simulate_lineup(lineup: Dict, n_sims: int = N_SIMS) -> Dict:
    """Run Monte Carlo simulation for a single lineup."""
    players = [lineup['QB'], lineup['RB1'], lineup['RB2'], lineup['WR1'],
               lineup['WR2'], lineup['TE'], lineup['FLEX'], lineup['DST']]

    # Extract means and stds
    means = []
    stds = []
    log_params = []

    for p in players:
        consensus = p['consensus']
        uncertainty = p['uncertainty'] if pd.notna(p['uncertainty']) else consensus * 0.3

        # Log-normal parameters
        mean = consensus
        std = uncertainty
        variance = std ** 2
        sigma_squared = np.log(1 + variance / (mean ** 2))
        mu = np.log(mean) - sigma_squared / 2
        sigma = np.sqrt(sigma_squared)

        means.append(consensus)
        stds.append(uncertainty)
        log_params.append((mu, sigma))

    # Build correlation matrix
    corr_matrix = build_correlation_matrix(lineup)

    # Generate correlated samples using Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, use uncorrelated
        L = np.eye(len(players))

    # Generate correlated normal samples
    standard_normals = np.random.standard_normal((n_sims, len(players)))
    correlated_normals = standard_normals @ L.T

    # Transform to log-normal
    samples = np.zeros((n_sims, len(players)))
    for i, (mu, sigma) in enumerate(log_params):
        samples[:, i] = np.exp(mu + sigma * correlated_normals[:, i])

    # Sum across players for each simulation
    lineup_scores = samples.sum(axis=1)

    # Calculate statistics
    return {
        'lineup': lineup,
        'mean': np.mean(lineup_scores),
        'median': np.median(lineup_scores),
        'p10': np.percentile(lineup_scores, 10),
        'p90': np.percentile(lineup_scores, 90),
        'std': np.std(lineup_scores),
        'max': np.max(lineup_scores),
    }

def simulate_all_lineups(lineups: List[Dict]) -> pd.DataFrame:
    """Simulate all lineups and return results."""
    print('\n' + '=' * 80)
    print(f'STEP 2: RUNNING MONTE CARLO SIMULATIONS ({N_SIMS:,} per lineup)')
    print('=' * 80)

    results = []

    for i, lineup in enumerate(lineups):
        if (i + 1) % 50 == 0 or (i + 1) == len(lineups):
            print(f'  Simulating lineup {i+1}/{len(lineups)}...', end='\r')

        sim_result = simulate_lineup(lineup)

        # Build result row
        result = {
            'lineup_id': i + 1,
            'qb': sim_result['lineup']['QB']['name'],
            'rb1': sim_result['lineup']['RB1']['name'],
            'rb2': sim_result['lineup']['RB2']['name'],
            'wr1': sim_result['lineup']['WR1']['name'],
            'wr2': sim_result['lineup']['WR2']['name'],
            'te': sim_result['lineup']['TE']['name'],
            'flex': sim_result['lineup']['FLEX']['name'],
            'flex_pos': sim_result['lineup']['FLEX']['position'],
            'dst': sim_result['lineup']['DST']['name'],
            'sim_mean': round(sim_result['mean'], 2),
            'sim_median': round(sim_result['median'], 2),
            'sim_p10': round(sim_result['p10'], 2),
            'sim_p90': round(sim_result['p90'], 2),
            'sim_std': round(sim_result['std'], 2),
            'sim_max': round(sim_result['max'], 2),
        }
        results.append(result)

    print()  # New line after progress

    df = pd.DataFrame(results)

    # Sort by P90 (highest ceiling)
    df = df.sort_values('sim_p90', ascending=False).reset_index(drop=True)

    print(f'\n✓ Completed {len(df)} lineup simulations')

    return df

# ============================================================================
# OUTPUT
# ============================================================================

def save_results(results_df: pd.DataFrame):
    """Save results to CSV and display top lineups."""
    print('\n' + '=' * 80)
    print('RESULTS')
    print('=' * 80)

    # Save full results
    output_file = 'espn_lineups_ranked.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n✓ Saved {len(results_df)} lineups to {output_file}')

    # Display top 10
    print('\n' + '=' * 80)
    print('TOP 10 LINEUPS BY P90 CEILING')
    print('=' * 80)

    for idx, row in results_df.head(10).iterrows():
        print(f"\nRank #{idx + 1}  (P90: {row['sim_p90']:.1f} pts)")
        print('-' * 80)
        print(f"  QB:   {row['qb']}")
        print(f"  RB1:  {row['rb1']}")
        print(f"  RB2:  {row['rb2']}")
        print(f"  WR1:  {row['wr1']}")
        print(f"  WR2:  {row['wr2']}")
        print(f"  TE:   {row['te']}")
        print(f"  FLEX: {row['flex']} ({row['flex_pos']})")
        print(f"  DST:  {row['dst']}")
        print(f"\n  Stats: Mean={row['sim_mean']:.1f}, Median={row['sim_median']:.1f}, "
              f"P10={row['sim_p10']:.1f}, P90={row['sim_p90']:.1f}, Std={row['sim_std']:.1f}")

    print('\n' + '=' * 80)
    print(f'COMPLETE! See {output_file} for all lineups.')
    print('=' * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    np.random.seed(42)

    # Load roster
    roster_df = load_roster_data()

    if len(roster_df) == 0:
        print('\n❌ No players loaded. Check MY_ROSTER configuration.')
        return

    # Generate all possible lineups
    lineups = generate_all_lineups(roster_df)

    if len(lineups) == 0:
        print('\n❌ No valid lineups generated. Check roster composition.')
        return

    # Simulate all lineups
    results_df = simulate_all_lineups(lineups)

    # Save and display results
    save_results(results_df)


if __name__ == '__main__':
    main()
