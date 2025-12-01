"""
Phase 1: Candidate Generation via MILP + Tiered Sampling

Generates 1000 diverse lineup candidates using:
- Lineups 1-20: Deterministic (expected game scripts)
- Lineups 21-100: Temperature-based variation
- Lineups 101-1000: Random weighted sampling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse

from utils.milp_solver import create_lineup_milp
from utils.monte_carlo import sample_game_script, determine_script_key


def calculate_projection_for_milp(
    player: pd.Series,
    game_script_probs: Dict[str, float],
    temperature: float = 1.0
) -> float:
    """
    Calculate projection for MILP - now uses pure consensus (expected value).

    Game script info is used for DIVERSIFICATION via temperature sampling,
    not for the optimization objective.

    Args:
        player: Player row with all stats
        game_script_probs: Dict with game script probabilities (unused now)
        temperature: Sampling temperature (unused now, kept for compatibility)

    Returns:
        Consensus projection (fpProjPts) - the true expected value
    """
    # Use consensus projection as MILP objective
    # This is mathematically sound: consensus IS the expected value
    # Game scripts are accounted for in Monte Carlo evaluation, not MILP optimization
    return player['fpProjPts']


def generate_candidates(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_lineups: int = 1000,
    salary_cap: float = 60000,
    output_path: str = 'outputs/lineups_candidates.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate diverse lineup candidates using tiered sampling.

    Args:
        players_df: DataFrame with integrated player data
        game_scripts_df: DataFrame with game script probabilities
        n_lineups: Number of lineups to generate (default: 1000)
        salary_cap: Salary cap (default: 60000)
        output_path: Where to save candidates CSV
        verbose: Print progress

    Returns:
        DataFrame with all candidate lineups
    """
    if verbose:
        print("=" * 80)
        print("PHASE 1: CANDIDATE GENERATION (MILP + TIERED SAMPLING)")
        print("=" * 80)

    # Preprocess: Fill missing floor/ceiling with defaults based on consensus
    if verbose:
        print("\nPreprocessing player data...")

    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
    missing_count = 0

    for idx in players_df.index:
        # Check if floor/ceiling are missing
        if pd.isna(players_df.loc[idx, 'floor_competitive']):
            missing_count += 1
            consensus = players_df.loc[idx, 'fpProjPts']
            # Use simple defaults: floor = 50% of consensus, ceiling = 150% of consensus
            for script in scripts:
                players_df.loc[idx, f'floor_{script}'] = consensus * 0.5
                players_df.loc[idx, f'ceiling_{script}'] = consensus * 1.5

    if verbose and missing_count > 0:
        print(f"  Filled {missing_count} players with default floor/ceiling values")

    # Build game script lookup
    game_script_lookup = {}
    for _, game in game_scripts_df.iterrows():
        game_script_lookup[game['game_id']] = game.to_dict()

    lineups = []
    previous_lineups = []

    # ===== TIER 1: Deterministic (Lineups 1-20) =====
    if verbose:
        print("\n=== TIER 1: Deterministic chalk lineups (1-20) ===")

    tier1_count = min(20, n_lineups)

    # Set projections to consensus (same for all tiers now)
    players_df['projection'] = players_df['fpProjPts']

    for i in range(tier1_count):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Generating lineup {i + 1}/{ tier1_count}...")

        lineup = create_lineup_milp(
            players_df,
            salary_cap=salary_cap,
            max_overlap_with=previous_lineups if i > 0 else None,
            max_overlap=7
        )

        if lineup is None:
            if verbose:
                print(f"  Warning: Failed to generate lineup {i + 1}")
            break

        lineups.append({
            'lineup_id': len(lineups),
            'tier': 1,
            'temperature': 0.0,
            'player_ids': ','.join(lineup['player_ids']),
            'total_salary': lineup['total_salary'],
            'total_projection': lineup['total_projection']
        })
        previous_lineups.append(lineup['player_ids'])

    if verbose:
        print(f"  Generated {len(lineups)} chalk lineups")

    # ===== TIER 2: Temperature-based (Lineups 21-100) =====
    if n_lineups > 20:
        if verbose:
            print("\n=== TIER 2: Temperature-based variation (21-100) ===")

        tier2_start = len(lineups)
        tier2_count = min(80, n_lineups - tier2_start)

        for i in range(tier2_count):
            # Temperature ramps from 0.3 to 1.1
            temperature = 0.3 + (1.1 - 0.3) * (i / tier2_count)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Generating lineup {tier2_start + i + 1}/{tier2_start + tier2_count} (temp={temperature:.2f})...")

            # Projections already set to consensus
            # Diversity comes from max_overlap constraint
            lineup = create_lineup_milp(
                players_df,
                salary_cap=salary_cap,
                max_overlap_with=previous_lineups,
                max_overlap=7
            )

            if lineup is None:
                continue

            lineups.append({
                'lineup_id': len(lineups),
                'tier': 2,
                'temperature': temperature,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            previous_lineups.append(lineup['player_ids'])

        if verbose:
            print(f"  Generated {len(lineups) - tier2_start} temperature-based lineups")

    # ===== TIER 3: Random weighted sampling (Lineups 101-1000) =====
    if n_lineups > 100:
        if verbose:
            print("\n=== TIER 3: Random weighted sampling (101-1000) ===")

        tier3_start = len(lineups)
        tier3_count = n_lineups - tier3_start

        for i in range(tier3_count):
            # High temperature = more random
            temperature = np.random.uniform(1.5, 3.0)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Generating lineup {tier3_start + i + 1}/{tier3_start + tier3_count}...")

            # Projections already set to consensus
            # Diversity comes from stricter max_overlap constraint
            lineup = create_lineup_milp(
                players_df,
                salary_cap=salary_cap,
                max_overlap_with=previous_lineups,
                max_overlap=6  # Stricter diversity for contrarian plays
            )

            if lineup is None:
                continue

            lineups.append({
                'lineup_id': len(lineups),
                'tier': 3,
                'temperature': temperature,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            previous_lineups.append(lineup['player_ids'])

        if verbose:
            print(f"  Generated {len(lineups) - tier3_start} contrarian lineups")

    # Convert to DataFrame
    lineups_df = pd.DataFrame(lineups)

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lineups_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"  Total lineups generated: {len(lineups_df)}")
        print(f"  Tier 1 (chalk): {len(lineups_df[lineups_df['tier'] == 1])}")
        print(f"  Tier 2 (temperature): {len(lineups_df[lineups_df['tier'] == 2])}")
        print(f"  Tier 3 (random): {len(lineups_df[lineups_df['tier'] == 3])}")
        print(f"  Saved to: {output_file}")
        print("=" * 80)

    return lineups_df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate lineup candidates")
    parser.add_argument('--input', default='players_integrated.csv', help='Input players CSV')
    parser.add_argument('--game-scripts', default='game_script_continuous.csv', help='Game scripts CSV')
    parser.add_argument('--output', default='outputs/lineups_candidates.csv', help='Output lineups CSV')
    parser.add_argument('--n-lineups', type=int, default=1000, help='Number of lineups to generate')
    parser.add_argument('--salary-cap', type=float, default=60000, help='Salary cap')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Load data
    players_df = pd.read_csv(args.input)
    game_scripts_df = pd.read_csv(args.game_scripts)

    # Generate candidates
    generate_candidates(
        players_df=players_df,
        game_scripts_df=game_scripts_df,
        n_lineups=args.n_lineups,
        salary_cap=args.salary_cap,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
