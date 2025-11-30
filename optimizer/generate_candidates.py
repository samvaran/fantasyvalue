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
    Calculate projection for MILP using 50/50 blend.

    Args:
        player: Player row with all stats
        game_script_probs: Dict with game script probabilities
        temperature: Sampling temperature (0=deterministic, >1=random)

    Returns:
        Projection value for MILP
    """
    # Sample game script based on temperature
    if temperature == 0:
        # Deterministic: use expected value
        script_probs = [
            game_script_probs.get('shootout_prob', 0.25),
            game_script_probs.get('defensive_prob', 0.25),
            game_script_probs.get('blowout_prob', 0.25),
            game_script_probs.get('competitive_prob', 0.25)
        ]
        # Weighted average across all scripts
        scripts = ['shootout', 'defensive', 'blowout', 'competitive']
        weighted_floor = 0
        weighted_ceiling = 0

        for script, prob in zip(scripts, script_probs):
            script_key = determine_script_key(script, player['is_favorite'])
            floor = player[f'floor_{script_key}'] * player['td_odds_floor_mult']
            ceiling = player[f'ceiling_{script_key}'] * player['td_odds_ceiling_mult']
            weighted_floor += floor * prob
            weighted_ceiling += ceiling * prob

        scenario_midpoint = (weighted_floor + weighted_ceiling) / 2
    else:
        # Apply temperature to probabilities
        script_probs = np.array([
            game_script_probs.get('shootout_prob', 0.25),
            game_script_probs.get('defensive_prob', 0.25),
            game_script_probs.get('blowout_prob', 0.25),
            game_script_probs.get('competitive_prob', 0.25)
        ])

        # Temperature adjustment: lower temp = more peaked, higher temp = more uniform
        if temperature != 1.0:
            script_probs = script_probs ** (1.0 / temperature)
            script_probs = script_probs / script_probs.sum()

        # Sample a script
        sampled_script = sample_game_script({
            'shootout_prob': script_probs[0],
            'defensive_prob': script_probs[1],
            'blowout_prob': script_probs[2],
            'competitive_prob': script_probs[3]
        })

        script_key = determine_script_key(sampled_script, player['is_favorite'])
        floor = player[f'floor_{script_key}'] * player['td_odds_floor_mult']
        ceiling = player[f'ceiling_{script_key}'] * player['td_odds_ceiling_mult']
        scenario_midpoint = (floor + ceiling) / 2

    # 50/50 blend with consensus
    consensus = player['fpProjPts']
    return (scenario_midpoint + consensus) / 2


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

    # Calculate deterministic projections (temperature=0)
    players_df['projection'] = players_df.apply(
        lambda p: calculate_projection_for_milp(
            p,
            game_script_lookup.get(p['game_id'], {}),
            temperature=0.0
        ),
        axis=1
    )

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

            # Recalculate projections with temperature
            players_df['projection'] = players_df.apply(
                lambda p: calculate_projection_for_milp(
                    p,
                    game_script_lookup.get(p['game_id'], {}),
                    temperature=temperature
                ),
                axis=1
            )

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

            # Recalculate projections with random temperature
            players_df['projection'] = players_df.apply(
                lambda p: calculate_projection_for_milp(
                    p,
                    game_script_lookup.get(p['game_id'], {}),
                    temperature=temperature
                ),
                axis=1
            )

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
