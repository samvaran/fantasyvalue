"""
Analyze how game script adjustments affect floor/ceiling values.
"""
import pandas as pd
import numpy as np

# Load integrated player data
players_df = pd.read_csv('data/intermediate/players_integrated.csv')

# Focus on high-value players
top_players = players_df.nlargest(10, 'fpProjPts')

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

print("=" * 100)
print("GAME SCRIPT FLOOR/CEILING ANALYSIS")
print("=" * 100)

for idx, player in top_players.iterrows():
    name = player['name']
    position = player['position']
    consensus = player['fpProjPts']
    original_floor = player.get('espnLowScore', consensus * 0.5)
    original_ceiling = player.get('espnHighScore', consensus * 1.5)
    original_midpoint = (original_floor + original_ceiling) / 2

    print(f"\n{name} ({position}) - Consensus: {consensus:.2f}")
    print(f"  Original: floor={original_floor:.2f}, ceiling={original_ceiling:.2f}, midpoint={original_midpoint:.2f}")
    print(f"  Original midpoint vs consensus: {original_midpoint - consensus:+.2f} ({(original_midpoint - consensus)/consensus*100:+.1f}%)")
    print()

    for scenario in scenarios:
        floor = player[f'floor_{scenario}']
        ceiling = player[f'ceiling_{scenario}']
        midpoint = (floor + ceiling) / 2

        floor_diff = floor - original_floor
        ceiling_diff = ceiling - original_ceiling
        midpoint_diff = midpoint - consensus

        # Calculate multipliers that were applied
        floor_mult = floor / original_floor if original_floor > 0 else 1.0
        ceiling_mult = ceiling / original_ceiling if original_ceiling > 0 else 1.0

        print(f"  {scenario:20s}: floor={floor:6.2f} ({floor_mult:.2f}x), ceiling={ceiling:6.2f} ({ceiling_mult:.2f}x), "
              f"midpoint={midpoint:6.2f} (vs consensus: {midpoint_diff:+6.2f} / {midpoint_diff/consensus*100:+5.1f}%)")

    # Calculate weighted average midpoint (assuming equal probabilities for simplicity)
    avg_midpoint = np.mean([
        (player[f'floor_{s}'] + player[f'ceiling_{s}']) / 2
        for s in scenarios
    ])
    avg_diff = avg_midpoint - consensus

    print(f"\n  Weighted avg midpoint: {avg_midpoint:.2f} (vs consensus: {avg_diff:+.2f} / {avg_diff/consensus*100:+.1f}%)")
    print("-" * 100)

print("\n" + "=" * 100)
print("SUMMARY: Do scenario midpoints preserve consensus on average?")
print("=" * 100)

for idx, player in top_players.iterrows():
    name = player['name']
    consensus = player['fpProjPts']

    avg_midpoint = np.mean([
        (player[f'floor_{s}'] + player[f'ceiling_{s}']) / 2
        for s in scenarios
    ])

    diff = avg_midpoint - consensus
    pct = (diff / consensus * 100) if consensus > 0 else 0

    print(f"{name:20s}: avg_midpoint={avg_midpoint:6.2f}, consensus={consensus:6.2f}, "
          f"diff={diff:+6.2f} ({pct:+5.1f}%)")
