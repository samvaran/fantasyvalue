"""
Analyze current game script multipliers to identify issues.

For a shifted log-normal with mean M, P10=F, P90=C:
- The relationship is constrained
- We can check if given (F, C) what mean is achievable
"""
import numpy as np
import pandas as pd

# Load current multipliers from 2_data_integration.py
GAME_SCRIPT_FLOOR = {
    'shootout': {'QB': 0.95, 'RB': 0.90, 'WR': 0.95, 'TE': 0.95, 'D': 0.85},
    'defensive': {'QB': 0.85, 'RB': 1.05, 'WR': 0.85, 'TE': 0.85, 'D': 1.15},
    'blowout_favorite': {'QB': 0.90, 'RB': 1.05, 'WR': 0.85, 'TE': 0.90, 'D': 1.10},
    'blowout_underdog': {'QB': 0.95, 'RB': 0.85, 'WR': 1.00, 'TE': 0.95, 'D': 0.90},
    'competitive': {'QB': 1.00, 'RB': 1.00, 'WR': 1.00, 'TE': 1.00, 'D': 1.00}
}

GAME_SCRIPT_CEILING = {
    'shootout': {'QB': 1.15, 'RB': 0.90, 'WR': 1.20, 'TE': 1.15, 'D': 0.80},
    'defensive': {'QB': 0.80, 'RB': 0.90, 'WR': 0.75, 'TE': 0.80, 'D': 1.25},
    'blowout_favorite': {'QB': 0.85, 'RB': 1.10, 'WR': 0.85, 'TE': 0.85, 'D': 1.15},
    'blowout_underdog': {'QB': 1.10, 'RB': 0.80, 'WR': 1.15, 'TE': 1.10, 'D': 0.85},
    'competitive': {'QB': 1.00, 'RB': 1.00, 'WR': 1.00, 'TE': 1.00, 'D': 1.00}
}

# For shifted log-normal: given P10 and P90, the implied mean is approximately:
# mean ≈ (P10 + P90) / 2 for symmetric, or closer to P10+shift for right-skewed
# A good heuristic: mean should be at least (P10 + 2*P90) / 3 and at most (2*P10 + P90) / 3
# Actually, for log-normal, mean > (P10 + P90) / 2 due to right skew

def estimate_implied_mean_range(p10, p90):
    """
    Estimate the range of feasible means for a shifted log-normal with given P10, P90.

    For shifted log-normal:
    - Lower bound: approximately (P10 + P90) / 2 (would be symmetric)
    - Upper bound: approximately P10 + 0.6*(P90 - P10) + extra for right tail

    Actually, let's compute it more precisely using the distribution properties.
    """
    midpoint = (p10 + p90) / 2
    range_val = p90 - p10

    # For log-normal, the mean is typically between midpoint and P90
    # Rough heuristic based on testing
    min_mean = midpoint - 0.1 * range_val  # Allow some left skew
    max_mean = p90 + 0.3 * range_val  # Allow right tail

    return min_mean, max_mean

print("=" * 100)
print("ANALYZING CURRENT GAME SCRIPT MULTIPLIERS")
print("=" * 100)
print("\nChecking if floor/ceiling combinations are compatible with mean = consensus\n")

# Test with representative players
test_players = [
    {'name': 'Josh Allen', 'position': 'QB', 'consensus': 21.90, 'floor': 14.25, 'ceiling': 27.27},
    {'name': 'Drake Maye', 'position': 'QB', 'consensus': 20.40, 'floor': 15.47, 'ceiling': 27.86},
    {'name': 'Devon Achane', 'position': 'RB', 'consensus': 19.90, 'floor': 21.43, 'ceiling': 24.38},
    {'name': 'Jonathan Taylor', 'position': 'RB', 'consensus': 18.30, 'floor': 11.64, 'ceiling': 21.82},
    {'name': 'Jaylen Waddle', 'position': 'WR', 'consensus': 15.50, 'floor': 10.36, 'ceiling': 18.82},
    {'name': 'Hunter Henry', 'position': 'TE', 'consensus': 11.40, 'floor': 6.92, 'ceiling': 14.59},
]

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

issues = []

for player in test_players:
    name = player['name']
    pos = player['position']
    consensus = player['consensus']
    base_floor = player['floor']
    base_ceiling = player['ceiling']

    print(f"\n{name} ({pos}) - Consensus: {consensus:.2f}")
    print("-" * 100)
    print(f"{'Scenario':<20s} {'Floor':<8s} {'Ceiling':<8s} {'Midpoint':<10s} {'Feasible Mean Range':<30s} {'Compatible?'}")
    print("-" * 100)

    for scenario in scenarios:
        floor_mult = GAME_SCRIPT_FLOOR[scenario][pos]
        ceiling_mult = GAME_SCRIPT_CEILING[scenario][pos]

        adj_floor = base_floor * floor_mult
        adj_ceiling = base_ceiling * ceiling_mult

        # Apply min/max constraints from data_integration
        adj_floor = min(adj_floor, consensus * 0.95)
        adj_ceiling = max(adj_ceiling, consensus * 1.05)

        midpoint = (adj_floor + adj_ceiling) / 2

        # Estimate feasible mean range
        min_mean, max_mean = estimate_implied_mean_range(adj_floor, adj_ceiling)

        # Check if consensus is within feasible range
        compatible = min_mean <= consensus <= max_mean
        status = "✓" if compatible else "✗"

        if not compatible:
            issues.append({
                'player': name,
                'position': pos,
                'scenario': scenario,
                'consensus': consensus,
                'floor': adj_floor,
                'ceiling': adj_ceiling,
                'min_mean': min_mean,
                'max_mean': max_mean,
                'gap': min(abs(consensus - min_mean), abs(consensus - max_mean))
            })

        print(f"{scenario:<20s} {adj_floor:6.2f}   {adj_ceiling:6.2f}   {midpoint:6.2f}     "
              f"[{min_mean:5.2f}, {max_mean:5.2f}]{'':>15s} {status}")

print("\n" + "=" * 100)
print("ISSUES FOUND")
print("=" * 100)

if issues:
    print(f"\nFound {len(issues)} incompatible scenario/player combinations:\n")
    for issue in issues[:10]:  # Show first 10
        print(f"  {issue['player']:15s} {issue['position']:3s} {issue['scenario']:20s}: "
              f"consensus={issue['consensus']:5.2f} not in [{issue['min_mean']:5.2f}, {issue['max_mean']:5.2f}] "
              f"(gap: {issue['gap']:.2f})")

    if len(issues) > 10:
        print(f"\n  ... and {len(issues) - 10} more")
else:
    print("\n✓ All scenarios are compatible!")

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print("\nThe main issue: ceiling multipliers that are TOO LOW")
print("Examples:")
print("  - QB defensive: ceiling 0.80x → ceiling below consensus → impossible!")
print("  - QB blowout_favorite: ceiling 0.85x → ceiling barely above consensus")
print("\nFor a right-skewed distribution, ceiling must be WELL ABOVE mean to accommodate the tail.")
print("Rule of thumb: ceiling should be at least mean * 1.15 for reasonable right skew")
