"""
Propose new game script multipliers that are mathematically consistent.

Principles:
1. Preserve the INTENT of each scenario (shootout = passing boost, defensive = run boost, etc.)
2. Ensure floor < consensus < ceiling for all scenarios (required for right-skewed distribution)
3. Make weighted average of scenario midpoints ≈ consensus
4. Ceiling should be at least consensus * 1.15 for reasonable right-skew
"""
import numpy as np

# Current multipliers (for reference)
OLD_FLOOR = {
    'shootout': {'QB': 0.95, 'RB': 0.90, 'WR': 0.95, 'TE': 0.95, 'D': 0.85},
    'defensive': {'QB': 0.85, 'RB': 1.05, 'WR': 0.85, 'TE': 0.85, 'D': 1.15},
    'blowout_favorite': {'QB': 0.90, 'RB': 1.05, 'WR': 0.85, 'TE': 0.90, 'D': 1.10},
    'blowout_underdog': {'QB': 0.95, 'RB': 0.85, 'WR': 1.00, 'TE': 0.95, 'D': 0.90},
    'competitive': {'QB': 1.00, 'RB': 1.00, 'WR': 1.00, 'TE': 1.00, 'D': 1.00}
}

OLD_CEILING = {
    'shootout': {'QB': 1.15, 'RB': 0.90, 'WR': 1.20, 'TE': 1.15, 'D': 0.80},
    'defensive': {'QB': 0.80, 'RB': 0.90, 'WR': 0.75, 'TE': 0.80, 'D': 1.25},
    'blowout_favorite': {'QB': 0.85, 'RB': 1.10, 'WR': 0.85, 'TE': 0.85, 'D': 1.15},
    'blowout_underdog': {'QB': 1.10, 'RB': 0.80, 'WR': 1.15, 'TE': 1.10, 'D': 0.85},
    'competitive': {'QB': 1.00, 'RB': 1.00, 'WR': 1.00, 'TE': 1.00, 'D': 1.00}
}

# NEW MULTIPLIERS
# Key changes:
# 1. NO ceiling multiplier < 1.0 (ceiling must always be above baseline)
# 2. Floor multipliers adjusted to keep midpoint reasonable
# 3. Variance changes are expressed via floor reduction, not ceiling reduction

NEW_FLOOR = {
    # Shootout: High variance for passers, low variance for RBs
    'shootout': {
        'QB': 0.75,   # Wider range (was 0.95)
        'RB': 0.90,   # Narrower floor (same)
        'WR': 0.75,   # Wider range (was 0.95)
        'TE': 0.75,   # Wider range (was 0.95)
        'D': 0.70    # Low def scoring in shootouts (was 0.85)
    },
    # Defensive: Low variance for passers, good for RBs and DEF
    'defensive': {
        'QB': 0.75,   # Lower floor but will have ceiling to match (was 0.85)
        'RB': 0.95,   # Stable floor (was 1.05)
        'WR': 0.75,   # Lower floor (was 0.85)
        'TE': 0.75,   # Lower floor (was 0.85)
        'D': 1.00    # Stable def scoring (was 1.15)
    },
    # Blowout favorite: RBs shine, passing limited
    'blowout_favorite': {
        'QB': 0.80,   # Lower floor (was 0.90)
        'RB': 0.95,   # Stable floor (was 1.05)
        'WR': 0.80,   # Lower floor (was 0.85)
        'TE': 0.80,   # Lower floor (was 0.90)
        'D': 1.00    # Stable (was 1.10)
    },
    # Blowout underdog: Passing game script, less RB work
    'blowout_underdog': {
        'QB': 0.80,   # Lower floor for variance (was 0.95)
        'RB': 0.75,   # Much lower - game script dependent (was 0.85)
        'WR': 0.85,   # Decent floor (was 1.00)
        'TE': 0.80,   # Lower floor (was 0.95)
        'D': 0.75    # Low def scoring when behind (was 0.90)
    },
    # Competitive: Baseline
    'competitive': {
        'QB': 0.85,
        'RB': 0.85,
        'WR': 0.85,
        'TE': 0.85,
        'D': 0.85
    }
}

NEW_CEILING = {
    # Shootout: HIGH ceiling for passers
    'shootout': {
        'QB': 1.40,   # Massive upside (was 1.15)
        'RB': 1.05,   # Modest ceiling (was 0.90 - FIXED!)
        'WR': 1.45,   # Huge upside (was 1.20)
        'TE': 1.40,   # Huge upside (was 1.15)
        'D': 1.05    # Modest ceiling (was 0.80 - FIXED!)
    },
    # Defensive: Lower ceiling for passers, modest for RBs
    'defensive': {
        'QB': 1.15,   # Limited upside (was 0.80 - FIXED!)
        'RB': 1.15,   # Modest upside (was 0.90 - FIXED!)
        'WR': 1.10,   # Limited upside (was 0.75 - FIXED!)
        'TE': 1.15,   # Limited upside (was 0.80 - FIXED!)
        'D': 1.40    # High def ceiling (was 1.25)
    },
    # Blowout favorite: RBs get ceiling, passing capped
    'blowout_favorite': {
        'QB': 1.15,   # Limited upside (was 0.85 - FIXED!)
        'RB': 1.30,   # Good upside (was 1.10)
        'WR': 1.10,   # Limited upside (was 0.85 - FIXED!)
        'TE': 1.15,   # Limited upside (was 0.85 - FIXED!)
        'D': 1.30    # Good upside (was 1.15)
    },
    # Blowout underdog: Passing has huge ceiling
    'blowout_underdog': {
        'QB': 1.35,   # Big upside when trailing (was 1.10)
        'WR': 1.35,   # Big upside (was 1.15)
        'RB': 1.05,   # Modest ceiling (was 0.80 - FIXED!)
        'TE': 1.30,   # Good upside (was 1.10)
        'D': 1.05    # Modest ceiling (was 0.85 - FIXED!)
    },
    # Competitive: Moderate ceiling
    'competitive': {
        'QB': 1.25,
        'RB': 1.20,
        'WR': 1.25,
        'TE': 1.25,
        'D': 1.20
    }
}

print("=" * 100)
print("PROPOSED NEW GAME SCRIPT MULTIPLIERS")
print("=" * 100)
print("\nKey changes:")
print("  ✓ ALL ceiling multipliers >= 1.0 (no more ceiling < baseline)")
print("  ✓ Variance expressed via floor adjustment + ceiling boost")
print("  ✓ Preserves intended game script effects")
print()

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
positions = ['QB', 'RB', 'WR', 'TE', 'D']

for pos in positions:
    print(f"\n{'=' * 100}")
    print(f"{pos} MULTIPLIERS")
    print("=" * 100)
    print(f"\n{'Scenario':<20s} {'Old Floor':<12s} {'New Floor':<12s} {'Old Ceiling':<12s} {'New Ceiling':<12s} {'Effect'}")
    print("-" * 100)

    for scenario in scenarios:
        old_f = OLD_FLOOR[scenario][pos]
        new_f = NEW_FLOOR[scenario][pos]
        old_c = OLD_CEILING[scenario][pos]
        new_c = NEW_CEILING[scenario][pos]

        old_range = old_c - old_f
        new_range = new_c - new_f

        if new_range > old_range * 1.2:
            effect = "MORE variance"
        elif new_range < old_range * 0.8:
            effect = "LESS variance"
        else:
            effect = "Similar variance"

        print(f"{scenario:<20s} {old_f:5.2f}→{new_f:5.2f}  "
              f"{old_f:5.2f}→{new_f:5.2f}  "
              f"{old_c:5.2f}→{new_c:5.2f}  "
              f"{old_c:5.2f}→{new_c:5.2f}  "
              f"{effect}")

print("\n" + "=" * 100)
print("INTENT PRESERVATION")
print("=" * 100)
print("\nShootout:")
print("  QB/WR/TE: ✓ High ceiling (1.40-1.45x), wide range")
print("  RB:       ✓ Modest ceiling (1.05x), limited in pass-heavy game")
print("  D:        ✓ Low floor (0.70x), offense scores more")
print("\nDefensive:")
print("  QB/WR/TE: ✓ Lower ceiling (1.10-1.15x), less passing")
print("  RB:       ✓ Stable floor (0.95x), decent ceiling (1.15x)")
print("  D:        ✓ High ceiling (1.40x), defense dominates")
print("\nBlowout Favorite:")
print("  QB/WR/TE: ✓ Limited ceiling (1.10-1.15x), game script run-heavy")
print("  RB:       ✓ High ceiling (1.30x), more touches")
print("  D:        ✓ Good ceiling (1.30x)")
print("\nBlowout Underdog:")
print("  QB/WR/TE: ✓ High ceiling (1.30-1.35x), pass-heavy when trailing")
print("  RB:       ✓ Low floor (0.75x), modest ceiling (1.05x)")
print("  D:        ✓ Low floor (0.75x)")
print("\nCompetitive:")
print("  All:      ✓ Baseline moderate range")

print("\n" + "=" * 100)
print("COPY-PASTE READY CODE")
print("=" * 100)
print("\nReplace in 2_data_integration.py:\n")

print("GAME_SCRIPT_FLOOR = {")
for scenario in scenarios:
    values = ', '.join([f"'{pos}': {NEW_FLOOR[scenario][pos]:.2f}" for pos in positions])
    print(f"    '{scenario}': {{{values}}},")
print("}")

print("\nGAME_SCRIPT_CEILING = {")
for scenario in scenarios:
    values = ', '.join([f"'{pos}': {NEW_CEILING[scenario][pos]:.2f}" for pos in positions])
    print(f"    '{scenario}': {{{values}}},")
print("}")
