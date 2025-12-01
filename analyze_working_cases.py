"""
Analyze cases where distribution fitting DOES work to understand the pattern.
"""
import pandas as pd
import numpy as np

players_df = pd.read_csv('data/intermediate/players_integrated.csv')

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

print("=" * 100)
print("ANALYZING WORKING vs FAILING CASES")
print("=" * 100)

# Test several high-value players
test_players = ['josh allen', 'devon achane', 'justin herbert', 'christian mccaffrey', 'drake maye']

for player_name in test_players:
    player = players_df[players_df['name'] == player_name].iloc[0]
    consensus = player['fpProjPts']

    print(f"\n{player_name.upper()} ({player['position']}) - Consensus: {consensus:.2f}")
    print("-" * 100)

    for scenario in scenarios:
        floor = player[f'floor_{scenario}']
        ceiling = player[f'ceiling_{scenario}']

        mu = player[f'mu_{scenario}']
        sigma = player[f'sigma_{scenario}']
        shift = player[f'shift_{scenario}']

        # Calculate fitted percentiles
        Z10 = -1.2815515655446004
        Z90 = 1.2815515655446004

        fitted_mean = np.exp(mu + sigma**2 / 2) + shift
        fitted_p10 = np.exp(mu + Z10 * sigma) + shift
        fitted_p90 = np.exp(mu + Z90 * sigma) + shift

        mean_err = (fitted_mean - consensus) / consensus * 100 if consensus > 0 else 0
        p10_err = (fitted_p10 - floor) / floor * 100 if floor > 0 else 0
        p90_err = (fitted_p90 - ceiling) / ceiling * 100 if ceiling > 0 else 0

        # Check if mean is close to or above P90 (problematic case)
        mean_vs_p90 = consensus - ceiling

        status = "✓" if abs(mean_err) < 10 else "✗"

        print(f"  {scenario:20s}: floor={floor:6.2f}, ceil={ceiling:6.2f}, "
              f"mean_err={mean_err:+6.1f}%, mean-P90={mean_vs_p90:+6.2f} {status}")

    print()

print("\n" + "=" * 100)
print("PATTERN ANALYSIS")
print("=" * 100)

print("\nCases where fitting WORKS (Devon Achane, CMC):")
print("  - Consensus is BELOW ceiling (mean < P90)")
print("  - Reasonable spread between floor and ceiling")
print("  - Right-skewed distribution makes sense")

print("\nCases where fitting FAILS (Josh Allen defensive, Justin Herbert):")
print("  - Consensus is CLOSE TO or ABOVE ceiling (mean ≈ P90)")
print("  - This requires LEFT-SKEWED distribution")
print("  - Shifted log-normal can't handle this!")

print("\n" + "=" * 100)
print("ROOT CAUSE")
print("=" * 100)

print("\nThe game script ceiling multipliers are TOO LOW for some positions!")
print("Example: Josh Allen defensive scenario")
print("  - Consensus: 21.90 (this is the EXPECTED VALUE across all scenarios)")
print("  - Defensive ceiling: 23.00 (ceiling multiplier 0.80x)")
print("  - Problem: ceiling (23.00) is barely above mean (21.90)")
print("  - This is saying 'in defensive scenarios, his 90th percentile is 23'")
print("  - But his overall expected value is 21.90")
print("  - This implies in THIS scenario, his distribution is LEFT-SKEWED")
print("  - That doesn't make sense for a QB in a defensive game!")

print("\nThe REAL issue: We're confusing scenario-specific means vs overall consensus")
print("  - Consensus (21.90) = weighted average across ALL scenarios")
print("  - Defensive scenario should have LOWER mean (not same mean)")
print("  - The floor/ceiling should reflect the scenario-specific distribution")
print("  - Not try to preserve the overall consensus mean!")
