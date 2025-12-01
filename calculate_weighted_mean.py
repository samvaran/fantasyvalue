"""Calculate weighted average of scenario means."""

# Josh Allen example - use the means we just calculated
scenario_means = {
    'shootout': 22.85,
    'defensive': 20.00,
    'blowout_favorite': None,  # Need to calculate
    'blowout_underdog': None,   # Need to calculate
    'competitive': None         # Need to calculate
}

# Calculate the other scenarios
import numpy as np

Z10 = -1.2816
Z90 = 1.2816

# From earlier data
floors = {
    'shootout': 10.69,
    'defensive': 10.69,
    'blowout_favorite': 11.41,  # Need to verify
    'blowout_underdog': 11.41,  # Need to verify
    'competitive': 12.11
}

ceilings = {
    'shootout': 38.18,
    'defensive': 31.36,
    'blowout_favorite': 31.36,  # Need to verify
    'blowout_underdog': 36.81,  # Need to verify
    'competitive': 34.09
}

# Actually let's use the multipliers directly
import pandas as pd
df = pd.read_csv('data/intermediate/players_integrated.csv')
josh = df[df['name'] == 'josh allen'].iloc[0]

consensus = josh['fpProjPts']
espn_floor = josh['espnLowScore']
espn_ceiling = josh['espnHighScore']

print("=" * 80)
print("JOSH ALLEN - SCENARIO-SPECIFIC MEANS")
print("=" * 80)
print(f"\nConsensus: {consensus:.2f}")
print(f"ESPN floor:  {espn_floor:.2f}")
print(f"ESPN ceiling: {espn_ceiling:.2f}")

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
scenario_means = {}

print(f"\n{'Scenario':<20s} {'Floor':<10s} {'Ceiling':<10s} {'Mean':<10s} {'vs Consensus'}")
print("-" * 80)

for scenario in scenarios:
    floor = josh[f'floor_{scenario}']
    ceiling = josh[f'ceiling_{scenario}']
    
    # Fit log-normal to (P10, P90) only
    sigma = np.log(ceiling / floor) / (Z90 - Z10)
    mu = np.log(floor) - Z10 * sigma
    mean = np.exp(mu + sigma**2 / 2)
    
    scenario_means[scenario] = mean
    diff = mean - consensus
    pct = diff / consensus * 100
    
    print(f"{scenario:<20s} {floor:>8.2f}  {ceiling:>8.2f}  {mean:>8.2f}  {diff:>+6.2f} ({pct:>+5.1f}%)")

# Calculate weighted average (assuming equal probabilities for simplicity)
avg_mean = np.mean(list(scenario_means.values()))
diff = avg_mean - consensus
pct = diff / consensus * 100

print("-" * 80)
print(f"{'AVERAGE':<20s} {'':>8s}  {'':>8s}  {avg_mean:>8.2f}  {diff:>+6.2f} ({pct:>+5.1f}%)")

if abs(pct) < 5:
    print("\n✓ Weighted average is within 5% of consensus")
else:
    print(f"\n✗ Weighted average differs by {pct:.1f}% from consensus")
