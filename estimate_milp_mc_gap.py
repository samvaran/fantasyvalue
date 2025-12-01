"""Estimate expected MILP vs MC gap with scenario-specific means."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/intermediate/players_integrated.csv')

# Calculate scenario means for each player
Z10 = -1.2816
Z90 = 1.2816

scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

# Sample 20 high-projection players
top_players = df.nlargest(20, 'fpProjPts')

print("=" * 80)
print("ESTIMATED MILP vs MC ALIGNMENT")
print("=" * 80)
print("\nFor each player, calculate weighted average of scenario means")
print("(assuming equal probability for all scenarios)")

total_consensus = 0
total_mc_mean = 0

for _, player in top_players.iterrows():
    consensus = player['fpProjPts']
    scenario_means = []
    
    for scenario in scenarios:
        mu = player[f'mu_{scenario}']
        sigma = player[f'sigma_{scenario}']
        shift = player[f'shift_{scenario}']
        
        mean = np.exp(mu + sigma**2 / 2) + shift
        scenario_means.append(mean)
    
    avg_mc_mean = np.mean(scenario_means)
    total_consensus += consensus
    total_mc_mean += avg_mc_mean

consensus_total = total_consensus
mc_total = total_mc_mean

diff = mc_total - consensus_total
pct = diff / consensus_total * 100

print(f"\nSample of 20 top players:")
print(f"  Sum of MILP projections (consensus): {consensus_total:.2f}")
print(f"  Sum of MC means (weighted avg):      {mc_total:.2f}")
print(f"  Difference:                          {diff:+.2f} ({pct:+.1f}%)")

if abs(pct) < 5:
    print("\n✓ MILP and MC are well-aligned (within 5%)!")
else:
    print(f"\n⚠ MILP and MC differ by {abs(pct):.1f}%")
    print("  This is expected when scenarios have different means")
    print("  The gap depends on game script probabilities")
