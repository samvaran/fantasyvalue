"""
Test if fitted distributions actually have mean = consensus.
"""
import pandas as pd
import numpy as np
from optimizer.utils.distribution_fit import sample_shifted_lognormal

# Load players with pre-computed distribution parameters
players_df = pd.read_csv('data/intermediate/players_integrated.csv')

# Test a few high-value players
test_players = players_df.nlargest(5, 'fpProjPts')[['name', 'fpProjPts']].copy()

print("Testing if fitted distributions preserve consensus mean...\n")
print("=" * 80)

for idx, player in players_df.nlargest(5, 'fpProjPts').iterrows():
    name = player['name']
    consensus = player['fpProjPts']

    print(f"\n{name} (Consensus: {consensus:.2f})")
    print("-" * 80)

    # Test each scenario
    scenarios = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']

    scenario_means = []
    for scenario in scenarios:
        mu = player[f'mu_{scenario}']
        sigma = player[f'sigma_{scenario}']
        shift = player[f'shift_{scenario}']

        # Theoretical mean of shifted log-normal: exp(mu + sigma^2/2) + shift
        theoretical_mean = np.exp(mu + sigma**2 / 2) + shift

        # Empirical mean from 10,000 samples
        samples = sample_shifted_lognormal(mu, sigma, shift, size=10000)
        empirical_mean = np.mean(samples)

        scenario_means.append(empirical_mean)

        diff = empirical_mean - consensus
        pct_diff = (diff / consensus * 100) if consensus > 0 else 0

        print(f"  {scenario:20s}: Empirical mean = {empirical_mean:6.2f}  "
              f"(vs consensus {consensus:.2f}, diff: {diff:+6.2f} / {pct_diff:+5.1f}%)")

    # Weighted average across scenarios (assuming equal probability)
    avg_mean = np.mean(scenario_means)
    avg_diff = avg_mean - consensus
    avg_pct = (avg_diff / consensus * 100) if consensus > 0 else 0

    print(f"\n  Average across scenarios: {avg_mean:.2f}  "
          f"(vs consensus {consensus:.2f}, diff: {avg_diff:+6.2f} / {avg_pct:+5.1f}%)")

print("\n" + "=" * 80)
