"""
Quick test to see what parameter combinations are causing numerical solver calls.
"""
import pandas as pd
from optimizer.utils.distribution_fit import fit_shifted_lognormal

# Load players data
df = pd.read_csv('data/intermediate/players_integrated.csv')

# Track how often we fall through to numerical solver
analytical_success = 0
numerical_needed = 0
problematic_cases = []

for idx, row in df.iterrows():
    mean = row['consensus']
    p10 = row['floor']
    p90 = row['ceiling']
    name = row['player_name']

    # Calculate what the analytical solution would give
    z10 = -1.2816
    z90 = 1.2816

    sigma = (p90 - p10) / mean  # range as % of mean

    # Try analytical
    try:
        sigma_est = max(0.001, min(10.0, sigma))  # Very wide bounds

        # Check if this would pass the 10% threshold
        # (simplified check - just look at the ratio)
        if sigma < 0.5:  # Reasonable range
            analytical_success += 1
        else:
            numerical_needed += 1
            problematic_cases.append({
                'player': name,
                'mean': mean,
                'p10': p10,
                'p90': p90,
                'range_pct': (p90 - p10) / mean * 100,
                'p90_p10_ratio': p90 / p10
            })
    except:
        numerical_needed += 1

print(f"Analytical likely to work: {analytical_success}/{len(df)} ({analytical_success/len(df)*100:.1f}%)")
print(f"Numerical needed: {numerical_needed}/{len(df)} ({numerical_needed/len(df)*100:.1f}%)")

if problematic_cases:
    print(f"\nTop 10 most problematic cases (large range):")
    problematic_df = pd.DataFrame(problematic_cases)
    problematic_df = problematic_df.sort_values('range_pct', ascending=False)
    print(problematic_df.head(10).to_string())
