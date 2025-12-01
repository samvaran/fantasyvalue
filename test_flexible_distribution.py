"""Test distribution approach that allows scenario-specific means."""
import numpy as np

# Josh Allen example - defensive scenario
floor = 10.69
ceiling = 31.36

print("=" * 80)
print("FLEXIBLE DISTRIBUTION - ALLOW SCENARIO-SPECIFIC MEANS")
print("=" * 80)
print(f"\nGiven constraints:")
print(f"  P10 (floor):  {floor:.2f}")
print(f"  P90 (ceiling): {ceiling:.2f}")

# Fit log-normal using ONLY the two percentile constraints
# For log-normal: P10 = exp(mu + Z10*sigma) + shift
#                 P90 = exp(mu + Z90*sigma) + shift

Z10 = -1.2816
Z90 = 1.2816

# Assume shift = 0 for simplicity (can be adjusted if needed)
shift = 0

# From the two equations:
# P10 = exp(mu + Z10*sigma)
# P90 = exp(mu + Z90*sigma)
# Take ratio: P90/P10 = exp((Z90-Z10)*sigma)

sigma = np.log(ceiling / floor) / (Z90 - Z10)
mu = np.log(floor) - Z10 * sigma

print(f"\nFitted parameters (shift=0):")
print(f"  mu:    {mu:.4f}")
print(f"  sigma: {sigma:.4f}")
print(f"  shift: {shift:.4f}")

# Calculate resulting mean
fitted_mean = np.exp(mu + sigma**2 / 2) + shift
fitted_p10 = np.exp(mu + Z10 * sigma) + shift
fitted_p90 = np.exp(mu + Z90 * sigma) + shift

print(f"\nResulting distribution:")
print(f"  Mean:  {fitted_mean:.2f}")
print(f"  P10:   {fitted_p10:.2f}  (target: {floor:.2f}, error: {abs(fitted_p10-floor):.2f})")
print(f"  P90:   {fitted_p90:.2f}  (target: {ceiling:.2f}, error: {abs(fitted_p90-ceiling):.2f})")

# Compare to consensus
consensus = 21.90
diff = fitted_mean - consensus
print(f"\nComparison to consensus:")
print(f"  Fitted mean: {fitted_mean:.2f}")
print(f"  Consensus:   {consensus:.2f}")
print(f"  Difference:  {diff:+.2f} ({diff/consensus*100:+.1f}%)")

print("\n" + "=" * 80)
print("SHOOTOUT SCENARIO (same player)")
print("=" * 80)
floor_shoot = 10.69
ceiling_shoot = 38.18

sigma_shoot = np.log(ceiling_shoot / floor_shoot) / (Z90 - Z10)
mu_shoot = np.log(floor_shoot) - Z10 * sigma_shoot

fitted_mean_shoot = np.exp(mu_shoot + sigma_shoot**2 / 2)
fitted_p10_shoot = np.exp(mu_shoot + Z10 * sigma_shoot)
fitted_p90_shoot = np.exp(mu_shoot + Z90 * sigma_shoot)

print(f"\nResulting distribution:")
print(f"  Mean:  {fitted_mean_shoot:.2f}")
print(f"  P10:   {fitted_p10_shoot:.2f}")
print(f"  P90:   {fitted_p90_shoot:.2f}")

diff_shoot = fitted_mean_shoot - consensus
print(f"\nComparison to consensus:")
print(f"  Fitted mean: {fitted_mean_shoot:.2f}")
print(f"  Consensus:   {consensus:.2f}")
print(f"  Difference:  {diff_shoot:+.2f} ({diff_shoot/consensus*100:+.1f}%)")

print(f"\n" + "=" * 80)
print("BOOM vs BUST GAP")
print("=" * 80)
gap = fitted_mean_shoot - fitted_mean
print(f"Shootout mean:  {fitted_mean_shoot:.2f}")
print(f"Defensive mean: {fitted_mean:.2f}")
print(f"Gap:            {gap:+.2f} ({gap/consensus*100:+.1f}%)")

if gap > consensus * 0.05:
    print("✓ SIGNIFICANT gap (>5% of consensus)!")
else:
    print("✗ Gap is too small")
