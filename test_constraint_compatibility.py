"""
Test if the constraints are mathematically compatible for shifted log-normal.

For a shifted log-normal distribution with parameters (μ, σ, shift):
- The relationship between mean, P10, and P90 is constrained
- Not every (mean, P10, P90) triple is achievable
"""
import numpy as np
from scipy.optimize import minimize

# Constraints
target_mean = 21.90
target_p10 = 12.11
target_p90 = 23.00

Z10 = -1.2815515655446004
Z90 = 1.2815515655446004

print("=" * 80)
print("TESTING CONSTRAINT COMPATIBILITY")
print("=" * 80)

# For shifted log-normal, we have:
# Mean = exp(μ + σ²/2) + shift ... (1)
# P10 = exp(μ + Z10*σ) + shift  ... (2)
# P90 = exp(μ + Z90*σ) + shift  ... (3)

# From (2) and (3):
# P10 - shift = exp(μ + Z10*σ)
# P90 - shift = exp(μ + Z90*σ)

# Dividing:
# (P90 - shift) / (P10 - shift) = exp(μ + Z90*σ) / exp(μ + Z10*σ)
#                                = exp((Z90 - Z10)*σ)
#                                = exp(2*Z90*σ)

# So: σ = ln((P90 - shift)/(P10 - shift)) / (2*Z90)

# From (2): μ = ln(P10 - shift) - Z10*σ

# Substitute into (1): Mean = exp(ln(P10 - shift) - Z10*σ + σ²/2) + shift
#                            = (P10 - shift) * exp(-Z10*σ + σ²/2) + shift

# This gives us Mean as a function of shift and the P10/P90 constraints!

def mean_as_function_of_shift(shift):
    """Given shift, calculate implied mean from P10/P90 constraints."""
    if shift >= target_p10:
        return None  # Invalid: shift must be < P10

    sigma = np.log((target_p90 - shift) / (target_p10 - shift)) / (2 * Z90)
    if sigma < 0:
        return None  # Invalid: sigma must be positive

    implied_mean = (target_p10 - shift) * np.exp(-Z10 * sigma + sigma**2 / 2) + shift
    return implied_mean

print("\nTesting different shift values:")
print("  If shift is constrained by P10/P90, what mean does that imply?")
print()

shifts_to_test = np.linspace(0, target_p10 * 0.95, 20)
valid_shifts = []
implied_means = []

for shift in shifts_to_test:
    implied_mean = mean_as_function_of_shift(shift)
    if implied_mean is not None:
        valid_shifts.append(shift)
        implied_means.append(implied_mean)
        diff = abs(implied_mean - target_mean)
        marker = " ← CLOSEST" if diff == min([abs(m - target_mean) for m in implied_means]) else ""
        print(f"  shift={shift:6.2f} → implied_mean={implied_mean:6.2f}  (target: {target_mean:.2f}, diff: {diff:+6.2f}){marker}")

print()
print("=" * 80)

# Find the shift that gives us closest to target mean
def objective(shift):
    implied_mean = mean_as_function_of_shift(shift)
    if implied_mean is None:
        return 1e10
    return (implied_mean - target_mean)**2

result = minimize(objective, x0=5.0, bounds=[(0, target_p10 * 0.99)])
best_shift = result.x[0]
best_implied_mean = mean_as_function_of_shift(best_shift)

print(f"\nOptimal shift to get closest to target mean:")
print(f"  shift: {best_shift:.4f}")
print(f"  Implied mean: {best_implied_mean:.2f}")
print(f"  Target mean: {target_mean:.2f}")
print(f"  Error: {best_implied_mean - target_mean:+.2f} ({(best_implied_mean - target_mean)/target_mean*100:+.1f}%)")

# Calculate the full parameters
sigma_best = np.log((target_p90 - best_shift) / (target_p10 - best_shift)) / (2 * Z90)
mu_best = np.log(target_p10 - best_shift) - Z10 * sigma_best

print(f"\n  Full parameters:")
print(f"    μ:     {mu_best:.6f}")
print(f"    σ:     {sigma_best:.6f}")
print(f"    shift: {best_shift:.6f}")

# Verify
fitted_mean = np.exp(mu_best + sigma_best**2 / 2) + best_shift
fitted_p10 = np.exp(mu_best + Z10 * sigma_best) + best_shift
fitted_p90 = np.exp(mu_best + Z90 * sigma_best) + best_shift

print(f"\n  Verification:")
print(f"    Mean: {fitted_mean:.2f}  (target: {target_mean:.2f})")
print(f"    P10:  {fitted_p10:.2f}  (target: {target_p10:.2f})")
print(f"    P90:  {fitted_p90:.2f}  (target: {target_p90:.2f})")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
if abs(best_implied_mean - target_mean) < 0.5:
    print("✓ Constraints are COMPATIBLE (within tolerance)")
else:
    print("✗ Constraints are INCOMPATIBLE")
    print(f"  The best possible mean is {best_implied_mean:.2f}, not {target_mean:.2f}")
    print(f"  Given P10={target_p10:.2f} and P90={target_p90:.2f}, the distribution shape")
    print(f"  is constrained such that the mean cannot equal {target_mean:.2f}")
