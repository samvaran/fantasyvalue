"""
Test if 3-parameter shifted log-normal can satisfy all 3 constraints.
"""
import numpy as np
from scipy.optimize import fsolve

# Target constraints
target_mean = 21.90
target_p10 = 12.11
target_p90 = 23.00

Z10 = -1.2815515655446004
Z90 = 1.2815515655446004

print("=" * 80)
print("CAN WE FIT ALL 3 CONSTRAINTS?")
print("=" * 80)
print(f"\nTarget constraints:")
print(f"  Mean: {target_mean:.2f}")
print(f"  P10:  {target_p10:.2f}")
print(f"  P90:  {target_p90:.2f}")
print()

# For shifted log-normal:
# Mean = exp(μ + σ²/2) + shift
# P10  = exp(μ + Z10*σ) + shift
# P90  = exp(μ + Z90*σ) + shift

# Define the system of equations
def equations(params):
    mu, sigma, shift = params

    eq1 = np.exp(mu + sigma**2 / 2) + shift - target_mean
    eq2 = np.exp(mu + Z10 * sigma) + shift - target_p10
    eq3 = np.exp(mu + Z90 * sigma) + shift - target_p90

    return [eq1, eq2, eq3]

# Try to solve
print("Attempting to solve system of equations...")
try:
    # Initial guess
    initial_guess = [2.0, 0.5, 5.0]

    solution = fsolve(equations, initial_guess, full_output=True)
    mu, sigma, shift = solution[0]
    info = solution[1]

    print(f"\nSolution found:")
    print(f"  μ:     {mu:.6f}")
    print(f"  σ:     {sigma:.6f}")
    print(f"  shift: {shift:.6f}")

    # Verify the solution
    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + Z10 * sigma) + shift
    fitted_p90 = np.exp(mu + Z90 * sigma) + shift

    print(f"\nVerification:")
    print(f"  Mean: {fitted_mean:.2f}  (target: {target_mean:.2f}, error: {abs(fitted_mean-target_mean):.6f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {target_p10:.2f}, error: {abs(fitted_p10-target_p10):.6f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {target_p90:.2f}, error: {abs(fitted_p90-target_p90):.6f})")

    # Check residuals
    residuals = equations([mu, sigma, shift])
    print(f"\nResiduals: {residuals}")
    print(f"Max residual: {max(abs(r) for r in residuals):.10f}")

    if max(abs(r) for r in residuals) < 0.01:
        print("\n✓ SUCCESS: All 3 constraints satisfied!")

        # Sample to verify empirically
        print("\nEmpirical verification (10,000 samples):")
        np.random.seed(42)
        z = np.random.randn(10000)
        samples = np.exp(mu + sigma * z) + shift

        empirical_mean = np.mean(samples)
        empirical_p10 = np.percentile(samples, 10)
        empirical_p90 = np.percentile(samples, 90)

        print(f"  Mean: {empirical_mean:.2f}  (target: {target_mean:.2f}, error: {abs(empirical_mean-target_mean):.2f})")
        print(f"  P10:  {empirical_p10:.2f}  (target: {target_p10:.2f}, error: {abs(empirical_p10-target_p10):.2f})")
        print(f"  P90:  {empirical_p90:.2f}  (target: {target_p90:.2f}, error: {abs(empirical_p90-target_p90):.2f})")
    else:
        print("\n✗ FAILED: Could not satisfy all constraints")
        print("The constraints may be mathematically incompatible!")

except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("COMPARISON: What does our current fitting function return?")
print("=" * 80)

from optimizer.utils.distribution_fit import fit_shifted_lognormal, sample_shifted_lognormal

mu_current, sigma_current, shift_current = fit_shifted_lognormal(target_mean, target_p10, target_p90)

print(f"\nCurrent implementation:")
print(f"  μ:     {mu_current:.6f}")
print(f"  σ:     {sigma_current:.6f}")
print(f"  shift: {shift_current:.6f}")

fitted_mean_current = np.exp(mu_current + sigma_current**2 / 2) + shift_current
fitted_p10_current = np.exp(mu_current + Z10 * sigma_current) + shift_current
fitted_p90_current = np.exp(mu_current + Z90 * sigma_current) + shift_current

print(f"\nCurrent implementation results:")
print(f"  Mean: {fitted_mean_current:.2f}  (target: {target_mean:.2f}, error: {abs(fitted_mean_current-target_mean):.2f})")
print(f"  P10:  {fitted_p10_current:.2f}  (target: {target_p10:.2f}, error: {abs(fitted_p10_current-target_p10):.2f})")
print(f"  P90:  {fitted_p90_current:.2f}  (target: {target_p90:.2f}, error: {abs(fitted_p90_current-target_p90):.2f})")

print("\n" + "=" * 80)
