"""
Test alternative distribution families that can satisfy (mean, P10, P90) constraints.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Target constraints
target_mean = 21.90
target_p10 = 12.11
target_p90 = 23.00

print("=" * 80)
print("TESTING ALTERNATIVE DISTRIBUTION FAMILIES")
print("=" * 80)
print(f"\nTarget constraints:")
print(f"  Mean: {target_mean:.2f}")
print(f"  P10:  {target_p10:.2f}")
print(f"  P90:  {target_p90:.2f}")
print()

# Calculate implied properties
implied_median = (target_p10 + target_p90) / 2  # rough estimate
implied_std = (target_p90 - target_p10) / 2.56  # rough estimate for normal

print(f"Implied properties:")
print(f"  Range: {target_p90 - target_p10:.2f}")
print(f"  Approximate median: {implied_median:.2f}")
print(f"  Approximate std: {implied_std:.2f}")
print(f"  Mean > P90? {target_mean > target_p90}")
print(f"  Mean - P90: {target_mean - target_p90:.2f}")
print()

# The issue: mean (21.90) is barely below P90 (23.00)
# This means we need a distribution with a LONG LEFT TAIL (left-skewed)
# or extremely concentrated around the mean with small variance

print("Analysis:")
print(f"  Mean ({target_mean:.2f}) is very close to P90 ({target_p90:.2f})")
print(f"  Only {target_p90 - target_mean:.2f} points of headroom above mean")
print(f"  But {target_mean - target_p10:.2f} points below mean to P10")
print(f"  This implies a LEFT-SKEWED or BOUNDED distribution")
print()

print("=" * 80)
print("Option 1: BETA DISTRIBUTION (bounded on [a, b])")
print("=" * 80)

# Beta distribution scaled to [a, b]
# X ~ a + (b-a) * Beta(α, β)
# Has 4 parameters but we can set a=0, leaving 3 parameters

def fit_beta(mean, p10, p90):
    """Fit beta distribution to match mean, P10, P90."""

    def objective(params):
        alpha, beta, scale = params
        if alpha <= 0 or beta <= 0 or scale <= 0:
            return 1e10

        # Beta on [0, scale]
        dist = stats.beta(alpha, beta, loc=0, scale=scale)

        fitted_mean = dist.mean()
        fitted_p10 = dist.ppf(0.10)
        fitted_p90 = dist.ppf(0.90)

        error = (
            (fitted_mean - mean)**2 +
            (fitted_p10 - p10)**2 +
            (fitted_p90 - p90)**2
        )
        return error

    # Initial guess
    x0 = [2.0, 2.0, 30.0]
    bounds = [(0.1, 20), (0.1, 20), (p90, 50)]

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    if result.success:
        alpha, beta, scale = result.x
        dist = stats.beta(alpha, beta, loc=0, scale=scale)
        return dist, result.fun
    else:
        return None, 1e10

print("\nFitting Beta distribution...")
beta_dist, beta_error = fit_beta(target_mean, target_p10, target_p90)

if beta_dist and beta_error < 1.0:
    fitted_mean = beta_dist.mean()
    fitted_p10 = beta_dist.ppf(0.10)
    fitted_p90 = beta_dist.ppf(0.90)

    print(f"  Parameters: α={beta_dist.args[0]:.4f}, β={beta_dist.args[1]:.4f}, scale={beta_dist.kwds['scale']:.4f}")
    print(f"  Mean: {fitted_mean:.2f}  (target: {target_mean:.2f}, error: {abs(fitted_mean-target_mean):.4f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {target_p10:.2f}, error: {abs(fitted_p10-target_p10):.4f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {target_p90:.2f}, error: {abs(fitted_p90-target_p90):.4f})")
    print(f"  Total error: {beta_error:.6f}")

    if max(abs(fitted_mean-target_mean), abs(fitted_p10-target_p10), abs(fitted_p90-target_p90)) < 0.5:
        print("  ✓ SUCCESS: All constraints satisfied!")
    else:
        print("  ✗ FAILED: Constraints not satisfied")
else:
    print("  ✗ FAILED: Could not fit")

print("\n" + "=" * 80)
print("Option 2: GAMMA DISTRIBUTION (3 parameters)")
print("=" * 80)

def fit_gamma(mean, p10, p90):
    """Fit gamma distribution to match mean, P10, P90."""

    def objective(params):
        shape, loc, scale = params
        if shape <= 0 or scale <= 0:
            return 1e10

        dist = stats.gamma(shape, loc=loc, scale=scale)

        fitted_mean = dist.mean()
        fitted_p10 = dist.ppf(0.10)
        fitted_p90 = dist.ppf(0.90)

        error = (
            (fitted_mean - mean)**2 +
            (fitted_p10 - p10)**2 +
            (fitted_p90 - p90)**2
        )
        return error

    # Initial guess
    x0 = [2.0, 5.0, 5.0]
    bounds = [(0.1, 20), (-10, 20), (0.1, 20)]

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    if result.success:
        shape, loc, scale = result.x
        dist = stats.gamma(shape, loc=loc, scale=scale)
        return dist, result.fun
    else:
        return None, 1e10

print("\nFitting Gamma distribution...")
gamma_dist, gamma_error = fit_gamma(target_mean, target_p10, target_p90)

if gamma_dist and gamma_error < 1.0:
    fitted_mean = gamma_dist.mean()
    fitted_p10 = gamma_dist.ppf(0.10)
    fitted_p90 = gamma_dist.ppf(0.90)

    print(f"  Parameters: shape={gamma_dist.args[0]:.4f}, loc={gamma_dist.kwds['loc']:.4f}, scale={gamma_dist.kwds['scale']:.4f}")
    print(f"  Mean: {fitted_mean:.2f}  (target: {target_mean:.2f}, error: {abs(fitted_mean-target_mean):.4f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {target_p10:.2f}, error: {abs(fitted_p10-target_p10):.4f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {target_p90:.2f}, error: {abs(fitted_p90-target_p90):.4f})")
    print(f"  Total error: {gamma_error:.6f}")

    if max(abs(fitted_mean-target_mean), abs(fitted_p10-target_p10), abs(fitted_p90-target_p90)) < 0.5:
        print("  ✓ SUCCESS: All constraints satisfied!")
    else:
        print("  ✗ FAILED: Constraints not satisfied")
else:
    print("  ✗ FAILED: Could not fit")

print("\n" + "=" * 80)
print("Option 3: EMPIRICAL / PIECEWISE (custom distribution)")
print("=" * 80)

print("\nIdea: Build a custom distribution that EXACTLY satisfies all 3 constraints")
print("  Approach: Use piecewise-linear CDF with control points at P10, median, P90")
print("  This gives us full control over mean, P10, P90")
print()

# Simple empirical approach: define quantiles
quantiles = [0, 0.10, 0.50, 0.90, 1.0]

# We want: P10=12.11, P90=23.00, mean=21.90
# For mean=21.90 to work, we need the median and upper tail to be high

# Rough approach: solve for median that gives correct mean
def compute_mean_from_quantile_values(values):
    """Approximate mean by averaging quantile values."""
    # Weighted average where each region contributes proportionally
    # This is a rough approximation
    return np.mean(values)

# Try different medians
print("  Testing different median values to achieve target mean:")
for median in [17, 18, 19, 20, 21, 22]:
    values = [0, target_p10, median, target_p90, 40]  # min, P10, P50, P90, max
    approx_mean = np.mean(values)
    print(f"    median={median:.2f} → approx_mean={approx_mean:.2f}")

# Better approach: Use Johnson Su distribution (unbounded, 4 parameters)
print("\n" + "=" * 80)
print("Option 4: JOHNSON SU DISTRIBUTION (4 parameters - unbounded)")
print("=" * 80)

def fit_johnson_su(mean, p10, p90):
    """Fit Johnson SU distribution."""

    def objective(params):
        gamma, delta, loc, scale = params
        if delta <= 0 or scale <= 0:
            return 1e10

        try:
            dist = stats.johnsonsu(gamma, delta, loc=loc, scale=scale)

            fitted_mean = dist.mean()
            fitted_p10 = dist.ppf(0.10)
            fitted_p90 = dist.ppf(0.90)

            error = (
                (fitted_mean - mean)**2 +
                (fitted_p10 - p10)**2 +
                (fitted_p90 - p90)**2
            )
            return error
        except:
            return 1e10

    # Initial guess
    x0 = [0.0, 1.0, 15.0, 5.0]
    bounds = [(-5, 5), (0.1, 10), (0, 30), (0.1, 20)]

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    if result.success:
        gamma, delta, loc, scale = result.x
        dist = stats.johnsonsu(gamma, delta, loc=loc, scale=scale)
        return dist, result.fun
    else:
        return None, 1e10

print("\nFitting Johnson SU distribution...")
try:
    johnsonsu_dist, johnsonsu_error = fit_johnson_su(target_mean, target_p10, target_p90)

    if johnsonsu_dist and johnsonsu_error < 1.0:
        fitted_mean = johnsonsu_dist.mean()
        fitted_p10 = johnsonsu_dist.ppf(0.10)
        fitted_p90 = johnsonsu_dist.ppf(0.90)

        print(f"  Parameters: γ={johnsonsu_dist.args[0]:.4f}, δ={johnsonsu_dist.args[1]:.4f}, loc={johnsonsu_dist.kwds['loc']:.4f}, scale={johnsonsu_dist.kwds['scale']:.4f}")
        print(f"  Mean: {fitted_mean:.2f}  (target: {target_mean:.2f}, error: {abs(fitted_mean-target_mean):.4f})")
        print(f"  P10:  {fitted_p10:.2f}  (target: {target_p10:.2f}, error: {abs(fitted_p10-target_p10):.4f})")
        print(f"  P90:  {fitted_p90:.2f}  (target: {target_p90:.2f}, error: {abs(fitted_p90-target_p90):.4f})")
        print(f"  Total error: {johnsonsu_error:.6f}")

        if max(abs(fitted_mean-target_mean), abs(fitted_p10-target_p10), abs(fitted_p90-target_p90)) < 0.5:
            print("  ✓ SUCCESS: All constraints satisfied!")
        else:
            print("  ✗ FAILED: Constraints not satisfied")
    else:
        print("  ✗ FAILED: Could not fit")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nThe issue: mean (21.90) is very close to P90 (23.00)")
print("This implies a LEFT-SKEWED or highly concentrated distribution")
print("Shifted log-normal is inherently RIGHT-SKEWED, so it can't fit this!")
print("\nBetter distribution families:")
print("  1. Johnson SU (4 params) - can handle any skewness")
print("  2. Beta (bounded) - can be left-skewed")
print("  3. Custom empirical distribution")
