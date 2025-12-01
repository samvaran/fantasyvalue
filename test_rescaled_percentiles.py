"""
Test rescaling floor/ceiling to be compatible with consensus mean.

Approach:
1. Game script gives us floor/ceiling which implies a certain variance/shape
2. We want to preserve that variance/shape
3. But rescale so the mean equals consensus
"""
import numpy as np
from optimizer.utils.distribution_fit import fit_shifted_lognormal, sample_shifted_lognormal

# Example: Josh Allen defensive scenario
consensus = 21.90
original_floor = 12.11
original_ceiling = 23.00

print("=" * 80)
print("RESCALING FLOOR/CEILING TO MATCH CONSENSUS MEAN")
print("=" * 80)

print(f"\nOriginal constraints (INCOMPATIBLE):")
print(f"  Consensus mean: {consensus:.2f}")
print(f"  Floor (P10):    {original_floor:.2f}")
print(f"  Ceiling (P90):  {original_ceiling:.2f}")

# Calculate what mean this floor/ceiling implies
original_midpoint = (original_floor + original_ceiling) / 2
original_range = original_ceiling - original_floor

print(f"  Midpoint:       {original_midpoint:.2f}")
print(f"  Range:          {original_range:.2f}")

# The signal we want to preserve: relative variance
# Coefficient of variation (CV) = std / mean ≈ range / (2.56 * mean)
# For P10-P90 range: range ≈ 2.56 * std for normal distribution
relative_range = original_range / original_midpoint
print(f"  Relative range: {relative_range:.2%} (signal to preserve)")

print("\n" + "=" * 80)
print("APPROACH 1: Scale floor/ceiling proportionally to consensus")
print("=" * 80)

# Scale factor to shift midpoint to consensus
scale_factor = consensus / original_midpoint
scaled_floor_v1 = original_floor * scale_factor
scaled_ceiling_v1 = original_ceiling * scale_factor

print(f"\nScale factor: {scale_factor:.4f}")
print(f"Scaled floor:   {scaled_floor_v1:.2f}")
print(f"Scaled ceiling: {scaled_ceiling_v1:.2f}")
print(f"Scaled midpoint: {(scaled_floor_v1 + scaled_ceiling_v1)/2:.2f} (should be ~{consensus:.2f})")

# Test if this fits
try:
    mu, sigma, shift = fit_shifted_lognormal(consensus, scaled_floor_v1, scaled_ceiling_v1)

    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + -1.2816 * sigma) + shift
    fitted_p90 = np.exp(mu + 1.2816 * sigma) + shift

    print(f"\nFitted distribution:")
    print(f"  Mean: {fitted_mean:.2f}  (target: {consensus:.2f}, error: {abs(fitted_mean-consensus):.2f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {scaled_floor_v1:.2f}, error: {abs(fitted_p10-scaled_floor_v1):.2f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {scaled_ceiling_v1:.2f}, error: {abs(fitted_p90-scaled_ceiling_v1):.2f})")

    if abs(fitted_mean - consensus) < 0.5:
        print("  ✓ SUCCESS!")
    else:
        print("  ✗ Still doesn't match consensus mean")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("APPROACH 2: Keep variance, center on consensus")
print("=" * 80)

# Use consensus as center, preserve range
half_range = original_range / 2
scaled_floor_v2 = consensus - half_range
scaled_ceiling_v2 = consensus + half_range

print(f"\nPreserve range: {original_range:.2f}")
print(f"Scaled floor:   {scaled_floor_v2:.2f}")
print(f"Scaled ceiling: {scaled_ceiling_v2:.2f}")
print(f"Scaled midpoint: {(scaled_floor_v2 + scaled_ceiling_v2)/2:.2f} (should be {consensus:.2f})")

# Test if this fits
try:
    mu, sigma, shift = fit_shifted_lognormal(consensus, scaled_floor_v2, scaled_ceiling_v2)

    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + -1.2816 * sigma) + shift
    fitted_p90 = np.exp(mu + 1.2816 * sigma) + shift

    print(f"\nFitted distribution:")
    print(f"  Mean: {fitted_mean:.2f}  (target: {consensus:.2f}, error: {abs(fitted_mean-consensus):.2f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {scaled_floor_v2:.2f}, error: {abs(fitted_p10-scaled_floor_v2):.2f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {scaled_ceiling_v2:.2f}, error: {abs(fitted_p90-scaled_ceiling_v2):.2f})")

    if abs(fitted_mean - consensus) < 0.5:
        print("  ✓ SUCCESS!")
    else:
        print("  ✗ Still doesn't match consensus mean")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("APPROACH 3: Preserve relative variance (CV), center on consensus")
print("=" * 80)

# For log-normal, approximate: range ≈ 2.56 * std
# We want to preserve std/mean ratio
original_std_approx = original_range / 2.56
original_cv = original_std_approx / original_midpoint

# Apply same CV to consensus
new_std = original_cv * consensus
new_range = new_std * 2.56

scaled_floor_v3 = consensus - new_range / 2
scaled_ceiling_v3 = consensus + new_range / 2

print(f"\nOriginal CV: {original_cv:.2%}")
print(f"New std: {new_std:.2f}")
print(f"New range: {new_range:.2f}")
print(f"Scaled floor:   {scaled_floor_v3:.2f}")
print(f"Scaled ceiling: {scaled_ceiling_v3:.2f}")
print(f"Scaled midpoint: {(scaled_floor_v3 + scaled_ceiling_v3)/2:.2f} (should be {consensus:.2f})")

# Test if this fits
try:
    mu, sigma, shift = fit_shifted_lognormal(consensus, scaled_floor_v3, scaled_ceiling_v3)

    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + -1.2816 * sigma) + shift
    fitted_p90 = np.exp(mu + 1.2816 * sigma) + shift

    print(f"\nFitted distribution:")
    print(f"  Mean: {fitted_mean:.2f}  (target: {consensus:.2f}, error: {abs(fitted_mean-consensus):.2f})")
    print(f"  P10:  {fitted_p10:.2f}  (target: {scaled_floor_v3:.2f}, error: {abs(fitted_p10-scaled_floor_v3):.2f})")
    print(f"  P90:  {fitted_p90:.2f}  (target: {scaled_ceiling_v3:.2f}, error: {abs(fitted_p90-scaled_ceiling_v3):.2f})")

    if abs(fitted_mean - consensus) < 0.5:
        print("  ✓ SUCCESS!")

        # Verify empirically
        samples = sample_shifted_lognormal(mu, sigma, shift, size=10000)
        empirical_mean = np.mean(samples)
        empirical_p10 = np.percentile(samples, 10)
        empirical_p90 = np.percentile(samples, 90)

        print(f"\nEmpirical verification (10,000 samples):")
        print(f"  Mean: {empirical_mean:.2f}  (target: {consensus:.2f}, error: {abs(empirical_mean-consensus):.2f})")
        print(f"  P10:  {empirical_p10:.2f}  (target: {scaled_floor_v3:.2f}, error: {abs(empirical_p10-scaled_floor_v3):.2f})")
        print(f"  P90:  {empirical_p90:.2f}  (target: {scaled_ceiling_v3:.2f}, error: {abs(empirical_p90-scaled_ceiling_v3):.2f})")
    else:
        print("  ✗ Still doesn't match consensus mean")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nThe game script floor/ceiling tells us about VARIANCE, not absolute values.")
print("We should rescale them to be compatible with consensus mean.")
print("\nBest approach: Preserve coefficient of variation (std/mean ratio)")
print("  - Tells us 'how much variance relative to the mean'")
print("  - Rescale around consensus as center")
print("  - This preserves the game script's variance signal")
