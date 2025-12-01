"""
Test distribution fitting speed after removing numerical optimization.
"""
import time
import numpy as np
from optimizer.utils.distribution_fit import fit_shifted_lognormal

# Test cases covering various scenarios
test_cases = [
    # (mean, p10, p90, description)
    (20, 15, 30, "Normal QB"),
    (15, 10, 25, "Normal RB"),
    (12, 8, 20, "Normal WR"),
    (8, 5, 15, "Normal TE"),
    (10, 5, 18, "High variance player"),
    (25, 22, 35, "Low variance player"),
    (18, 10, 40, "Very high ceiling"),
    (5, 4, 8, "Low scoring"),
]

print("Testing distribution fitting speed...")
print("=" * 60)

total_time = 0
for mean, p10, p90, desc in test_cases:
    start = time.time()

    # Run 1000 fits to get measurable time
    for _ in range(1000):
        mu, sigma, shift = fit_shifted_lognormal(mean, p10, p90)

    elapsed = time.time() - start
    total_time += elapsed

    print(f"{desc:25s}: {elapsed*1000:.1f}ms for 1000 fits ({elapsed*1000000:.1f}µs per fit)")

print("=" * 60)
print(f"Average: {total_time*1000/len(test_cases):.1f}ms for 1000 fits")
print(f"         {total_time*1000000/(len(test_cases)*1000):.1f}µs per fit")

# Estimate for full optimization
print("\n" + "=" * 60)
print("Estimated times for full optimization:")
print("=" * 60)

fits_per_sim = 9  # 9 players per lineup
sims_per_lineup = 10000
num_lineups = 100

total_fits = fits_per_sim * sims_per_lineup * num_lineups
time_per_fit_us = total_time * 1000000 / (len(test_cases) * 1000)
estimated_total_time = total_fits * time_per_fit_us / 1000000  # Convert to seconds

print(f"Total distribution fits: {total_fits:,}")
print(f"Time per fit: {time_per_fit_us:.1f}µs")
print(f"Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} minutes)")
print(f"With 12 cores: {estimated_total_time/12:.1f}s ({estimated_total_time/12/60:.1f} minutes)")
