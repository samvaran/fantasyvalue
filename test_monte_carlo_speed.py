"""
Test Monte Carlo simulation speed with caching.
"""
import time
import numpy as np
from optimizer.utils.monte_carlo import _get_distribution_params_cached, sample_shifted_lognormal

# Simulate realistic scenario:
# - 100 lineups
# - 10,000 simulations each
# - 9 players per lineup
# - ~200 unique players total
# - Each player has 5 possible game script scenarios
# - So ~1000 unique (consensus, floor, ceiling) combinations

print("Simulating realistic Monte Carlo workload...")
print("=" * 60)

# Generate test cases (mimic real player distributions)
np.random.seed(42)
unique_combinations = []
for i in range(1000):
    consensus = np.random.uniform(5, 25)
    floor = consensus * np.random.uniform(0.6, 0.9)
    ceiling = consensus * np.random.uniform(1.2, 2.0)
    unique_combinations.append((consensus, floor, ceiling))

print(f"Generated {len(unique_combinations)} unique (consensus, floor, ceiling) combinations")
print()

# Test 1: First pass (cold cache) - should be slow
print("Test 1: Cold cache (first time fitting each distribution)")
start = time.time()
for consensus, floor, ceiling in unique_combinations:
    mu, sigma, shift = _get_distribution_params_cached(consensus, floor, ceiling)
cold_time = time.time() - start
print(f"  Time: {cold_time:.2f}s for {len(unique_combinations)} fits")
print(f"  Per fit: {cold_time*1000/len(unique_combinations):.1f}ms")
print()

# Test 2: Simulate 100 lineups × 10,000 sims × 9 players
# With caching, we should only hit the cache, not recompute
print("Test 2: Simulating 100 lineups × 10,000 sims × 9 players")
print("  (should be MUCH faster with caching)")

num_lineups = 100
sims_per_lineup = 10000
players_per_lineup = 9
total_calls = num_lineups * sims_per_lineup * players_per_lineup

start = time.time()
for lineup_idx in range(num_lineups):
    # Each lineup has 9 players
    lineup_combos = np.random.choice(len(unique_combinations), size=players_per_lineup, replace=False)

    for sim_idx in range(sims_per_lineup):
        for player_idx in lineup_combos:
            consensus, floor, ceiling = unique_combinations[player_idx]
            # This should be instant (cache hit)
            mu, sigma, shift = _get_distribution_params_cached(consensus, floor, ceiling)
            # Sample from distribution (this is the actual work)
            score = sample_shifted_lognormal(mu, sigma, shift, size=1)[0]

hot_time = time.time() - start
print(f"  Time: {hot_time:.2f}s for {total_calls:,} distribution calls + samples")
print(f"  Per call: {hot_time*1000000/total_calls:.1f}µs")
print()

# Cache stats
cache_info = _get_distribution_params_cached.cache_info()
print("Cache statistics:")
print(f"  Hits: {cache_info.hits:,}")
print(f"  Misses: {cache_info.misses:,}")
print(f"  Hit rate: {cache_info.hits/(cache_info.hits + cache_info.misses)*100:.1f}%")
print()

print("=" * 60)
print("SUMMARY:")
print(f"  Without caching: {cold_time/1000*total_calls:.1f}s (~{cold_time/1000*total_calls/60:.1f} minutes)")
print(f"  With caching:    {hot_time:.1f}s ({hot_time/60:.1f} minutes)")
print(f"  Speedup:         {cold_time/1000*total_calls/hot_time:.0f}x faster")
