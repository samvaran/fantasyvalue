#!/usr/bin/env python3
"""Test simulation variance to diagnose the issue"""
import sys

try:
    import numpy as np
    from scipy import stats

    np.random.seed(42)

    print("=" * 80)
    print("TESTING SIMULATION VARIANCE")
    print("=" * 80)

    # Test 1: Single player variance
    print("\n1. SINGLE PLAYER TEST:")
    consensus = 20.0
    uncertainty = 6.6  # 33% of consensus

    print(f"   Player: consensus={consensus:.2f}, uncertainty={uncertainty:.2f}")
    print(f"   Uncertainty: {(uncertainty/consensus)*100:.1f}% of mean")

    # Log-normal parameters
    mean = consensus
    std = uncertainty
    variance = std ** 2
    sigma_squared = np.log(1 + variance / (mean ** 2))
    mu = np.log(mean) - sigma_squared / 2
    sigma = np.sqrt(sigma_squared)

    # Generate samples
    samples = np.random.lognormal(mu, sigma, 10000)
    samples = np.maximum(samples, 0)

    print(f"\n   Simulation results:")
    print(f"     Mean: {np.mean(samples):.2f} (expected: {consensus:.2f})")
    print(f"     P10: {np.percentile(samples, 10):.2f}")
    print(f"     P50: {np.percentile(samples, 50):.2f}")
    print(f"     P75: {np.percentile(samples, 75):.2f}")
    print(f"     P90: {np.percentile(samples, 90):.2f}")
    print(f"\n   Spreads:")
    print(f"     P90 - P50: {np.percentile(samples, 90) - np.percentile(samples, 50):.2f} pts")
    print(f"     P90 - Mean: {np.percentile(samples, 90) - np.mean(samples):.2f} pts")

    # Test 2: Full lineup variance
    print("\n2. FULL LINEUP TEST (9 players):")

    # Typical lineup
    players = [
        {'name': 'QB', 'consensus': 20, 'uncertainty': 6.6},
        {'name': 'RB1', 'consensus': 15, 'uncertainty': 5.0},
        {'name': 'RB2', 'consensus': 12, 'uncertainty': 4.0},
        {'name': 'WR1', 'consensus': 14, 'uncertainty': 4.5},
        {'name': 'WR2', 'consensus': 12, 'uncertainty': 4.0},
        {'name': 'WR3', 'consensus': 10, 'uncertainty': 3.5},
        {'name': 'TE', 'consensus': 9, 'uncertainty': 3.0},
        {'name': 'FLEX', 'consensus': 11, 'uncertainty': 3.5},
        {'name': 'DEF', 'consensus': 7, 'uncertainty': 2.5},
    ]

    lineup_consensus = sum(p['consensus'] for p in players)
    print(f"   Lineup consensus total: {lineup_consensus:.2f}")

    # Simulate lineup WITHOUT correlations (independent)
    N_SIMS = 10000
    lineup_totals = np.zeros(N_SIMS)

    for player in players:
        mean = player['consensus']
        std = player['uncertainty']
        variance = std ** 2
        sigma_squared = np.log(1 + variance / (mean ** 2))
        mu = np.log(mean) - sigma_squared / 2
        sigma = np.sqrt(sigma_squared)

        samples = np.random.lognormal(mu, sigma, N_SIMS)
        lineup_totals += np.maximum(samples, 0)

    print(f"\n   Simulation results (INDEPENDENT):")
    print(f"     Mean: {np.mean(lineup_totals):.2f} (expected: {lineup_consensus:.2f})")
    print(f"     P10: {np.percentile(lineup_totals, 10):.2f}")
    print(f"     P50: {np.percentile(lineup_totals, 50):.2f}")
    print(f"     P75: {np.percentile(lineup_totals, 75):.2f}")
    print(f"     P90: {np.percentile(lineup_totals, 90):.2f}")
    print(f"\n   Spreads:")
    print(f"     P90 - P50: {np.percentile(lineup_totals, 90) - np.percentile(lineup_totals, 50):.2f} pts")
    print(f"     P90 - Mean: {np.percentile(lineup_totals, 90) - np.mean(lineup_totals):.2f} pts")
    print(f"     P50 - P10: {np.percentile(lineup_totals, 50) - np.percentile(lineup_totals, 10):.2f} pts")

    # Test 3: Check if correlation code is the issue
    print("\n3. TESTING CORRELATION TRANSFORMATION:")

    # Generate independent samples for 2 players
    independent = np.zeros((N_SIMS, 2))

    for i, player in enumerate(players[:2]):
        mean = player['consensus']
        std = player['uncertainty']
        variance = std ** 2
        sigma_squared = np.log(1 + variance / (mean ** 2))
        mu = np.log(mean) - sigma_squared / 2
        sigma = np.sqrt(sigma_squared)

        samples = np.random.lognormal(mu, sigma, N_SIMS)
        independent[:, i] = np.maximum(samples, 0)

    print(f"   Before correlation:")
    print(f"     Player 1: Mean={np.mean(independent[:,0]):.2f}, Std={np.std(independent[:,0]):.2f}")
    print(f"     Player 2: Mean={np.mean(independent[:,1]):.2f}, Std={np.std(independent[:,1]):.2f}")

    # Apply correlation (using same logic as league_optimizer.py)
    corr_matrix = np.array([[1.0, 0.65], [0.65, 1.0]])  # QB-WR same team

    # Convert to standard normal
    standardized = np.zeros_like(independent)
    for i in range(independent.shape[1]):
        ranks = stats.rankdata(independent[:, i])
        standardized[:, i] = stats.norm.ppf(ranks / (len(independent[:, i]) + 1))

    # Apply correlation
    L = np.linalg.cholesky(corr_matrix)
    correlated_std = standardized @ L.T

    # Convert back
    correlated = np.zeros_like(independent)
    for i in range(correlated_std.shape[1]):
        ranks = stats.rankdata(correlated_std[:, i])
        correlated[:, i] = np.sort(independent[:, i])[ranks.astype(int) - 1]

    print(f"\n   After correlation (0.65):")
    print(f"     Player 1: Mean={np.mean(correlated[:,0]):.2f}, Std={np.std(correlated[:,0]):.2f}")
    print(f"     Player 2: Mean={np.mean(correlated[:,1]):.2f}, Std={np.std(correlated[:,1]):.2f}")
    print(f"     Actual correlation: {np.corrcoef(correlated[:,0], correlated[:,1])[0,1]:.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    if np.percentile(lineup_totals, 90) - np.percentile(lineup_totals, 50) < 5:
        print("❌ PROBLEM: Lineup variance is TOO LOW!")
        print("   P90-P50 spread should be 10-20+ points for proper variance")
    else:
        print("✅ Variance looks reasonable")

    if abs(np.mean(lineup_totals) - lineup_consensus) > 2:
        print("❌ PROBLEM: Simulation mean doesn't match consensus!")
    else:
        print("✅ Simulation mean matches consensus")

except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install numpy scipy")
    sys.exit(1)
