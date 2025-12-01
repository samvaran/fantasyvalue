"""
Debug why fitted distributions don't preserve consensus mean.
"""
import pandas as pd
import numpy as np
from optimizer.utils.distribution_fit import fit_shifted_lognormal, sample_shifted_lognormal

# Load players
players_df = pd.read_csv('data/intermediate/players_integrated.csv')

# Test Josh Allen defensive scenario (we know this fails badly)
player = players_df[players_df['name'] == 'josh allen'].iloc[0]
consensus = player['fpProjPts']
floor_defensive = player['floor_defensive']
ceiling_defensive = player['ceiling_defensive']

print("=" * 80)
print("DEBUGGING JOSH ALLEN DEFENSIVE SCENARIO")
print("=" * 80)
print(f"\nInputs:")
print(f"  Consensus (target mean): {consensus:.2f}")
print(f"  Floor (target P10):      {floor_defensive:.2f}")
print(f"  Ceiling (target P90):    {ceiling_defensive:.2f}")
print(f"  Midpoint:                {(floor_defensive + ceiling_defensive)/2:.2f}")
print()

# Try to fit distribution
print("Attempting to fit distribution...")
try:
    mu, sigma, shift = fit_shifted_lognormal(consensus, floor_defensive, ceiling_defensive)

    print(f"\nFitted parameters:")
    print(f"  mu:    {mu:.4f}")
    print(f"  sigma: {sigma:.4f}")
    print(f"  shift: {shift:.4f}")

    # Calculate what we actually got
    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + -1.2815515655446004 * sigma) + shift
    fitted_p90 = np.exp(mu + 1.2815515655446004 * sigma) + shift

    print(f"\nFitted distribution:")
    print(f"  Mean: {fitted_mean:.2f}  (target: {consensus:.2f}, error: {(fitted_mean-consensus)/consensus*100:+.1f}%)")
    print(f"  P10:  {fitted_p10:.2f}   (target: {floor_defensive:.2f}, error: {(fitted_p10-floor_defensive)/floor_defensive*100:+.1f}%)")
    print(f"  P90:  {fitted_p90:.2f}  (target: {ceiling_defensive:.2f}, error: {(fitted_p90-ceiling_defensive)/ceiling_defensive*100:+.1f}%)")

    # Sample to verify
    samples = sample_shifted_lognormal(mu, sigma, shift, size=10000)
    empirical_mean = np.mean(samples)
    empirical_p10 = np.percentile(samples, 10)
    empirical_p90 = np.percentile(samples, 90)

    print(f"\nEmpirical from 10,000 samples:")
    print(f"  Mean: {empirical_mean:.2f}  (target: {consensus:.2f}, error: {(empirical_mean-consensus)/consensus*100:+.1f}%)")
    print(f"  P10:  {empirical_p10:.2f}   (target: {floor_defensive:.2f}, error: {(empirical_p10-floor_defensive)/floor_defensive*100:+.1f}%)")
    print(f"  P90:  {empirical_p90:.2f}  (target: {ceiling_defensive:.2f}, error: {(empirical_p90-ceiling_defensive)/ceiling_defensive*100:+.1f}%)")

except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 80)
print("QUESTION: Why doesn't the mean match consensus?")
print("=" * 80)

# Let's check what the theoretical constraints imply
print(f"\nFor a shifted log-normal distribution:")
print(f"  If P10 = {floor_defensive:.2f} and P90 = {ceiling_defensive:.2f}")
print(f"  Then sigma ≈ ln(P90/P10) / (2*1.2816) = ln({ceiling_defensive:.2f}/{floor_defensive:.2f}) / 2.563")
sigma_implied = np.log(ceiling_defensive / floor_defensive) / (2 * 1.2815515655446004)
print(f"  σ ≈ {sigma_implied:.4f}")

print(f"\n  For a log-normal: mean = exp(μ + σ²/2) + shift")
print(f"  For a log-normal: P10  = exp(μ - 1.2816*σ) + shift")
print(f"  For a log-normal: P90  = exp(μ + 1.2816*σ) + shift")

print(f"\n  From P10 and P90 constraints:")
print(f"    exp(μ - 1.2816*σ) + shift = {floor_defensive:.2f}")
print(f"    exp(μ + 1.2816*σ) + shift = {ceiling_defensive:.2f}")
print(f"\n  Subtracting: exp(μ + 1.2816*σ) - exp(μ - 1.2816*σ) = {ceiling_defensive - floor_defensive:.2f}")
print(f"               exp(μ) * [exp(1.2816*σ) - exp(-1.2816*σ)] = {ceiling_defensive - floor_defensive:.2f}")

exp_term = np.exp(1.2816 * sigma_implied) - np.exp(-1.2816 * sigma_implied)
mu_implied = np.log((ceiling_defensive - floor_defensive) / exp_term)
print(f"               μ ≈ {mu_implied:.4f}")

shift_implied = floor_defensive - np.exp(mu_implied - 1.2816 * sigma_implied)
print(f"               shift ≈ {shift_implied:.4f}")

mean_implied = np.exp(mu_implied + sigma_implied**2 / 2) + shift_implied
print(f"\n  This gives mean = {mean_implied:.2f}")
print(f"  But we want mean = {consensus:.2f}")
print(f"\n  ERROR: {mean_implied:.2f} ≠ {consensus:.2f}")
print(f"         Difference: {mean_implied - consensus:.2f} ({(mean_implied-consensus)/consensus*100:+.1f}%)")

print(f"\n  CONCLUSION: The constraints (P10={floor_defensive:.2f}, P90={ceiling_defensive:.2f})")
print(f"              MATHEMATICALLY IMPLY mean ≈ {mean_implied:.2f}")
print(f"              This is INCOMPATIBLE with target mean = {consensus:.2f}")
print(f"\n  The fitting function CANNOT satisfy all three constraints!")
print(f"  It must sacrifice one of them. Currently it's sacrificing the mean constraint.")

print("\n" + "=" * 80)
