"""
Shifted log-normal distribution fitting and sampling.

Fits a 3-parameter shifted log-normal distribution with constraints:
- Mean = consensus projection
- P10 = floor (10th percentile)
- P90 = ceiling (90th percentile)
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple
import warnings


def fit_shifted_lognormal(mean: float, p10: float, p90: float) -> Tuple[float, float, float]:
    """
    Fit shifted log-normal distribution to match mean, P10, and P90.

    Distribution: X = exp(mu + sigma * Z) + shift, where Z ~ N(0,1)

    Args:
        mean: Target mean (consensus projection)
        p10: Target 10th percentile (floor)
        p90: Target 90th percentile (ceiling)

    Returns:
        (mu, sigma, shift): Parameters for shifted log-normal

    Raises:
        ValueError: If constraints are invalid (p10 >= mean or p90 <= mean)
    """
    # Validate inputs
    if p10 >= mean:
        raise ValueError(f"P10 ({p10:.2f}) must be < mean ({mean:.2f})")
    if p90 <= mean:
        raise ValueError(f"P90 ({p90:.2f}) must be > mean ({mean:.2f})")
    if p10 >= p90:
        raise ValueError(f"P10 ({p10:.2f}) must be < P90 ({p90:.2f})")
    if mean <= 0 or p10 <= 0 or p90 <= 0:
        raise ValueError(f"All values must be positive: mean={mean}, p10={p10}, p90={p90}")

    # Standard normal quantiles for P10 and P90
    z10 = stats.norm.ppf(0.10)  # ≈ -1.28
    z90 = stats.norm.ppf(0.90)  # ≈ +1.28

    # Step 1: Fit standard log-normal to P10/P90 ratio
    # P90/P10 = exp(sigma * (z90 - z10))
    # => sigma = ln(P90/P10) / (z90 - z10)
    sigma = np.log(p90 / p10) / (z90 - z10)

    # Step 2: Solve for shift to match P10
    # P10 = exp(mu + sigma * z10) + shift
    # We need to also match the mean constraint
    # Mean = exp(mu + sigma^2/2) + shift

    # From P10 constraint: shift = P10 - exp(mu + sigma * z10)
    # From mean constraint: mean = exp(mu + sigma^2/2) + shift
    # Substituting: mean = exp(mu + sigma^2/2) + P10 - exp(mu + sigma * z10)
    # => exp(mu + sigma^2/2) - exp(mu + sigma * z10) = mean - P10
    # => exp(mu) * [exp(sigma^2/2) - exp(sigma * z10)] = mean - P10
    # => mu = ln((mean - P10) / (exp(sigma^2/2) - exp(sigma * z10)))

    numerator = mean - p10
    denominator = np.exp(sigma**2 / 2) - np.exp(sigma * z10)

    if denominator <= 0:
        # Fallback: use simple approximation
        warnings.warn(f"Denominator <= 0 for mean={mean}, p10={p10}, p90={p90}. Using approximation.")
        mu = np.log((mean + p90) / 2) - sigma**2 / 2
        shift = p10 - np.exp(mu + sigma * z10)
    else:
        mu = np.log(numerator / denominator)
        shift = p10 - np.exp(mu + sigma * z10)

    # Validate the fit
    fitted_mean = np.exp(mu + sigma**2 / 2) + shift
    fitted_p10 = np.exp(mu + sigma * z10) + shift
    fitted_p90 = np.exp(mu + sigma * z90) + shift

    # Allow small tolerance
    mean_error = abs(fitted_mean - mean) / mean
    p10_error = abs(fitted_p10 - p10) / p10
    p90_error = abs(fitted_p90 - p90) / p90

    if mean_error > 0.05 or p10_error > 0.05 or p90_error > 0.05:
        warnings.warn(
            f"Large fitting error: "
            f"mean_err={mean_error:.1%}, p10_err={p10_error:.1%}, p90_err={p90_error:.1%}"
        )

    return mu, sigma, shift


def sample_shifted_lognormal(mu: float, sigma: float, shift: float, size: int = 1) -> np.ndarray:
    """
    Sample from shifted log-normal distribution.

    Args:
        mu: Log-normal mu parameter
        sigma: Log-normal sigma parameter
        shift: Shift parameter (minimum value)
        size: Number of samples

    Returns:
        Array of samples
    """
    # Sample from standard log-normal
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=size)

    # Add shift
    return samples + shift


def validate_distribution(mean: float, p10: float, p90: float, mu: float, sigma: float, shift: float, n_samples: int = 100000) -> dict:
    """
    Validate fitted distribution by sampling and comparing statistics.

    Args:
        mean: Target mean
        p10: Target P10
        p90: Target P90
        mu, sigma, shift: Fitted parameters
        n_samples: Number of samples for validation

    Returns:
        Dict with validation statistics
    """
    samples = sample_shifted_lognormal(mu, sigma, shift, size=n_samples)

    return {
        'target_mean': mean,
        'sample_mean': np.mean(samples),
        'mean_error': abs(np.mean(samples) - mean) / mean,
        'target_p10': p10,
        'sample_p10': np.percentile(samples, 10),
        'p10_error': abs(np.percentile(samples, 10) - p10) / p10,
        'target_p90': p90,
        'sample_p90': np.percentile(samples, 90),
        'p90_error': abs(np.percentile(samples, 90) - p90) / p90,
        'sample_std': np.std(samples),
        'sample_median': np.median(samples),
    }


if __name__ == '__main__':
    # Test with example player
    print("Testing shifted log-normal fitting...")

    # Example: Player with consensus 15, floor 8, ceiling 25
    mean, p10, p90 = 15.0, 8.0, 25.0

    print(f"\nTarget: mean={mean}, p10={p10}, p90={p90}")

    # Fit distribution
    mu, sigma, shift = fit_shifted_lognormal(mean, p10, p90)
    print(f"Fitted parameters: mu={mu:.4f}, sigma={sigma:.4f}, shift={shift:.4f}")

    # Validate
    validation = validate_distribution(mean, p10, p90, mu, sigma, shift)
    print("\nValidation (100k samples):")
    print(f"  Mean: {validation['sample_mean']:.2f} (error: {validation['mean_error']:.1%})")
    print(f"  P10:  {validation['sample_p10']:.2f} (error: {validation['p10_error']:.1%})")
    print(f"  P90:  {validation['sample_p90']:.2f} (error: {validation['p90_error']:.1%})")
    print(f"  Median: {validation['sample_median']:.2f}")
    print(f"  Std: {validation['sample_std']:.2f}")
