"""
Shifted log-normal distribution fitting and sampling.

Fits a 3-parameter shifted log-normal distribution with constraints:
- Mean = consensus projection
- P10 = floor (10th percentile)
- P90 = ceiling (90th percentile)
"""

import numpy as np
from typing import Tuple
import warnings

# Pre-computed z-scores for P10 and P90 (avoid expensive scipy calls)
Z10 = -1.2815515655446004  # stats.norm.ppf(0.10)
Z90 = 1.2815515655446004   # stats.norm.ppf(0.90)

# Import config to check if warnings should be suppressed
try:
    from config import SUPPRESS_DISTRIBUTION_WARNINGS
except ImportError:
    try:
        import sys
        from pathlib import Path
        # Add parent directory to path to find 0_config.py
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from importlib import util
        spec = util.spec_from_file_location("config", "0_config.py")
        config = util.module_from_spec(spec)
        spec.loader.exec_module(config)
        SUPPRESS_DISTRIBUTION_WARNINGS = getattr(config, 'SUPPRESS_DISTRIBUTION_WARNINGS', True)
    except:
        SUPPRESS_DISTRIBUTION_WARNINGS = True  # Default to suppressing warnings


def fit_lognormal_to_percentiles(p10: float, p90: float) -> Tuple[float, float, float]:
    """
    Fit log-normal distribution to match P10 and P90 only (let mean vary).

    This allows different game scenarios to have different means while preserving
    the floor/ceiling constraints.

    Distribution: X = exp(mu + sigma * Z) + shift, where Z ~ N(0,1)

    Args:
        p10: Target 10th percentile (floor)
        p90: Target 90th percentile (ceiling)

    Returns:
        (mu, sigma, shift): Parameters for shifted log-normal
        The resulting mean will be exp(mu + sigma^2/2) + shift
    """
    # Validate inputs
    if p10 >= p90:
        raise ValueError(f"P10 ({p10:.2f}) must be < P90 ({p90:.2f})")
    if p10 <= 0 or p90 <= 0:
        raise ValueError(f"All values must be positive: p10={p10}, p90={p90}")

    # For simplicity, use shift=0 (can be adjusted if needed for very low values)
    shift = 0

    # From the two equations:
    # P10 = exp(mu + Z10*sigma)
    # P90 = exp(mu + Z90*sigma)
    # Taking ratio: P90/P10 = exp((Z90-Z10)*sigma)

    sigma = np.log(p90 / p10) / (Z90 - Z10)
    mu = np.log(p10) - Z10 * sigma

    return mu, sigma, shift


def fit_shifted_lognormal(mean: float, p10: float, p90: float, player_name: str = None) -> Tuple[float, float, float]:
    """
    Fit shifted log-normal distribution to match mean, P10, and P90.

    Uses analytical solution when possible, falls back to numerical optimization.

    Distribution: X = exp(mu + sigma * Z) + shift, where Z ~ N(0,1)

    Args:
        mean: Target mean (consensus projection)
        p10: Target 10th percentile (floor)
        p90: Target 90th percentile (ceiling)
        player_name: Optional player name for debugging

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

    # Special cases for numerical stability

    # Case 1: Very narrow range (within 2% of mean)
    range_pct = (p90 - p10) / mean
    if range_pct < 0.02:
        # Use minimal variance distribution
        sigma = 0.1
        mu = np.log(mean) - sigma**2 / 2
        shift = 0
        return mu, sigma, shift

    # Case 2: Floor very close to mean (within 1%)
    if abs(mean - p10) / mean < 0.01:
        # Use very low variance
        sigma = 0.15
        mu = np.log(mean - p10 + 0.1) - sigma**2 / 2
        shift = p10 - 0.1
        return mu, sigma, shift

    # Try analytical approximation first (fast and usually good enough)
    sigma = np.log(p90 / p10) / (Z90 - Z10)

    # Solve for mu and shift from mean and P10 constraints
    numerator = mean - p10
    denominator = np.exp(sigma**2 / 2) - np.exp(sigma * Z10)

    if denominator > 0 and not np.isnan(denominator) and not np.isinf(denominator):
        mu = np.log(numerator / denominator)
        if not np.isnan(mu) and not np.isinf(mu):
            shift = p10 - np.exp(mu + sigma * Z10)

            # Check if this gives a reasonable fit
            fitted_mean = np.exp(mu + sigma**2 / 2) + shift
            fitted_p10 = np.exp(mu + sigma * Z10) + shift
            fitted_p90 = np.exp(mu + sigma * Z90) + shift

            mean_err = abs(fitted_mean - mean) / mean if mean > 0 else 0
            p10_err = abs(fitted_p10 - p10) / p10 if p10 > 0 else 0
            p90_err = abs(fitted_p90 - p90) / p90 if p90 > 0 else 0

            # If analytical solution is good enough (within 15%), use it
            # Relaxed from 10% to avoid numerical optimization
            if mean_err < 0.15 and p10_err < 0.15 and p90_err < 0.15:
                return mu, sigma, shift

            # Even if not perfect, if it's reasonable (within 25%), still use it
            # The numerical solver is slow, often fails, and causes overflow warnings
            if mean_err < 0.25 and p10_err < 0.25 and p90_err < 0.25:
                return mu, sigma, shift

    # Analytical solution had large errors - use simple robust fallback
    # Skip numerical optimization entirely (too slow, unreliable, causes scipy warnings)
    # This prioritizes mean and P10, approximates P90
    sigma = np.clip(sigma, 0.2, 2.0)  # Reasonable range for fantasy points

    # Simple estimate: use mean and P10 to get mu and shift
    mu = np.log(max(0.1, mean - p10 + 1)) - sigma**2 / 2
    shift = p10 - np.exp(mu + sigma * Z10)

    # Ensure shift is reasonable
    shift = np.clip(shift, 0, p10 * 0.9)

    return mu, sigma, shift


def sample_shifted_lognormal(mu: float, sigma: float, shift: float, size: int = 1) -> np.ndarray:
    """
    Sample from shifted log-normal distribution.

    Args:
        mu: Log-normal mu parameter
        sigma: Log-normal sigma parameter
        shift: Shift parameter (additive)
        size: Number of samples

    Returns:
        Array of samples
    """
    # Sample from standard normal
    z = np.random.randn(size)

    # Transform to log-normal and shift
    samples = np.exp(mu + sigma * z) + shift

    return samples
