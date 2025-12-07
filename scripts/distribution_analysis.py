#!/usr/bin/env python3
"""
Distribution Shape Analysis

Analyzes the shape of player fantasy point distributions to inform our modeling:

1. Base Distribution Shape:
   - Overall residual distribution (actual - expected)
   - Residual distribution by projection tier (do better players have different variance?)

2. Game Line Signal Analysis:
   - Which betting features best predict residual?
   - Feature importance for mu and sigma adjustments

Rolling Average Formula:
    expected_pts = weighted_rolling_mean(previous 4 games)
    weights = [0.8^3, 0.8^2, 0.8^1, 0.8^0] = [0.512, 0.64, 0.8, 1.0] (normalized)

    For a player with last 4 games scoring [10, 12, 15, 20]:
    expected = (10*0.512 + 12*0.64 + 15*0.8 + 20*1.0) / (0.512 + 0.64 + 0.8 + 1.0)
             = (5.12 + 7.68 + 12.0 + 20.0) / 2.952
             = 44.8 / 2.952 = 15.18 points

    This gives more weight to recent games (decay=0.8 means each older game
    is worth 80% of the next newer game).

Filters:
- Only players with expected (rolling avg) >= 5 points
- Only players with actual points > 0 (removes DNP)
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import norm, skewnorm
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_historical_data(seasons: List[int] = None) -> pl.DataFrame:
    """Load historical player stats from nflreadpy."""
    if seasons is None:
        seasons = list(range(2016, 2025))

    print(f"Loading player stats for seasons {seasons[0]}-{seasons[-1]}...")

    df = nfl.load_player_stats(seasons=seasons, summary_level='week')
    # Already returns polars DataFrame

    # Filter to skill positions
    df = df.filter(pl.col('position').is_in(['QB', 'RB', 'WR', 'TE']))

    # Keep relevant columns (team column name varies)
    cols = ['player_id', 'player_name', 'position', 'team',
            'season', 'week', 'fantasy_points_ppr']
    df = df.select([c for c in cols if c in df.columns])

    print(f"  Total records: {len(df):,}")

    return df


def load_betting_data(seasons: List[int] = None) -> pd.DataFrame:
    """Load historical betting lines."""
    if seasons is None:
        seasons = list(range(2016, 2025))

    print(f"\nLoading betting data for seasons {seasons[0]}-{seasons[-1]}...")

    # nflreadpy has schedules with betting info
    try:
        df = nfl.load_schedules(seasons=seasons)
        # Convert to pandas if polars
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        print(f"  Loaded schedules")
    except Exception as e:
        print(f"  Warning: Could not load schedules: {e}")
        return pd.DataFrame()

    # Relevant betting columns
    betting_cols = ['game_id', 'season', 'week', 'home_team', 'away_team',
                   'spread_line', 'total_line', 'home_moneyline', 'away_moneyline',
                   'home_score', 'away_score']

    available_cols = [c for c in betting_cols if c in df.columns]
    df = df[available_cols].copy()

    print(f"  Games with betting data: {len(df):,}")

    return df


def calculate_rolling_expected(df: pl.DataFrame, window: int = 4, decay: float = 0.8) -> pd.DataFrame:
    """
    Calculate rolling weighted average as expected performance.

    Formula: expected = sum(weight_i * points_i) / sum(weight_i)
    where weight_i = decay^(window - 1 - i) for i in [0, window-1]

    Example with decay=0.8, window=4:
    weights = [0.8^3, 0.8^2, 0.8^1, 0.8^0] = [0.512, 0.64, 0.8, 1.0]
    normalized = [0.173, 0.217, 0.271, 0.339]

    Most recent game gets ~34% weight, oldest gets ~17%
    """
    print(f"\nCalculating rolling expected (window={window}, decay={decay})...")

    pdf = df.to_pandas()
    pdf = pdf.sort_values(['player_id', 'season', 'week'])

    # Create weights: older games get lower weight
    weights = np.array([decay ** i for i in range(window - 1, -1, -1)])
    weights = weights / weights.sum()

    print(f"  Weights (oldest to newest): {weights.round(3)}")

    def weighted_rolling_mean(series):
        result = []
        values = series.values
        for i in range(len(values)):
            if i < 1:
                # Need at least 1 previous game
                result.append(np.nan)
            else:
                # Use previous games (not current)
                start_idx = max(0, i - window)
                prev_values = values[start_idx:i]
                if len(prev_values) == 0:
                    result.append(np.nan)
                else:
                    # Use appropriate weights for available games
                    w = weights[-len(prev_values):]
                    w = w / w.sum()
                    result.append(np.average(prev_values, weights=w))
        return pd.Series(result, index=series.index)

    pdf['expected_pts'] = pdf.groupby('player_id')['fantasy_points_ppr'].transform(weighted_rolling_mean)
    pdf['residual'] = pdf['fantasy_points_ppr'] - pdf['expected_pts']

    print(f"  Players with expected: {pdf['expected_pts'].notna().sum():,}")

    return pdf


def merge_with_betting(players: pd.DataFrame, betting: pd.DataFrame) -> pd.DataFrame:
    """Merge player data with betting lines."""
    if betting.empty:
        print("\nNo betting data to merge")
        return players

    print("\nMerging player data with betting lines...")

    # Create home and away dataframes for merging
    home_betting = betting.copy()
    home_betting['team'] = home_betting['home_team']
    home_betting['is_home'] = True
    home_betting['spread'] = -home_betting['spread_line']  # Home spread (negative = favorite)
    home_betting['moneyline'] = home_betting['home_moneyline']
    home_betting['opp_moneyline'] = home_betting['away_moneyline']

    away_betting = betting.copy()
    away_betting['team'] = away_betting['away_team']
    away_betting['is_home'] = False
    away_betting['spread'] = away_betting['spread_line']  # Away spread
    away_betting['moneyline'] = away_betting['away_moneyline']
    away_betting['opp_moneyline'] = away_betting['home_moneyline']

    all_betting = pd.concat([home_betting, away_betting], ignore_index=True)

    # Calculate implied points
    all_betting['implied_team_pts'] = all_betting['total_line'] / 2 - all_betting['spread'] / 2
    all_betting['implied_opp_pts'] = all_betting['total_line'] / 2 + all_betting['spread'] / 2
    all_betting['abs_spread'] = all_betting['spread'].abs()

    # Merge with players (use 'team' column from player data)
    merged = players.merge(
        all_betting[['season', 'week', 'team', 'is_home', 'spread', 'total_line',
                    'moneyline', 'opp_moneyline', 'implied_team_pts', 'implied_opp_pts',
                    'abs_spread', 'home_score', 'away_score']],
        left_on=['season', 'week', 'team'],
        right_on=['season', 'week', 'team'],
        how='left'
    )

    # Calculate actual margin for the player's team
    merged['team_score'] = np.where(merged['is_home'], merged['home_score'], merged['away_score'])
    merged['opp_score'] = np.where(merged['is_home'], merged['away_score'], merged['home_score'])
    merged['actual_margin'] = merged['team_score'] - merged['opp_score']

    print(f"  Merged records with betting: {merged['total_line'].notna().sum():,}")

    return merged


def analyze_distribution_shape(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Analyze the shape of residual distributions.

    Creates two views:
    1. Overall distribution - assumes all players similar
    2. By projection tier - checks if better players have different variance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Filter to valid data
    valid = df[(df['expected_pts'] >= 5.0) &
               (df['fantasy_points_ppr'] > 0) &
               (df['residual'].notna())].copy()

    print(f"\n{'='*70}")
    print("DISTRIBUTION SHAPE ANALYSIS")
    print(f"{'='*70}")
    print(f"Valid observations: {len(valid):,}")

    positions = ['QB', 'RB', 'WR', 'TE']

    # =========================================================================
    # FIGURE 1: Overall Residual Distribution (assumes all players similar)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    overall_stats = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos]['residual'].dropna()

        if len(pos_data) < 100:
            ax.text(0.5, 0.5, f'{pos}\nInsufficient data', ha='center', va='center')
            continue

        # Fit distributions
        mu, std = norm.fit(pos_data)

        # Try skew-normal fit
        try:
            skew_params = skewnorm.fit(pos_data)
            skew_a, skew_loc, skew_scale = skew_params
        except:
            skew_a, skew_loc, skew_scale = 0, mu, std

        # Calculate percentiles
        p10 = np.percentile(pos_data, 10)
        p25 = np.percentile(pos_data, 25)
        p50 = np.percentile(pos_data, 50)
        p75 = np.percentile(pos_data, 75)
        p90 = np.percentile(pos_data, 90)

        # Actual skewness and kurtosis
        actual_skew = stats.skew(pos_data)
        actual_kurt = stats.kurtosis(pos_data)

        overall_stats[pos] = {
            'n': len(pos_data),
            'mean': float(mu),
            'std': float(std),
            'skewness': float(actual_skew),
            'kurtosis': float(actual_kurt),
            'p10': float(p10),
            'p25': float(p25),
            'p50': float(p50),
            'p75': float(p75),
            'p90': float(p90),
            'skewnorm_a': float(skew_a),
            'skewnorm_loc': float(skew_loc),
            'skewnorm_scale': float(skew_scale),
        }

        # Plot histogram
        ax.hist(pos_data, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', label='Actual')

        # Overlay normal fit
        x = np.linspace(pos_data.min(), pos_data.max(), 200)
        ax.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2,
                label=f'Normal (μ={mu:.1f}, σ={std:.1f})')

        # Overlay skew-normal fit
        ax.plot(x, skewnorm.pdf(x, skew_a, skew_loc, skew_scale), 'g--', linewidth=2,
                label=f'Skew-Normal (α={skew_a:.2f})')

        # Add percentile lines
        for pct, val, color in [(10, p10, 'orange'), (50, p50, 'black'), (90, p90, 'purple')]:
            ax.axvline(val, color=color, linestyle=':', alpha=0.7,
                      label=f'P{pct}={val:.1f}')

        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Residual (Actual - Expected)')
        ax.set_ylabel('Density')
        ax.set_title(f'{pos}: Residual Distribution (n={len(pos_data):,})\n'
                    f'Skewness={actual_skew:.2f}, Kurtosis={actual_kurt:.2f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-25, 35)

    plt.suptitle('Overall Residual Distribution by Position\n(Assumes all players have similar distribution shape)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_overall.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'distribution_overall.png'}")
    plt.close()

    results['overall'] = overall_stats

    # =========================================================================
    # FIGURE 1b: Actual vs Expected Scatter (with P10/P50/P90 trendlines)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    scatter_stats = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        if len(pos_data) < 100:
            ax.text(0.5, 0.5, f'{pos}\nInsufficient data', ha='center', va='center')
            continue

        # Sample for plotting (too many points otherwise)
        sample = pos_data.sample(min(3000, len(pos_data)), random_state=42)

        x = sample['expected_pts']
        y = sample['fantasy_points_ppr']

        # Simple scatter plot (no color coding)
        ax.scatter(x, y, alpha=0.3, s=10, c='steelblue', edgecolors='none')

        # Perfect prediction line (y = x)
        max_val = max(x.max(), y.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x (perfect)')

        # Calculate rolling bucket percentiles for P10/P50/P90 trendlines
        pos_data_sorted = pos_data.sort_values('expected_pts')
        # Create bins based on expected points - extend to max value
        max_exp = min(35, pos_data['expected_pts'].max())
        bin_edges = np.linspace(5, max_exp, 12)
        pos_data_sorted['exp_bin'] = pd.cut(pos_data_sorted['expected_pts'], bins=bin_edges)

        bin_stats = pos_data_sorted.groupby('exp_bin', observed=True).agg({
            'expected_pts': 'mean',
            'fantasy_points_ppr': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90), 'count']
        }).reset_index()
        bin_stats.columns = ['bin', 'exp_mean', 'p50', 'p10', 'p90', 'n']
        # Lower threshold to 15 to include high-projection bins with fewer samples
        bin_stats = bin_stats[bin_stats['n'] >= 15].dropna()

        if len(bin_stats) >= 3:
            # Plot P10, P50, P90 lines
            ax.plot(bin_stats['exp_mean'], bin_stats['p50'], 'b-', linewidth=1.5,
                   label='Median (P50)')
            ax.plot(bin_stats['exp_mean'], bin_stats['p10'], 'r-', linewidth=1,
                   label='Floor (P10)', alpha=0.8)
            ax.plot(bin_stats['exp_mean'], bin_stats['p90'], 'g-', linewidth=1,
                   label='Ceiling (P90)', alpha=0.8)

            # Fill between P10 and P90
            ax.fill_between(bin_stats['exp_mean'], bin_stats['p10'], bin_stats['p90'],
                           alpha=0.15, color='gray')

        # Regression line (thin)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pos_data['expected_pts'], pos_data['fantasy_points_ppr'])

        scatter_stats[pos] = {
            'n': len(pos_data),
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r_value**2),
            'correlation': float(pos_data['expected_pts'].corr(pos_data['fantasy_points_ppr'])),
        }

        ax.set_xlabel('Expected Points (Rolling Avg)')
        ax.set_ylabel('Actual Fantasy Points')
        ax.set_title(f'{pos}: Actual vs Expected (n={len(pos_data):,})\n'
                    f'R²={r_value**2:.3f}, slope={slope:.2f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set reasonable limits
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 50)

    plt.suptitle('Actual vs Expected Fantasy Points\n(with P10/P50/P90 trendlines from rolling buckets)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_expected_scatter.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'actual_vs_expected_scatter.png'}")
    plt.close()

    results['actual_vs_expected'] = scatter_stats

    # =========================================================================
    # FIGURE 2: Distribution by Projection Tier
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    tier_stats = {}

    # Define projection tiers
    tiers = [
        ('5-10', 5, 10),
        ('10-15', 10, 15),
        ('15-20', 15, 20),
        ('20+', 20, 50),
    ]
    tier_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        tier_stats[pos] = {}

        for (tier_name, low, high), color in zip(tiers, tier_colors):
            tier_data = pos_data[(pos_data['expected_pts'] >= low) &
                                (pos_data['expected_pts'] < high)]['residual'].dropna()

            if len(tier_data) < 30:
                continue

            mu, std = norm.fit(tier_data)
            actual_skew = stats.skew(tier_data)

            # Store tier stats
            tier_stats[pos][tier_name] = {
                'n': len(tier_data),
                'mean': float(mu),
                'std': float(std),
                'skewness': float(actual_skew),
                'p10': float(np.percentile(tier_data, 10)),
                'p50': float(np.percentile(tier_data, 50)),
                'p90': float(np.percentile(tier_data, 90)),
                'cv': float(std / (pos_data[(pos_data['expected_pts'] >= low) &
                                           (pos_data['expected_pts'] < high)]['expected_pts'].mean())),
            }

            # Plot
            ax.hist(tier_data, bins=30, density=True, alpha=0.5, color=color,
                   label=f'{tier_name} pts (n={len(tier_data)}, σ={std:.1f})')

        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Residual (Actual - Expected)')
        ax.set_ylabel('Density')
        ax.set_title(f'{pos}: Residual by Projection Tier')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-25, 35)

    plt.suptitle('Residual Distribution by Projection Tier\n(Do better players have different variance?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_by_tier.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'distribution_by_tier.png'}")
    plt.close()

    results['by_tier'] = tier_stats

    # =========================================================================
    # FIGURE 3: Sigma vs Expected (Coefficient of Variation analysis)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    cv_stats = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        # Bin by expected points and calculate std in each bin
        pos_data['expected_bin'] = pd.cut(pos_data['expected_pts'],
                                          bins=[5, 8, 11, 14, 17, 20, 25, 35],
                                          labels=['5-8', '8-11', '11-14', '14-17', '17-20', '20-25', '25+'])

        bin_stats = pos_data.groupby('expected_bin').agg({
            'residual': ['mean', 'std', 'count'],
            'expected_pts': 'mean'
        }).reset_index()
        bin_stats.columns = ['bin', 'residual_mean', 'residual_std', 'n', 'expected_mean']
        bin_stats = bin_stats[bin_stats['n'] >= 30]

        if len(bin_stats) < 3:
            continue

        # Calculate coefficient of variation
        bin_stats['cv'] = bin_stats['residual_std'] / bin_stats['expected_mean']

        cv_stats[pos] = {
            'bins': bin_stats[['bin', 'expected_mean', 'residual_std', 'cv', 'n']].to_dict('records')
        }

        # Plot: Expected vs Std Dev
        ax.errorbar(bin_stats['expected_mean'], bin_stats['residual_std'],
                   fmt='o-', markersize=10, capsize=5, color='steelblue', linewidth=2)

        # Add sample size labels
        for _, row in bin_stats.iterrows():
            ax.annotate(f"n={int(row['n'])}",
                       (row['expected_mean'], row['residual_std']),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        # Fit linear relationship: std = a + b * expected
        if len(bin_stats) >= 3:
            z = np.polyfit(bin_stats['expected_mean'], bin_stats['residual_std'], 1)
            x_line = np.linspace(bin_stats['expected_mean'].min(), bin_stats['expected_mean'].max(), 50)
            ax.plot(x_line, z[0] * x_line + z[1], 'r--', linewidth=2,
                   label=f'σ = {z[0]:.3f}×E[pts] + {z[1]:.2f}')

            cv_stats[pos]['sigma_slope'] = float(z[0])
            cv_stats[pos]['sigma_intercept'] = float(z[1])
            cv_stats[pos]['avg_cv'] = float(bin_stats['cv'].mean())

        ax.set_xlabel('Expected Points (Projection)')
        ax.set_ylabel('Standard Deviation of Residual')
        ax.set_title(f'{pos}: Variance Scaling with Projection\n'
                    f'Avg CV = {bin_stats["cv"].mean():.2f}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('How Does Variance Scale with Projection?\n'
                '(If CV is constant, σ is proportional to projection)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_sigma_scaling.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'distribution_sigma_scaling.png'}")
    plt.close()

    results['sigma_scaling'] = cv_stats

    return results


def analyze_betting_signals(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Analyze which betting features predict residuals.

    Features:
    - total_line (game total)
    - spread (team spread, negative = favorite)
    - abs_spread (margin of favorite)
    - implied_team_pts
    - moneyline, opp_moneyline
    - is_home
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Filter to valid data with betting info
    valid = df[(df['expected_pts'] >= 5.0) &
               (df['fantasy_points_ppr'] > 0) &
               (df['residual'].notna()) &
               (df['total_line'].notna())].copy()

    print(f"\n{'='*70}")
    print("BETTING SIGNAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Valid observations with betting data: {len(valid):,}")

    if len(valid) < 1000:
        print("  Insufficient betting data for analysis")
        return results

    positions = ['QB', 'RB', 'WR', 'TE']

    # Features to analyze
    features = ['total_line', 'spread', 'abs_spread', 'implied_team_pts',
                'implied_opp_pts', 'is_home']

    # =========================================================================
    # FIGURE 4: Correlation of betting features with residual
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    corr_results = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        if len(pos_data) < 500:
            ax.text(0.5, 0.5, f'{pos}\nInsufficient data', ha='center', va='center')
            continue

        # Calculate correlations
        correlations = {}
        for feat in features:
            if feat in pos_data.columns and pos_data[feat].notna().sum() > 100:
                if feat == 'is_home':
                    # Convert boolean to numeric
                    r = pos_data['is_home'].astype(int).corr(pos_data['residual'])
                else:
                    r = pos_data[feat].corr(pos_data['residual'])
                correlations[feat] = r

        corr_results[pos] = correlations

        # Bar chart of correlations
        feats = list(correlations.keys())
        corrs = [correlations[f] for f in feats]
        colors = ['green' if c > 0 else 'red' for c in corrs]

        bars = ax.barh(feats, corrs, color=colors, alpha=0.7)
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Correlation with Residual')
        ax.set_title(f'{pos}: Betting Feature Correlations (n={len(pos_data):,})')
        ax.set_xlim(-0.15, 0.15)

        # Add correlation values
        for bar, corr in zip(bars, corrs):
            ax.text(corr + 0.005 if corr >= 0 else corr - 0.005,
                   bar.get_y() + bar.get_height()/2,
                   f'{corr:.3f}', va='center', ha='left' if corr >= 0 else 'right', fontsize=9)

    plt.suptitle('Betting Feature Correlations with Residual\n'
                '(Which betting signals predict over/under-performance?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'betting_correlations.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'betting_correlations.png'}")
    plt.close()

    results['correlations'] = corr_results

    # =========================================================================
    # FIGURE 5: Scatter plots of key betting features vs residual
    # =========================================================================
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))

    key_features = ['total_line', 'implied_team_pts', 'spread']

    for row, pos in enumerate(positions):
        pos_data = valid[valid['position'] == pos].copy()

        for col, feat in enumerate(key_features):
            ax = axes[row, col]

            if len(pos_data) < 100 or feat not in pos_data.columns:
                continue

            # Sample for plotting
            sample = pos_data.sample(min(2000, len(pos_data)), random_state=42)

            ax.scatter(sample[feat], sample['residual'], alpha=0.2, s=10, c='steelblue')

            # Trendline
            x = pos_data[feat].dropna()
            y = pos_data.loc[x.index, 'residual']
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 100:
                z = np.polyfit(x[mask], y[mask], 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, z[0] * x_line + z[1], 'r-', linewidth=2,
                       label=f'slope={z[0]:.4f}')

            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(feat)
            ax.set_ylabel('Residual')
            ax.set_title(f'{pos}: {feat} vs Residual')
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Betting Features vs Residual by Position', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'betting_scatter.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'betting_scatter.png'}")
    plt.close()

    # =========================================================================
    # FIGURE 6: Multivariate regression - all betting features
    # =========================================================================
    import statsmodels.api as sm

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    regression_results = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        # Prepare features
        feature_cols = ['total_line', 'spread', 'implied_team_pts', 'is_home']
        pos_data['is_home_int'] = pos_data['is_home'].astype(int)

        X = pos_data[['total_line', 'spread', 'implied_team_pts', 'is_home_int']].dropna()
        y = pos_data.loc[X.index, 'residual']

        if len(X) < 500:
            ax.text(0.5, 0.5, f'{pos}\nInsufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Fit OLS
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        regression_results[pos] = {
            'r2': float(model.rsquared),
            'coefficients': {
                'intercept': float(model.params['const']),
                'total_line': float(model.params['total_line']),
                'spread': float(model.params['spread']),
                'implied_team_pts': float(model.params['implied_team_pts']),
                'is_home': float(model.params['is_home_int']),
            },
            'pvalues': {
                'total_line': float(model.pvalues['total_line']),
                'spread': float(model.pvalues['spread']),
                'implied_team_pts': float(model.pvalues['implied_team_pts']),
                'is_home': float(model.pvalues['is_home_int']),
            },
            'n': len(X),
        }

        # Plot coefficients with significance
        coef_names = ['total_line', 'spread', 'implied_team_pts', 'is_home_int']
        coefs = [model.params[c] for c in coef_names]
        pvals = [model.pvalues[c] for c in coef_names]

        colors = ['green' if c > 0 else 'red' for c in coefs]
        alphas = [1.0 if p < 0.05 else 0.4 for p in pvals]

        bars = ax.barh(range(len(coef_names)), coefs, color=colors)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.set_yticks(range(len(coef_names)))
        ax.set_yticklabels(['Total', 'Spread', 'Implied Pts', 'Is Home'])
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Coefficient')
        ax.set_title(f'{pos}: Multivariate Regression (R²={model.rsquared:.4f})\n'
                    f'n={len(X):,} | Faded = not significant (p>0.05)')

        # Add coefficient values
        for i, (coef, pval) in enumerate(zip(coefs, pvals)):
            sig = '*' if pval < 0.05 else ''
            ax.text(coef + 0.01 if coef >= 0 else coef - 0.01, i,
                   f'{coef:.4f}{sig}', va='center',
                   ha='left' if coef >= 0 else 'right', fontsize=9)

    plt.suptitle('Multivariate Regression: Betting Features → Residual\n'
                '(How much do all features combined explain?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'betting_regression.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'betting_regression.png'}")
    plt.close()

    results['regression'] = regression_results

    # =========================================================================
    # FIGURE 7: Sigma prediction from betting features
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sigma_results = {}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_data = valid[valid['position'] == pos].copy()

        if len(pos_data) < 1000:
            continue

        # Bin by total_line and calculate residual std in each bin
        pos_data['total_bin'] = pd.cut(pos_data['total_line'], bins=8)

        bin_stats = pos_data.groupby('total_bin').agg({
            'residual': ['std', 'count'],
            'total_line': 'mean'
        }).reset_index()
        bin_stats.columns = ['bin', 'residual_std', 'n', 'total_mean']
        bin_stats = bin_stats[bin_stats['n'] >= 50]

        sigma_results[pos] = {
            'total_line_bins': bin_stats[['total_mean', 'residual_std', 'n']].to_dict('records')
        }

        # Plot
        ax.errorbar(bin_stats['total_mean'], bin_stats['residual_std'],
                   fmt='o-', markersize=10, color='steelblue', linewidth=2)

        # Fit line
        if len(bin_stats) >= 3:
            z = np.polyfit(bin_stats['total_mean'], bin_stats['residual_std'], 1)
            x_line = np.linspace(bin_stats['total_mean'].min(), bin_stats['total_mean'].max(), 50)
            ax.plot(x_line, z[0] * x_line + z[1], 'r--', linewidth=2,
                   label=f'σ = {z[0]:.4f}×total + {z[1]:.2f}')
            sigma_results[pos]['sigma_vs_total_slope'] = float(z[0])
            sigma_results[pos]['sigma_vs_total_intercept'] = float(z[1])

        ax.set_xlabel('Game Total Line')
        ax.set_ylabel('Residual Std Dev')
        ax.set_title(f'{pos}: Does Game Total Affect Variance?')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Variance (σ) vs Game Total\n'
                '(Do high-total games have more variance?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'betting_sigma_by_total.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'betting_sigma_by_total.png'}")
    plt.close()

    results['sigma_prediction'] = sigma_results

    return results


def save_results_markdown(dist_results: Dict, betting_results: Dict, output_dir: Path):
    """Save all results to a markdown file for later reference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'distribution_analysis_results.md'

    with open(md_path, 'w') as f:
        f.write("# Distribution Analysis Results\n\n")
        f.write("This file contains the numerical results from the distribution shape and betting signal analysis.\n")
        f.write("These values can be used to parameterize the player distribution models.\n\n")

        # Rolling Average Formula
        f.write("## Rolling Average Formula (Expected Points Proxy)\n\n")
        f.write("We use a weighted rolling average of the previous 4 games as a proxy for projections:\n\n")
        f.write("```\n")
        f.write("expected_pts = Σ(weight_i × points_i) / Σ(weight_i)\n")
        f.write("where weight_i = 0.8^(3-i) for i in [0,1,2,3] (most recent = i=3)\n\n")
        f.write("Weights (oldest to newest): [0.173, 0.217, 0.271, 0.339]\n")
        f.write("```\n\n")
        f.write("Example: For last 4 games = [10, 12, 15, 20] points:\n")
        f.write("- expected = (10×0.173 + 12×0.217 + 15×0.271 + 20×0.339) = 15.2 points\n\n")

        # Actual vs Expected Regression
        f.write("## 0. Actual vs Expected (Projection Quality)\n\n")
        f.write("How well does rolling average predict actual performance?\n\n")

        if 'actual_vs_expected' in dist_results:
            f.write("| Position | n | Slope | Intercept | R² | Correlation |\n")
            f.write("|----------|---|-------|-----------|-----|-------------|\n")
            for pos, stats in dist_results['actual_vs_expected'].items():
                f.write(f"| {pos} | {stats['n']:,} | {stats['slope']:.3f} | {stats['intercept']:.2f} | "
                       f"{stats['r2']:.3f} | {stats['correlation']:.3f} |\n")
            f.write("\n")
            f.write("**Interpretation**: Slope < 1 means regression to mean (high projections underperform, low projections overperform).\n\n")

        # Overall Distribution Stats
        f.write("## 1. Overall Residual Distribution\n\n")
        f.write("Residual = Actual Points - Expected Points\n\n")

        if 'overall' in dist_results:
            f.write("| Position | n | Mean | Std | Skewness | Kurtosis | P10 | P50 | P90 |\n")
            f.write("|----------|---|------|-----|----------|----------|-----|-----|-----|\n")
            for pos, stats in dist_results['overall'].items():
                f.write(f"| {pos} | {stats['n']:,} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                       f"{stats['skewness']:.2f} | {stats['kurtosis']:.2f} | "
                       f"{stats['p10']:.1f} | {stats['p50']:.1f} | {stats['p90']:.1f} |\n")
            f.write("\n")

            f.write("### Skew-Normal Fit Parameters\n\n")
            f.write("| Position | α (skewness) | loc | scale |\n")
            f.write("|----------|--------------|-----|-------|\n")
            for pos, stats in dist_results['overall'].items():
                f.write(f"| {pos} | {stats['skewnorm_a']:.3f} | {stats['skewnorm_loc']:.2f} | "
                       f"{stats['skewnorm_scale']:.2f} |\n")
            f.write("\n")

        # Tier Stats
        f.write("## 2. Distribution by Projection Tier\n\n")
        f.write("Does variance scale with projection level?\n\n")

        if 'by_tier' in dist_results:
            for pos, tiers in dist_results['by_tier'].items():
                f.write(f"### {pos}\n\n")
                f.write("| Tier | n | Mean | Std | CV | P10 | P90 |\n")
                f.write("|------|---|------|-----|-----|-----|-----|\n")
                for tier, stats in tiers.items():
                    f.write(f"| {tier} | {stats['n']:,} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                           f"{stats['cv']:.2f} | {stats['p10']:.1f} | {stats['p90']:.1f} |\n")
                f.write("\n")

        # Sigma Scaling
        f.write("## 3. Sigma Scaling with Projection\n\n")
        f.write("Formula: σ = slope × E[pts] + intercept\n\n")

        if 'sigma_scaling' in dist_results:
            f.write("| Position | Slope | Intercept | Avg CV |\n")
            f.write("|----------|-------|-----------|--------|\n")
            for pos, stats in dist_results['sigma_scaling'].items():
                if 'sigma_slope' in stats:
                    f.write(f"| {pos} | {stats['sigma_slope']:.4f} | {stats['sigma_intercept']:.2f} | "
                           f"{stats['avg_cv']:.3f} |\n")
            f.write("\n")

        # Betting Correlations
        f.write("## 4. Betting Feature Correlations with Residual\n\n")

        if betting_results and 'correlations' in betting_results:
            # Get all features
            all_features = set()
            for pos_corrs in betting_results['correlations'].values():
                all_features.update(pos_corrs.keys())
            all_features = sorted(all_features)

            header = "| Feature | " + " | ".join(['QB', 'RB', 'WR', 'TE']) + " |\n"
            f.write(header)
            f.write("|---------|" + "|".join(["---"] * 4) + "|\n")

            for feat in all_features:
                row = f"| {feat} |"
                for pos in ['QB', 'RB', 'WR', 'TE']:
                    if pos in betting_results['correlations'] and feat in betting_results['correlations'][pos]:
                        row += f" {betting_results['correlations'][pos][feat]:.4f} |"
                    else:
                        row += " - |"
                f.write(row + "\n")
            f.write("\n")

        # Regression Results
        f.write("## 5. Multivariate Regression: Betting → Residual\n\n")
        f.write("Model: residual ~ intercept + β₁×total + β₂×spread + β₃×implied_pts + β₄×is_home\n\n")

        if betting_results and 'regression' in betting_results:
            f.write("| Position | R² | n | Intercept | β(total) | β(spread) | β(implied) | β(home) |\n")
            f.write("|----------|-----|---|-----------|----------|-----------|------------|--------|\n")
            for pos, stats in betting_results['regression'].items():
                coef = stats['coefficients']
                pval = stats['pvalues']
                f.write(f"| {pos} | {stats['r2']:.4f} | {stats['n']:,} | "
                       f"{coef['intercept']:.3f} | "
                       f"{coef['total_line']:.4f}{'*' if pval['total_line'] < 0.05 else ''} | "
                       f"{coef['spread']:.4f}{'*' if pval['spread'] < 0.05 else ''} | "
                       f"{coef['implied_team_pts']:.4f}{'*' if pval['implied_team_pts'] < 0.05 else ''} | "
                       f"{coef['is_home']:.4f}{'*' if pval['is_home'] < 0.05 else ''} |\n")
            f.write("\n*asterisk indicates p < 0.05\n\n")

        # Sigma vs Total
        f.write("## 6. Variance Scaling with Game Total\n\n")
        f.write("Formula: σ = slope × game_total + intercept\n\n")

        if betting_results and 'sigma_prediction' in betting_results:
            f.write("| Position | Slope | Intercept |\n")
            f.write("|----------|-------|----------|\n")
            for pos, stats in betting_results['sigma_prediction'].items():
                if 'sigma_vs_total_slope' in stats:
                    f.write(f"| {pos} | {stats['sigma_vs_total_slope']:.4f} | "
                           f"{stats['sigma_vs_total_intercept']:.2f} |\n")
            f.write("\n")

        # Key Takeaways
        f.write("## Key Takeaways\n\n")
        f.write("1. **Base Distribution**: Use skew-normal with position-specific parameters\n")
        f.write("2. **Sigma Scaling**: σ scales roughly proportionally with projection (CV ≈ constant)\n")
        f.write("3. **Betting Adjustments**: Game total and spread provide small but significant signal\n")
        f.write("4. **For CVaR**: Use these parameters to construct player-specific distributions\n")

    print(f"\n  Saved results: {md_path}")


def main():
    """Run distribution analysis."""
    print("=" * 70)
    print("DISTRIBUTION SHAPE ANALYSIS")
    print("=" * 70)

    # Load data
    seasons = list(range(2016, 2025))
    players = load_historical_data(seasons)
    betting = load_betting_data(seasons)

    # Calculate rolling expected
    players_df = calculate_rolling_expected(players)

    # Merge with betting
    merged = merge_with_betting(players_df, betting)

    # Output directory
    output_dir = Path(__file__).parent / 'charts' / 'distributions'

    # Analyze distribution shape
    dist_results = analyze_distribution_shape(merged, output_dir)

    # Analyze betting signals
    betting_results = analyze_betting_signals(merged, output_dir)

    # Save results to markdown
    save_results_markdown(dist_results, betting_results, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    print("  - distribution_overall.png")
    print("  - distribution_by_tier.png")
    print("  - distribution_sigma_scaling.png")
    print("  - betting_correlations.png")
    print("  - betting_scatter.png")
    print("  - betting_regression.png")
    print("  - betting_sigma_by_total.png")
    print("  - distribution_analysis_results.md")


if __name__ == '__main__':
    main()
