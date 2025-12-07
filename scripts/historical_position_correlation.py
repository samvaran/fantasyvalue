#!/usr/bin/env python3
"""
Historical Position Correlation Analysis

Uses nflreadpy to download historical player stats and analyze:
1. Raw fantasy point correlations between positions on same team
2. Residual correlations (actual - rolling expected) between positions

Rolling weighted average is used as proxy for "expected" performance.
Only includes players with expected >= 5 points.
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple


def load_historical_data(seasons: list) -> pl.DataFrame:
    """Load historical player stats for given seasons."""
    print(f"Loading player stats for seasons {seasons[0]}-{seasons[-1]}...")
    df = nfl.load_player_stats(seasons=seasons, summary_level='week')
    print(f"  Loaded {len(df):,} player-game records")
    return df


def calculate_rolling_expected(df: pl.DataFrame, window: int = 4, decay: float = 0.8) -> pl.DataFrame:
    """
    Calculate rolling weighted average as expected performance.

    Uses exponential decay weighting for recent games.
    Only looks at previous games (not current game).
    """
    print(f"\nCalculating rolling expected (window={window}, decay={decay})...")

    # Convert to pandas for easier rolling calculations
    pdf = df.to_pandas()

    # Sort by player and time
    pdf = pdf.sort_values(['player_id', 'season', 'week'])

    # Calculate weights for exponential decay
    weights = np.array([decay ** i for i in range(window - 1, -1, -1)])
    weights = weights / weights.sum()

    def weighted_rolling_mean(series):
        """Calculate weighted rolling mean, excluding current value."""
        result = []
        values = series.values
        for i in range(len(values)):
            if i < 1:
                result.append(np.nan)
            else:
                # Look at previous games only
                start = max(0, i - window)
                prev_values = values[start:i]
                if len(prev_values) == 0:
                    result.append(np.nan)
                else:
                    # Use appropriate weights for available history
                    w = weights[-len(prev_values):]
                    w = w / w.sum()
                    result.append(np.average(prev_values, weights=w))
        return pd.Series(result, index=series.index)

    # Calculate expected for fantasy_points_ppr
    pdf['expected_pts'] = pdf.groupby('player_id')['fantasy_points_ppr'].transform(weighted_rolling_mean)

    # Calculate residual
    pdf['residual'] = pdf['fantasy_points_ppr'] - pdf['expected_pts']

    print(f"  Players with expected: {pdf['expected_pts'].notna().sum():,}")

    return pl.from_pandas(pdf)


def aggregate_by_team_game(df: pl.DataFrame, min_expected: float = 5.0) -> pd.DataFrame:
    """
    Aggregate to best player per position per team per game.

    Filter to players with expected >= min_expected.
    """
    print(f"\nAggregating by team/game (min expected >= {min_expected})...")

    pdf = df.to_pandas()

    # Filter to skill positions
    positions = ['QB', 'RB', 'WR', 'TE']
    pdf = pdf[pdf['position'].isin(positions)].copy()

    # Filter to players with expected >= threshold
    pdf = pdf[pdf['expected_pts'] >= min_expected].copy()
    print(f"  After filtering: {len(pdf):,} player-games")

    # Create game identifier
    pdf['game_id'] = pdf['season'].astype(str) + '_' + pdf['week'].astype(str) + '_' + pdf['team']

    # Get best player per team per position per game (by expected, not actual)
    idx = pdf.groupby(['game_id', 'position'])['expected_pts'].idxmax()
    aggregated = pdf.loc[idx].copy()

    print(f"  Aggregated: {len(aggregated):,} team-game-position records")

    return aggregated


def build_correlation_matrices(
    df: pd.DataFrame,
    positions: list = ['QB', 'RB', 'WR', 'TE']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build correlation matrices for raw points and residuals.

    Returns:
        corr_raw, corr_residual, pivot_raw, pivot_residual
    """
    print("\nBuilding correlation matrices...")

    # Pivot: game_id as rows, positions as columns for actual points
    pivot_raw = df.pivot_table(
        index='game_id',
        columns='position',
        values='fantasy_points_ppr',
        aggfunc='first'
    )
    pivot_raw = pivot_raw[[p for p in positions if p in pivot_raw.columns]]
    pivot_raw = pivot_raw.dropna()

    # Pivot for residuals
    pivot_residual = df.pivot_table(
        index='game_id',
        columns='position',
        values='residual',
        aggfunc='first'
    )
    pivot_residual = pivot_residual[[p for p in positions if p in pivot_residual.columns]]
    pivot_residual = pivot_residual.dropna()

    print(f"  Games with all positions (raw): {len(pivot_raw):,}")
    print(f"  Games with all positions (residual): {len(pivot_residual):,}")

    corr_raw = pivot_raw.corr()
    corr_residual = pivot_residual.corr()

    return corr_raw, corr_residual, pivot_raw, pivot_residual


def create_visualizations(
    corr_raw: pd.DataFrame,
    corr_residual: pd.DataFrame,
    pivot_raw: pd.DataFrame,
    pivot_residual: pd.DataFrame,
    output_dir: Path,
    n_games: int
):
    """Create correlation matrix heatmaps and scatter plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw correlations
    ax = axes[0]
    sns.heatmap(
        corr_raw,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.3,
        ax=ax,
        square=True,
        annot_kws={'size': 14}
    )
    ax.set_title(f'Raw Fantasy Points (PPR)\n(n={len(pivot_raw):,} team-games)', fontsize=12)

    # Residual correlations
    ax = axes[1]
    sns.heatmap(
        corr_residual,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.3,
        ax=ax,
        square=True,
        annot_kws={'size': 14}
    )
    ax.set_title(f'Residual (Actual - Expected)\n(n={len(pivot_residual):,} team-games)', fontsize=12)

    plt.suptitle(f'Position Correlations (Same Team, Expected >= 5 pts)\nSeasons 2016-2024', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'historical_position_correlations_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'historical_position_correlations_heatmap.png'}")
    plt.close()

    # Figure 2: Scatter plots for key relationships (sample for readability)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    pairs = [
        ('QB', 'WR', 'QB vs WR'),
        ('QB', 'RB', 'QB vs RB'),
        ('QB', 'TE', 'QB vs TE'),
        ('RB', 'WR', 'RB vs WR'),
        ('WR', 'TE', 'WR vs TE'),
        ('RB', 'TE', 'RB vs TE'),
    ]

    # Sample for scatter plot (too many points otherwise)
    sample_raw = pivot_raw.sample(min(1000, len(pivot_raw)), random_state=42)

    for idx, (pos1, pos2, title) in enumerate(pairs):
        ax = axes.flat[idx]

        if pos1 in sample_raw.columns and pos2 in sample_raw.columns:
            x = sample_raw[pos1]
            y = sample_raw[pos2]

            ax.scatter(x, y, alpha=0.3, s=20, c='blue')

            # Trendline
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)

                # Full correlation (not just sample)
                r = pivot_raw[pos1].corr(pivot_raw[pos2])
                ax.set_title(f'{title}\nr = {r:.3f}')

        ax.set_xlabel(f'{pos1} Points')
        ax.set_ylabel(f'{pos2} Points')

    plt.suptitle('Raw Fantasy Points: Position Scatter Plots (sampled)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'historical_position_scatter_raw.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'historical_position_scatter_raw.png'}")
    plt.close()

    # Figure 3: Residual scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sample_residual = pivot_residual.sample(min(1000, len(pivot_residual)), random_state=42)

    for idx, (pos1, pos2, title) in enumerate(pairs):
        ax = axes.flat[idx]

        if pos1 in sample_residual.columns and pos2 in sample_residual.columns:
            x = sample_residual[pos1]
            y = sample_residual[pos2]

            ax.scatter(x, y, alpha=0.3, s=20, c='green')

            # Reference lines at 0
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Trendline
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)

                # Full correlation
                r = pivot_residual[pos1].corr(pivot_residual[pos2])
                ax.set_title(f'{title}\nr = {r:.3f}')

        ax.set_xlabel(f'{pos1} Residual')
        ax.set_ylabel(f'{pos2} Residual')

    plt.suptitle('Residual (Actual - Expected): Position Scatter Plots (sampled)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'historical_position_scatter_residual.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'historical_position_scatter_residual.png'}")
    plt.close()


def analyze_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlations by season to see stability."""
    print("\nAnalyzing correlations by season...")

    results = []
    positions = ['QB', 'RB', 'WR', 'TE']
    pairs = [('QB', 'WR'), ('QB', 'RB'), ('QB', 'TE'), ('RB', 'WR')]

    # Extract season from game_id
    df['season'] = df['game_id'].str.split('_').str[0].astype(int)

    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]

        # Pivot for this season
        pivot = season_df.pivot_table(
            index='game_id',
            columns='position',
            values='residual',
            aggfunc='first'
        )
        pivot = pivot[[p for p in positions if p in pivot.columns]].dropna()

        if len(pivot) < 50:
            continue

        row = {'season': season, 'n_games': len(pivot)}
        for pos1, pos2 in pairs:
            if pos1 in pivot.columns and pos2 in pivot.columns:
                row[f'{pos1}_{pos2}'] = pivot[pos1].corr(pivot[pos2])
        results.append(row)

    return pd.DataFrame(results)


def main():
    """Run historical position correlation analysis."""
    print("=" * 70)
    print("HISTORICAL POSITION CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data for multiple seasons
    seasons = list(range(2016, 2025))  # 2016-2024
    df = load_historical_data(seasons)

    # Calculate rolling expected
    df = calculate_rolling_expected(df, window=4, decay=0.8)

    # Aggregate to best player per team per game
    aggregated = aggregate_by_team_game(df, min_expected=5.0)

    # Build correlation matrices
    corr_raw, corr_residual, pivot_raw, pivot_residual = build_correlation_matrices(aggregated)

    print("\n" + "=" * 70)
    print("RAW FANTASY POINTS CORRELATION")
    print("=" * 70)
    print(corr_raw.round(3).to_string())

    print("\n" + "=" * 70)
    print("RESIDUAL CORRELATION (Actual - Rolling Expected)")
    print("=" * 70)
    print(corr_residual.round(3).to_string())

    # Analyze by season
    season_corrs = analyze_by_season(aggregated)
    print("\n" + "=" * 70)
    print("RESIDUAL CORRELATIONS BY SEASON")
    print("=" * 70)
    print(season_corrs.round(3).to_string(index=False))

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    output_dir = Path(__file__).parent / 'charts' / 'correlations'
    create_visualizations(
        corr_raw, corr_residual,
        pivot_raw, pivot_residual,
        output_dir,
        len(pivot_raw)
    )

    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\n1. RAW POINT CORRELATIONS (what we see on the field):")
    for pos1, pos2 in [('QB', 'WR'), ('QB', 'RB'), ('QB', 'TE'), ('RB', 'WR')]:
        r = corr_raw.loc[pos1, pos2]
        print(f"   {pos1}-{pos2}: {r:+.3f}")

    print("\n2. RESIDUAL CORRELATIONS (boom/bust together after controlling for expectation):")
    for pos1, pos2 in [('QB', 'WR'), ('QB', 'RB'), ('QB', 'TE'), ('RB', 'WR')]:
        r = corr_residual.loc[pos1, pos2]
        print(f"   {pos1}-{pos2}: {r:+.3f}")

    print("\n3. DFS STACKING IMPLICATIONS:")
    qb_wr_raw = corr_raw.loc['QB', 'WR']
    qb_wr_res = corr_residual.loc['QB', 'WR']
    qb_rb_res = corr_residual.loc['QB', 'RB']
    rb_wr_res = corr_residual.loc['RB', 'WR']

    if qb_wr_res > 0.05:
        print(f"   - QB+WR stacking supported (residual r={qb_wr_res:+.3f})")
    else:
        print(f"   - QB+WR stacking weak (residual r={qb_wr_res:+.3f})")

    if qb_rb_res < -0.02:
        print(f"   - Avoid QB+RB same team (residual r={qb_rb_res:+.3f})")
    elif qb_rb_res > 0.02:
        print(f"   - QB+RB correlation positive (residual r={qb_rb_res:+.3f})")

    if rb_wr_res < -0.02:
        print(f"   - RB+WR inversely correlated (residual r={rb_wr_res:+.3f})")


if __name__ == '__main__':
    main()
