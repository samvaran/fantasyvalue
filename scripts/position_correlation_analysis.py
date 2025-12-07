#!/usr/bin/env python3
"""
Position Correlation Analysis

Analyzes correlations between positions on the same team:
1. Raw fantasy point correlations (when QB booms, do WRs boom?)
2. Residual correlations (when QB beats projection, do WRs beat projection?)

Only includes players with projections >= 5 points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple


def load_data(week_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load projection and actual data."""
    players_file = week_dir / 'intermediate' / '1_players.csv'
    actuals_file = week_dir / 'intermediate' / '4_actual_players.csv'

    projections = pd.read_csv(players_file)
    actuals = pd.read_csv(actuals_file)

    return projections, actuals


def merge_data(projections: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Merge projections with actuals."""
    # Match on name and team
    merged = projections.merge(
        actuals[['name', 'team', 'actual_points']],
        on=['name', 'team'],
        how='inner'
    )

    # Filter to players with projections >= 5
    merged = merged[merged['consensus'] >= 5.0].copy()

    # Calculate residual (actual - projected)
    merged['residual'] = merged['actual_points'] - merged['consensus']

    return merged


def aggregate_by_team_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to best player per position per team.

    For positions with multiple players (WR, RB), take the highest projected player.
    """
    # Get best player per team per position (by projection)
    team_pos = df.loc[df.groupby(['team', 'position'])['consensus'].idxmax()]

    return team_pos


def build_correlation_matrix(
    df: pd.DataFrame,
    value_col: str,
    positions: list = ['QB', 'RB', 'WR', 'TE']
) -> pd.DataFrame:
    """
    Build correlation matrix between positions.

    Pivots data so each team is a row with position columns,
    then computes correlation matrix.
    """
    # Pivot: teams as rows, positions as columns
    pivot = df.pivot_table(
        index='team',
        columns='position',
        values=value_col,
        aggfunc='first'  # Already aggregated to one per team
    )

    # Filter to requested positions
    pivot = pivot[[p for p in positions if p in pivot.columns]]

    # Drop teams missing any position
    pivot = pivot.dropna()

    # Compute correlation matrix
    corr = pivot.corr()

    return corr, pivot


def create_visualizations(
    corr_raw: pd.DataFrame,
    corr_residual: pd.DataFrame,
    pivot_raw: pd.DataFrame,
    pivot_residual: pd.DataFrame,
    output_dir: Path
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
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True
    )
    ax.set_title(f'Raw Fantasy Points Correlation\n(n={len(pivot_raw)} teams)')

    # Residual correlations
    ax = axes[1]
    sns.heatmap(
        corr_residual,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True
    )
    ax.set_title(f'Residual Correlation (Actual - Projected)\n(n={len(pivot_residual)} teams)')

    plt.suptitle('Position Correlations (Same Team, Projections >= 5 pts)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'position_correlations_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'position_correlations_heatmap.png'}")
    plt.close()

    # Figure 2: Scatter plots for key relationships
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    pairs = [
        ('QB', 'WR', 'QB vs WR'),
        ('QB', 'RB', 'QB vs RB'),
        ('QB', 'TE', 'QB vs TE'),
        ('RB', 'WR', 'RB vs WR'),
        ('WR', 'TE', 'WR vs TE'),
        ('RB', 'TE', 'RB vs TE'),
    ]

    for idx, (pos1, pos2, title) in enumerate(pairs):
        ax = axes.flat[idx]

        if pos1 in pivot_raw.columns and pos2 in pivot_raw.columns:
            x = pivot_raw[pos1]
            y = pivot_raw[pos2]

            ax.scatter(x, y, alpha=0.6, s=60, c='blue', label='Raw')

            # Add team labels
            for team, xi, yi in zip(pivot_raw.index, x, y):
                ax.annotate(team, (xi, yi), fontsize=7, alpha=0.7)

            # Trendline
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

                # Correlation
                r = x.corr(y)
                ax.set_title(f'{title}\nr = {r:.2f}')
            else:
                ax.set_title(title)
        else:
            ax.set_title(f'{title}\n(insufficient data)')

        ax.set_xlabel(f'{pos1} Points')
        ax.set_ylabel(f'{pos2} Points')

    plt.suptitle('Raw Fantasy Points: Position Scatter Plots', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'position_scatter_raw.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'position_scatter_raw.png'}")
    plt.close()

    # Figure 3: Residual scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (pos1, pos2, title) in enumerate(pairs):
        ax = axes.flat[idx]

        if pos1 in pivot_residual.columns and pos2 in pivot_residual.columns:
            x = pivot_residual[pos1]
            y = pivot_residual[pos2]

            ax.scatter(x, y, alpha=0.6, s=60, c='green', label='Residual')

            # Add team labels
            for team, xi, yi in zip(pivot_residual.index, x, y):
                ax.annotate(team, (xi, yi), fontsize=7, alpha=0.7)

            # Add reference lines at 0
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Trendline
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

                # Correlation
                r = x.corr(y)
                ax.set_title(f'{title}\nr = {r:.2f}')
            else:
                ax.set_title(title)
        else:
            ax.set_title(f'{title}\n(insufficient data)')

        ax.set_xlabel(f'{pos1} Residual')
        ax.set_ylabel(f'{pos2} Residual')

    plt.suptitle('Residual (Actual - Projected): Position Scatter Plots', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'position_scatter_residual.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'position_scatter_residual.png'}")
    plt.close()


def main():
    """Run position correlation analysis."""
    print("=" * 70)
    print("POSITION CORRELATION ANALYSIS")
    print("=" * 70)

    # Find most recent week with actual data
    data_dir = Path(__file__).parent.parent / 'data'
    week_dirs = sorted(data_dir.glob('2025_*'), reverse=True)

    if not week_dirs:
        print("No week data found!")
        return

    # Find first week that has both players.csv and actual_players.csv
    week_dir = None
    for wd in week_dirs:
        players_file = wd / 'intermediate' / '1_players.csv'
        actuals_file = wd / 'intermediate' / '4_actual_players.csv'
        if players_file.exists() and actuals_file.exists():
            week_dir = wd
            break

    if week_dir is None:
        print("No week with complete data found!")
        return

    print(f"\nUsing week: {week_dir.name}")

    # Load and merge data
    print("\nLoading data...")
    projections, actuals = load_data(week_dir)
    print(f"  Projections: {len(projections)} players")
    print(f"  Actuals: {len(actuals)} players")

    merged = merge_data(projections, actuals)
    print(f"  Merged (projections >= 5): {len(merged)} players")

    # Show position breakdown
    print(f"\n  Position breakdown:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'D']:
        count = len(merged[merged['position'] == pos])
        print(f"    {pos}: {count} players")

    # Aggregate to best player per team per position
    print("\nAggregating to best player per team per position...")
    aggregated = aggregate_by_team_position(merged)
    print(f"  Aggregated: {len(aggregated)} player-team-position combinations")

    # Build correlation matrices
    print("\nBuilding correlation matrices...")

    # Raw points
    corr_raw, pivot_raw = build_correlation_matrix(aggregated, 'actual_points')
    print(f"\nRaw Fantasy Points Correlation (n={len(pivot_raw)} teams):")
    print(corr_raw.round(2).to_string())

    # Residuals
    corr_residual, pivot_residual = build_correlation_matrix(aggregated, 'residual')
    print(f"\nResidual Correlation (Actual - Projected) (n={len(pivot_residual)} teams):")
    print(corr_residual.round(2).to_string())

    # Create visualizations
    print("\nCreating visualizations...")
    output_dir = Path(__file__).parent / 'charts' / 'correlations'
    create_visualizations(
        corr_raw, corr_residual,
        pivot_raw, pivot_residual,
        output_dir
    )

    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\nRaw Points Correlations:")
    print("  - QB-WR: High positive suggests stacking value")
    print("  - QB-RB: Negative suggests they cannibalize each other")
    print("  - RB-WR: Typically negative (run vs pass game)")

    print("\nResidual Correlations:")
    print("  - If QB beats expectation, do WRs also beat expectation?")
    print("  - Positive = boom together, bust together")
    print("  - Negative = inverse relationship after controlling for projection")

    print(f"\nNote: Only 1 week of data (n={len(pivot_raw)} teams with all positions)")
    print("More weeks needed for statistically reliable correlations.")


if __name__ == '__main__':
    main()
