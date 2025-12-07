"""
One-off analysis charts for Fantasy DFS Optimizer.

Generates matplotlib charts to analyze relationships in the data.

Usage:
    python scripts/analysis_charts.py --week-dir data/2025_11_30
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))


def linear_regression(x, y):
    """Simple linear regression returning slope, intercept, r2."""
    n = len(x)
    sum_x, sum_y = x.sum(), y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x ** 2).sum()

    mean_x, mean_y = x.mean(), y.mean()

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0, mean_y, 0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = mean_y - slope * mean_x

    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - mean_y) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, intercept, r2


def chart_1_floor_ceiling_regression(espn_df, output_dir):
    """
    Chart 1: Floor and Ceiling regression models with data points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter valid data - require projection > 5
    valid = espn_df[
        (espn_df['espnScoreProjection'].notna()) &
        (espn_df['espnScoreProjection'] > 5) &
        (espn_df['espnLowScore'].notna()) &
        (espn_df['espnHighScore'].notna())
    ].copy()

    x = valid['espnScoreProjection'].values
    floor = valid['espnLowScore'].values
    ceiling = valid['espnHighScore'].values

    # Floor regression
    slope_f, int_f, r2_f = linear_regression(x, floor)
    x_line = np.linspace(x.min(), x.max(), 100)

    axes[0].scatter(x, floor, alpha=0.6, s=30, c='blue', label='Data points')
    axes[0].plot(x_line, int_f + slope_f * x_line, 'r-', linewidth=2,
                 label=f'y = {int_f:.2f} + {slope_f:.2f}x\nR² = {r2_f:.3f}')
    axes[0].set_xlabel('ESPN Score Projection', fontsize=12)
    axes[0].set_ylabel('ESPN Low Score (Floor)', fontsize=12)
    axes[0].set_title('Floor Regression Model', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ceiling regression
    slope_c, int_c, r2_c = linear_regression(x, ceiling)

    axes[1].scatter(x, ceiling, alpha=0.6, s=30, c='green', label='Data points')
    axes[1].plot(x_line, int_c + slope_c * x_line, 'r-', linewidth=2,
                 label=f'y = {int_c:.2f} + {slope_c:.2f}x\nR² = {r2_c:.3f}')
    axes[1].set_xlabel('ESPN Score Projection', fontsize=12)
    axes[1].set_ylabel('ESPN High Score (Ceiling)', fontsize=12)
    axes[1].set_title('Ceiling Regression Model', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_1_floor_ceiling_regression.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_2_espn_to_fp_regression(players_df, output_dir):
    """
    Chart 2: ESPN projections to FantasyPros scale regression.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Filter valid data - require projection > 5
    valid = players_df[
        (players_df['fpProjPts'].notna()) &
        (players_df['fpProjPts'] > 5)
    ].copy()

    projections = [
        ('espnOutsideProjection', 'ESPN Outside', 'blue'),
        ('espnScoreProjection', 'ESPN Score', 'green'),
        ('espnSimulationProjection', 'ESPN Simulation', 'orange'),
    ]

    for i, (col, name, color) in enumerate(projections):
        subset = valid[valid[col].notna() & (valid[col] > 0)]
        if len(subset) < 10:
            continue

        x = subset[col].values
        y = subset['fpProjPts'].values

        slope, intercept, r2 = linear_regression(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)

        axes[i].scatter(x, y, alpha=0.5, s=30, c=color, label='Data points')
        axes[i].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                     label=f'FP = {intercept:.2f} + {slope:.2f} × ESPN\nR² = {r2:.3f}')
        axes[i].plot([0, 30], [0, 30], 'k--', alpha=0.3, label='y = x')
        axes[i].set_xlabel(f'{name} Projection', fontsize=11)
        axes[i].set_ylabel('FantasyPros Projection', fontsize=11)
        axes[i].set_title(f'{name} → FP Conversion', fontsize=12)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_2_espn_to_fp_regression.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_3_td_odds_vs_skew(players_df, output_dir):
    """
    Chart 3: TD odds vs skew ratio.
    Skew = (ceiling - consensus) / (consensus - floor)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate skew ratio - filter to projection > 5
    valid = players_df[
        (players_df['tdProbability'].notna()) &
        (players_df['tdProbability'] > 0) &
        (players_df['ceiling_competitive'].notna()) &
        (players_df['floor_competitive'].notna()) &
        (players_df['consensus'].notna()) &
        (players_df['consensus'] > 5)  # Filter low-projection players
    ].copy()

    valid['skew'] = (valid['ceiling_competitive'] - valid['consensus']) / \
                    (valid['consensus'] - valid['floor_competitive']).replace(0, 0.01)

    # Filter out extreme skew values
    valid = valid[(valid['skew'] > 0) & (valid['skew'] < 10)]

    x = valid['tdProbability'].values
    y = valid['skew'].values

    # Color by position
    positions = valid['position'].values
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange', 'D': 'purple'}
    c = [colors.get(p, 'gray') for p in positions]

    scatter = ax.scatter(x, y, alpha=0.6, s=40, c=c)

    # Add regression line
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'k-', linewidth=2,
            label=f'Skew = {intercept:.2f} + {slope:.4f} × TD%\nR² = {r2:.3f}')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Symmetric (skew=1)')

    ax.set_xlabel('TD Probability (%)', fontsize=12)
    ax.set_ylabel('Skew Ratio (ceiling-cons)/(cons-floor)', fontsize=12)
    ax.set_title('TD Odds vs Distribution Skew', fontsize=14)

    # Legend for positions
    for pos, color in colors.items():
        ax.scatter([], [], c=color, label=pos, s=40)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_3_td_odds_vs_skew.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_4_consensus_vs_actual(players_df, actuals_df, output_dir):
    """
    Chart 4: Consensus projection vs actual scores.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to projection > 5 first
    players_filtered = players_df[players_df['consensus'] > 5].copy()

    # Merge with actuals
    merged = players_filtered.merge(actuals_df[['name', 'actual']],
                               left_on='name', right_on='name', how='inner')

    if len(merged) < 10:
        print("  ⚠️ Not enough actual data for chart 4")
        return

    x = merged['consensus'].values
    y = merged['actual'].values
    positions = merged['position'].values

    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange', 'D': 'purple'}
    c = [colors.get(p, 'gray') for p in positions]

    ax.scatter(x, y, alpha=0.6, s=40, c=c)

    # Add regression line
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(0, x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'k-', linewidth=2,
            label=f'Actual = {intercept:.2f} + {slope:.2f} × Consensus\nR² = {r2:.3f}')

    # Perfect prediction line
    ax.plot([0, 30], [0, 30], 'g--', alpha=0.5, label='Perfect prediction')

    ax.set_xlabel('Consensus Projection', fontsize=12)
    ax.set_ylabel('Actual FanDuel Points', fontsize=12)
    ax.set_title('Consensus vs Actual Scores', fontsize=14)

    for pos, color in colors.items():
        ax.scatter([], [], c=color, label=pos, s=40)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_4_consensus_vs_actual.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_5_skew_vs_actual(players_df, actuals_df, output_dir):
    """
    Chart 5: Skew ratio vs actual scores (relative to projection).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to projection > 5 first
    players_filtered = players_df[players_df['consensus'] > 5].copy()

    # Merge with actuals
    merged = players_filtered.merge(actuals_df[['name', 'actual']],
                               left_on='name', right_on='name', how='inner')

    if len(merged) < 10:
        print("  ⚠️ Not enough actual data for chart 5")
        return

    # Calculate skew
    merged['skew'] = (merged['ceiling_competitive'] - merged['consensus']) / \
                     (merged['consensus'] - merged['floor_competitive']).replace(0, 0.01)
    merged = merged[(merged['skew'] > 0) & (merged['skew'] < 10)]

    # Calculate outperformance
    merged['outperformance'] = merged['actual'] - merged['consensus']
    merged['outperformance_pct'] = merged['outperformance'] / merged['consensus'].replace(0, 1) * 100

    positions = merged['position'].values
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange', 'D': 'purple'}
    c = [colors.get(p, 'gray') for p in positions]

    # Left: Skew vs raw outperformance
    x = merged['skew'].values
    y = merged['outperformance'].values

    axes[0].scatter(x, y, alpha=0.6, s=40, c=c)
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'k-', linewidth=2,
                 label=f'Outperf = {intercept:.2f} + {slope:.2f} × Skew\nR² = {r2:.3f}')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Skew Ratio', fontsize=12)
    axes[0].set_ylabel('Outperformance (Actual - Consensus)', fontsize=12)
    axes[0].set_title('Skew vs Raw Outperformance', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Skew vs actual
    y2 = merged['actual'].values
    axes[1].scatter(x, y2, alpha=0.6, s=40, c=c)
    slope2, intercept2, r2_2 = linear_regression(x, y2)
    axes[1].plot(x_line, intercept2 + slope2 * x_line, 'k-', linewidth=2,
                 label=f'Actual = {intercept2:.2f} + {slope2:.2f} × Skew\nR² = {r2_2:.3f}')
    axes[1].set_xlabel('Skew Ratio', fontsize=12)
    axes[1].set_ylabel('Actual FanDuel Points', fontsize=12)
    axes[1].set_title('Skew vs Actual Score', fontsize=14)

    for pos, color in colors.items():
        axes[1].scatter([], [], c=color, label=pos, s=40)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_5_skew_vs_actual.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_6_espn_range_accuracy(espn_df, actuals_df, output_dir):
    """
    Chart 6: How often do actuals fall within ESPN's predicted range?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to projection > 5 first
    espn_filtered = espn_df[espn_df['espnScoreProjection'] > 5].copy()

    # Merge ESPN with actuals
    espn_filtered['name_lower'] = espn_filtered['name'].str.lower().str.strip()
    actuals_df['name_lower'] = actuals_df['name'].str.lower().str.strip()

    merged = espn_filtered.merge(actuals_df[['name_lower', 'actual']], on='name_lower', how='inner')

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 6")
        return

    merged['in_range'] = (merged['actual'] >= merged['espnLowScore']) & \
                         (merged['actual'] <= merged['espnHighScore'])
    merged['below_floor'] = merged['actual'] < merged['espnLowScore']
    merged['above_ceiling'] = merged['actual'] > merged['espnHighScore']

    # Left: Pie chart of range accuracy
    in_range = merged['in_range'].sum()
    below = merged['below_floor'].sum()
    above = merged['above_ceiling'].sum()

    labels = [f'In Range\n({in_range})', f'Below Floor\n({below})', f'Above Ceiling\n({above})']
    sizes = [in_range, below, above]
    colors_pie = ['#2ecc71', '#e74c3c', '#3498db']
    explode = (0.05, 0.05, 0.05)

    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'ESPN Range Accuracy (n={len(merged)})', fontsize=14)

    # Right: Scatter of predicted range vs actual
    axes[1].scatter(merged['espnScoreProjection'], merged['actual'],
                    alpha=0.6, s=40, c='blue', label='Actual')

    # Plot floor and ceiling lines
    x_sorted = merged.sort_values('espnScoreProjection')['espnScoreProjection']
    axes[1].fill_between(x_sorted,
                         merged.sort_values('espnScoreProjection')['espnLowScore'],
                         merged.sort_values('espnScoreProjection')['espnHighScore'],
                         alpha=0.2, color='green', label='ESPN Range')

    axes[1].plot([0, 30], [0, 30], 'k--', alpha=0.3, label='y = x')
    axes[1].set_xlabel('ESPN Projection', fontsize=12)
    axes[1].set_ylabel('Actual Score', fontsize=12)
    axes[1].set_title('ESPN Range vs Actual Scores', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_6_espn_range_accuracy.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_7_position_variance(players_df, actuals_df, output_dir):
    """
    Chart 7: Variance by position - who booms/busts most?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to projection > 5 first
    players_filtered = players_df[players_df['consensus'] > 5].copy()

    merged = players_filtered.merge(actuals_df[['name', 'actual']], on='name', how='inner')

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 7")
        return

    merged['error'] = merged['actual'] - merged['consensus']
    merged['abs_error'] = merged['error'].abs()
    merged['error_pct'] = merged['error'] / merged['consensus'].replace(0, 1) * 100

    positions = ['QB', 'RB', 'WR', 'TE', 'D']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Left: Box plot of errors by position
    data = [merged[merged['position'] == pos]['error'].values for pos in positions]
    bp = axes[0].boxplot(data, labels=positions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Position', fontsize=12)
    axes[0].set_ylabel('Prediction Error (Actual - Consensus)', fontsize=12)
    axes[0].set_title('Prediction Error by Position', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Mean and std of errors
    stats = merged.groupby('position').agg({
        'error': ['mean', 'std', 'count'],
        'abs_error': 'mean'
    }).round(2)

    x_pos = np.arange(len(positions))
    means = [merged[merged['position'] == pos]['error'].mean() for pos in positions]
    stds = [merged[merged['position'] == pos]['error'].std() for pos in positions]

    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(positions)
    axes[1].set_xlabel('Position', fontsize=12)
    axes[1].set_ylabel('Mean Error ± Std Dev', fontsize=12)
    axes[1].set_title('Mean Prediction Error by Position', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_7_position_variance.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_8_ceiling_floor_spread(espn_df, output_dir):
    """
    Chart 8: Relationship between projection level and floor-ceiling spread.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to projection > 5
    valid = espn_df[
        (espn_df['espnScoreProjection'].notna()) &
        (espn_df['espnScoreProjection'] > 5) &
        (espn_df['espnLowScore'].notna()) &
        (espn_df['espnHighScore'].notna())
    ].copy()

    valid['range'] = valid['espnHighScore'] - valid['espnLowScore']
    valid['range_pct'] = valid['range'] / valid['espnScoreProjection'] * 100

    # Left: Range vs projection
    x = valid['espnScoreProjection'].values
    y = valid['range'].values

    axes[0].scatter(x, y, alpha=0.6, s=40, c='blue')
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                 label=f'Range = {intercept:.2f} + {slope:.2f} × Proj\nR² = {r2:.3f}')
    axes[0].set_xlabel('ESPN Projection', fontsize=12)
    axes[0].set_ylabel('Floor-Ceiling Range (pts)', fontsize=12)
    axes[0].set_title('Range Width vs Projection', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Range % vs projection
    y2 = valid['range_pct'].values
    axes[1].scatter(x, y2, alpha=0.6, s=40, c='green')
    slope2, intercept2, r2_2 = linear_regression(x, y2)
    axes[1].plot(x_line, intercept2 + slope2 * x_line, 'r-', linewidth=2,
                 label=f'Range% = {intercept2:.1f} + {slope2:.2f} × Proj\nR² = {r2_2:.3f}')
    axes[1].set_xlabel('ESPN Projection', fontsize=12)
    axes[1].set_ylabel('Range as % of Projection', fontsize=12)
    axes[1].set_title('Relative Uncertainty vs Projection', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_8_ceiling_floor_spread.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


# ============================================================================
# PREDICTIVE SIGNAL CHARTS (9-15)
# ============================================================================

def chart_9_projection_source_comparison(players_df, actuals_df, output_dir):
    """
    Chart 9: Compare prediction accuracy of different projection sources.
    Which source has the best signal? FP, ESPN Score, ESPN Sim, ESPN Outside?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter and merge
    players_filtered = players_df[players_df['consensus'] > 5].copy()
    # actuals_df may have 'actual' or 'actual_points' column
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 9")
        plt.close()
        return

    sources = [
        ('fpProjPts', 'FantasyPros', 'blue', axes[0, 0]),
        ('espnScoreProjection_fp', 'ESPN Score (FP-scaled)', 'green', axes[0, 1]),
        ('espnSimulationProjection_fp', 'ESPN Simulation (FP-scaled)', 'orange', axes[1, 0]),
        ('consensus', 'Consensus (weighted)', 'purple', axes[1, 1]),
    ]

    results = []
    for col, name, color, ax in sources:
        valid = merged[merged[col].notna() & (merged[col] > 0)]
        if len(valid) < 10:
            continue

        x = valid[col].values
        y = valid['actual'].values

        # Calculate metrics
        slope, intercept, r2 = linear_regression(x, y)
        mae = np.abs(y - x).mean()
        rmse = np.sqrt(((y - x) ** 2).mean())

        results.append({'source': name, 'r2': r2, 'mae': mae, 'rmse': rmse, 'slope': slope})

        ax.scatter(x, y, alpha=0.5, s=30, c=color)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                label=f'R² = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}')
        ax.plot([0, 35], [0, 35], 'k--', alpha=0.3)
        ax.set_xlabel(f'{name}', fontsize=11)
        ax.set_ylabel('Actual Points', fontsize=11)
        ax.set_title(f'{name} vs Actual', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 40)

    plt.tight_layout()
    output_path = output_dir / 'chart_9_projection_source_comparison.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Print summary
    if results:
        print(f"  ✓ Saved {output_path.name}")
        print("    Source Rankings (by R²):")
        for r in sorted(results, key=lambda x: -x['r2']):
            print(f"      {r['source']}: R²={r['r2']:.3f}, MAE={r['mae']:.1f}, slope={r['slope']:.2f}")


def chart_9b_raw_espn_comparison(players_df, actuals_df, output_dir):
    """
    Chart 9b: Compare prediction accuracy using RAW ESPN projections.
    ESPN uses a different scoring system, so this gives them a fair comparison
    without the FP-scale conversion that may introduce artifacts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter and merge
    players_filtered = players_df[players_df['consensus'] > 5].copy()
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 9b")
        plt.close()
        return

    # Compare raw ESPN projections directly to actuals
    sources = [
        ('fpProjPts', 'FantasyPros', 'blue', axes[0, 0]),
        ('espnScoreProjection', 'ESPN Score (RAW)', 'green', axes[0, 1]),
        ('espnSimulationProjection', 'ESPN Simulation (RAW)', 'orange', axes[1, 0]),
        ('espnOutsideProjection', 'ESPN Outside (RAW)', 'red', axes[1, 1]),
    ]

    results = []
    for col, name, color, ax in sources:
        if col not in merged.columns:
            ax.text(0.5, 0.5, f'{name}\nColumn not found', ha='center', va='center', transform=ax.transAxes)
            continue

        valid = merged[merged[col].notna() & (merged[col] > 0)]
        if len(valid) < 10:
            ax.text(0.5, 0.5, f'{name}\nInsufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        x = valid[col].values
        y = valid['actual'].values

        # Calculate metrics
        slope, intercept, r2 = linear_regression(x, y)
        mae = np.abs(y - x).mean()
        rmse = np.sqrt(((y - x) ** 2).mean())

        results.append({'source': name, 'r2': r2, 'mae': mae, 'rmse': rmse, 'slope': slope, 'intercept': intercept, 'n': len(valid)})

        ax.scatter(x, y, alpha=0.5, s=30, c=color)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                label=f'R² = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}\nslope = {slope:.2f}')
        ax.plot([0, 40], [0, 40], 'k--', alpha=0.3, label='Perfect (y=x)')
        ax.set_xlabel(f'{name}', fontsize=11)
        ax.set_ylabel('Actual Points (PPR)', fontsize=11)
        ax.set_title(f'{name} vs Actual (n={len(valid)})', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 45)

    plt.suptitle('Raw Projection Source Comparison (No Scale Conversion)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'chart_9b_raw_espn_comparison.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Print summary
    if results:
        print(f"  ✓ Saved {output_path.name}")
        print("    Raw Source Rankings (by R²):")
        for r in sorted(results, key=lambda x: -x['r2']):
            print(f"      {r['source']}: R²={r['r2']:.3f}, MAE={r['mae']:.1f}, slope={r['slope']:.2f}, n={r['n']}")


def chart_10_td_odds_signal(players_df, actuals_df, output_dir):
    """
    Chart 10: Do TD odds predict actual scoring?
    Look at correlation between TD probability and outperformance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Filter and merge
    players_filtered = players_df[
        (players_df['consensus'] > 5) &
        (players_df['tdProbability'].notna()) &
        (players_df['tdProbability'] > 0)
    ].copy()

    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})
    merged['outperformance'] = merged['actual'] - merged['consensus']
    merged['outperf_pct'] = merged['outperformance'] / merged['consensus'] * 100

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 10")
        plt.close()
        return

    # Left: TD probability vs actual points
    x = merged['tdProbability'].values
    y = merged['actual'].values

    axes[0].scatter(x, y, alpha=0.6, s=40, c='blue')
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                 label=f'Actual = {intercept:.1f} + {slope:.2f} × TD%\nR² = {r2:.3f}')
    axes[0].set_xlabel('TD Probability (%)', fontsize=12)
    axes[0].set_ylabel('Actual Points', fontsize=12)
    axes[0].set_title('TD Odds → Actual Points', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle: TD probability vs outperformance
    y2 = merged['outperformance'].values
    axes[1].scatter(x, y2, alpha=0.6, s=40, c='green')
    slope2, intercept2, r2_2 = linear_regression(x, y2)
    axes[1].plot(x_line, intercept2 + slope2 * x_line, 'r-', linewidth=2,
                 label=f'Outperf = {intercept2:.1f} + {slope2:.2f} × TD%\nR² = {r2_2:.3f}')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('TD Probability (%)', fontsize=12)
    axes[1].set_ylabel('Outperformance (Actual - Consensus)', fontsize=12)
    axes[1].set_title('TD Odds → Outperformance', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Right: Binned analysis - average outperformance by TD odds bucket
    merged['td_bucket'] = pd.cut(merged['tdProbability'],
                                  bins=[0, 10, 20, 30, 40, 100],
                                  labels=['0-10%', '10-20%', '20-30%', '30-40%', '40%+'])
    bucket_stats = merged.groupby('td_bucket', observed=True).agg({
        'outperformance': ['mean', 'std', 'count']
    }).round(2)

    buckets = bucket_stats.index.tolist()
    means = bucket_stats[('outperformance', 'mean')].values
    stds = bucket_stats[('outperformance', 'std')].values
    counts = bucket_stats[('outperformance', 'count')].values

    x_pos = np.arange(len(buckets))
    bars = axes[2].bar(x_pos, means, yerr=stds, capsize=5, color='teal', alpha=0.7)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f'{b}\n(n={int(c)})' for b, c in zip(buckets, counts)])
    axes[2].set_xlabel('TD Probability Bucket', fontsize=12)
    axes[2].set_ylabel('Mean Outperformance', fontsize=12)
    axes[2].set_title('Avg Outperformance by TD Odds', fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_10_td_odds_signal.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    TD odds → actual R² = {r2:.3f}")
    print(f"    TD odds → outperformance R² = {r2_2:.3f}")


def chart_11_floor_ceiling_hit_rates(players_df, actuals_df, output_dir):
    """
    Chart 11: How often do players hit floor/ceiling by position?
    Calibration check: are our floor/ceiling estimates accurate?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter and merge
    players_filtered = players_df[players_df['consensus'] > 5].copy()
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 11")
        plt.close()
        return

    # Calculate where actual fell relative to floor/ceiling
    merged['hit_floor'] = merged['actual'] <= merged['floor_competitive']
    merged['hit_ceiling'] = merged['actual'] >= merged['ceiling_competitive']
    merged['in_range'] = (merged['actual'] > merged['floor_competitive']) & \
                         (merged['actual'] < merged['ceiling_competitive'])

    # Left: Hit rates by position
    positions = ['QB', 'RB', 'WR', 'TE', 'D']
    floor_rates = []
    ceiling_rates = []
    in_range_rates = []

    for pos in positions:
        pos_data = merged[merged['position'] == pos]
        if len(pos_data) > 0:
            floor_rates.append(pos_data['hit_floor'].mean() * 100)
            ceiling_rates.append(pos_data['hit_ceiling'].mean() * 100)
            in_range_rates.append(pos_data['in_range'].mean() * 100)
        else:
            floor_rates.append(0)
            ceiling_rates.append(0)
            in_range_rates.append(0)

    x_pos = np.arange(len(positions))
    width = 0.25

    axes[0].bar(x_pos - width, floor_rates, width, label='Below Floor', color='red', alpha=0.7)
    axes[0].bar(x_pos, in_range_rates, width, label='In Range', color='green', alpha=0.7)
    axes[0].bar(x_pos + width, ceiling_rates, width, label='Above Ceiling', color='blue', alpha=0.7)

    # Add expected lines (for P10/P90, expect 10%/80%/10%)
    axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Expected 10%')
    axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Expected 80%')

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(positions)
    axes[0].set_xlabel('Position', fontsize=12)
    axes[0].set_ylabel('Percentage of Players', fontsize=12)
    axes[0].set_title('Floor/Ceiling Hit Rates by Position', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Overall calibration
    overall = {
        'Below Floor': merged['hit_floor'].mean() * 100,
        'In Range': merged['in_range'].mean() * 100,
        'Above Ceiling': merged['hit_ceiling'].mean() * 100,
    }
    expected = {'Below Floor': 10, 'In Range': 80, 'Above Ceiling': 10}

    categories = list(overall.keys())
    actual_vals = list(overall.values())
    expected_vals = [expected[c] for c in categories]

    x_pos = np.arange(len(categories))
    width = 0.35

    axes[1].bar(x_pos - width/2, actual_vals, width, label='Actual', color='teal', alpha=0.7)
    axes[1].bar(x_pos + width/2, expected_vals, width, label='Expected (P10/P90)', color='gray', alpha=0.7)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(categories)
    axes[1].set_xlabel('Category', fontsize=12)
    axes[1].set_ylabel('Percentage', fontsize=12)
    axes[1].set_title(f'Overall Calibration (n={len(merged)})', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (a, e) in enumerate(zip(actual_vals, expected_vals)):
        axes[1].text(i - width/2, a + 1, f'{a:.1f}%', ha='center', fontsize=10)
        axes[1].text(i + width/2, e + 1, f'{e:.0f}%', ha='center', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'chart_11_floor_ceiling_hit_rates.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Calibration: {overall['Below Floor']:.1f}% below floor, {overall['In Range']:.1f}% in range, {overall['Above Ceiling']:.1f}% above ceiling")


def chart_12_value_prediction(players_df, actuals_df, output_dir):
    """
    Chart 12: Does projected value (pts/$) predict actual value?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter and merge
    players_filtered = players_df[
        (players_df['consensus'] > 5) &
        (players_df['salary'] > 0)
    ].copy()

    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 12")
        plt.close()
        return

    # Calculate value metrics
    merged['proj_value'] = merged['consensus'] / (merged['salary'] / 1000)
    merged['actual_value'] = merged['actual'] / (merged['salary'] / 1000)

    # Left: Projected value vs actual value
    x = merged['proj_value'].values
    y = merged['actual_value'].values

    positions = merged['position'].values
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange', 'D': 'purple'}
    c = [colors.get(p, 'gray') for p in positions]

    axes[0].scatter(x, y, alpha=0.6, s=40, c=c)
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'k-', linewidth=2,
                 label=f'R² = {r2:.3f}\nslope = {slope:.2f}')
    axes[0].plot([0, 6], [0, 6], 'k--', alpha=0.3)
    axes[0].set_xlabel('Projected Value (pts per $1k)', fontsize=12)
    axes[0].set_ylabel('Actual Value (pts per $1k)', fontsize=12)
    axes[0].set_title('Value Prediction Accuracy', fontsize=14)
    for pos, color in colors.items():
        axes[0].scatter([], [], c=color, label=pos, s=40)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: Value by salary tier
    merged['salary_tier'] = pd.cut(merged['salary'],
                                    bins=[0, 5000, 6000, 7000, 8000, 15000],
                                    labels=['<$5k', '$5-6k', '$6-7k', '$7-8k', '$8k+'])

    tier_stats = merged.groupby('salary_tier', observed=True).agg({
        'proj_value': 'mean',
        'actual_value': 'mean',
        'salary': 'count'
    }).round(2)

    tiers = tier_stats.index.tolist()
    proj_vals = tier_stats['proj_value'].values
    actual_vals = tier_stats['actual_value'].values
    counts = tier_stats['salary'].values

    x_pos = np.arange(len(tiers))
    width = 0.35

    axes[1].bar(x_pos - width/2, proj_vals, width, label='Projected', color='blue', alpha=0.7)
    axes[1].bar(x_pos + width/2, actual_vals, width, label='Actual', color='green', alpha=0.7)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{t}\n(n={int(c)})' for t, c in zip(tiers, counts)])
    axes[1].set_xlabel('Salary Tier', fontsize=12)
    axes[1].set_ylabel('Pts per $1k', fontsize=12)
    axes[1].set_title('Value by Salary Tier', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_12_value_prediction.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Value prediction R² = {r2:.3f}")


def chart_13_sigma_vs_variance(players_df, actuals_df, output_dir):
    """
    Chart 13: Does our modeled sigma (variance) predict actual variance?
    Players with higher sigma should have more extreme outcomes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter and merge
    players_filtered = players_df[
        (players_df['consensus'] > 5) &
        (players_df['sigma_competitive'].notna())
    ].copy()

    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 13")
        plt.close()
        return

    # Calculate absolute error (proxy for realized variance)
    merged['abs_error'] = np.abs(merged['actual'] - merged['consensus'])
    merged['error_pct'] = merged['abs_error'] / merged['consensus'] * 100

    # Left: Sigma vs absolute error
    x = merged['sigma_competitive'].values
    y = merged['abs_error'].values

    axes[0].scatter(x, y, alpha=0.6, s=40, c='blue')
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                 label=f'|Error| = {intercept:.1f} + {slope:.1f} × σ\nR² = {r2:.3f}')
    axes[0].set_xlabel('Modeled Sigma (σ)', fontsize=12)
    axes[0].set_ylabel('Absolute Error |Actual - Consensus|', fontsize=12)
    axes[0].set_title('Does Sigma Predict Variance?', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Binned analysis
    merged['sigma_bucket'] = pd.cut(merged['sigma_competitive'],
                                     bins=[0, 0.25, 0.35, 0.45, 1.0],
                                     labels=['Low (<0.25)', 'Med (0.25-0.35)',
                                            'High (0.35-0.45)', 'Very High (>0.45)'])

    bucket_stats = merged.groupby('sigma_bucket', observed=True).agg({
        'abs_error': ['mean', 'std'],
        'error_pct': 'mean',
        'sigma_competitive': 'count'
    }).round(2)

    buckets = bucket_stats.index.tolist()
    means = bucket_stats[('abs_error', 'mean')].values
    stds = bucket_stats[('abs_error', 'std')].values
    counts = bucket_stats[('sigma_competitive', 'count')].values

    x_pos = np.arange(len(buckets))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color='teal', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{b}\n(n={int(c)})' for b, c in zip(buckets, counts)], fontsize=9)
    axes[1].set_xlabel('Sigma Bucket', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('Prediction Error by Sigma Level', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_13_sigma_vs_variance.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Sigma → |error| R² = {r2:.3f}")


def chart_14_correlation_matrix(players_df, actuals_df, output_dir):
    """
    Chart 14: Correlation matrix of all predictive features vs actual.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Filter and merge
    players_filtered = players_df[players_df['consensus'] > 5].copy()
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 14")
        plt.close()
        return

    # Select features for correlation
    features = [
        'actual',
        'consensus',
        'fpProjPts',
        'espnScoreProjection_fp',
        'espnSimulationProjection_fp',
        'tdProbability',
        'sigma_competitive',
        'ceiling_competitive',
        'floor_competitive',
        'salary',
    ]

    # Filter to available features
    available = [f for f in features if f in merged.columns and merged[f].notna().sum() > 10]
    corr_data = merged[available].dropna()

    if len(corr_data) < 10:
        print("  ⚠️ Not enough complete data for chart 14")
        plt.close()
        return

    # Compute correlation matrix
    corr_matrix = corr_data.corr()

    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)

    # Add labels
    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels(available, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(available, fontsize=10)

    # Add correlation values
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.colorbar(im, label='Correlation')

    plt.tight_layout()
    output_path = output_dir / 'chart_14_correlation_matrix.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")

    # Print correlations with actual
    actual_corrs = corr_matrix['actual'].drop('actual').sort_values(ascending=False)
    print("    Correlations with Actual:")
    for feat, corr in actual_corrs.items():
        print(f"      {feat}: {corr:.3f}")


def chart_15_residual_analysis(players_df, actuals_df, output_dir):
    """
    Chart 15: Residual analysis - where do predictions fail?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter and merge
    players_filtered = players_df[players_df['consensus'] > 5].copy()
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    merged = players_filtered.merge(actuals_df[['name', actual_col]],
                                     on='name', how='inner')
    if actual_col == 'actual_points':
        merged = merged.rename(columns={'actual_points': 'actual'})

    if len(merged) < 10:
        print("  ⚠️ Not enough data for chart 15")
        plt.close()
        return

    merged['residual'] = merged['actual'] - merged['consensus']

    # Top left: Residuals vs predicted
    axes[0, 0].scatter(merged['consensus'], merged['residual'], alpha=0.6, s=40, c='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Consensus Projection', fontsize=12)
    axes[0, 0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Residual distribution
    axes[0, 1].hist(merged['residual'], bins=30, color='teal', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=merged['residual'].mean(), color='orange', linestyle='-', linewidth=2,
                       label=f'Mean = {merged["residual"].mean():.1f}')
    axes[0, 1].set_xlabel('Residual', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Residual Distribution', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Biggest misses (under-predicted)
    top_under = merged.nlargest(10, 'residual')[['name', 'position', 'consensus', 'actual', 'residual']]
    y_pos = np.arange(len(top_under))
    axes[1, 0].barh(y_pos, top_under['residual'], color='green', alpha=0.7)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([f"{row['name']} ({row['position']})" for _, row in top_under.iterrows()], fontsize=9)
    axes[1, 0].set_xlabel('Residual (Under-predicted)', fontsize=12)
    axes[1, 0].set_title('Top 10 Under-Predicted (Booms)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Bottom right: Biggest misses (over-predicted)
    top_over = merged.nsmallest(10, 'residual')[['name', 'position', 'consensus', 'actual', 'residual']]
    y_pos = np.arange(len(top_over))
    axes[1, 1].barh(y_pos, top_over['residual'], color='red', alpha=0.7)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f"{row['name']} ({row['position']})" for _, row in top_over.iterrows()], fontsize=9)
    axes[1, 1].set_xlabel('Residual (Over-predicted)', fontsize=12)
    axes[1, 1].set_title('Top 10 Over-Predicted (Busts)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / 'chart_15_residual_analysis.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")

    # Print summary stats
    print(f"    Mean residual: {merged['residual'].mean():.2f}")
    print(f"    Std residual: {merged['residual'].std():.2f}")
    print(f"    Top boom: {top_under.iloc[0]['name']} (+{top_under.iloc[0]['residual']:.1f})")
    print(f"    Top bust: {top_over.iloc[0]['name']} ({top_over.iloc[0]['residual']:.1f})")


# ============================================================================
# GAME SCRIPT & BETTING SIGNAL CHARTS (16-21)
# ============================================================================

def chart_16_game_script_accuracy(game_scripts_df, actual_games_df, output_dir):
    """
    Chart 16: Did our predicted game scripts match actual outcomes?
    Compare predicted primary_script to actual_script.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Need to merge predicted and actual game scripts
    # First, create a game_id mapping from actual_games
    actual_games_df = actual_games_df.copy()
    actual_games_df['game_key'] = actual_games_df['away_team'] + '@' + actual_games_df['home_team']

    game_scripts_df = game_scripts_df.copy()
    game_scripts_df['game_key'] = game_scripts_df['game_id']

    merged = game_scripts_df.merge(
        actual_games_df[['game_key', 'actual_script', 'total_points', 'point_differential']],
        on='game_key', how='inner'
    )

    if len(merged) < 3:
        print("  ⚠️ Not enough game data for chart 16")
        plt.close()
        return

    # Left: Confusion matrix of predicted vs actual scripts
    scripts = ['shootout', 'defensive', 'blowout', 'competitive']
    # Normalize actual script names (blowout_favorite → blowout)
    merged['actual_script_norm'] = merged['actual_script'].apply(
        lambda x: 'blowout' if 'blowout' in str(x) else x
    )

    confusion = pd.crosstab(
        merged['primary_script'],
        merged['actual_script_norm'],
        dropna=False
    )

    # Reindex to ensure all scripts are present
    for script in scripts:
        if script not in confusion.index:
            confusion.loc[script] = 0
        if script not in confusion.columns:
            confusion[script] = 0
    confusion = confusion.reindex(scripts, axis=0).reindex(scripts, axis=1).fillna(0)

    im = axes[0].imshow(confusion.values, cmap='Blues')
    axes[0].set_xticks(range(len(scripts)))
    axes[0].set_yticks(range(len(scripts)))
    axes[0].set_xticklabels(scripts, rotation=45, ha='right')
    axes[0].set_yticklabels(scripts)
    axes[0].set_xlabel('Actual Script', fontsize=12)
    axes[0].set_ylabel('Predicted Script', fontsize=12)
    axes[0].set_title('Game Script Prediction Accuracy', fontsize=14)

    # Add counts in cells
    for i in range(len(scripts)):
        for j in range(len(scripts)):
            val = int(confusion.iloc[i, j])
            color = 'white' if val > confusion.values.max() / 2 else 'black'
            axes[0].text(j, i, str(val), ha='center', va='center', color=color, fontsize=12)

    # Calculate accuracy
    correct = sum(merged['primary_script'] == merged['actual_script_norm'])
    total = len(merged)
    accuracy = correct / total * 100

    axes[0].text(0.5, -0.15, f'Accuracy: {correct}/{total} = {accuracy:.1f}%',
                 ha='center', transform=axes[0].transAxes, fontsize=11)

    # Right: Script probability vs actual outcome
    # For each game, what was our probability for the script that actually happened?
    merged['prob_for_actual'] = merged.apply(
        lambda row: row[f"{row['actual_script_norm']}_prob"] if f"{row['actual_script_norm']}_prob" in row else 0.25,
        axis=1
    )

    x = np.arange(len(merged))
    colors = ['green' if row['primary_script'] == row['actual_script_norm'] else 'red'
              for _, row in merged.iterrows()]

    bars = axes[1].bar(x, merged['prob_for_actual'] * 100, color=colors, alpha=0.7)
    axes[1].axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Random (25%)')
    axes[1].axhline(y=merged['prob_for_actual'].mean() * 100, color='blue', linestyle='-',
                    alpha=0.7, label=f"Mean: {merged['prob_for_actual'].mean()*100:.1f}%")

    axes[1].set_xlabel('Game', fontsize=12)
    axes[1].set_ylabel('Probability Assigned to Actual Outcome (%)', fontsize=12)
    axes[1].set_title('Script Probability Calibration', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(merged['game_key'], rotation=45, ha='right', fontsize=8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_16_game_script_accuracy.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Script accuracy: {accuracy:.1f}%")
    print(f"    Avg probability for actual outcome: {merged['prob_for_actual'].mean()*100:.1f}%")


def chart_17_vegas_total_accuracy(game_lines_df, actual_games_df, output_dir):
    """
    Chart 17: How accurate were Vegas totals (O/U)?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Merge game lines with actuals
    # Need to match by teams
    game_lines_df = game_lines_df.copy()
    actual_games_df = actual_games_df.copy()

    # Create lookup from actuals
    actuals_by_team = {}
    for _, row in actual_games_df.iterrows():
        actuals_by_team[row['away_team']] = row['total_points']
        actuals_by_team[row['home_team']] = row['total_points']

    # Add actual total to game lines
    game_lines_df['actual_total'] = game_lines_df['team_abbr'].map(actuals_by_team)
    valid = game_lines_df[game_lines_df['actual_total'].notna()].copy()

    # Dedupe (each game appears twice - once per team)
    valid = valid.drop_duplicates(subset=['total', 'actual_total'])

    if len(valid) < 3:
        print("  ⚠️ Not enough data for chart 17")
        plt.close()
        return

    # Left: Predicted total vs actual total
    x = valid['total'].values
    y = valid['actual_total'].values

    axes[0].scatter(x, y, s=100, alpha=0.7, c='blue')
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                 label=f'R² = {r2:.3f}\nslope = {slope:.2f}')
    axes[0].plot([30, 60], [30, 60], 'k--', alpha=0.3, label='Perfect')
    axes[0].set_xlabel('Vegas Total (O/U)', fontsize=12)
    axes[0].set_ylabel('Actual Total Points', fontsize=12)
    axes[0].set_title('Vegas Total Accuracy', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add game labels
    for i, row in valid.iterrows():
        axes[0].annotate(f"{row['team_abbr']}", (row['total'], row['actual_total']),
                         fontsize=8, alpha=0.7)

    # Right: Error distribution
    valid['total_error'] = valid['actual_total'] - valid['total']
    mae = valid['total_error'].abs().mean()
    rmse = np.sqrt((valid['total_error'] ** 2).mean())

    axes[1].hist(valid['total_error'], bins=10, color='teal', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect')
    axes[1].axvline(x=valid['total_error'].mean(), color='blue', linestyle='-', linewidth=2,
                    label=f"Mean: {valid['total_error'].mean():.1f}")

    axes[1].set_xlabel('Total Error (Actual - Vegas)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'Total Prediction Error\nMAE={mae:.1f}, RMSE={rmse:.1f}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'chart_17_vegas_total_accuracy.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Vegas total R² = {r2:.3f}, MAE = {mae:.1f}")


def chart_18_vegas_spread_accuracy(game_lines_df, actual_games_df, output_dir):
    """
    Chart 18: How accurate were Vegas spreads?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    game_lines_df = game_lines_df.copy()
    actual_games_df = actual_games_df.copy()

    # Build actuals lookup - we need point differential per team
    actual_diffs = {}
    for _, row in actual_games_df.iterrows():
        # Away team diff = away_score - home_score (positive if away won)
        actual_diffs[row['away_team']] = row['away_score'] - row['home_score']
        actual_diffs[row['home_team']] = row['home_score'] - row['away_score']

    game_lines_df['actual_margin'] = game_lines_df['team_abbr'].map(actual_diffs)
    valid = game_lines_df[game_lines_df['actual_margin'].notna()].copy()

    # Keep only favorites (negative spread)
    valid = valid[valid['spread'] < 0].copy()

    if len(valid) < 3:
        print("  ⚠️ Not enough data for chart 18")
        plt.close()
        return

    # Spread is negative for favorites (e.g., -7 means they should win by 7)
    # Actual margin is positive if they won
    # So if spread=-7 and actual_margin=10, they covered (won by more than expected)
    valid['spread_abs'] = -valid['spread']  # Convert to expected margin
    valid['covered'] = valid['actual_margin'] > valid['spread_abs']
    valid['cover_margin'] = valid['actual_margin'] - valid['spread_abs']

    # Left: Expected margin vs actual margin
    x = valid['spread_abs'].values
    y = valid['actual_margin'].values

    colors = ['green' if c else 'red' for c in valid['covered']]
    axes[0].scatter(x, y, s=100, alpha=0.7, c=colors)
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, 'b-', linewidth=2,
                 label=f'R² = {r2:.3f}')
    axes[0].plot([0, 15], [0, 15], 'k--', alpha=0.3, label='Perfect')
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    axes[0].set_xlabel('Vegas Expected Margin (Spread)', fontsize=12)
    axes[0].set_ylabel('Actual Margin', fontsize=12)
    axes[0].set_title('Spread Accuracy (Green=Covered)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add labels
    for _, row in valid.iterrows():
        axes[0].annotate(row['team_abbr'], (row['spread_abs'], row['actual_margin']),
                         fontsize=8, alpha=0.7)

    # Right: Cover rate by spread size
    valid['spread_bucket'] = pd.cut(valid['spread_abs'],
                                     bins=[0, 4, 7, 10, 20],
                                     labels=['1-3.5', '4-6.5', '7-9.5', '10+'])

    bucket_stats = valid.groupby('spread_bucket', observed=True).agg({
        'covered': ['mean', 'count'],
        'cover_margin': 'mean'
    }).round(3)

    buckets = bucket_stats.index.tolist()
    cover_rates = bucket_stats[('covered', 'mean')].values * 100
    counts = bucket_stats[('covered', 'count')].values

    x_pos = np.arange(len(buckets))
    bars = axes[1].bar(x_pos, cover_rates, color='teal', alpha=0.7)
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% (random)')

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{b}\n(n={int(c)})' for b, c in zip(buckets, counts)])
    axes[1].set_xlabel('Spread Size', fontsize=12)
    axes[1].set_ylabel('Cover Rate (%)', fontsize=12)
    axes[1].set_title('Favorite Cover Rate by Spread Size', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 100)

    overall_cover = valid['covered'].mean() * 100
    axes[1].text(0.5, 0.95, f'Overall: {overall_cover:.1f}% covered',
                 ha='center', va='top', transform=axes[1].transAxes, fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'chart_18_vegas_spread_accuracy.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Spread R² = {r2:.3f}, Cover rate = {overall_cover:.1f}%")


def chart_19_game_environment_vs_scoring(game_lines_df, actual_games_df, players_df, actuals_df, output_dir):
    """
    Chart 19: How does game environment (total, spread) affect player scoring?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Merge players with actuals
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    players_merged = players_df.merge(actuals_df[['name', actual_col]], on='name', how='inner')
    if actual_col == 'actual_points':
        players_merged = players_merged.rename(columns={'actual_points': 'actual'})

    players_merged = players_merged[players_merged['consensus'] > 5].copy()

    if len(players_merged) < 20:
        print("  ⚠️ Not enough data for chart 19")
        plt.close()
        return

    # Build game environment lookup
    game_lines_df = game_lines_df.copy()
    env_by_team = {}
    for _, row in game_lines_df.iterrows():
        env_by_team[row['team_abbr']] = {
            'total': row['total'],
            'spread': row['spread'],
            'projected_pts': row['projected_pts']
        }

    # Add game environment to players
    players_merged['vegas_total'] = players_merged['team'].map(lambda t: env_by_team.get(t, {}).get('total', 44))
    players_merged['vegas_spread'] = players_merged['team'].map(lambda t: env_by_team.get(t, {}).get('spread', 0))
    players_merged['outperformance'] = players_merged['actual'] - players_merged['consensus']

    # Top left: Total vs outperformance (all positions)
    x = players_merged['vegas_total'].values
    y = players_merged['outperformance'].values

    axes[0, 0].scatter(x, y, alpha=0.5, s=30, c='blue')
    slope, intercept, r2 = linear_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0, 0].plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
                    label=f'R² = {r2:.3f}')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[0, 0].set_xlabel('Vegas Total', fontsize=12)
    axes[0, 0].set_ylabel('Outperformance (Actual - Projected)', fontsize=12)
    axes[0, 0].set_title('Game Total vs Player Outperformance', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Spread vs outperformance by position
    positions = ['QB', 'RB', 'WR', 'TE']
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange'}

    for pos in positions:
        pos_data = players_merged[players_merged['position'] == pos]
        if len(pos_data) > 5:
            axes[0, 1].scatter(pos_data['vegas_spread'], pos_data['outperformance'],
                              alpha=0.5, s=30, c=colors[pos], label=pos)

    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    axes[0, 1].set_xlabel('Vegas Spread (negative = favorite)', fontsize=12)
    axes[0, 1].set_ylabel('Outperformance', fontsize=12)
    axes[0, 1].set_title('Spread vs Outperformance by Position', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Total buckets by position
    players_merged['total_bucket'] = pd.cut(players_merged['vegas_total'],
                                             bins=[30, 40, 44, 48, 60],
                                             labels=['Low (<40)', 'Mid-Low', 'Mid-High', 'High (>48)'])

    bucket_outperf = players_merged.groupby(['total_bucket', 'position'], observed=True)['outperformance'].mean().unstack()

    if not bucket_outperf.empty:
        bucket_outperf.plot(kind='bar', ax=axes[1, 0], alpha=0.7, width=0.8)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Total Bucket', fontsize=12)
        axes[1, 0].set_ylabel('Mean Outperformance', fontsize=12)
        axes[1, 0].set_title('Outperformance by Total Bucket & Position', fontsize=14)
        axes[1, 0].legend(title='Position')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Bottom right: Favorites vs underdogs
    players_merged['team_type'] = players_merged['vegas_spread'].apply(
        lambda x: 'Favorite' if x < -3 else ('Underdog' if x > 3 else 'Pick-em')
    )

    team_stats = players_merged.groupby(['team_type', 'position'], observed=True).agg({
        'outperformance': 'mean',
        'actual': 'mean'
    }).round(2)

    if not team_stats.empty:
        outperf_pivot = players_merged.groupby(['team_type', 'position'], observed=True)['outperformance'].mean().unstack()
        outperf_pivot.plot(kind='bar', ax=axes[1, 1], alpha=0.7, width=0.8)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Team Type', fontsize=12)
        axes[1, 1].set_ylabel('Mean Outperformance', fontsize=12)
        axes[1, 1].set_title('Outperformance: Favorites vs Underdogs', fontsize=14)
        axes[1, 1].legend(title='Position')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_19_game_environment_vs_scoring.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")
    print(f"    Total → outperformance R² = {r2:.3f}")


def chart_20_blowout_impact(game_lines_df, actual_games_df, players_df, actuals_df, output_dir):
    """
    Chart 20: How did blowouts affect player scoring vs close games?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get actual game results and classify
    actual_games_df = actual_games_df.copy()
    actual_games_df['margin'] = actual_games_df['point_differential'].abs() if 'point_differential' in actual_games_df.columns else (actual_games_df['away_score'] - actual_games_df['home_score']).abs()
    actual_games_df['game_type'] = actual_games_df['margin'].apply(
        lambda x: 'Blowout (>14)' if x > 14 else ('Comfortable (7-14)' if x >= 7 else 'Close (<7)')
    )

    # Create team → game_type mapping
    game_type_by_team = {}
    for _, row in actual_games_df.iterrows():
        game_type_by_team[row['away_team']] = row['game_type']
        game_type_by_team[row['home_team']] = row['game_type']

    # Merge players with actuals
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    players_merged = players_df.merge(actuals_df[['name', actual_col]], on='name', how='inner')
    if actual_col == 'actual_points':
        players_merged = players_merged.rename(columns={'actual_points': 'actual'})

    players_merged = players_merged[players_merged['consensus'] > 5].copy()
    players_merged['game_type'] = players_merged['team'].map(game_type_by_team)
    players_merged['outperformance'] = players_merged['actual'] - players_merged['consensus']

    valid = players_merged[players_merged['game_type'].notna()].copy()

    if len(valid) < 20:
        print("  ⚠️ Not enough data for chart 20")
        plt.close()
        return

    # Left: Outperformance by game type and position
    game_types = ['Close (<7)', 'Comfortable (7-14)', 'Blowout (>14)']
    positions = ['QB', 'RB', 'WR', 'TE']

    x_pos = np.arange(len(positions))
    width = 0.25

    for i, gt in enumerate(game_types):
        gt_data = valid[valid['game_type'] == gt]
        means = [gt_data[gt_data['position'] == pos]['outperformance'].mean() if len(gt_data[gt_data['position'] == pos]) > 0 else 0
                 for pos in positions]
        axes[0].bar(x_pos + i * width, means, width, label=gt, alpha=0.7)

    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xticks(x_pos + width)
    axes[0].set_xticklabels(positions)
    axes[0].set_xlabel('Position', fontsize=12)
    axes[0].set_ylabel('Mean Outperformance', fontsize=12)
    axes[0].set_title('Outperformance by Game Margin', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Variance by game type
    variance_by_type = valid.groupby('game_type', observed=True)['outperformance'].agg(['mean', 'std', 'count']).round(2)

    x_pos = np.arange(len(game_types))
    means = [variance_by_type.loc[gt, 'mean'] if gt in variance_by_type.index else 0 for gt in game_types]
    stds = [variance_by_type.loc[gt, 'std'] if gt in variance_by_type.index else 0 for gt in game_types]
    counts = [variance_by_type.loc[gt, 'count'] if gt in variance_by_type.index else 0 for gt in game_types]

    axes[1].bar(x_pos, stds, color='teal', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{gt}\n(n={int(c)})' for gt, c in zip(game_types, counts)])
    axes[1].set_xlabel('Game Type', fontsize=12)
    axes[1].set_ylabel('Std Dev of Outperformance', fontsize=12)
    axes[1].set_title('Scoring Variance by Game Margin', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'chart_20_blowout_impact.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def chart_21_position_game_script_effects(actual_games_df, players_df, actuals_df, output_dir):
    """
    Chart 21: How do different game outcomes affect each position?
    The key question: do blowouts/shootouts/defensive games affect positions the way we model?
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Merge players with actuals
    actual_col = 'actual' if 'actual' in actuals_df.columns else 'actual_points'
    players_merged = players_df.merge(actuals_df[['name', actual_col]], on='name', how='inner')
    if actual_col == 'actual_points':
        players_merged = players_merged.rename(columns={'actual_points': 'actual'})

    players_merged = players_merged[players_merged['consensus'] > 5].copy()
    players_merged['outperformance'] = players_merged['actual'] - players_merged['consensus']
    players_merged['outperf_pct'] = players_merged['outperformance'] / players_merged['consensus'] * 100

    # Add actual game results to players
    actual_games_df = actual_games_df.copy()

    # Create lookups for game results
    game_results_by_team = {}
    for _, row in actual_games_df.iterrows():
        total = row['total_points']
        margin = abs(row['away_score'] - row['home_score'])
        actual_script = row.get('actual_script', 'unknown')

        # Determine winner/loser
        if row['away_score'] > row['home_score']:
            winner, loser = row['away_team'], row['home_team']
            winner_margin, loser_margin = margin, -margin
        else:
            winner, loser = row['home_team'], row['away_team']
            winner_margin, loser_margin = margin, -margin

        game_results_by_team[row['away_team']] = {
            'total': total, 'margin': margin,
            'team_margin': row['away_score'] - row['home_score'],
            'actual_script': actual_script,
            'won': row['away_score'] > row['home_score']
        }
        game_results_by_team[row['home_team']] = {
            'total': total, 'margin': margin,
            'team_margin': row['home_score'] - row['away_score'],
            'actual_script': actual_script,
            'won': row['home_score'] > row['away_score']
        }

    # Map to players
    players_merged['game_total'] = players_merged['team'].map(lambda t: game_results_by_team.get(t, {}).get('total', 44))
    players_merged['game_margin'] = players_merged['team'].map(lambda t: game_results_by_team.get(t, {}).get('margin', 0))
    players_merged['team_margin'] = players_merged['team'].map(lambda t: game_results_by_team.get(t, {}).get('team_margin', 0))
    players_merged['actual_script'] = players_merged['team'].map(lambda t: game_results_by_team.get(t, {}).get('actual_script', 'unknown'))
    players_merged['won'] = players_merged['team'].map(lambda t: game_results_by_team.get(t, {}).get('won', False))

    # Normalize script names
    players_merged['script_norm'] = players_merged['actual_script'].apply(
        lambda x: 'blowout' if 'blowout' in str(x) else x
    )

    valid = players_merged[players_merged['game_total'].notna()].copy()

    if len(valid) < 30:
        print("  ⚠️ Not enough data for chart 21")
        plt.close()
        return

    positions = ['QB', 'RB', 'WR', 'TE']
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange'}

    # =========================================================================
    # Top Left: Outperformance by actual game script and position
    # =========================================================================
    scripts = ['shootout', 'defensive', 'blowout', 'competitive']
    x_pos = np.arange(len(scripts))
    width = 0.2

    for i, pos in enumerate(positions):
        pos_data = valid[valid['position'] == pos]
        means = []
        for script in scripts:
            script_data = pos_data[pos_data['script_norm'] == script]
            if len(script_data) > 0:
                means.append(script_data['outperf_pct'].mean())
            else:
                means.append(0)
        axes[0, 0].bar(x_pos + i * width, means, width, label=pos, color=colors[pos], alpha=0.7)

    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xticks(x_pos + width * 1.5)
    axes[0, 0].set_xticklabels(scripts)
    axes[0, 0].set_xlabel('Actual Game Script', fontsize=12)
    axes[0, 0].set_ylabel('Mean Outperformance (%)', fontsize=12)
    axes[0, 0].set_title('Position Performance by Actual Game Script', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add sample sizes
    for script in scripts:
        count = len(valid[valid['script_norm'] == script])
        axes[0, 0].text(scripts.index(script) + width * 1.5, axes[0, 0].get_ylim()[0] - 2,
                        f'n={count}', ha='center', fontsize=9)

    # =========================================================================
    # Top Right: Total points vs outperformance by position (with trend lines)
    # =========================================================================
    for pos in positions:
        pos_data = valid[valid['position'] == pos]
        if len(pos_data) > 5:
            x = pos_data['game_total'].values
            y = pos_data['outperf_pct'].values
            axes[0, 1].scatter(x, y, alpha=0.5, s=40, c=colors[pos], label=pos)

            # Add trend line
            slope, intercept, r2 = linear_regression(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            axes[0, 1].plot(x_line, intercept + slope * x_line, c=colors[pos], linestyle='--', alpha=0.7)

    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Actual Game Total Points', fontsize=12)
    axes[0, 1].set_ylabel('Outperformance (%)', fontsize=12)
    axes[0, 1].set_title('Game Scoring Level vs Position Performance', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # =========================================================================
    # Bottom Left: Point margin (blowout indicator) vs outperformance
    # Split by winning/losing team
    # =========================================================================
    for pos in positions:
        pos_data = valid[valid['position'] == pos]
        if len(pos_data) > 5:
            # Winning team players (team_margin > 0)
            winners = pos_data[pos_data['won'] == True]
            losers = pos_data[pos_data['won'] == False]

            if len(winners) > 3:
                axes[1, 0].scatter(winners['game_margin'], winners['outperf_pct'],
                                  alpha=0.6, s=40, c=colors[pos], marker='o')
            if len(losers) > 3:
                axes[1, 0].scatter(losers['game_margin'], losers['outperf_pct'],
                                  alpha=0.4, s=40, c=colors[pos], marker='x')

    # Add legend entries manually
    for pos in positions:
        axes[1, 0].scatter([], [], c=colors[pos], marker='o', label=f'{pos} (winner)')
        axes[1, 0].scatter([], [], c=colors[pos], marker='x', alpha=0.5, label=f'{pos} (loser)')

    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Game Margin (Blowout Size)', fontsize=12)
    axes[1, 0].set_ylabel('Outperformance (%)', fontsize=12)
    axes[1, 0].set_title('Blowout Margin vs Performance (circles=winners, X=losers)', fontsize=14)
    axes[1, 0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)

    # =========================================================================
    # Bottom Right: Summary table - expected vs actual effects
    # =========================================================================
    # Our model assumptions:
    # - Shootouts: WR/TE boom, RB slightly hurt (pass-heavy)
    # - Defensive games: Everyone suppressed, especially WR
    # - Blowouts: Winners' RB boom (garbage time carries), losers' WR boom (garbage time passing)
    # - Competitive: Neutral

    summary_data = []
    for pos in positions:
        pos_data = valid[valid['position'] == pos]
        for script in scripts:
            script_data = pos_data[pos_data['script_norm'] == script]
            if len(script_data) >= 3:
                summary_data.append({
                    'Position': pos,
                    'Script': script,
                    'Outperf %': script_data['outperf_pct'].mean(),
                    'N': len(script_data)
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        pivot = summary_df.pivot(index='Position', columns='Script', values='Outperf %')

        # Create heatmap
        im = axes[1, 1].imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_xticklabels(pivot.columns)
        axes[1, 1].set_yticklabels(pivot.index)
        axes[1, 1].set_xlabel('Actual Game Script', fontsize=12)
        axes[1, 1].set_ylabel('Position', fontsize=12)
        axes[1, 1].set_title('Position Outperformance by Game Script (%)', fontsize=14)

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    color = 'white' if abs(val) > 15 else 'black'
                    axes[1, 1].text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=11)

        plt.colorbar(im, ax=axes[1, 1], label='Outperformance %')

    plt.tight_layout()
    output_path = output_dir / 'chart_21_position_game_script_effects.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")

    # Print key insights
    if summary_data:
        print("    Position × Script Effects (Outperformance %):")
        for script in scripts:
            script_summary = [s for s in summary_data if s['Script'] == script]
            if script_summary:
                effects = ', '.join([f"{s['Position']}:{s['Outperf %']:+.1f}%" for s in script_summary])
                print(f"      {script.capitalize()}: {effects}")


def chart_22_betting_signal_summary(game_lines_df, game_scripts_df, actual_games_df, output_dir):
    """
    Chart 22: Summary of all betting signals - which ones are predictive?
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Build a summary of betting signal accuracy
    results = []

    # 1. Total prediction accuracy
    game_lines_unique = game_lines_df.drop_duplicates(subset=['total']).copy()
    actual_totals = {}
    for _, row in actual_games_df.iterrows():
        actual_totals[row['away_team']] = row['total_points']
        actual_totals[row['home_team']] = row['total_points']

    game_lines_df_with_actual = game_lines_df.copy()
    game_lines_df_with_actual['actual_total'] = game_lines_df_with_actual['team_abbr'].map(actual_totals)
    valid_totals = game_lines_df_with_actual.dropna(subset=['actual_total']).drop_duplicates(subset=['total', 'actual_total'])

    if len(valid_totals) > 3:
        x = valid_totals['total'].values
        y = valid_totals['actual_total'].values
        _, _, r2_total = linear_regression(x, y)
        mae_total = np.abs(y - x).mean()
        results.append({'Signal': 'Vegas Total (O/U)', 'R²': r2_total, 'MAE': mae_total, 'Type': 'Game Level'})

    # 2. Spread prediction accuracy (margin)
    actual_margins = {}
    for _, row in actual_games_df.iterrows():
        actual_margins[row['away_team']] = row['away_score'] - row['home_score']
        actual_margins[row['home_team']] = row['home_score'] - row['away_score']

    game_lines_df_with_margin = game_lines_df.copy()
    game_lines_df_with_margin['actual_margin'] = game_lines_df_with_margin['team_abbr'].map(actual_margins)
    valid_spread = game_lines_df_with_margin.dropna(subset=['actual_margin'])

    if len(valid_spread) > 3:
        # Spread is expected margin (negative for underdog)
        x = -valid_spread['spread'].values  # Convert to expected win margin
        y = valid_spread['actual_margin'].values
        _, _, r2_spread = linear_regression(x, y)
        mae_spread = np.abs(y - x).mean()
        results.append({'Signal': 'Vegas Spread', 'R²': r2_spread, 'MAE': mae_spread, 'Type': 'Game Level'})

    # 3. Game script prediction accuracy
    game_scripts_df = game_scripts_df.copy()
    actual_games_df = actual_games_df.copy()

    game_scripts_df['game_key'] = game_scripts_df['game_id']
    actual_games_df['game_key'] = actual_games_df['away_team'] + '@' + actual_games_df['home_team']

    script_merged = game_scripts_df.merge(
        actual_games_df[['game_key', 'actual_script']],
        on='game_key', how='inner'
    )

    if len(script_merged) > 3:
        script_merged['actual_script_norm'] = script_merged['actual_script'].apply(
            lambda x: 'blowout' if 'blowout' in str(x) else x
        )
        accuracy = (script_merged['primary_script'] == script_merged['actual_script_norm']).mean()
        results.append({'Signal': 'Game Script Prediction', 'R²': accuracy, 'MAE': 1 - accuracy, 'Type': 'Game Level'})

        # Script probability calibration
        script_merged['prob_for_actual'] = script_merged.apply(
            lambda row: row.get(f"{row['actual_script_norm']}_prob", 0.25),
            axis=1
        )
        avg_prob = script_merged['prob_for_actual'].mean()
        results.append({'Signal': 'Script Probability Calibration', 'R²': avg_prob, 'MAE': 0.25 - avg_prob, 'Type': 'Probability'})

    if not results:
        print("  ⚠️ Not enough data for chart 21")
        plt.close()
        return

    # Create summary chart
    df_results = pd.DataFrame(results)

    colors = ['blue' if t == 'Game Level' else 'green' for t in df_results['Type']]
    y_pos = np.arange(len(df_results))

    bars = ax.barh(y_pos, df_results['R²'], color=colors, alpha=0.7)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.7, label='Random baseline')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_results['Signal'])
    ax.set_xlabel('R² / Accuracy', fontsize=12)
    ax.set_title('Betting Signal Accuracy Summary', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (r2, mae) in enumerate(zip(df_results['R²'], df_results['MAE'])):
        ax.text(r2 + 0.02, i, f'R²={r2:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'chart_22_betting_signal_summary.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_path.name}")

    # Print summary
    print("    Betting Signal Summary:")
    for _, row in df_results.iterrows():
        print(f"      {row['Signal']}: R²={row['R²']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Generate analysis charts')
    parser.add_argument('--week-dir', required=True, help='Week directory')
    args = parser.parse_args()

    week_dir = Path(args.week_dir)
    inputs_dir = week_dir / 'inputs'
    intermediate_dir = week_dir / 'intermediate'

    # Output charts to scripts/charts/analysis/ alongside this script
    output_dir = Path(__file__).parent / 'charts' / 'analysis' / week_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ANALYSIS CHARTS")
    print(f"{'='*60}")
    print(f"Week: {week_dir.name}")
    print(f"Output: {output_dir}\n")

    # Load data
    print("Loading data...")
    espn_df = pd.read_csv(inputs_dir / 'espn_projections.csv')
    players_df = pd.read_csv(intermediate_dir / '1_players.csv')

    # Try to load actuals if available
    actuals_path = intermediate_dir / '4_actual_players.csv'
    if not actuals_path.exists():
        actuals_path = week_dir / 'actuals.csv'
    if not actuals_path.exists():
        # Try to find in outputs
        for output_run in (week_dir / 'outputs').glob('run_*'):
            possible_actuals = output_run / 'actuals.csv'
            if possible_actuals.exists():
                actuals_path = possible_actuals
                break

    actuals_df = None
    if actuals_path.exists():
        actuals_df = pd.read_csv(actuals_path)
        # Normalize name column
        if 'Player' in actuals_df.columns:
            actuals_df = actuals_df.rename(columns={'Player': 'name', 'FD Points': 'actual'})
        if 'actual_points' in actuals_df.columns:
            actuals_df = actuals_df.rename(columns={'actual_points': 'actual'})
        print(f"  ✓ Loaded {len(actuals_df)} actual scores")
    else:
        print("  ⚠️ No actuals file found - some charts will be skipped")

    print(f"  ✓ Loaded {len(espn_df)} ESPN projections")
    print(f"  ✓ Loaded {len(players_df)} integrated players\n")

    print("Generating charts...")

    # Chart 1: Floor/Ceiling regression
    chart_1_floor_ceiling_regression(espn_df, output_dir)

    # Chart 2: ESPN to FP regression
    chart_2_espn_to_fp_regression(players_df, output_dir)

    # Chart 3: TD odds vs skew
    chart_3_td_odds_vs_skew(players_df, output_dir)

    # Charts requiring actuals
    if actuals_df is not None:
        chart_4_consensus_vs_actual(players_df, actuals_df, output_dir)
        chart_5_skew_vs_actual(players_df, actuals_df, output_dir)
        chart_6_espn_range_accuracy(espn_df, actuals_df, output_dir)
        chart_7_position_variance(players_df, actuals_df, output_dir)

    # Chart 8: Ceiling-floor spread
    chart_8_ceiling_floor_spread(espn_df, output_dir)

    # Predictive signal charts (require actuals)
    if actuals_df is not None:
        print("\n--- Predictive Signal Charts ---")
        chart_9_projection_source_comparison(players_df, actuals_df, output_dir)
        chart_9b_raw_espn_comparison(players_df, actuals_df, output_dir)
        chart_10_td_odds_signal(players_df, actuals_df, output_dir)
        chart_11_floor_ceiling_hit_rates(players_df, actuals_df, output_dir)
        chart_12_value_prediction(players_df, actuals_df, output_dir)
        chart_13_sigma_vs_variance(players_df, actuals_df, output_dir)
        chart_14_correlation_matrix(players_df, actuals_df, output_dir)
        chart_15_residual_analysis(players_df, actuals_df, output_dir)

    # =========================================================================
    # GAME SCRIPT & BETTING SIGNAL CHARTS (16-22)
    # =========================================================================

    # Load game-level data
    game_lines_path = inputs_dir / 'game_lines.csv'
    game_scripts_path = intermediate_dir / '2_game_scripts.csv'

    # Prefer backtest_games.csv (has actual_script) over 4_actual_games.csv
    actual_games_path = None
    for output_run in sorted((week_dir / 'outputs').glob('run_*'), reverse=True):
        possible_path = output_run / '6_backtest_games.csv'
        if possible_path.exists():
            actual_games_path = possible_path
            break

    # Fallback to intermediate if no backtest file found
    if actual_games_path is None:
        actual_games_path = intermediate_dir / '4_actual_games.csv'

    game_lines_df = None
    game_scripts_df = None
    actual_games_df = None

    if game_lines_path.exists():
        game_lines_df = pd.read_csv(game_lines_path)
        print(f"  ✓ Loaded {len(game_lines_df)} game lines")

    if game_scripts_path.exists():
        game_scripts_df = pd.read_csv(game_scripts_path)
        print(f"  ✓ Loaded {len(game_scripts_df)} game script predictions")

    if actual_games_path.exists():
        actual_games_df = pd.read_csv(actual_games_path)
        print(f"  ✓ Loaded {len(actual_games_df)} actual game results")

    # Generate game-level charts
    if game_scripts_df is not None and actual_games_df is not None:
        print("\n--- Game Script & Betting Signal Charts ---")
        chart_16_game_script_accuracy(game_scripts_df, actual_games_df, output_dir)

    if game_lines_df is not None and actual_games_df is not None:
        chart_17_vegas_total_accuracy(game_lines_df, actual_games_df, output_dir)
        chart_18_vegas_spread_accuracy(game_lines_df, actual_games_df, output_dir)

        if actuals_df is not None:
            chart_19_game_environment_vs_scoring(game_lines_df, actual_games_df, players_df, actuals_df, output_dir)
            chart_20_blowout_impact(game_lines_df, actual_games_df, players_df, actuals_df, output_dir)

    if actual_games_df is not None and actuals_df is not None:
        chart_21_position_game_script_effects(actual_games_df, players_df, actuals_df, output_dir)

    if game_lines_df is not None and game_scripts_df is not None and actual_games_df is not None:
        chart_22_betting_signal_summary(game_lines_df, game_scripts_df, actual_games_df, output_dir)

    print(f"\n✅ Charts saved to {output_dir}")


if __name__ == '__main__':
    main()
