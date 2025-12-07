#!/usr/bin/env python3
"""
Shootout Index vs Position Performance Analysis

Analyzes the relationship between a continuous "shootout index" and:
1. Absolute fantasy points by position
2. Residuals (actual - expected) by position

Shootout Index = 1 - (margin / total_score)
- Close to 1.0 = shootout (high scoring, close game)
- Close to 0.0 = blowout (margin approaches total)
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def load_data(seasons: list) -> tuple:
    """Load player stats and game schedules."""
    print(f"Loading data for seasons {seasons[0]}-{seasons[-1]}...")

    # Player stats
    players = nfl.load_player_stats(seasons=seasons, summary_level='week')
    print(f"  Player stats: {len(players):,} records")

    # Schedule for game scores
    schedules = nfl.load_schedules(seasons=seasons)
    print(f"  Schedules: {len(schedules):,} games")

    return players, schedules


def calculate_shootout_index(schedules: pl.DataFrame) -> pd.DataFrame:
    """
    Calculate shootout index for each game.

    Shootout Index = 1 - (margin / total_score)
    """
    print("\nCalculating shootout index...")

    sdf = schedules.to_pandas()

    # Filter to completed games
    sdf = sdf[sdf['home_score'].notna() & sdf['away_score'].notna()].copy()

    # Calculate total and margin
    sdf['total_score'] = sdf['home_score'] + sdf['away_score']
    sdf['margin'] = abs(sdf['home_score'] - sdf['away_score'])

    # Shootout index: 1 - (margin / total)
    # Handle edge case where total_score = 0
    sdf['shootout_index'] = np.where(
        sdf['total_score'] > 0,
        1 - (sdf['margin'] / sdf['total_score']),
        0.5  # Default for 0-0 games (rare)
    )

    print(f"  Games with scores: {len(sdf):,}")
    print(f"  Shootout index range: {sdf['shootout_index'].min():.3f} - {sdf['shootout_index'].max():.3f}")
    print(f"  Mean shootout index: {sdf['shootout_index'].mean():.3f}")

    # Create game lookup keys for both home and away teams
    home_games = sdf[['season', 'week', 'home_team', 'shootout_index', 'total_score', 'margin']].copy()
    home_games = home_games.rename(columns={'home_team': 'team'})

    away_games = sdf[['season', 'week', 'away_team', 'shootout_index', 'total_score', 'margin']].copy()
    away_games = away_games.rename(columns={'away_team': 'team'})

    game_index = pd.concat([home_games, away_games], ignore_index=True)

    return game_index


def calculate_rolling_expected(df: pl.DataFrame, window: int = 4, decay: float = 0.8) -> pd.DataFrame:
    """Calculate rolling weighted average as expected performance."""
    print(f"\nCalculating rolling expected (window={window}, decay={decay})...")

    pdf = df.to_pandas()
    pdf = pdf.sort_values(['player_id', 'season', 'week'])

    weights = np.array([decay ** i for i in range(window - 1, -1, -1)])
    weights = weights / weights.sum()

    def weighted_rolling_mean(series):
        result = []
        values = series.values
        for i in range(len(values)):
            if i < 1:
                result.append(np.nan)
            else:
                start = max(0, i - window)
                prev_values = values[start:i]
                if len(prev_values) == 0:
                    result.append(np.nan)
                else:
                    w = weights[-len(prev_values):]
                    w = w / w.sum()
                    result.append(np.average(prev_values, weights=w))
        return pd.Series(result, index=series.index)

    pdf['expected_pts'] = pdf.groupby('player_id')['fantasy_points_ppr'].transform(weighted_rolling_mean)
    pdf['residual'] = pdf['fantasy_points_ppr'] - pdf['expected_pts']

    print(f"  Players with expected: {pdf['expected_pts'].notna().sum():,}")

    return pdf


def merge_with_shootout_index(players: pd.DataFrame, game_index: pd.DataFrame) -> pd.DataFrame:
    """Merge player stats with shootout index."""
    print("\nMerging player stats with shootout index...")

    merged = players.merge(
        game_index,
        on=['season', 'week', 'team'],
        how='inner'
    )

    print(f"  Merged records: {len(merged):,}")

    return merged


def analyze_correlations(df: pd.DataFrame, min_expected: float = 5.0) -> dict:
    """
    Analyze correlations between shootout index and position performance.
    """
    print(f"\nAnalyzing correlations (min expected >= {min_expected})...")

    positions = ['QB', 'RB', 'WR', 'TE']
    results = {'raw': {}, 'residual': {}}

    for pos in positions:
        pos_df = df[(df['position'] == pos) & (df['expected_pts'] >= min_expected)].copy()

        if len(pos_df) < 100:
            continue

        # Raw correlation: shootout_index vs fantasy_points_ppr
        r_raw, p_raw = stats.pearsonr(pos_df['shootout_index'], pos_df['fantasy_points_ppr'])
        results['raw'][pos] = {'r': r_raw, 'p': p_raw, 'n': len(pos_df)}

        # Residual correlation: shootout_index vs residual
        valid = pos_df['residual'].notna()
        if valid.sum() > 100:
            r_res, p_res = stats.pearsonr(pos_df.loc[valid, 'shootout_index'], pos_df.loc[valid, 'residual'])
            results['residual'][pos] = {'r': r_res, 'p': p_res, 'n': valid.sum()}

    return results


def analyze_by_shootout_bucket(df: pd.DataFrame, min_expected: float = 5.0) -> pd.DataFrame:
    """
    Analyze average performance by shootout index buckets.
    """
    print("\nAnalyzing by shootout bucket...")

    # Create shootout buckets
    df = df.copy()
    df['shootout_bucket'] = pd.cut(
        df['shootout_index'],
        bins=[0, 0.5, 0.7, 0.85, 1.0],
        labels=['Blowout (<0.5)', 'Moderate (0.5-0.7)', 'Competitive (0.7-0.85)', 'Shootout (>0.85)']
    )

    positions = ['QB', 'RB', 'WR', 'TE']
    results = []

    for pos in positions:
        pos_df = df[(df['position'] == pos) & (df['expected_pts'] >= min_expected)]

        for bucket in pos_df['shootout_bucket'].unique():
            if pd.isna(bucket):
                continue
            bucket_df = pos_df[pos_df['shootout_bucket'] == bucket]

            results.append({
                'position': pos,
                'bucket': bucket,
                'n': len(bucket_df),
                'avg_pts': bucket_df['fantasy_points_ppr'].mean(),
                'avg_expected': bucket_df['expected_pts'].mean(),
                'avg_residual': bucket_df['residual'].mean(),
                'std_residual': bucket_df['residual'].std()
            })

    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, correlations: dict, bucket_analysis: pd.DataFrame, output_dir: Path):
    """Create visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = ['QB', 'RB', 'WR', 'TE']
    colors = {'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange'}

    # Figure 1: Scatter plots - Shootout Index vs Raw Points
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_df = df[(df['position'] == pos) & (df['expected_pts'] >= 5.0)]

        # Sample for readability
        sample = pos_df.sample(min(2000, len(pos_df)), random_state=42)

        ax.scatter(sample['shootout_index'], sample['fantasy_points_ppr'],
                   alpha=0.2, s=10, c=colors[pos])

        # Trendline
        z = np.polyfit(pos_df['shootout_index'], pos_df['fantasy_points_ppr'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0.2, 1.0, 50)
        ax.plot(x_line, p(x_line), 'k-', linewidth=2)

        r = correlations['raw'].get(pos, {}).get('r', 0)
        n = correlations['raw'].get(pos, {}).get('n', 0)
        ax.set_title(f'{pos}: Shootout Index vs Fantasy Points\nr = {r:.3f} (n={n:,})')
        ax.set_xlabel('Shootout Index (1 = shootout, 0 = blowout)')
        ax.set_ylabel('Fantasy Points (PPR)')
        ax.set_xlim(0.2, 1.0)

    plt.suptitle('Shootout Index vs Raw Fantasy Points by Position', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'shootout_vs_raw_points.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'shootout_vs_raw_points.png'}")
    plt.close()

    # Figure 2: Scatter plots - Shootout Index vs Residuals
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        pos_df = df[(df['position'] == pos) & (df['expected_pts'] >= 5.0) & (df['residual'].notna())]

        sample = pos_df.sample(min(2000, len(pos_df)), random_state=42)

        ax.scatter(sample['shootout_index'], sample['residual'],
                   alpha=0.2, s=10, c=colors[pos])
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Trendline
        z = np.polyfit(pos_df['shootout_index'], pos_df['residual'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0.2, 1.0, 50)
        ax.plot(x_line, p(x_line), 'k-', linewidth=2)

        r = correlations['residual'].get(pos, {}).get('r', 0)
        n = correlations['residual'].get(pos, {}).get('n', 0)
        ax.set_title(f'{pos}: Shootout Index vs Residual\nr = {r:.3f} (n={n:,})')
        ax.set_xlabel('Shootout Index (1 = shootout, 0 = blowout)')
        ax.set_ylabel('Residual (Actual - Expected)')
        ax.set_xlim(0.2, 1.0)

    plt.suptitle('Shootout Index vs Residual (Boom/Bust) by Position', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'shootout_vs_residual.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'shootout_vs_residual.png'}")
    plt.close()

    # Figure 3: Bar chart - Average residual by shootout bucket
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average points by bucket
    ax = axes[0]
    pivot = bucket_analysis.pivot(index='bucket', columns='position', values='avg_pts')
    pivot = pivot[positions]  # Order positions
    pivot.plot(kind='bar', ax=ax, color=[colors[p] for p in positions], width=0.8)
    ax.set_xlabel('Game Type')
    ax.set_ylabel('Average Fantasy Points (PPR)')
    ax.set_title('Average Fantasy Points by Game Type')
    ax.legend(title='Position')
    ax.tick_params(axis='x', rotation=45)

    # Average residual by bucket
    ax = axes[1]
    pivot = bucket_analysis.pivot(index='bucket', columns='position', values='avg_residual')
    pivot = pivot[positions]
    pivot.plot(kind='bar', ax=ax, color=[colors[p] for p in positions], width=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Game Type')
    ax.set_ylabel('Average Residual (Actual - Expected)')
    ax.set_title('Average Over/Under-Performance by Game Type')
    ax.legend(title='Position')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'shootout_bucket_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'shootout_bucket_analysis.png'}")
    plt.close()

    # Figure 4: Combined correlation summary
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(positions))
    width = 0.35

    raw_corrs = [correlations['raw'].get(p, {}).get('r', 0) for p in positions]
    res_corrs = [correlations['residual'].get(p, {}).get('r', 0) for p in positions]

    bars1 = ax.bar(x - width/2, raw_corrs, width, label='Raw Points', color='steelblue')
    bars2 = ax.bar(x + width/2, res_corrs, width, label='Residual', color='coral')

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Correlation with Shootout Index')
    ax.set_title('Correlation: Shootout Index vs Fantasy Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'shootout_correlation_summary.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'shootout_correlation_summary.png'}")
    plt.close()


def main():
    """Run shootout index analysis."""
    print("=" * 70)
    print("SHOOTOUT INDEX vs POSITION PERFORMANCE ANALYSIS")
    print("=" * 70)
    print("\nShootout Index = 1 - (margin / total_score)")
    print("  1.0 = perfect shootout (high scoring, no margin)")
    print("  0.0 = complete blowout (margin = total)")

    # Load data
    seasons = list(range(2016, 2025))
    players, schedules = load_data(seasons)

    # Calculate shootout index for each game
    game_index = calculate_shootout_index(schedules)

    # Calculate rolling expected for players
    players_df = calculate_rolling_expected(players)

    # Merge
    merged = merge_with_shootout_index(players_df, game_index)

    # Analyze correlations
    correlations = analyze_correlations(merged, min_expected=5.0)

    print("\n" + "=" * 70)
    print("CORRELATION: SHOOTOUT INDEX vs RAW FANTASY POINTS")
    print("=" * 70)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in correlations['raw']:
            r = correlations['raw'][pos]['r']
            n = correlations['raw'][pos]['n']
            p = correlations['raw'][pos]['p']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {pos}: r = {r:+.3f} {sig} (n={n:,})")

    print("\n" + "=" * 70)
    print("CORRELATION: SHOOTOUT INDEX vs RESIDUAL (Actual - Expected)")
    print("=" * 70)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in correlations['residual']:
            r = correlations['residual'][pos]['r']
            n = correlations['residual'][pos]['n']
            p = correlations['residual'][pos]['p']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {pos}: r = {r:+.3f} {sig} (n={n:,})")

    # Bucket analysis
    bucket_analysis = analyze_by_shootout_bucket(merged, min_expected=5.0)

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE BY GAME TYPE")
    print("=" * 70)
    print(bucket_analysis.to_string(index=False))

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    output_dir = Path(__file__).parent / 'charts' / 'shootout_analysis'
    create_visualizations(merged, correlations, bucket_analysis, output_dir)

    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    qb_raw = correlations['raw'].get('QB', {}).get('r', 0)
    rb_raw = correlations['raw'].get('RB', {}).get('r', 0)
    wr_raw = correlations['raw'].get('WR', {}).get('r', 0)
    te_raw = correlations['raw'].get('TE', {}).get('r', 0)

    qb_res = correlations['residual'].get('QB', {}).get('r', 0)
    rb_res = correlations['residual'].get('RB', {}).get('r', 0)
    wr_res = correlations['residual'].get('WR', {}).get('r', 0)
    te_res = correlations['residual'].get('TE', {}).get('r', 0)

    print("\n1. RAW POINT CORRELATIONS (higher = better in shootouts):")
    print(f"   QB:  {qb_raw:+.3f} - {'Benefits from shootouts' if qb_raw > 0.05 else 'Neutral'}")
    print(f"   RB:  {rb_raw:+.3f} - {'Benefits from shootouts' if rb_raw > 0.05 else 'Hurt by shootouts' if rb_raw < -0.05 else 'Neutral'}")
    print(f"   WR:  {wr_raw:+.3f} - {'Benefits from shootouts' if wr_raw > 0.05 else 'Neutral'}")
    print(f"   TE:  {te_raw:+.3f} - {'Benefits from shootouts' if te_raw > 0.05 else 'Neutral'}")

    print("\n2. RESIDUAL CORRELATIONS (positive = boom in shootouts):")
    print(f"   QB:  {qb_res:+.3f} - {'Exceeds expectations in shootouts' if qb_res > 0.03 else 'Neutral'}")
    print(f"   RB:  {rb_res:+.3f} - {'Exceeds expectations in shootouts' if rb_res > 0.03 else 'Below expectations in shootouts' if rb_res < -0.03 else 'Neutral'}")
    print(f"   WR:  {wr_res:+.3f} - {'Exceeds expectations in shootouts' if wr_res > 0.03 else 'Neutral'}")
    print(f"   TE:  {te_res:+.3f} - {'Exceeds expectations in shootouts' if te_res > 0.03 else 'Neutral'}")

    print("\n3. DFS IMPLICATIONS:")
    if qb_raw > 0.05:
        print("   - Target QBs in projected shootouts")
    if rb_raw < -0.03:
        print("   - Fade RBs in projected shootouts (game script negative)")
    if wr_raw > 0.05:
        print("   - Target WRs in projected shootouts")


if __name__ == '__main__':
    main()
