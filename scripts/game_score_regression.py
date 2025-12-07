#!/usr/bin/env python3
"""
Game Score Regression Analysis

Direct regression analysis of position performance against:
- Team score (points scored by player's team)
- Relative margin (positive = team won, negative = team lost)

Separate analysis for home vs away players.
Models both raw fantasy points and residuals (actual - expected).

Filters:
- Only players with expected (rolling avg) >= 5 points
- Only players with actual points > 0 (removes players who didn't play)
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def load_data(seasons: list) -> tuple:
    """Load player stats and game schedules."""
    print(f"Loading data for seasons {seasons[0]}-{seasons[-1]}...")

    players = nfl.load_player_stats(seasons=seasons, summary_level='week')
    print(f"  Player stats: {len(players):,} records")

    schedules = nfl.load_schedules(seasons=seasons)
    print(f"  Schedules: {len(schedules):,} games")

    return players, schedules


def prepare_game_data(schedules: pl.DataFrame) -> pd.DataFrame:
    """
    Prepare game-level data with team-specific scores and relative margin.

    Returns separate rows for home and away teams with:
    - team_score: points scored by that team
    - opp_score: points scored by opponent
    - relative_margin: team_score - opp_score (positive = won, negative = lost)
    - is_home: whether the team is home
    """
    print("\nPreparing game data...")

    sdf = schedules.to_pandas()
    sdf = sdf[sdf['home_score'].notna() & sdf['away_score'].notna()].copy()

    # Home team perspective
    home_games = pd.DataFrame({
        'season': sdf['season'],
        'week': sdf['week'],
        'team': sdf['home_team'],
        'team_score': sdf['home_score'],
        'opp_score': sdf['away_score'],
        'relative_margin': sdf['home_score'] - sdf['away_score'],
        'total_score': sdf['home_score'] + sdf['away_score'],
        'is_home': True
    })

    # Away team perspective
    away_games = pd.DataFrame({
        'season': sdf['season'],
        'week': sdf['week'],
        'team': sdf['away_team'],
        'team_score': sdf['away_score'],
        'opp_score': sdf['home_score'],
        'relative_margin': sdf['away_score'] - sdf['home_score'],
        'total_score': sdf['home_score'] + sdf['away_score'],
        'is_home': False
    })

    game_data = pd.concat([home_games, away_games], ignore_index=True)

    print(f"  Team-games prepared: {len(game_data):,}")
    print(f"  Home games: {home_games['is_home'].sum():,}")
    print(f"  Away games: {len(away_games):,}")

    return game_data


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


def merge_data(players: pd.DataFrame, game_data: pd.DataFrame) -> pd.DataFrame:
    """Merge player stats with game data."""
    print("\nMerging player and game data...")

    merged = players.merge(game_data, on=['season', 'week', 'team'], how='inner')
    print(f"  Merged records: {len(merged):,}")
    print(f"  Home player-games: {merged['is_home'].sum():,}")
    print(f"  Away player-games: {(~merged['is_home']).sum():,}")

    return merged


def run_regression_analysis(df: pd.DataFrame, min_expected: float = 5.0) -> dict:
    """
    Run regression analysis for each position, split by home/away.

    Models:
    1. Raw points ~ team_score + relative_margin
    2. Residual ~ team_score + relative_margin
    """
    print(f"\nRunning regression analysis (min expected >= {min_expected})...")

    positions = ['QB', 'RB', 'WR', 'TE']
    locations = ['home', 'away', 'all']
    results = {}

    for pos in positions:
        results[pos] = {}

        for loc in locations:
            # Filter: expected >= min, and actual > 0 (0 usually means didn't play)
            pos_df = df[(df['position'] == pos) &
                       (df['expected_pts'] >= min_expected) &
                       (df['fantasy_points_ppr'] > 0)].copy()

            if loc == 'home':
                pos_df = pos_df[pos_df['is_home'] == True]
            elif loc == 'away':
                pos_df = pos_df[pos_df['is_home'] == False]
            # else 'all' - use all data

            pos_df = pos_df.dropna(subset=['team_score', 'relative_margin', 'fantasy_points_ppr', 'residual'])

            if len(pos_df) < 100:
                continue

            results[pos][loc] = {}

            # Prepare features
            X = pos_df[['team_score', 'relative_margin']].values
            X_with_const = sm.add_constant(X)

            # Model 1: Raw points
            y_raw = pos_df['fantasy_points_ppr'].values
            model_raw = sm.OLS(y_raw, X_with_const).fit()

            results[pos][loc]['raw'] = {
                'intercept': model_raw.params[0],
                'coef_team_score': model_raw.params[1],
                'coef_margin': model_raw.params[2],
                'r2': model_raw.rsquared,
                'pvalue_team_score': model_raw.pvalues[1],
                'pvalue_margin': model_raw.pvalues[2],
                'n': len(pos_df)
            }

            # Model 2: Residual
            y_res = pos_df['residual'].values
            model_res = sm.OLS(y_res, X_with_const).fit()

            results[pos][loc]['residual'] = {
                'intercept': model_res.params[0],
                'coef_team_score': model_res.params[1],
                'coef_margin': model_res.params[2],
                'r2': model_res.rsquared,
                'pvalue_team_score': model_res.pvalues[1],
                'pvalue_margin': model_res.pvalues[2],
                'n': len(pos_df)
            }

    return results


def create_visualizations(df: pd.DataFrame, results: dict, output_dir: Path):
    """Create regression visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = ['QB', 'RB', 'WR', 'TE']
    colors = {'home': 'red', 'away': 'blue', 'all': 'gray'}

    # Figure 1: Coefficient comparison bar chart - Home vs Away
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    width = 0.35
    x = np.arange(len(positions))

    # Top left: Team Score coefficient (Raw)
    ax = axes[0, 0]
    home_coefs = [results[p]['home']['raw']['coef_team_score'] for p in positions]
    away_coefs = [results[p]['away']['raw']['coef_team_score'] for p in positions]

    bars1 = ax.bar(x - width/2, home_coefs, width, label='Home', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, away_coefs, width, label='Away', color='blue', alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Coefficient')
    ax.set_title('Raw Points: Team Score Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points", ha='center', fontsize=8)

    # Top right: Margin coefficient (Raw)
    ax = axes[0, 1]
    home_coefs = [results[p]['home']['raw']['coef_margin'] for p in positions]
    away_coefs = [results[p]['away']['raw']['coef_margin'] for p in positions]

    bars1 = ax.bar(x - width/2, home_coefs, width, label='Home', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, away_coefs, width, label='Away', color='blue', alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Coefficient')
    ax.set_title('Raw Points: Relative Margin Effect\n(positive margin = team won)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points", ha='center', fontsize=8)

    # Bottom left: Team Score coefficient (Residual)
    ax = axes[1, 0]
    home_coefs = [results[p]['home']['residual']['coef_team_score'] for p in positions]
    away_coefs = [results[p]['away']['residual']['coef_team_score'] for p in positions]

    bars1 = ax.bar(x - width/2, home_coefs, width, label='Home', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, away_coefs, width, label='Away', color='blue', alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Coefficient')
    ax.set_title('Residual: Team Score Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points", ha='center', fontsize=8)

    # Bottom right: Margin coefficient (Residual)
    ax = axes[1, 1]
    home_coefs = [results[p]['home']['residual']['coef_margin'] for p in positions]
    away_coefs = [results[p]['away']['residual']['coef_margin'] for p in positions]

    bars1 = ax.bar(x - width/2, home_coefs, width, label='Home', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, away_coefs, width, label='Away', color='blue', alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Coefficient')
    ax.set_title('Residual: Relative Margin Effect\n(positive margin = team won)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points", ha='center', fontsize=8)

    plt.suptitle('Regression Coefficients: Home vs Away Players', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_coefficients_home_away.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'regression_coefficients_home_away.png'}")
    plt.close()

    # Figure 2: Scatter plots with overlaid home/away regression lines
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for idx, pos in enumerate(positions):
        # Top row: Points vs Team Score
        ax = axes[0, idx]

        corr_text = []
        for loc, color in [('home', 'red'), ('away', 'blue')]:
            is_home = (loc == 'home')
            pos_df = df[(df['position'] == pos) &
                       (df['expected_pts'] >= 5.0) &
                       (df['fantasy_points_ppr'] > 0) &
                       (df['is_home'] == is_home)].copy()
            pos_df = pos_df.dropna(subset=['team_score', 'fantasy_points_ppr'])

            # Calculate correlation
            r = pos_df['team_score'].corr(pos_df['fantasy_points_ppr'])
            corr_text.append(f'{loc[0].upper()}: r={r:.3f}')

            sample = pos_df.sample(min(1000, len(pos_df)), random_state=42)
            ax.scatter(sample['team_score'], sample['fantasy_points_ppr'],
                       alpha=0.15, s=8, c=color, label=loc.capitalize())

            # Regression line
            coef = results[pos][loc]['raw']
            mean_margin = pos_df['relative_margin'].mean()
            x_line = np.linspace(0, 50, 50)
            y_line = coef['intercept'] + coef['coef_team_score'] * x_line + coef['coef_margin'] * mean_margin
            ax.plot(x_line, y_line, color=color, linewidth=2, linestyle='-')

        ax.set_xlabel('Team Score')
        ax.set_ylabel('Fantasy Points (PPR)')
        ax.set_title(f'{pos}: Team Score Effect\n{", ".join(corr_text)}')
        ax.legend(loc='upper left')

        # Bottom row: Points vs Relative Margin
        ax = axes[1, idx]

        corr_text = []
        for loc, color in [('home', 'red'), ('away', 'blue')]:
            is_home = (loc == 'home')
            pos_df = df[(df['position'] == pos) &
                       (df['expected_pts'] >= 5.0) &
                       (df['fantasy_points_ppr'] > 0) &
                       (df['is_home'] == is_home)].copy()
            pos_df = pos_df.dropna(subset=['relative_margin', 'fantasy_points_ppr'])

            # Calculate correlation
            r = pos_df['relative_margin'].corr(pos_df['fantasy_points_ppr'])
            corr_text.append(f'{loc[0].upper()}: r={r:.3f}')

            sample = pos_df.sample(min(1000, len(pos_df)), random_state=42)
            ax.scatter(sample['relative_margin'], sample['fantasy_points_ppr'],
                       alpha=0.15, s=8, c=color, label=loc.capitalize())

            # Simple univariate trendline (what correlation measures)
            z = np.polyfit(pos_df['relative_margin'], pos_df['fantasy_points_ppr'], 1)
            x_line = np.linspace(-30, 30, 50)
            ax.plot(x_line, z[0] * x_line + z[1], color=color, linewidth=2, linestyle='-')

        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Relative Margin (+ = won)')
        ax.set_ylabel('Fantasy Points (PPR)')
        ax.set_title(f'{pos}: Margin Effect (univariate)\n{", ".join(corr_text)}')
        ax.legend(loc='upper left')

    plt.suptitle('Raw Fantasy Points vs Team Outcomes (Home=Red, Away=Blue)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_scatter_home_away.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'regression_scatter_home_away.png'}")
    plt.close()

    # Figure 3: Same for residuals
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for idx, pos in enumerate(positions):
        # Top row: Residual vs Team Score
        ax = axes[0, idx]

        corr_text = []
        for loc, color in [('home', 'red'), ('away', 'blue')]:
            is_home = (loc == 'home')
            pos_df = df[(df['position'] == pos) &
                       (df['expected_pts'] >= 5.0) &
                       (df['fantasy_points_ppr'] > 0) &
                       (df['is_home'] == is_home)].copy()
            pos_df = pos_df.dropna(subset=['team_score', 'residual'])

            # Calculate correlation
            r = pos_df['team_score'].corr(pos_df['residual'])
            corr_text.append(f'{loc[0].upper()}: r={r:.3f}')

            sample = pos_df.sample(min(1000, len(pos_df)), random_state=42)
            ax.scatter(sample['team_score'], sample['residual'],
                       alpha=0.15, s=8, c=color, label=loc.capitalize())

            # Regression line
            coef = results[pos][loc]['residual']
            mean_margin = pos_df['relative_margin'].mean()
            x_line = np.linspace(0, 50, 50)
            y_line = coef['intercept'] + coef['coef_team_score'] * x_line + coef['coef_margin'] * mean_margin
            ax.plot(x_line, y_line, color=color, linewidth=2, linestyle='-')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Team Score')
        ax.set_ylabel('Residual')
        ax.set_title(f'{pos}: Team Score Effect\n{", ".join(corr_text)}')
        ax.legend(loc='upper left')

        # Bottom row: Residual vs Relative Margin
        ax = axes[1, idx]

        corr_text = []
        for loc, color in [('home', 'red'), ('away', 'blue')]:
            is_home = (loc == 'home')
            pos_df = df[(df['position'] == pos) &
                       (df['expected_pts'] >= 5.0) &
                       (df['fantasy_points_ppr'] > 0) &
                       (df['is_home'] == is_home)].copy()
            pos_df = pos_df.dropna(subset=['relative_margin', 'residual'])

            # Calculate correlation
            r = pos_df['relative_margin'].corr(pos_df['residual'])
            corr_text.append(f'{loc[0].upper()}: r={r:.3f}')

            sample = pos_df.sample(min(1000, len(pos_df)), random_state=42)
            ax.scatter(sample['relative_margin'], sample['residual'],
                       alpha=0.15, s=8, c=color, label=loc.capitalize())

            # Simple univariate trendline (what correlation measures)
            z = np.polyfit(pos_df['relative_margin'], pos_df['residual'], 1)
            x_line = np.linspace(-30, 30, 50)
            ax.plot(x_line, z[0] * x_line + z[1], color=color, linewidth=2, linestyle='-')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Relative Margin (+ = won)')
        ax.set_ylabel('Residual')
        ax.set_title(f'{pos}: Margin Effect (univariate)\n{", ".join(corr_text)}')
        ax.legend(loc='upper left')

    plt.suptitle('Residual vs Team Outcomes (Home=Red, Away=Blue)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_scatter_residual_home_away.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'regression_scatter_residual_home_away.png'}")
    plt.close()

    # Figure 4: Heatmap of predicted residual across team_score/margin space
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    team_score_range = np.linspace(10, 40, 30)
    margin_range = np.linspace(-20, 20, 40)
    TeamScore, Margin = np.meshgrid(team_score_range, margin_range)

    for idx, pos in enumerate(positions):
        # Home heatmap
        ax = axes[0, idx]
        coef = results[pos]['home']['residual']
        Z = coef['intercept'] + coef['coef_team_score'] * TeamScore + coef['coef_margin'] * Margin

        im = ax.contourf(TeamScore, Margin, Z, levels=20, cmap='RdBu_r', vmin=-5, vmax=5)
        plt.colorbar(im, ax=ax, label='Residual')
        ax.axhline(0, color='white', linestyle='--', linewidth=1)
        ax.set_xlabel('Team Score')
        ax.set_ylabel('Relative Margin')
        ax.set_title(f'{pos} HOME: Predicted Residual')

        # Away heatmap
        ax = axes[1, idx]
        coef = results[pos]['away']['residual']
        Z = coef['intercept'] + coef['coef_team_score'] * TeamScore + coef['coef_margin'] * Margin

        im = ax.contourf(TeamScore, Margin, Z, levels=20, cmap='RdBu_r', vmin=-5, vmax=5)
        plt.colorbar(im, ax=ax, label='Residual')
        ax.axhline(0, color='white', linestyle='--', linewidth=1)
        ax.set_xlabel('Team Score')
        ax.set_ylabel('Relative Margin')
        ax.set_title(f'{pos} AWAY: Predicted Residual')

    plt.suptitle('Predicted Residual Across Game Scenarios (Home vs Away)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_heatmaps_home_away.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'regression_heatmaps_home_away.png'}")
    plt.close()


def create_3d_surface_plots(df: pd.DataFrame, results: dict, output_dir: Path, interactive: bool = True):
    """
    Create 3D surface plots showing residual as function of team_score and margin.

    Shows how players perform relative to expectation across different game scenarios:
    - High team score + positive margin = winning shootout
    - High team score + negative margin = losing shootout
    - Low team score + positive margin = defensive win
    - Low team score + negative margin = getting blown out

    Uses Plotly for fast interactive WebGL rendering, matplotlib for static images.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir.mkdir(parents=True, exist_ok=True)
    positions = ['QB', 'RB', 'WR', 'TE']
    colors_plotly = {'QB': 'Reds', 'RB': 'Greens', 'WR': 'Blues', 'TE': 'Oranges'}
    colors_mpl = {'QB': 'red', 'RB': 'green', 'WR': 'blue', 'TE': 'orange'}

    # Create grid for surface
    # Use absolute margin: 0 = close game (shootout potential), 25+ = blowout
    team_score_range = np.linspace(10, 45, 40)
    abs_margin_range = np.linspace(0, 30, 40)  # 0 = close, 30 = blowout
    TeamScore, AbsMargin = np.meshgrid(team_score_range, abs_margin_range)

    # =========================================================================
    # INTERACTIVE PLOTLY VERSION (fast WebGL rendering)
    # =========================================================================
    if interactive:
        # Combined plot with all positions
        fig = go.Figure()

        for pos in positions:
            coef = results[pos]['all']['residual']
            # For absolute margin, we use the magnitude of the margin effect
            # Since margin effect is typically negative for pass-catchers (they do better when losing),
            # we model: close games (abs_margin=0) vs blowouts (abs_margin=30)
            # The coefficient represents change per point of margin, so for absolute margin
            # we take the average of winning and losing scenarios
            # Actually simpler: just use abs(coef_margin) * abs_margin for symmetric effect
            # But the real insight is: low abs_margin = close game = more passing = higher residual for QBs
            # So we want: Z = intercept + coef_team_score * TeamScore + |coef_margin| * (-AbsMargin)
            # This way, low AbsMargin (close game) gives higher Z for positions with negative margin coef
            Z = coef['intercept'] + coef['coef_team_score'] * TeamScore + abs(coef['coef_margin']) * (-AbsMargin)

            fig.add_trace(go.Surface(
                x=team_score_range,
                y=abs_margin_range,
                z=Z,
                name=pos,
                colorscale=colors_plotly[pos],
                opacity=0.7,
                showscale=False,
                hovertemplate=f'{pos}<br>Team Score: %{{x:.0f}}<br>Abs Margin: %{{y:.0f}}<br>Residual: %{{z:.2f}}<extra></extra>'
            ))

        # Add reference plane at z=0
        fig.add_trace(go.Surface(
            x=team_score_range,
            y=abs_margin_range,
            z=np.zeros_like(TeamScore),
            name='Zero Reference',
            colorscale=[[0, 'gray'], [1, 'gray']],
            opacity=0.2,
            showscale=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            title='Position Residuals vs Game Script<br><sub>Front-left = SHOOTOUT (high score + close game) | Back-right = low-scoring blowout</sub>',
            scene=dict(
                xaxis_title='Team Score',
                yaxis_title='Absolute Margin (0=close, 30=blowout)',
                zaxis_title='Residual (Actual - Expected)',
                # Camera: high team score + low abs margin (shootout) in front-left
                camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
                # Make the 3D scene fill more of the layout
                domain=dict(x=[0, 1], y=[0, 1])
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=50, b=0),
            # Auto-size to fill container
            autosize=True
        )

        # Save interactive HTML with full-screen config
        html_path = output_dir / 'regression_3d_interactive.html'
        fig.write_html(
            str(html_path),
            full_html=True,
            include_plotlyjs=True,
            config={'responsive': True}
        )
        print(f"  Saved: {html_path}")

        # Show in browser
        print("  Opening interactive 3D plot in browser...")
        fig.show()

    # =========================================================================
    # STATIC MATPLOTLIB VERSION (for saved images)
    # =========================================================================
    from mpl_toolkits.mplot3d import Axes3D

    # Figure: Individual position 3D plots (2x2 grid)
    fig = plt.figure(figsize=(16, 14))

    for idx, pos in enumerate(positions):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        coef = results[pos]['all']['residual']
        # Same formula as Plotly: close games (low abs margin) benefit pass-catchers
        Z = coef['intercept'] + coef['coef_team_score'] * TeamScore + abs(coef['coef_margin']) * (-AbsMargin)

        # Surface plot
        surf = ax.plot_surface(TeamScore, AbsMargin, Z, cmap='RdBu_r',
                              vmin=-8, vmax=8, alpha=0.8)

        # Add contour lines on the bottom
        ax.contour(TeamScore, AbsMargin, Z, zdir='z', offset=Z.min() - 2,
                  levels=10, cmap='RdBu_r', alpha=0.5)

        # Reference plane at z=0
        ax.plot_surface(TeamScore, AbsMargin, np.zeros_like(TeamScore),
                       alpha=0.15, color='gray')

        ax.set_xlabel('Team Score')
        ax.set_ylabel('Abs Margin (0=close)')
        ax.set_zlabel('Residual')

        # Add key scenario labels
        r2 = coef['r2']
        ax.set_title(f'{pos}: Residual Surface (R²={r2:.3f})\n'
                    f'β(score)={coef["coef_team_score"]:.3f}, β(|margin|)={abs(coef["coef_margin"]):.3f}')

        ax.view_init(elev=25, azim=-135)  # View from opposite corner

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Residual')

    plt.suptitle('Position Residuals: Shootout (high score, close game) vs Blowout\nFront corner = SHOOTOUT', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_3d_individual.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'regression_3d_individual.png'}")
    plt.close()

    # Print shootout scenario analysis
    print("\n  GAME SCRIPT SCENARIO ANALYSIS:")
    print(f"  {'Position':<6} {'Shootout':>12} {'Blowout':>12} {'Difference':>12}")
    print(f"  {'':6} {'(35pts, close)':>12} {'(35pts, +25)':>12} {'':>12}")
    print("  " + "-" * 54)
    for pos in positions:
        coef = results[pos]['all']['residual']
        # Shootout: high score (35), close game (abs_margin=3)
        shootout = coef['intercept'] + coef['coef_team_score'] * 35 + abs(coef['coef_margin']) * (-3)
        # Blowout: high score (35), big margin (abs_margin=25)
        blowout = coef['intercept'] + coef['coef_team_score'] * 35 + abs(coef['coef_margin']) * (-25)
        diff = shootout - blowout
        print(f"  {pos:<6} {shootout:>+12.2f} {blowout:>+12.2f} {diff:>+12.2f}")


def print_regression_table(results: dict):
    """Print formatted regression results table."""

    positions = ['QB', 'RB', 'WR', 'TE']

    for model_type in ['raw', 'residual']:
        print("\n" + "=" * 100)
        if model_type == 'raw':
            print("RAW FANTASY POINTS REGRESSION: points ~ intercept + β₁*team_score + β₂*relative_margin")
        else:
            print("RESIDUAL REGRESSION: residual ~ intercept + β₁*team_score + β₂*relative_margin")
        print("=" * 100)

        for loc in ['home', 'away']:
            print(f"\n{loc.upper()} PLAYERS:")
            print(f"{'Position':<10} {'Intercept':>10} {'β(TeamScore)':>14} {'p-value':>10} {'β(Margin)':>12} {'p-value':>10} {'R²':>8} {'n':>8}")
            print("-" * 100)

            for pos in positions:
                r = results[pos][loc][model_type]
                sig_t = '***' if r['pvalue_team_score'] < 0.001 else '**' if r['pvalue_team_score'] < 0.01 else '*' if r['pvalue_team_score'] < 0.05 else ''
                sig_m = '***' if r['pvalue_margin'] < 0.001 else '**' if r['pvalue_margin'] < 0.01 else '*' if r['pvalue_margin'] < 0.05 else ''
                print(f"{pos:<10} {r['intercept']:>10.3f} {r['coef_team_score']:>12.4f}{sig_t:<2} {r['pvalue_team_score']:>10.4f} {r['coef_margin']:>10.4f}{sig_m:<2} {r['pvalue_margin']:>10.4f} {r['r2']:>8.4f} {r['n']:>8,}")


def main():
    """Run game score regression analysis."""
    print("=" * 70)
    print("GAME SCORE REGRESSION ANALYSIS (Home vs Away)")
    print("=" * 70)
    print("\nModels: performance ~ intercept + β₁*team_score + β₂*relative_margin")
    print("  team_score = points scored by player's team")
    print("  relative_margin = team_score - opponent_score (positive = won)")

    # Load data
    seasons = list(range(2016, 2025))
    players, schedules = load_data(seasons)

    # Prepare game data with team-specific scores
    game_data = prepare_game_data(schedules)

    # Calculate rolling expected
    players_df = calculate_rolling_expected(players)

    # Merge
    merged = merge_data(players_df, game_data)

    # Run regression
    results = run_regression_analysis(merged, min_expected=5.0)

    # Print results
    print_regression_table(results)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    output_dir = Path(__file__).parent / 'charts' / 'regression'
    create_visualizations(merged, results, output_dir)

    # Create 3D surface plots
    print("\n  Creating 3D surface plots...")
    create_3d_surface_plots(merged, results, output_dir, interactive=True)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("\n1. TEAM SCORE EFFECT (per point scored by team):")
    print(f"   {'Position':<6} {'Home':>10} {'Away':>10} {'Diff':>10}")
    print("   " + "-" * 40)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        home = results[pos]['home']['residual']['coef_team_score']
        away = results[pos]['away']['residual']['coef_team_score']
        print(f"   {pos:<6} {home:>+10.4f} {away:>+10.4f} {home-away:>+10.4f}")

    print("\n2. MARGIN EFFECT (per point of win/loss margin):")
    print(f"   {'Position':<6} {'Home':>10} {'Away':>10} {'Diff':>10}")
    print("   " + "-" * 40)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        home = results[pos]['home']['residual']['coef_margin']
        away = results[pos]['away']['residual']['coef_margin']
        print(f"   {pos:<6} {home:>+10.4f} {away:>+10.4f} {home-away:>+10.4f}")

    print("\n3. SCENARIO ANALYSIS (Residual):")
    print("\n   Team scores 28 and WINS by 14 vs Team scores 14 and LOSES by 14:")
    print(f"   {'Position':<6} {'Home Win':>10} {'Home Loss':>10} {'Away Win':>10} {'Away Loss':>10}")
    print("   " + "-" * 55)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        h = results[pos]['home']['residual']
        a = results[pos]['away']['residual']
        home_win = h['intercept'] + h['coef_team_score'] * 28 + h['coef_margin'] * 14
        home_loss = h['intercept'] + h['coef_team_score'] * 14 + h['coef_margin'] * (-14)
        away_win = a['intercept'] + a['coef_team_score'] * 28 + a['coef_margin'] * 14
        away_loss = a['intercept'] + a['coef_team_score'] * 14 + a['coef_margin'] * (-14)
        print(f"   {pos:<6} {home_win:>+10.2f} {home_loss:>+10.2f} {away_win:>+10.2f} {away_loss:>+10.2f}")


if __name__ == '__main__':
    main()
