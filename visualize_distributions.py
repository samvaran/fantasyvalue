#!/usr/bin/env python3
"""
Interactive Player Distribution Visualizer

Visualizes the spliced log-normal distributions for each player.
Shows how consensus, floor, ceiling, and variance parameters create
the final distribution used in simulations.

Usage: python visualize_distributions.py
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import plotly.express as px

# Load player data
print("Loading player data...")
players_df = pd.read_csv('knapsack.csv')
players_df = players_df[
    (players_df['consensus'].notna()) &
    (players_df['salary'].notna()) &
    (players_df['mu'].notna()) &
    (players_df['sigma_lower'].notna()) &
    (players_df['sigma_upper'].notna())
].copy()

print(f"Loaded {len(players_df)} players with complete distribution data\n")


def generate_spliced_distribution(mu, sigma_lower, sigma_upper, n_points=10000):
    """Generate spliced log-normal distribution samples."""
    # Generate uniform samples
    uniform_samples = np.linspace(0.001, 0.999, n_points)

    # Convert to z-scores
    z_scores = stats.norm.ppf(uniform_samples)

    # Use sigma_lower for samples <= 0.5, sigma_upper for samples > 0.5
    sigmas = np.where(uniform_samples <= 0.5, sigma_lower, sigma_upper)

    # Calculate distribution values
    samples = np.exp(mu + z_scores * sigmas)

    return samples, uniform_samples * 100  # Return percentiles


def create_player_chart(player_row):
    """Create distribution chart for a single player."""
    name = player_row['name']
    position = player_row['position']
    team = player_row['team']
    salary = player_row['salary']
    consensus = player_row['consensus']
    mu = player_row['mu']
    sigma_lower = player_row['sigma_lower']
    sigma_upper = player_row['sigma_upper']

    # Generate distribution
    samples, percentiles = generate_spliced_distribution(mu, sigma_lower, sigma_upper)

    # Calculate key percentiles
    p10 = np.percentile(samples, 10)
    p25 = np.percentile(samples, 25)
    p50 = np.percentile(samples, 50)
    p75 = np.percentile(samples, 75)
    p90 = np.percentile(samples, 90)
    p99 = np.percentile(samples, 99)

    # Create figure
    fig = go.Figure()

    # Add distribution line
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=samples,
        mode='lines',
        name='Distribution',
        line=dict(color='royalblue', width=3),
        hovertemplate='<b>Percentile:</b> %{x:.1f}%<br><b>Points:</b> %{y:.2f}<extra></extra>'
    ))

    # Add consensus line
    fig.add_hline(
        y=consensus,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Consensus: {consensus:.1f}",
        annotation_position="right"
    )

    # Add P90 line
    fig.add_hline(
        y=p90,
        line_dash="dash",
        line_color="green",
        annotation_text=f"P90 Ceiling: {p90:.1f}",
        annotation_position="right"
    )

    # Add P10 line (floor)
    fig.add_hline(
        y=p10,
        line_dash="dash",
        line_color="red",
        annotation_text=f"P10 Floor: {p10:.1f}",
        annotation_position="right"
    )

    # Shade regions
    fig.add_vrect(
        x0=0, x1=50,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below", line_width=0,
        annotation_text="Downside (σ_lower)", annotation_position="top left"
    )

    fig.add_vrect(
        x0=50, x1=100,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below", line_width=0,
        annotation_text="Upside (σ_upper)", annotation_position="top right"
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b> ({position} - {team})<br>" +
                 f"<sup>Salary: ${salary:,} | Consensus: {consensus:.1f} pts | " +
                 f"σ_lower: {sigma_lower:.3f} | σ_upper: {sigma_upper:.3f}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Percentile",
        yaxis_title="Fantasy Points",
        hovermode='x unified',
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            range=[0, min(p99 * 1.1, 60)]  # Cap at 60 for readability
        )
    )

    # Add annotations for key percentiles
    annotations_data = [
        (10, p10, f"P10: {p10:.1f}"),
        (25, p25, f"P25: {p25:.1f}"),
        (50, p50, f"P50: {p50:.1f}"),
        (75, p75, f"P75: {p75:.1f}"),
        (90, p90, f"P90: {p90:.1f}"),
    ]

    for percentile, value, text in annotations_data:
        fig.add_annotation(
            x=percentile,
            y=value,
            text=text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='black',
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        )

    return fig


def create_comparison_chart(players_subset):
    """Create comparison chart for multiple players."""
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for idx, (_, player) in enumerate(players_subset.iterrows()):
        name = player['name']
        mu = player['mu']
        sigma_lower = player['sigma_lower']
        sigma_upper = player['sigma_upper']

        samples, percentiles = generate_spliced_distribution(mu, sigma_lower, sigma_upper)

        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=percentiles,
            y=samples,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{name}</b><br>Percentile: %{{x:.1f}}%<br>Points: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title="Player Distribution Comparison",
        xaxis_title="Percentile",
        yaxis_title="Fantasy Points",
        hovermode='x unified',
        height=700,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis=dict(gridcolor='lightgray'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_variance_scatter():
    """Create scatter plot showing floor vs ceiling variance."""
    fig = go.Figure()

    # Color by position
    positions = players_df['position'].unique()
    colors = px.colors.qualitative.Plotly

    for idx, pos in enumerate(positions):
        pos_data = players_df[players_df['position'] == pos]

        fig.add_trace(go.Scatter(
            x=pos_data['sigma_lower'],
            y=pos_data['sigma_upper'],
            mode='markers',
            name=pos,
            marker=dict(
                size=pos_data['salary'] / 500,  # Size by salary
                color=colors[idx % len(colors)],
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=pos_data['name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'σ_lower: %{x:.3f}<br>' +
                         'σ_upper: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

    # Add diagonal line (equal variance)
    max_val = max(players_df['sigma_lower'].max(), players_df['sigma_upper'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Equal Variance',
        showlegend=True
    ))

    fig.update_layout(
        title="Floor Variance (σ_lower) vs Ceiling Variance (σ_upper)<br><sup>Marker size = salary</sup>",
        xaxis_title="σ_lower (Downside Risk)",
        yaxis_title="σ_upper (Upside Potential)",
        hovermode='closest',
        height=700,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


# ============================================================================
# MAIN INTERACTIVE APP
# ============================================================================

print("=" * 80)
print("INTERACTIVE PLAYER DISTRIBUTION VISUALIZER")
print("=" * 80)
print("\nSelect visualization mode:")
print("  1. Single player distribution")
print("  2. Compare multiple players")
print("  3. Variance scatter plot (all players)")
print("  4. Top 10 by ceiling value (comparison)")
print("  5. Top 10 by salary (comparison)")

mode = input("\nEnter mode (1-5): ").strip()

if mode == '1':
    # Single player mode
    print("\nAvailable players:")
    for idx, row in players_df.head(20).iterrows():
        print(f"  {row['name']:<30} {row['position']:<3} ${row['salary']:>5} {row['consensus']:>5.1f} pts")

    search = input("\nEnter player name (partial match): ").strip().lower()
    matches = players_df[players_df['name'].str.lower().str.contains(search)]

    if len(matches) == 0:
        print("No matches found!")
    elif len(matches) == 1:
        player = matches.iloc[0]
        print(f"\nGenerating chart for {player['name']}...")
        fig = create_player_chart(player)
        fig.show()
    else:
        print(f"\nFound {len(matches)} matches:")
        for idx, row in matches.iterrows():
            print(f"  {row['name']:<30} {row['position']:<3} ${row['salary']:>5}")

        exact = input("\nEnter exact player name: ").strip()
        player = players_df[players_df['name'].str.lower() == exact.lower()].iloc[0]
        print(f"\nGenerating chart for {player['name']}...")
        fig = create_player_chart(player)
        fig.show()

elif mode == '2':
    # Comparison mode
    print("\nEnter player names (comma separated):")
    names = input("> ").strip().split(',')
    names = [n.strip().lower() for n in names]

    selected = []
    for name in names:
        matches = players_df[players_df['name'].str.lower().str.contains(name)]
        if len(matches) > 0:
            selected.append(matches.iloc[0])

    if len(selected) > 0:
        selected_df = pd.DataFrame(selected)
        print(f"\nComparing {len(selected)} players...")
        fig = create_comparison_chart(selected_df)
        fig.show()
    else:
        print("No players found!")

elif mode == '3':
    # Variance scatter
    print("\nGenerating variance scatter plot...")
    fig = create_variance_scatter()
    fig.show()

elif mode == '4':
    # Top 10 by ceiling value
    print("\nGenerating top 10 by ceiling value...")

    # Calculate ceiling values
    z90 = 1.2815515655446004
    players_df['p90'] = np.exp(players_df['mu'] + players_df['sigma_upper'] * z90)
    players_df['ceiling_value'] = players_df['p90'] / (players_df['salary'] / 1000)

    top10 = players_df.nlargest(10, 'ceiling_value')

    print("\nTop 10 players:")
    for _, row in top10.iterrows():
        print(f"  {row['name']:<30} {row['position']:<3} ${row['salary']:>5} P90: {row['p90']:>5.1f} Value: {row['ceiling_value']:>5.2f}")

    fig = create_comparison_chart(top10)
    fig.show()

elif mode == '5':
    # Top 10 by salary
    print("\nGenerating top 10 by salary...")
    top10 = players_df.nlargest(10, 'salary')

    print("\nTop 10 players:")
    for _, row in top10.iterrows():
        print(f"  {row['name']:<30} {row['position']:<3} ${row['salary']:>5} {row['consensus']:>5.1f} pts")

    fig = create_comparison_chart(top10)
    fig.show()

else:
    print("Invalid mode!")
