#!/usr/bin/env python3
"""
Generate Interactive Dashboard

Creates a single HTML file with:
1. Filterable player data table from knapsack.csv
2. Top lineups table from LEAGUE_LINEUPS.csv
3. Player distribution visualizations

Usage: python generate_dashboard.py
Output: dashboard.html
"""
import pandas as pd
import json
from pathlib import Path

def generate_html_dashboard():
    """Generate a self-contained HTML dashboard."""

    print("Loading data...")

    # Load player data
    players_df = pd.read_csv('knapsack.csv')
    players_df = players_df.fillna('')

    # Load lineups if available
    lineups_exist = Path('LEAGUE_LINEUPS.csv').exists()
    if lineups_exist:
        lineups_df = pd.read_csv('LEAGUE_LINEUPS.csv')
        lineups_df = lineups_df.fillna('')
    else:
        lineups_df = pd.DataFrame()

    # Load game lines if available
    game_lines_exist = Path('game_lines.csv').exists()
    if game_lines_exist:
        game_lines_df = pd.read_csv('game_lines.csv')
        game_lines_df = game_lines_df.fillna('')
    else:
        game_lines_df = pd.DataFrame()

    # Convert to JSON for embedding
    players_json = players_df.to_json(orient='records')
    lineups_json = lineups_df.to_json(orient='records') if lineups_exist else '[]'
    game_lines_json = game_lines_df.to_json(orient='records') if game_lines_exist else '[]'

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fantasy Football Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}

        h1 {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #334155;
        }}

        .tab {{
            padding: 12px 24px;
            background: transparent;
            border: none;
            color: #94a3b8;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}

        .tab:hover {{
            color: #e2e8f0;
            background: #1e293b;
        }}

        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .filters {{
            background: #1e293b;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}

        .filter-group label {{
            font-size: 0.85em;
            color: #94a3b8;
            font-weight: 500;
        }}

        input[type="text"], select {{
            padding: 8px 12px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.95em;
        }}

        input[type="text"]:focus, select:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .table-container {{
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        thead {{
            background: #334155;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        th {{
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{
            background: #3f4f68;
        }}

        th.sortable::after {{
            content: ' ⇅';
            opacity: 0.3;
        }}

        th.sorted-asc::after {{
            content: ' ↑';
            opacity: 1;
        }}

        th.sorted-desc::after {{
            content: ' ↓';
            opacity: 1;
        }}

        td {{
            padding: 12px 16px;
            border-top: 1px solid #334155;
            font-size: 0.95em;
        }}

        tbody tr {{
            transition: background 0.2s;
        }}

        tbody tr:hover {{
            background: #2d3748;
        }}

        .position-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .position-QB {{ background: #ef4444; color: white; }}
        .position-RB {{ background: #10b981; color: white; }}
        .position-WR {{ background: #3b82f6; color: white; }}
        .position-TE {{ background: #f59e0b; color: white; }}
        .position-D {{ background: #8b5cf6; color: white; }}

        .stat-card {{
            background: #1e293b;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .stat-label {{
            font-size: 0.85em;
            color: #94a3b8;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #667eea;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .lineup-card {{
            background: #1e293b;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }}

        .lineup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #334155;
        }}

        .lineup-title {{
            font-size: 1.2em;
            font-weight: 600;
        }}

        .lineup-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.9em;
        }}

        .lineup-stat {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }}

        .lineup-stat-label {{
            color: #94a3b8;
            font-size: 0.85em;
        }}

        .lineup-stat-value {{
            font-weight: 600;
            color: #667eea;
        }}

        .lineup-players {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}

        .lineup-player {{
            background: #0f172a;
            padding: 10px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .player-info {{
            flex: 1;
        }}

        .player-name {{
            font-weight: 600;
            margin-bottom: 2px;
        }}

        .player-meta {{
            font-size: 0.85em;
            color: #94a3b8;
        }}

        .no-data {{
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
        }}

        .no-data-icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}

        .dist-heatmap {{
            position: relative;
            height: 24px;
            width: 100%;
            min-width: 250px;
            background: #0f172a;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #334155;
        }}

        .dist-gradient {{
            position: absolute;
            height: 100%;
            width: 100%;
        }}

        .dist-marker {{
            position: absolute;
            height: 100%;
            width: 2px;
            background: white;
            box-shadow: 0 0 4px rgba(0,0,0,0.5);
        }}

        .dist-marker.consensus {{
            background: #fbbf24;
            width: 3px;
            z-index: 3;
        }}

        .dist-marker.floor {{
            background: #ef4444;
            z-index: 1;
        }}

        .dist-marker.ceiling {{
            background: #8b5cf6;
            z-index: 2;
        }}

        /* Game Lines Styles */
        .game-container {{
            max-width: 900px;
            margin: 0 auto;
        }}

        .game-row {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.2s;
        }}

        .game-row:hover {{
            border-color: #60a5fa;
            box-shadow: 0 4px 12px rgba(96, 165, 250, 0.1);
        }}

        .game-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            color: #94a3b8;
            font-size: 0.9em;
        }}

        .game-total {{
            font-weight: 600;
            color: #60a5fa;
        }}

        .game-bars {{
            display: flex;
            align-items: center;
            gap: 0;
            margin-bottom: 10px;
        }}

        .team-bar {{
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.95em;
            transition: all 0.2s;
            position: relative;
        }}

        .team-bar.away {{
            background: linear-gradient(to right, #1e293b, #3b82f6);
            color: white;
            justify-content: flex-end;
            padding-right: 15px;
            border-radius: 6px 0 0 6px;
        }}

        .team-bar.home {{
            background: linear-gradient(to left, #1e293b, #8b5cf6);
            color: white;
            justify-content: flex-start;
            padding-left: 15px;
            border-radius: 0 6px 6px 0;
        }}

        .team-bar:hover {{
            opacity: 0.9;
            transform: scale(1.02);
        }}

        .spread-indicator {{
            position: absolute;
            top: -8px;
            background: #fbbf24;
            color: #0f172a;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 700;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}

        .away .spread-indicator {{
            right: 10px;
        }}

        .home .spread-indicator {{
            left: 10px;
        }}

        .game-details {{
            display: flex;
            justify-content: space-around;
            color: #64748b;
            font-size: 0.85em;
            padding-top: 10px;
            border-top: 1px solid #334155;
        }}

        .game-stat {{
            text-align: center;
        }}

        .game-stat-label {{
            display: block;
            margin-bottom: 3px;
            color: #94a3b8;
        }}

        .game-stat-value {{
            font-weight: 600;
            color: #e2e8f0;
        }}

        @media (max-width: 768px) {{
            .filters {{
                flex-direction: column;
                align-items: stretch;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}

            table {{
                font-size: 0.85em;
            }}

            th, td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ Fantasy Football Dashboard</h1>
            <div class="subtitle">Player Analysis & Lineup Optimizer</div>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('players')">📊 Players</button>
            <button class="tab" onclick="switchTab('lineups')">🏆 Lineups</button>
            <button class="tab" onclick="switchTab('gamelines')">🏈 Game Lines</button>
            <button class="tab" onclick="switchTab('methodology')">📖 Methodology</button>
        </div>

        <!-- PLAYERS TAB -->
        <div id="players-tab" class="tab-content active">

            <div class="filters">
                <div class="filter-group">
                    <label>Search Player</label>
                    <input type="text" id="search-input" placeholder="Search by name..." oninput="filterPlayers()">
                </div>
                <div class="filter-group">
                    <label>Position</label>
                    <select id="position-filter" onchange="filterPlayers()">
                        <option value="">All Positions</option>
                        <option value="QB">QB</option>
                        <option value="RB">RB</option>
                        <option value="WR">WR</option>
                        <option value="TE">TE</option>
                        <option value="D">D</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Team</label>
                    <select id="team-filter" onchange="filterPlayers()">
                        <option value="">All Teams</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Min Salary</label>
                    <input type="text" id="min-salary" placeholder="e.g. 7000" oninput="filterPlayers()">
                </div>
                <div class="filter-group">
                    <label>Min Projection</label>
                    <input type="text" id="min-projection" placeholder="e.g. 15" oninput="filterPlayers()">
                </div>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th class="sortable" onclick="sortTable('position')">Pos</th>
                            <th class="sortable" onclick="sortTable('name')">Name</th>
                            <th class="sortable" onclick="sortTable('game')">Matchup</th>
                            <th class="sortable" onclick="sortTable('salary')">Salary</th>
                            <th class="sortable" onclick="sortTable('tdProbability')">TD%</th>
                            <th class="sortable" onclick="sortTable('gameTotal')">O/U</th>
                            <th class="sortable" onclick="sortTable('spread')">Spread</th>
                            <th class="sortable" onclick="sortTable('p10')">P10</th>
                            <th class="sortable" onclick="sortTable('mu')">μ</th>
                            <th class="sortable" onclick="sortTable('p90')">P90</th>
                            <th class="sortable" onclick="sortTable('ceilingValue')">P90 Value</th>
                            <th>Distribution</th>
                        </tr>
                    </thead>
                    <tbody id="players-tbody">
                    </tbody>
                </table>
            </div>
        </div>

        <!-- LINEUPS TAB -->
        <div id="lineups-tab" class="tab-content">
            <div id="lineups-container"></div>
        </div>

        <!-- GAME LINES TAB -->
        <div id="gamelines-tab" class="tab-content">
            <div class="game-container" id="gamelines-container"></div>
        </div>

        <!-- METHODOLOGY TAB -->
        <div id="methodology-tab" class="tab-content">
            <div style="max-width: 1000px; margin: 0 auto;">
                <div style="background: #1e293b; padding: 30px; border-radius: 8px; border-left: 4px solid #667eea;">
                    <h2 style="margin-top: 0; color: #667eea;">Algorithm Methodology</h2>
                    <p style="color: #94a3b8; margin-bottom: 30px;">
                        Our optimization algorithm uses asymmetric log-normal distributions to model player outcomes,
                        accounting for game script, touchdown probability, player archetypes, and correlations.
                    </p>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            1. Data Collection & Consensus Building
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>FantasyPros Projections</strong>: Expert consensus (weight = 2.0)</li>
                            <li><strong>ESPN Outside (IBM Watson AI)</strong>: Advanced projection (weight = 2.0)</li>
                            <li><strong>ESPN Score Projection</strong>: Base ESPN projection (weight = 1.0)</li>
                            <li><strong>ESPN Simulation</strong>: Monte Carlo projection (weight = 1.0)</li>
                            <li><strong>Game Lines Data</strong>: Spread, over/under, implied team totals</li>
                            <li><strong>TD Odds</strong>: Player-specific touchdown probability from betting markets</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            2. Player Archetype Calculation
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Floor Variance</strong>: Calculated from (consensus - ESPN Low) / consensus
                                <ul style="margin-top: 5px;">
                                    <li>Range: 0.1 to 1.0</li>
                                    <li>Low variance (0.1-0.3) = Stable floor, can't bust hard</li>
                                    <li>High variance (0.6-1.0) = Risky floor, can bust hard</li>
                                </ul>
                            </li>
                            <li><strong>Ceiling Variance</strong>: Calculated from (ESPN High - consensus) / consensus
                                <ul style="margin-top: 5px;">
                                    <li>Range: 0.3 to 2.0</li>
                                    <li>Low variance (0.3-0.5) = Limited upside</li>
                                    <li>High variance (1.0-2.0) = Explosive upside potential</li>
                                </ul>
                            </li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            3. Game Script Adjustments
                        </h3>
                        <p style="color: #cbd5e1; margin-bottom: 15px;">
                            <strong>Game Pace Factor</strong>: Scales TD probability based on game total
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Formula: <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">0.85 + (total - 42) × 0.025</code></li>
                            <li>Range: 0.8 to 1.2</li>
                            <li>High-scoring games (52+) = 1.1x pace factor</li>
                            <li>Low-scoring games (42-) = 0.9x pace factor</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>Running Backs (Floor Adjustments)</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Trailing by 3+ (spread < -3): Floor penalty = floorVariance × 0.20 (up to -20%)</li>
                            <li>Favored by 3+ (spread > 3): Floor boost = (1 - floorVariance) × 0.05 (up to +5%)</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>Running Backs (Ceiling Adjustments)</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Favored by 3+ (spread > 3): Ceiling boost = ceilingVariance × 0.15 (up to +30%)</li>
                            <li>High total (total > 50): Shootout boost = ceilingVariance × 0.10 (up to +20%)</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>WR/TE (Floor Adjustments)</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Trailing by 3+ (spread < -3): Floor boost = (1 - floorVariance) × 0.08 (up to +8%)</li>
                            <li>Heavy favorite (spread > 7): Floor penalty = floorVariance × 0.12 (up to -12%)</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>WR/TE (Ceiling Adjustments)</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Trailing by 3+ (spread < -3): Volume boost = ceilingVariance × 0.18 (up to +36%)</li>
                            <li>Heavy favorite (spread > 7): Ceiling penalty = ceilingVariance × 0.10 (up to -20%)</li>
                            <li>High total (total > 50): Shootout boost = ceilingVariance × 0.20 (up to +40%)</li>
                            <li>Low total (total < 44): Low scoring penalty = ceilingVariance × 0.15 (up to -30%)</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>Quarterbacks</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>High total (total > 50): Ceiling × 1.12</li>
                            <li>Low total (total < 42): Ceiling × 0.92</li>
                        </ul>

                        <p style="color: #cbd5e1; margin: 20px 0 15px 0;">
                            <strong>Defenses</strong>:
                        </p>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li>Opponent < 18 pts: Ceiling × 1.3, Floor × 1.15</li>
                            <li>Opponent > 26 pts: Ceiling × 0.75, Floor × 0.70</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            4. Touchdown Probability Integration
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Floor TD Boost</strong> (RB/WR/TE):
                                <ul style="margin-top: 5px;">
                                    <li>Formula: <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">(TD% / 100) × 0.05 × (1 - floorVariance)</code></li>
                                    <li>Maximum: +5% for high-probability, stable-floor players</li>
                                </ul>
                            </li>
                            <li><strong>Ceiling TD Boost</strong> (RB/WR/TE):
                                <ul style="margin-top: 5px;">
                                    <li>Formula: <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">(TD% / 100) × 0.50 × (1 + ceilingVariance × 0.4) × gamePaceFactor</code></li>
                                    <li>Maximum: +50% for high-probability, explosive players in high-pace games</li>
                                </ul>
                            </li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            5. Spliced Log-Normal Distribution Fitting
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>μ (mu)</strong>: Location parameter = ln(consensus)</li>
                            <li><strong>σ_lower</strong>: Downside scale parameter (P0-P50)
                                <ul style="margin-top: 5px;">
                                    <li>Formula: <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">(μ - ln(floor)) / 1.2816</code></li>
                                    <li>Controls bust potential below median</li>
                                </ul>
                            </li>
                            <li><strong>σ_upper</strong>: Upside scale parameter (P50-P100)
                                <ul style="margin-top: 5px;">
                                    <li>Formula: <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">(ln(ceiling) - μ) / 1.2816</code></li>
                                    <li>Controls boom potential above median</li>
                                </ul>
                            </li>
                            <li><strong>P10</strong> (Floor): <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">exp(μ - 1.2816 × σ_lower)</code></li>
                            <li><strong>P50</strong> (Median): <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">exp(μ)</code></li>
                            <li><strong>P90</strong> (Ceiling): <code style="background: #0f172a; padding: 2px 6px; border-radius: 3px;">exp(μ + 1.2816 × σ_upper)</code></li>
                            <li><strong>Z-score constant</strong>: 1.2815515655446004 (90th/10th percentile)</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            6. Monte Carlo Simulation with Correlations
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Same-Team QB-Skill Correlation</strong>: +0.3 to +0.5
                                <ul style="margin-top: 5px;">
                                    <li>QB boom → teammates more likely to boom</li>
                                </ul>
                            </li>
                            <li><strong>Same-Game Opposing Players</strong>: +0.15 to +0.25
                                <ul style="margin-top: 5px;">
                                    <li>Shootouts benefit both teams</li>
                                </ul>
                            </li>
                            <li><strong>Same-Position Same-Team</strong>: -0.1 to -0.3
                                <ul style="margin-top: 5px;">
                                    <li>RB1 boom → RB2 more likely to bust (shared workload)</li>
                                </ul>
                            </li>
                            <li><strong>Simulation Count</strong>: 10,000 iterations per lineup</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #e2e8f0; border-bottom: 2px solid #334155; padding-bottom: 10px;">
                            7. Lineup Optimization
                        </h3>
                        <ul style="color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Method</strong>: Mixed-Integer Linear Programming (MILP)</li>
                            <li><strong>Objective</strong>: Maximize simulated P90 score</li>
                            <li><strong>Constraints</strong>:
                                <ul style="margin-top: 5px;">
                                    <li>Salary cap: $60,000</li>
                                    <li>Roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 D/ST</li>
                                    <li>Uniqueness: Each lineup differs by ≥1 player</li>
                                </ul>
                            </li>
                            <li><strong>Stack Preference</strong>: QB + same-team skill players (correlation bonus)</li>
                        </ul>
                    </div>

                    <div style="background: #0f172a; padding: 20px; border-radius: 6px; border-left: 3px solid #60a5fa;">
                        <h4 style="color: #60a5fa; margin-top: 0;">Key Innovation: Asymmetric Distributions</h4>
                        <p style="color: #cbd5e1; margin: 0;">
                            Unlike traditional symmetric models, our spliced log-normal approach captures the reality
                            that fantasy outcomes are rarely symmetric. A "boom-bust" WR has tight downside (catches = floor)
                            but explosive upside (long TDs). This is modeled with σ_lower ≈ 0.3 (stable floor) and
                            σ_upper ≈ 1.5 (explosive ceiling). Game script and TD probability further adjust these parameters
                            to reflect matchup-specific opportunities.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data
        const playersData = {players_json};
        const lineupsData = {lineups_json};
        const gameLinesData = {game_lines_json};

        // Create game lines lookup by team
        const gameLinesByTeam = {{}};
        gameLinesData.forEach(line => {{
            gameLinesByTeam[line.team_abbr] = {{
                total: line.total,
                spread: line.spread
            }};
        }});

        let currentSort = {{ column: 'ceilingValue', direction: 'desc' }};
        let filteredPlayers = [...playersData];

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            initializeFilters();
            renderPlayers();
            renderLineups();
            renderGameLines();
            updateStats();
        }});

        function initializeFilters() {{
            // Populate team filter
            const teams = [...new Set(playersData.map(p => p.team))].filter(t => t).sort();
            const teamFilter = document.getElementById('team-filter');
            teams.forEach(team => {{
                const option = document.createElement('option');
                option.value = team;
                option.textContent = team;
                teamFilter.appendChild(option);
            }});
        }}

        function renderGameLines() {{
            const container = document.getElementById('gamelines-container');

            if (!gameLinesData || gameLinesData.length === 0) {{
                container.innerHTML = `
                    <div class="no-data">
                        <div class="no-data-icon">🏈</div>
                        <h2>No Game Lines Available</h2>
                        <p>Run <code>node fetch_data.js</code> to fetch game lines</p>
                    </div>
                `;
                return;
            }}

            // Group games by matchup (pair away/home teams)
            const games = {{}};
            gameLinesData.forEach(team => {{
                const gameKey = [team.team_abbr, team.opponent_abbr].sort().join('-');
                if (!games[gameKey]) {{
                    games[gameKey] = {{}};
                }}
                // Determine if this team is away or home based on spread sign
                const isAway = team.spread > 0;
                if (isAway) {{
                    games[gameKey].away = team;
                }} else {{
                    games[gameKey].home = team;
                }}
            }});

            // Convert to array and sort by total descending
            const gamesArray = Object.values(games)
                .filter(g => g.away && g.home)
                .sort((a, b) => parseFloat(b.away.total || 0) - parseFloat(a.away.total || 0));

            container.innerHTML = '';

            gamesArray.forEach(game => {{
                const away = game.away;
                const home = game.home;
                const total = parseFloat(away.total || 0);
                const awayProj = parseFloat(away.projected_pts || 0);
                const homeProj = parseFloat(home.projected_pts || 0);
                const spread = Math.abs(parseFloat(away.spread || 0));

                // Calculate bar widths as percentages (based on projected points)
                const awayWidth = (awayProj / total) * 100;
                const homeWidth = (homeProj / total) * 100;

                const gameRow = document.createElement('div');
                gameRow.className = 'game-row';

                gameRow.innerHTML = `
                    <div class="game-header">
                        <span>${{away.team_abbr}} @ ${{home.team_abbr}}</span>
                        <span class="game-total">O/U: ${{total.toFixed(1)}}</span>
                    </div>
                    <div class="game-bars">
                        <div class="team-bar away" style="width: ${{awayWidth}}%;">
                            <span class="spread-indicator">${{away.spread > 0 ? '+' : ''}}${{away.spread}}</span>
                            ${{away.team_abbr}} ${{awayProj.toFixed(1)}}
                        </div>
                        <div class="team-bar home" style="width: ${{homeWidth}}%;">
                            <span class="spread-indicator">${{home.spread > 0 ? '+' : ''}}${{home.spread}}</span>
                            ${{home.team_abbr}} ${{homeProj.toFixed(1)}}
                        </div>
                    </div>
                    <div class="game-details">
                        <div class="game-stat">
                            <span class="game-stat-label">Spread</span>
                            <span class="game-stat-value">${{home.team_abbr}} ${{home.spread > 0 ? '+' : ''}}${{home.spread}}</span>
                        </div>
                        <div class="game-stat">
                            <span class="game-stat-label">Total</span>
                            <span class="game-stat-value">${{total.toFixed(1)}}</span>
                        </div>
                        <div class="game-stat">
                            <span class="game-stat-label">Implied Total</span>
                            <span class="game-stat-value">${{awayProj.toFixed(1)}} - ${{homeProj.toFixed(1)}}</span>
                        </div>
                    </div>
                `;

                container.appendChild(gameRow);
            }});
        }}

        function switchTab(tab) {{
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tab + '-tab').classList.add('active');
        }}

        function filterPlayers() {{
            const search = document.getElementById('search-input').value.toLowerCase();
            const position = document.getElementById('position-filter').value;
            const team = document.getElementById('team-filter').value;
            const minSalary = parseFloat(document.getElementById('min-salary').value) || 0;
            const minProjection = parseFloat(document.getElementById('min-projection').value) || 0;

            filteredPlayers = playersData.filter(player => {{
                if (search && !player.name.toLowerCase().includes(search)) return false;
                if (position && player.position !== position) return false;
                if (team && player.team !== team) return false;
                if (player.salary < minSalary) return false;
                if ((player.consensus || 0) < minProjection) return false;
                return true;
            }});

            renderPlayers();
            updateStats();
        }}

        function sortTable(column) {{
            if (currentSort.column === column) {{
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSort.column = column;
                currentSort.direction = 'desc';
            }}

            filteredPlayers.sort((a, b) => {{
                let aVal = a[column];
                let bVal = b[column];

                // Handle empty/null values
                if (aVal === null || aVal === undefined || aVal === '') aVal = -Infinity;
                if (bVal === null || bVal === undefined || bVal === '') bVal = -Infinity;

                // Compare
                if (typeof aVal === 'string') {{
                    return currentSort.direction === 'asc'
                        ? aVal.localeCompare(bVal)
                        : bVal.localeCompare(aVal);
                }} else {{
                    return currentSort.direction === 'asc'
                        ? aVal - bVal
                        : bVal - aVal;
                }}
            }});

            renderPlayers();
            updateSortIndicators();
        }}

        function updateSortIndicators() {{
            document.querySelectorAll('th.sortable').forEach(th => {{
                th.classList.remove('sorted-asc', 'sorted-desc');
            }});

            // Map column names to header text patterns
            const columnToHeaderMap = {{
                'position': 'pos',
                'name': 'name',
                'game': 'matchup',
                'salary': 'salary',
                'tdProbability': 'td%',
                'gameTotal': 'o/u',
                'spread': 'spread',
                'p10': 'p10',
                'mu': 'μ',
                'p90': 'p90',
                'ceilingValue': 'p90 value'
            }};

            const column = currentSort.column;
            const headerPattern = columnToHeaderMap[column] || column.substring(0, 3);
            const th = Array.from(document.querySelectorAll('th.sortable'))
                .find(t => t.textContent.toLowerCase().includes(headerPattern.toLowerCase()));

            if (th) {{
                th.classList.add('sorted-' + currentSort.direction);
            }}
        }}

        function generateDistributionHeatmap(player, globalMin, globalMax) {{
            const p10 = parseFloat(player.p10) || 0;
            const consensus = parseFloat(player.consensus) || 0;
            const p90 = parseFloat(player.p90) || 0;
            const mu = parseFloat(player.mu);
            const sigmaLower = parseFloat(player.sigma_lower);
            const sigmaUpper = parseFloat(player.sigma_upper);

            if (!p10 || !consensus || !p90 || !globalMin || !globalMax || !mu || !sigmaLower || !sigmaUpper) {{
                return '<div class="dist-heatmap"></div>';
            }}

            const range = globalMax - globalMin;
            if (range === 0) return '<div class="dist-heatmap"></div>';

            // Sample the log-normal PDF at multiple points to create gradient
            const numSamples = 50;
            const samples = [];

            for (let i = 0; i < numSamples; i++) {{
                const percent = i / (numSamples - 1);
                const x = globalMin + percent * range;

                if (x <= 0) {{
                    samples.push({{ percent, density: 0 }});
                    continue;
                }}

                // Calculate log-normal PDF at x
                // Use spliced distribution: sigma_lower for x < median, sigma_upper for x >= median
                const median = Math.exp(mu);
                const sigma = x < median ? sigmaLower : sigmaUpper;

                // Log-normal PDF: (1 / (x * sigma * sqrt(2π))) * exp(-((ln(x) - mu)^2) / (2 * sigma^2))
                const lnX = Math.log(x);
                const exponent = -Math.pow(lnX - mu, 2) / (2 * sigma * sigma);
                const density = (1 / (x * sigma * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);

                samples.push({{ percent, density }});
            }}

            // Normalize densities to 0-1 range for opacity
            const maxDensity = Math.max(...samples.map(s => s.density));
            const normalizedSamples = samples.map(s => ({{
                percent: s.percent,
                opacity: maxDensity > 0 ? s.density / maxDensity : 0
            }}));

            // Create CSS gradient with opacity representing density
            const gradientStops = normalizedSamples.map(s =>
                `rgba(96, 165, 250, ${{s.opacity.toFixed(3)}}) ${{(s.percent * 100).toFixed(1)}}%`
            ).join(', ');

            // Calculate marker positions
            const p10Pos = ((p10 - globalMin) / range) * 100;
            const consensusPos = ((consensus - globalMin) / range) * 100;
            const p90Pos = ((p90 - globalMin) / range) * 100;

            return `
                <div class="dist-heatmap">
                    <div class="dist-gradient" style="background: linear-gradient(to right, ${{gradientStops}});"></div>
                    <div class="dist-marker floor" style="left: ${{p10Pos}}%;" title="Floor (P10): ${{p10.toFixed(1)}}"></div>
                    <div class="dist-marker consensus" style="left: ${{consensusPos}}%;" title="Consensus: ${{consensus.toFixed(1)}}"></div>
                    <div class="dist-marker ceiling" style="left: ${{p90Pos}}%;" title="Ceiling (P90): ${{p90.toFixed(1)}}"></div>
                </div>
            `;
        }}


        function renderPlayers() {{
            const tbody = document.getElementById('players-tbody');
            tbody.innerHTML = '';

            // Calculate global min/max across all filtered players for consistent scale
            let globalMin = Infinity;
            let globalMax = -Infinity;

            filteredPlayers.forEach(player => {{
                const p10 = parseFloat(player.p10) || 0;
                const p90 = parseFloat(player.p90) || 0;
                if (p10 < globalMin) globalMin = p10;
                if (p90 > globalMax) globalMax = p90;
            }});

            // Handle edge cases
            if (!isFinite(globalMin)) globalMin = 0;
            if (!isFinite(globalMax)) globalMax = 50;

            filteredPlayers.forEach(player => {{
                const row = document.createElement('tr');
                const p10 = parseFloat(player.p10) || 0;
                const mu = parseFloat(player.mu) || 0;
                const p90 = parseFloat(player.p90) || 0;

                const distHeatmap = generateDistributionHeatmap(player, globalMin, globalMax);

                // Get game line data for this player's team
                const gameLine = gameLinesByTeam[player.team] || {{}};
                const gameTotal = gameLine.total ? gameLine.total.toFixed(1) : '-';
                const spread = gameLine.spread ? (gameLine.spread > 0 ? '+' : '') + gameLine.spread : '-';

                // Column order: Pos | Name | Matchup | Salary | TD% | O/U | Spread | P10 | μ | P90 | P90 Value | Distribution
                row.innerHTML = `
                    <td><span class="position-badge position-${{player.position}}">${{player.position}}</span></td>
                    <td>${{player.name || '-'}}</td>
                    <td>${{player.game || '-'}}</td>
                    <td>$${{(player.salary || 0).toLocaleString()}}</td>
                    <td>${{player.tdProbability ? parseFloat(player.tdProbability).toFixed(1) + '%' : '-'}}</td>
                    <td>${{gameTotal}}</td>
                    <td>${{spread}}</td>
                    <td>${{p10.toFixed(1)}}</td>
                    <td>${{mu.toFixed(3)}}</td>
                    <td>${{p90.toFixed(1)}}</td>
                    <td>${{(player.ceilingValue || 0).toFixed(2)}}</td>
                    <td>${{distHeatmap}}</td>
                `;
                tbody.appendChild(row);
            }});
        }}

        function updateStats() {{
            const total = filteredPlayers.length;
            const avgSalary = filteredPlayers.reduce((sum, p) => sum + (p.salary || 0), 0) / total;
            const avgProj = filteredPlayers.reduce((sum, p) => sum + (p.consensus || 0), 0) / total;
            const avgP90 = filteredPlayers.reduce((sum, p) => sum + (p.p90 || 0), 0) / total;

            document.getElementById('total-players').textContent = total;
            document.getElementById('avg-salary').textContent = '$' + Math.round(avgSalary).toLocaleString();
            document.getElementById('avg-projection').textContent = avgProj.toFixed(1);
            document.getElementById('avg-p90').textContent = avgP90.toFixed(1);
        }}

        function renderLineups() {{
            const container = document.getElementById('lineups-container');

            if (!lineupsData || lineupsData.length === 0) {{
                container.innerHTML = `
                    <div class="no-data">
                        <div class="no-data-icon">📋</div>
                        <h2>No Lineups Generated Yet</h2>
                        <p>Run <code>python league_optimizer.py</code> to generate optimal lineups</p>
                    </div>
                `;
                return;
            }}

            container.innerHTML = '';

            // Display top lineups (limit to 20 for performance)
            const topLineups = lineupsData.slice(0, 20);

            topLineups.forEach((lineup, index) => {{
                const card = document.createElement('div');
                card.className = 'lineup-card';

                // Extract player names from individual columns (player_1_qb, player_2_rb1, etc.)
                const playerNames = [];
                Object.keys(lineup).forEach(key => {{
                    if (key.startsWith('player_')) {{
                        const name = lineup[key];
                        if (name) playerNames.push(name);
                    }}
                }});

                const playerDetails = playerNames.map(name => {{
                    const player = playersData.find(p => p.name.toLowerCase() === name.toLowerCase());
                    return player || {{ name, position: '?', salary: 0, consensus: 0 }};
                }});

                const totalSalary = lineup.salary || playerDetails.reduce((sum, p) => sum + (p.salary || 0), 0);
                const totalProj = lineup.consensus_total || playerDetails.reduce((sum, p) => sum + (p.consensus || 0), 0);

                card.innerHTML = `
                    <div class="lineup-header">
                        <div class="lineup-title">Lineup #${{index + 1}}</div>
                        <div class="lineup-stats">
                            <div class="lineup-stat">
                                <span class="lineup-stat-label">Salary</span>
                                <span class="lineup-stat-value">$${{totalSalary.toLocaleString()}}</span>
                            </div>
                            <div class="lineup-stat">
                                <span class="lineup-stat-label">Proj Points</span>
                                <span class="lineup-stat-value">${{totalProj.toFixed(1)}}</span>
                            </div>
                            <div class="lineup-stat">
                                <span class="lineup-stat-label">Sim P90</span>
                                <span class="lineup-stat-value">${{(lineup.sim_p90 || 0).toFixed(1)}}</span>
                            </div>
                            <div class="lineup-stat">
                                <span class="lineup-stat-label">Mean</span>
                                <span class="lineup-stat-value">${{(lineup.sim_mean || 0).toFixed(1)}}</span>
                            </div>
                        </div>
                    </div>
                    <div class="lineup-players">
                        ${{playerDetails.map(p => `
                            <div class="lineup-player">
                                <div class="player-info">
                                    <div class="player-name">${{p.name}}</div>
                                    <div class="player-meta">
                                        <span class="position-badge position-${{p.position}}">${{p.position}}</span>
                                        $${{(p.salary || 0).toLocaleString()}} • ${{(p.consensus || 0).toFixed(1)}} pts
                                    </div>
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                `;

                container.appendChild(card);
            }});
        }}
    </script>
</body>
</html>
"""

    # Write to file
    output_file = 'dashboard.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Dashboard generated: {output_file}")
    print(f"  - {len(players_df)} players loaded")
    if lineups_exist:
        print(f"  - {len(lineups_df)} lineups loaded")
    else:
        print(f"  - No lineups found (run league_optimizer.py to generate)")
    print(f"\nOpen {output_file} in your browser to view the dashboard!")

if __name__ == '__main__':
    generate_html_dashboard()
