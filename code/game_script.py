"""
Continuous game script scoring system.

Instead of bucketing games into discrete categories, this calculates
continuous scores (0-1) for each game script type:
- Shootout probability
- Defensive probability
- Blowout probability
- Competitive probability

This allows for more nuanced analysis - a game can be 70% shootout, 20% competitive, etc.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import math


def sigmoid(x: float, midpoint: float = 0, steepness: float = 1) -> float:
    """
    Smooth sigmoid function for continuous transitions.

    Args:
        x: Input value
        midpoint: Center point of sigmoid (where output = 0.5)
        steepness: How steep the transition (higher = more binary)

    Returns:
        Value between 0 and 1
    """
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def calculate_shootout_score(total: float, spread: float, total_over_odds: int) -> float:
    """
    Calculate continuous shootout probability (0-1).

    High shootout indicators:
    - High total (>50 is strong, >48 is moderate)
    - Competitive spread (closer to 0 is better, >7 reduces score)
    - Market favors over (more negative odds = higher probability)

    Args:
        total: Game total points
        spread: Absolute value of spread
        total_over_odds: Over odds (e.g., -115)

    Returns:
        Shootout score 0-1 (1 = definite shootout)
    """
    abs_spread = abs(spread)

    # Total component (0-1)
    # Center at 48, steep transition
    # >52 = 0.95+, <44 = 0.05-
    total_score = sigmoid(total, midpoint=48, steepness=0.5)

    # Spread component (0-1)
    # Close games (spread < 3) = 1.0
    # Medium spreads (3-7) = 0.5-0.7
    # Large spreads (>10) = 0.2-
    spread_score = 1 - sigmoid(abs_spread, midpoint=5, steepness=0.4)

    # Market sentiment (0-1)
    # Convert over odds to probability
    if total_over_odds < 0:
        over_prob = abs(total_over_odds) / (abs(total_over_odds) + 100)
    else:
        over_prob = 100 / (total_over_odds + 100)

    # Scale so 0.5 (balanced) = 0.5, >0.53 starts favoring shootout
    market_score = sigmoid(over_prob, midpoint=0.50, steepness=20)

    # Weighted combination
    # Total is most important (50%), spread (30%), market (20%)
    shootout_score = (
        total_score * 0.50 +
        spread_score * 0.30 +
        market_score * 0.20
    )

    return shootout_score


def calculate_defensive_score(total: float, total_under_odds: int) -> float:
    """
    Calculate continuous defensive game probability (0-1).

    High defensive indicators:
    - Low total (<42 is strong, <45 is moderate)
    - Market favors under

    Args:
        total: Game total points
        total_under_odds: Under odds (e.g., -110)

    Returns:
        Defensive score 0-1 (1 = definite defensive slugfest)
    """
    # Total component (inverted - lower is better)
    # Center at 43, <38 = 0.95+, >48 = 0.05-
    total_score = 1 - sigmoid(total, midpoint=43, steepness=0.4)

    # Market sentiment
    if total_under_odds < 0:
        under_prob = abs(total_under_odds) / (abs(total_under_odds) + 100)
    else:
        under_prob = 100 / (total_under_odds + 100)

    market_score = sigmoid(under_prob, midpoint=0.50, steepness=20)

    # Weighted combination (total matters more)
    defensive_score = total_score * 0.7 + market_score * 0.3

    return defensive_score


def calculate_blowout_score(
    spread: float,
    moneyline_fav: int,
    spread_odds_fav: int,
    spread_odds_dog: int
) -> Tuple[float, float]:
    """
    Calculate continuous blowout probability (0-1) and confidence.

    High blowout indicators:
    - Large spread (>10 is strong, >7 is moderate)
    - Large moneyline gap (>500 is strong)
    - Balanced spread odds (market agrees)

    Args:
        spread: Favorite's spread (negative)
        moneyline_fav: Favorite's moneyline (negative)
        spread_odds_fav: Favorite's spread odds
        spread_odds_dog: Underdog's spread odds

    Returns:
        Tuple of (blowout_score, confidence_score)
        - blowout_score: 0-1 (1 = definite blowout)
        - confidence_score: 0-1 (1 = market very confident, 0 = "trap game")
    """
    abs_spread = abs(spread)

    # Spread magnitude component
    # Center at 7, >12 = 0.9+, <3 = 0.1-
    spread_score = sigmoid(abs_spread, midpoint=7, steepness=0.4)

    # Moneyline magnitude component
    abs_ml = abs(moneyline_fav)
    # >600 = 0.9+, <200 = 0.2-
    ml_score = sigmoid(abs_ml, midpoint=350, steepness=0.008)

    # Confidence: how much do spread odds agree?
    # Balanced odds (-110/-110) = high confidence
    # Imbalanced odds (-120/+100) = low confidence / trap game
    fav_prob = abs(spread_odds_fav) / (abs(spread_odds_fav) + 100) if spread_odds_fav < 0 else 100 / (spread_odds_fav + 100)
    dog_prob = abs(spread_odds_dog) / (abs(spread_odds_dog) + 100) if spread_odds_dog < 0 else 100 / (spread_odds_dog + 100)

    # Measure how balanced the market is
    # Both around 0.52 = balanced = confident
    # One at 0.55, other at 0.48 = imbalanced = uncertain
    balance = 1 - abs(fav_prob - dog_prob) / 0.1  # Normalize to 0-1
    confidence_score = max(0, min(1, balance))

    # Blowout score is combination of spread and moneyline
    blowout_score = spread_score * 0.6 + ml_score * 0.4

    return blowout_score, confidence_score


def calculate_competitive_score(spread: float) -> float:
    """
    Calculate how competitive/close the game is expected to be (0-1).

    Args:
        spread: Absolute value of spread

    Returns:
        Competitive score 0-1 (1 = very close game)
    """
    abs_spread = abs(spread)

    # Inverted sigmoid - closer to 0 is better
    # <3 = 0.9+, >7 = 0.2-
    competitive_score = 1 - sigmoid(abs_spread, midpoint=4, steepness=0.5)

    return competitive_score


def analyze_game_continuous(game1: Dict, game2: Dict) -> Dict:
    """
    Perform continuous game script analysis on a game.

    Args:
        game1: First team's game data
        game2: Second team's game data

    Returns:
        Dict with continuous scores for each game script type
    """
    # Determine favorite
    if game1['spread'] < 0:
        fav = game1
        dog = game2
    else:
        fav = game2
        dog = game1

    # Calculate all scores
    shootout_score = calculate_shootout_score(
        fav['total'],
        fav['spread'],
        fav['total_over_odds']
    )

    defensive_score = calculate_defensive_score(
        fav['total'],
        fav['total_under_odds']
    )

    blowout_score, blowout_confidence = calculate_blowout_score(
        fav['spread'],
        fav['moneyline'],
        fav['spread_odds'],
        dog['spread_odds']
    )

    competitive_score = calculate_competitive_score(fav['spread'])

    # Normalize scores to sum to 1 (treat as probabilities)
    total = shootout_score + defensive_score + blowout_score + competitive_score

    if total > 0:
        shootout_prob = shootout_score / total
        defensive_prob = defensive_score / total
        blowout_prob = blowout_score / total
        competitive_prob = competitive_score / total
    else:
        # Fallback
        shootout_prob = defensive_prob = blowout_prob = competitive_prob = 0.25

    # Determine primary game script (highest probability)
    scores = {
        'shootout': shootout_prob,
        'defensive': defensive_prob,
        'blowout': blowout_prob,
        'competitive': competitive_prob
    }
    primary_script = max(scores, key=scores.get)

    return {
        'game_id': f"{dog['team_abbr']}@{fav['team_abbr']}",
        'favorite': fav['team'],
        'underdog': dog['team'],
        'spread': fav['spread'],
        'total': fav['total'],

        # Continuous scores (0-1, sum to 1)
        'shootout_prob': shootout_prob,
        'defensive_prob': defensive_prob,
        'blowout_prob': blowout_prob,
        'competitive_prob': competitive_prob,

        # Raw scores (before normalization)
        'shootout_score_raw': shootout_score,
        'defensive_score_raw': defensive_score,
        'blowout_score_raw': blowout_score,
        'competitive_score_raw': competitive_score,

        # Metadata
        'primary_script': primary_script,
        'blowout_confidence': blowout_confidence,
        'script_strength': max(scores.values()),  # How dominant is primary script
    }


def main():
    print("=" * 80)
    print("CONTINUOUS GAME SCRIPT ANALYSIS")
    print("=" * 80)

    # Load game lines (from main data pipeline)
    csv_path = Path('game_lines.csv')
    if not csv_path.exists():
        print(f"Error: {csv_path.name} not found")
        print("Please run fetch_data.py first to generate game lines")
        return

    df = pd.read_csv(csv_path)
    games = df.to_dict('records')

    print(f"\nAnalyzing {len(games) // 2} games...")

    # Analyze each game
    results = []
    for i in range(0, len(games), 2):
        result = analyze_game_continuous(games[i], games[i+1])
        results.append(result)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Sort by shootout probability (descending)
    results_df = results_df.sort_values('shootout_prob', ascending=False)

    # Display results
    print("\n" + "=" * 80)
    print("GAMES RANKED BY SHOOTOUT PROBABILITY")
    print("=" * 80)

    for idx, row in results_df.iterrows():
        print(f"\n{row['game_id']}: {row['underdog']} @ {row['favorite']}")
        print(f"  Spread: {row['spread']:.1f} | Total: {row['total']:.1f}")
        print(f"  Primary Script: {row['primary_script'].upper()} ({row['script_strength']:.1%} strength)")
        print(f"  Probabilities:")
        print(f"    Shootout:    {row['shootout_prob']:>6.1%}  {'█' * int(row['shootout_prob'] * 40)}")
        print(f"    Defensive:   {row['defensive_prob']:>6.1%}  {'█' * int(row['defensive_prob'] * 40)}")
        print(f"    Blowout:     {row['blowout_prob']:>6.1%}  {'█' * int(row['blowout_prob'] * 40)}")
        print(f"    Competitive: {row['competitive_prob']:>6.1%}  {'█' * int(row['competitive_prob'] * 40)}")
        if row['blowout_prob'] > 0.3:
            print(f"  Blowout Confidence: {row['blowout_confidence']:.1%}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nAverage shootout probability: {results_df['shootout_prob'].mean():.1%}")
    print(f"Highest shootout game: {results_df.iloc[0]['game_id']} ({results_df.iloc[0]['shootout_prob']:.1%})")
    print(f"Lowest shootout game: {results_df.iloc[-1]['game_id']} ({results_df.iloc[-1]['shootout_prob']:.1%})")

    print(f"\nGames by primary script:")
    for script in ['shootout', 'defensive', 'blowout', 'competitive']:
        count = (results_df['primary_script'] == script).sum()
        avg_prob = results_df[results_df['primary_script'] == script]['script_strength'].mean() if count > 0 else 0
        print(f"  {script:12s}: {count:2d} games (avg strength: {avg_prob:.1%})")

    print(f"\nHigh-confidence shootouts (>50% shootout prob):")
    high_shootout = results_df[results_df['shootout_prob'] > 0.5]
    if len(high_shootout) > 0:
        for _, row in high_shootout.iterrows():
            print(f"  {row['game_id']:10s} - {row['shootout_prob']:.1%}")
    else:
        print("  None")

    print(f"\nHigh-confidence defensive games (>50% defensive prob):")
    high_defensive = results_df[results_df['defensive_prob'] > 0.5]
    if len(high_defensive) > 0:
        for _, row in high_defensive.iterrows():
            print(f"  {row['game_id']:10s} - {row['defensive_prob']:.1%}")
    else:
        print("  None")

    # Save to CSV
    output_dir = Path('data/intermediate')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'game_script.csv'

    # Reorder columns for CSV
    output_cols = [
        'game_id', 'favorite', 'underdog', 'spread', 'total',
        'shootout_prob', 'defensive_prob', 'blowout_prob', 'competitive_prob',
        'primary_script', 'script_strength', 'blowout_confidence',
        'shootout_score_raw', 'defensive_score_raw', 'blowout_score_raw', 'competitive_score_raw'
    ]

    results_df[output_cols].to_csv(output_path, index=False)
    print(f"\n✓ Saved continuous analysis to {output_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
