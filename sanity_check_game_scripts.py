"""
Sanity check: Do game script adjustments actually affect simulated scores correctly?

Test if:
1. Players in "boom" scenarios (favorable game scripts) actually score higher
2. Players in "bust" scenarios (unfavorable game scripts) actually score lower
"""
import pandas as pd
import numpy as np
from optimizer.utils.monte_carlo import simulate_player_score

# Load data
players_df = pd.read_csv('data/intermediate/players_integrated.csv')
game_scripts_df = pd.read_csv('data/intermediate/game_scripts_enhanced.csv')

print("=" * 100)
print("GAME SCRIPT SANITY CHECK")
print("=" * 100)
print("\nTesting if favorable/unfavorable game scripts actually affect simulated scores\n")

# Define boom/bust scenarios for each position
# Based on GAME_SCRIPT_FLOOR and GAME_SCRIPT_CEILING from 2_data_integration.py
position_scenarios = {
    'QB': {
        'boom': ['shootout', 'blowout_underdog'],  # High ceiling multipliers
        'bust': ['defensive', 'blowout_favorite']   # Low ceiling multipliers
    },
    'RB': {
        'boom': ['blowout_favorite'],  # High floor + ceiling
        'bust': ['shootout', 'blowout_underdog']  # Low floor/ceiling
    },
    'WR': {
        'boom': ['shootout', 'blowout_underdog'],  # High ceiling
        'bust': ['defensive', 'blowout_favorite']   # Low ceiling
    },
    'TE': {
        'boom': ['shootout', 'blowout_underdog'],
        'bust': ['defensive', 'blowout_favorite']
    }
}

# Test top players at each position
positions = ['QB', 'RB', 'WR', 'TE']
n_sims = 10000

for position in positions:
    print(f"\n{'=' * 100}")
    print(f"{position} ANALYSIS")
    print("=" * 100)

    # Get top 5 players at this position
    pos_players = players_df[players_df['position'] == position].nlargest(5, 'fpProjPts')

    boom_scenarios = position_scenarios[position]['boom']
    bust_scenarios = position_scenarios[position]['bust']

    for _, player in pos_players.iterrows():
        name = player['name']
        consensus = player['fpProjPts']
        is_favorite = True  # Assume favorite for simplicity

        print(f"\n{name} (Consensus: {consensus:.2f})")
        print("-" * 100)

        # Simulate in boom scenarios
        boom_scores = []
        for scenario in boom_scenarios:
            scores = []
            for _ in range(n_sims):
                score = simulate_player_score(
                    player.to_dict(),
                    scenario,
                    is_favorite
                )
                scores.append(score)
            avg_score = np.mean(scores)
            boom_scores.append(avg_score)
            diff = avg_score - consensus
            pct = (diff / consensus * 100) if consensus > 0 else 0
            print(f"  {scenario:20s} (BOOM): {avg_score:6.2f}  (vs consensus: {diff:+6.2f} / {pct:+5.1f}%)")

        avg_boom = np.mean(boom_scores)

        # Simulate in bust scenarios
        bust_scores = []
        for scenario in bust_scenarios:
            scores = []
            for _ in range(n_sims):
                score = simulate_player_score(
                    player.to_dict(),
                    scenario,
                    is_favorite
                )
                scores.append(score)
            avg_score = np.mean(scores)
            bust_scores.append(avg_score)
            diff = avg_score - consensus
            pct = (diff / consensus * 100) if consensus > 0 else 0
            print(f"  {scenario:20s} (BUST): {avg_score:6.2f}  (vs consensus: {diff:+6.2f} / {pct:+5.1f}%)")

        avg_bust = np.mean(bust_scores)

        # Simulate in competitive (neutral)
        competitive_scores = []
        for _ in range(n_sims):
            score = simulate_player_score(
                player.to_dict(),
                'competitive',
                is_favorite
            )
            competitive_scores.append(score)
        avg_competitive = np.mean(competitive_scores)
        diff = avg_competitive - consensus
        pct = (diff / consensus * 100) if consensus > 0 else 0
        print(f"  {'competitive':20s} (BASE): {avg_competitive:6.2f}  (vs consensus: {diff:+6.2f} / {pct:+5.1f}%)")

        # Summary
        boom_vs_consensus = avg_boom - consensus
        bust_vs_consensus = avg_bust - consensus
        boom_vs_bust = avg_boom - avg_bust

        print(f"\n  SUMMARY:")
        print(f"    Avg BOOM scenarios: {avg_boom:6.2f}  ({boom_vs_consensus:+5.2f} vs consensus, {boom_vs_consensus/consensus*100:+.1f}%)")
        print(f"    Avg BUST scenarios: {avg_bust:6.2f}  ({bust_vs_consensus:+5.2f} vs consensus, {bust_vs_consensus/consensus*100:+.1f}%)")
        print(f"    Competitive:        {avg_competitive:6.2f}")
        print(f"    BOOM - BUST gap:    {boom_vs_bust:+6.2f} pts ({boom_vs_bust/consensus*100:+.1f}%)")

        # Validation
        if boom_vs_bust > 0:
            print(f"    ✓ VALID: BOOM scenarios score {boom_vs_bust:.2f} pts higher than BUST")
        else:
            print(f"    ✗ INVALID: BOOM scenarios should score higher than BUST!")

        if abs(avg_competitive - consensus) < consensus * 0.05:  # Within 5%
            print(f"    ✓ VALID: Competitive scenario close to consensus")
        else:
            print(f"    ✗ WARNING: Competitive scenario differs from consensus by {abs(avg_competitive - consensus):.2f} pts")

print("\n" + "=" * 100)
print("OVERALL VALIDATION")
print("=" * 100)
print("\nExpected behavior:")
print("  1. BOOM scenarios should score HIGHER than consensus")
print("  2. BUST scenarios should score LOWER than consensus")
print("  3. Competitive scenario should be CLOSE to consensus")
print("  4. BOOM - BUST gap should be significant (>5% of consensus)")
print("\nIf these patterns hold, the game script adjustments are working correctly!")
