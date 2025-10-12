import pandas as pd
import pulp
import csv
import numpy as np
import random
from itertools import combinations

BUDGET = 60000

QB_SHORTLIST = []
D_SHORTLIST = []

INCLUDE_LIST = []
BLOCK_LIST = []

# PROJECTED_PTS_WEIGHTS = [0.95, 0.5, 0.05]
PROJECTED_PTS_WEIGHTS = [0.67]

# Diversity parameters
NUM_LINEUPS_PER_CONFIG = 20  # Number of diverse lineups to generate per configuration
OPTIMALITY_GAP = 0.3  # Accept solutions within 5% of optimal
PERTURBATION_FACTOR = 0.3  # Random perturbation range for objective coefficients
MIN_PLAYER_DIFFERENCE = 1  # Minimum number of different players between lineups

# Load player data from CSV
df = pd.read_csv('knapsack.csv')

# Convert tdProbability from percentage string to float
df['tdProbability'] = df['tdProbability'].astype(float) / 100.0
df['tdProbSigmoid'] = df['tdProbSigmoid'].astype(float) / 100.0

# Create a list of player dictionaries
players = df.to_dict(orient='records')
players = list(filter(lambda p: p['name'] not in BLOCK_LIST, players))
players_by_name = {player["name"]: player for player in players}

# Define position constraints
positions_needed = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "D": 1, "FLEX": 1}

projPts_values = [p["projPts"] for p in players]
mean_projPts = np.mean(projPts_values)
std_projPts = np.std(projPts_values)

def standardizeProjPts(x):
    z = (x - mean_projPts) / std_projPts
    z = (z + 1) / 2
    return z

def generateLineupWithExclusions(players, proj_pts_weight, excluded_lineups=None, use_perturbation=False, target_value=None):
    """
    Generate a lineup with optional exclusion constraints and objective perturbation.
    
    Args:
        players: List of player dictionaries
        proj_pts_weight: Weight for projected points vs TD probability
        excluded_lineups: List of previously found lineups to exclude
        use_perturbation: Whether to add random perturbation to objective coefficients
        target_value: If provided, find solutions within OPTIMALITY_GAP of this value
    """
    # Create the optimization problem
    prob = pulp.LpProblem("Fantasy_Football_Lineup", pulp.LpMaximize)

    # Create a decision variable for each player (1 if selected, 0 otherwise)
    player_vars = {p["name"]: pulp.LpVariable(
        f"select_{p['name']}", cat="Binary") for p in players}

    # Add constraints to force selection of "include" players
    for player_name in INCLUDE_LIST:
        if player_name in player_vars:
            prob += player_vars[player_name] == 1
        else:
            print(f"Warning: {player_name} not found in player list.")

    # Calculate base objective coefficients
    obj_coeffs = {}
    for p in players:
        base_value = (int(p["position"] not in ["QB", "D"]) * 
                     (proj_pts_weight * standardizeProjPts(p["projPts"]) + 
                      (1-proj_pts_weight) * p["tdProbSigmoid"]) + 
                     int(p["position"] in ["QB", "D"]) * standardizeProjPts(p["projPts"]))
        
        # Apply random perturbation if requested
        if use_perturbation:
            perturbation = random.uniform(1 - PERTURBATION_FACTOR, 1 + PERTURBATION_FACTOR)
            obj_coeffs[p["name"]] = base_value * perturbation
        else:
            obj_coeffs[p["name"]] = base_value

    # Objective function: maximize weighted sum with potentially perturbed coefficients
    prob += pulp.lpSum(player_vars[p["name"]] * obj_coeffs[p["name"]] for p in players)

    # If we have a target value, constrain the solution to be within gap
    if target_value is not None:
        prob += pulp.lpSum(player_vars[p["name"]] * obj_coeffs[p["name"]] for p in players) >= target_value * (1 - OPTIMALITY_GAP)

    # Salary constraint
    prob += pulp.lpSum(player_vars[p["name"]] * p["salary"] for p in players) <= BUDGET

    # Position constraints
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "QB") == positions_needed["QB"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "D") == positions_needed["D"]

    # RB and WR constraints
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "RB") >= positions_needed["RB"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "WR") >= positions_needed["WR"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "TE") >= positions_needed["TE"]

    # FLEX constraint
    prob += pulp.lpSum(
        player_vars[p["name"]] for p in players if p["position"] in ["RB", "WR", "TE"]
    ) >= positions_needed["FLEX"]

    # Total players constraint
    prob += pulp.lpSum(player_vars[p["name"]] for p in players) == 9

    # Add exclusion constraints for previously found lineups
    if excluded_lineups:
        for i, lineup in enumerate(excluded_lineups):
            # Ensure at least MIN_PLAYER_DIFFERENCE players are different
            prob += pulp.lpSum(player_vars[p] for p in lineup) <= 9 - MIN_PLAYER_DIFFERENCE

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, options=["gapRel=0.0001"]))

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Output the chosen lineup
    selected_players = [p["name"] for p in players if player_vars[p["name"]].value() == 1]
    total_cost = sum(p["salary"] for p in players if player_vars[p["name"]].value() == 1)
    total_projPts = sum(p["projPts"] for p in players if player_vars[p["name"]].value() == 1)
    
    # Calculate true score (without perturbation)
    true_score = sum([player_vars[p["name"]].value() * 
                     (int(p["position"] not in ["QB", "D"]) * (proj_pts_weight * standardizeProjPts(p["projPts"]) + (1-proj_pts_weight) * p["tdProbSigmoid"]) 
                      + int(p["position"] in ["QB", "D"]) * standardizeProjPts(p["projPts"])) 
                      for p in players])

    output = [round(true_score, 5), round(total_projPts, 2), total_cost, proj_pts_weight] + selected_players

    return output

def generateDiverseLineups(players, proj_pts_weight, num_lineups=NUM_LINEUPS_PER_CONFIG):
    """
    Generate multiple diverse lineups using various strategies.
    """
    lineups = []
    excluded_player_sets = []
    
    # Strategy 1: Find optimal solution first
    optimal_lineup = generateLineupWithExclusions(players, proj_pts_weight)
    if optimal_lineup:
        lineups.append(optimal_lineup)
        excluded_player_sets.append(set(optimal_lineup[4:]))
        optimal_value = optimal_lineup[0]
    else:
        return lineups
    
    # Strategy 2: Use exclusion constraints to force different lineups
    for i in range(num_lineups - 1):
        # Convert player sets to lists for constraint
        excluded_lineups_lists = [list(s) for s in excluded_player_sets]
        
        new_lineup = generateLineupWithExclusions(
            players, 
            proj_pts_weight, 
            excluded_lineups=excluded_lineups_lists,
            target_value=optimal_value
        )
        
        if new_lineup:
            lineups.append(new_lineup)
            excluded_player_sets.append(set(new_lineup[4:]))
    
    # Strategy 3: Add perturbation-based solutions
    for i in range(num_lineups // 2):
        perturbed_lineup = generateLineupWithExclusions(
            players, 
            proj_pts_weight,
            use_perturbation=True,
            target_value=optimal_value
        )
        
        if perturbed_lineup:
            # Check if it's sufficiently different from existing lineups
            perturbed_set = set(perturbed_lineup[4:])
            is_different = all(
                len(perturbed_set.symmetric_difference(existing)) >= MIN_PLAYER_DIFFERENCE 
                for existing in excluded_player_sets
            )
            
            if is_different:
                lineups.append(perturbed_lineup)
                excluded_player_sets.append(perturbed_set)
    
    return lineups

def appendSerializedLineup(lineup):
    lineup_copy = lineup.copy()
    lineup_copy = lineup_copy[4:]
    lineup_copy.sort()
    lineup_copy = str(lineup_copy)
    lineup.append(lineup_copy)
    return lineup

lineups = []

qbs = list(filter(lambda p: p['name'] in QB_SHORTLIST, players)) if len(QB_SHORTLIST) > 0 else list(filter(lambda p: p['position'] == 'QB', players))
ds = list(filter(lambda p: p['name'] in D_SHORTLIST, players)) if len(D_SHORTLIST) > 0 else list(filter(lambda p: p['position'] == 'D', players))
skillPlayers = list(filter(lambda p: p['position'] not in ['QB', 'D'], players))

for proj_pts_weight in PROJECTED_PTS_WEIGHTS:
    for qb in qbs:
        if len(D_SHORTLIST) > 0:
            for d in ds:
                players_subset = [qb] + skillPlayers + [d]
                diverse_lineups = generateDiverseLineups(players_subset, proj_pts_weight)
                for lineup in diverse_lineups:
                    if lineup:
                        lineup = appendSerializedLineup(lineup)
                        lineups.append(lineup)
        else:
            players_subset = [qb] + skillPlayers + ds
            diverse_lineups = generateDiverseLineups(players_subset, proj_pts_weight)
            for lineup in diverse_lineups:
                if lineup:
                    lineup = appendSerializedLineup(lineup)
                    lineups.append(lineup)

# Deduplicate based on the serialized list
seen = set()
deduped_lineups = []
for lineup in lineups:
    if lineup[-1] not in seen:
        seen.add(lineup[-1])
        deduped_lineups.append(lineup[:-1])
lineups = deduped_lineups

# Sort by score
lineups.sort(key=lambda x: x[0], reverse=True)

# Print summary
print(f"Generated {len(lineups)} unique lineups")
if lineups:
    print(f"Best score: {lineups[0][0]:.5f}")
    print(f"Score range: {lineups[0][0]:.5f} - {lineups[-1][0]:.5f}")

headers = ["Score", "Total Projected Pts", "Total Cost", "Proj Pts Weight", "QB", "SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "D"]
output = [headers] + lineups
with open("LINEUPS_DIVERSE.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(output)

print(f"Results saved to LINEUPS_DIVERSE.csv")