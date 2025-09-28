import pandas as pd
import pulp
import csv
import numpy as np

BUDGET = 60000

QB_SHORTLIST = ['lamar jackson']
D_SHORTLIST = []

INCLUDE_LIST = []
BLOCK_LIST = ['tetairoa mcmillan', 'davante adams']

# PROJECTED_PTS_WEIGHTS = [0.95, 0.5, 0.05]
PROJECTED_PTS_WEIGHTS = [0.5]

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

# def standardizeProjPts(x):
#     return 1
def standardizeProjPts(x):
    z = (x - mean_projPts) / std_projPts
    z = (z + 1) / 2
    return z

# def sigmoid(x):
#     scaling_factor = 10
#     return 1 / (1 + math.exp((-1 * scaling_factor) * (x - 0.45)))

def generateLineup(players, proj_pts_weight):
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

    # for p in players:
    #     print(p["name"], p["projPts"], standardizeProjPts(p["projPts"]),  p["tdProbability"], sigmoid(p["tdProbability"]))

    # Objective function: maximize weighted sum of projected points and TD probability (sigmoid)
    prob += pulp.lpSum(player_vars[p["name"]] * (int(p["position"] not in ["QB", "D"]) * (proj_pts_weight * standardizeProjPts(p["projPts"]) + (1-proj_pts_weight) * p["tdProbSigmoid"]) + int(p["position"] in ["QB", "D"]) * standardizeProjPts(p["projPts"])) for p in players)

    # Salary constraint
    prob += pulp.lpSum(player_vars[p["name"]] * p["salary"] for p in players) <= BUDGET

    # Position constraints
    # QB, TE, and D are straightforward as they require exact counts
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "QB") == positions_needed["QB"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "D") == positions_needed["D"]

    # RB and WR constraints (at least the specified number of each)
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "RB") >= positions_needed["RB"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "WR") >= positions_needed["WR"]
    prob += pulp.lpSum(player_vars[p["name"]] for p in players if p["position"] == "TE") >= positions_needed["TE"]

    # FLEX constraint (can be RB, WR, or TE)
    prob += pulp.lpSum(
        player_vars[p["name"]] for p in players if p["position"] in ["RB", "WR", "TE"]
    ) >= positions_needed["FLEX"]

    # Total players constraint (exactly 9 players)
    prob += pulp.lpSum(player_vars[p["name"]] for p in players) == 9

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, options=["gapRel=0.0001"]))
    # prob.solve(pulp.GLPK(msg=False, options=["gapRel=0.01"]))

    # Output the chosen lineup
    selected_players = [p["name"] for p in players if player_vars[p["name"]].value() == 1]
    total_cost = sum(p["salary"] for p in players if player_vars[p["name"]].value() == 1)
    total_projPts = sum(p["projPts"] for p in players if player_vars[p["name"]].value() == 1)
    # sum_tdProbability_sigmoid_skill_players = sum(p["tdProbSigmoid"] for p in players if player_vars[p["name"]].value() == 1 and p["position"] not in ["QB", "D"])
    score = sum([player_vars[p["name"]].value() * 
                 (int(p["position"] not in ["QB", "D"]) * (proj_pts_weight * standardizeProjPts(p["projPts"]) + (1-proj_pts_weight) * p["tdProbSigmoid"]) 
                  + int(p["position"] in ["QB", "D"]) * standardizeProjPts(p["projPts"])) 
                  for p in players])

    output = [round(score, 5), round(total_projPts, 2), total_cost, proj_pts_weight] + selected_players

    return output

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
                players = [qb] + skillPlayers + [d]
                lineup = generateLineup(players, proj_pts_weight)
                lineup = appendSerializedLineup(lineup)
                lineups.append(lineup)
        else:
            players = [qb] + skillPlayers + ds
            lineup = generateLineup(players, proj_pts_weight)
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

# print(seen)

lineups.sort(key=lambda x: x[0], reverse=True)
headers = ["Score", "Total Projected Pts", "Total Cost", "Proj Pts Weight", "QB", "SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "D"]
output = [headers] + lineups
with open("LINEUPS.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(output)

# for row in output:
#     print(row)
