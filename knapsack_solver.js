import solver from "javascript-lp-solver";

function optimizeLineup(players) {
  try {
    // First, let's verify we have enough players at each position
    const positionCounts = players.reduce((counts, player) => {
      counts[player.position] = (counts[player.position] || 0) + 1;
      return counts;
    }, {});

    console.log("Position counts in data:", positionCounts);

    if (positionCounts.QB < 1) throw new Error("Not enough QBs");
    if (positionCounts.RB < 2) throw new Error("Not enough RBs");
    if (positionCounts.WR < 3) throw new Error("Not enough WRs");
    if (positionCounts.TE < 1) throw new Error("Not enough TEs");
    if (positionCounts.DEF < 1) throw new Error("Not enough DEFs");

    // Create variables object for each player
    const variables = {};
    players.forEach((player) => {
      const varName = player.name.replace(/\s+/g, "_");
      variables[varName] = {
        total: 1, // This is for counting total players
        value: player.projPts + player.tdProbability * 2,
        salary: player.salary,
        QB: player.position === "QB" ? 1 : 0,
        RB: player.position === "RB" ? 1 : 0,
        WR: player.position === "WR" ? 1 : 0,
        TE: player.position === "TE" ? 1 : 0,
        DEF: player.position === "DEF" ? 1 : 0,
        FLEX_eligible: ["RB", "WR", "TE"].includes(player.position) ? 1 : 0,
      };
    });

    const model = {
      optimize: "value",
      opType: "max",
      constraints: {
        total: { equal: 9 }, // Must select exactly 9 players
        salary: { max: 100000 },
        QB: { equal: 1 }, // Exactly 1 QB
        RB: { equal: 2 }, // Exactly 2 RB (not counting FLEX)
        WR: { equal: 3 }, // Exactly 3 WR (not counting FLEX)
        TE: { equal: 1 }, // Exactly 1 TE (not counting FLEX)
        DEF: { equal: 1 }, // Exactly 1 DEF
        FLEX_eligible: { min: 1 }, // At least 1 additional FLEX player
      },
      variables: variables,
      binaries: Object.keys(variables),
    };

    console.log("Solving with constraints:", model.constraints);

    const solution = solver.Solve(model);
    console.log("Raw solution:", solution);

    if (!solution.feasible) {
      throw new Error(
        "No feasible solution found. Check if salary cap is too low or position requirements can't be met."
      );
    }

    // Extract selected players
    const selectedPlayers = Object.keys(solution)
      .filter(
        (key) =>
          solution[key] === 1 &&
          !["feasible", "bounded", "result", "total"].includes(key)
      )
      .map((playerName) =>
        players.find((p) => p.name.replace(/\s+/g, "_") === playerName)
      )
      .filter(Boolean);

    console.log("Selected players count:", selectedPlayers.length);

    const finalCounts = selectedPlayers.reduce((counts, player) => {
      counts[player.position] = (counts[player.position] || 0) + 1;
      return counts;
    }, {});

    console.log("Final position counts:", finalCounts);

    // Validate the final lineup
    if (selectedPlayers.length !== 9) {
      throw new Error(
        `Invalid number of players selected: ${selectedPlayers.length}`
      );
    }

    const totalSalary = selectedPlayers.reduce(
      (sum, player) => sum + player.salary,
      0
    );

    return {
      feasible: true,
      players: selectedPlayers.map((player) => ({
        ...player,
        role: determineRole(player, selectedPlayers, finalCounts),
      })),
      totalSalary,
      totalProjectedPoints: selectedPlayers.reduce(
        (sum, p) => sum + p.projPts,
        0
      ),
      totalTDProbability: selectedPlayers.reduce(
        (sum, p) => sum + p.tdProbability,
        0
      ),
      positionCounts: finalCounts,
    };
  } catch (error) {
    console.error("Optimization error:", error.message);
    return {
      feasible: false,
      error: error.message,
    };
  }
}

function determineRole(player, selectedPlayers, counts) {
  if (player.position === "QB") return "QB";
  if (player.position === "DEF") return "DEF";

  // Count how many players of this position have already been assigned their primary role
  const assignedCount = selectedPlayers.filter(
    (p) => p.role === player.position && p.name !== player.name
  ).length;

  const positionMinimums = {
    RB: 2,
    WR: 3,
    TE: 1,
  };

  // If we haven't met the minimum for this position, assign primary role
  if (assignedCount < positionMinimums[player.position]) {
    return player.position;
  }

  // Otherwise, this player is a FLEX
  return "FLEX";
}

//prettier-ignore
const players = [
    // QBs
    { name: "QB1", position: "QB", salary: 8000, projPts: 25.5, tdProbability: 0.8 },
    { name: "QB2", position: "QB", salary: 7500, projPts: 23.1, tdProbability: 0.7 },
    
    // RBs
    { name: "RB1", position: "RB", salary: 7500, projPts: 20.1, tdProbability: 0.7 },
    { name: "RB2", position: "RB", salary: 7000, projPts: 18.5, tdProbability: 0.6 },
    { name: "RB3", position: "RB", salary: 6500, projPts: 17.2, tdProbability: 0.5 },
    { name: "RB4", position: "RB", salary: 6000, projPts: 15.8, tdProbability: 0.4 },
    
    // WRs
    { name: "WR1", position: "WR", salary: 8500, projPts: 22.3, tdProbability: 0.75 },
    { name: "WR2", position: "WR", salary: 7200, projPts: 19.8, tdProbability: 0.65 },
    { name: "WR3", position: "WR", salary: 6800, projPts: 17.2, tdProbability: 0.55 },
    { name: "WR4", position: "WR", salary: 6300, projPts: 16.5, tdProbability: 0.5 },
    
    // TEs
    { name: "TE1", position: "TE", salary: 6500, projPts: 15.9, tdProbability: 0.5 },
    { name: "TE2", position: "TE", salary: 6000, projPts: 14.5, tdProbability: 0.45 },
    
    // DEFs
    { name: "DEF1", position: "DEF", salary: 4500, projPts: 8.5, tdProbability: 0.3 },
    { name: "DEF2", position: "DEF", salary: 4000, projPts: 7.8, tdProbability: 0.25 }
];

const result = optimizeLineup(players);
console.log(JSON.stringify(result, null, 2));
