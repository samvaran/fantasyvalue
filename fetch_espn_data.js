#!/usr/bin/env node

/**
 * ESPN-SPECIFIC DATA PIPELINE
 *
 * Generates espn_players_full.csv with ALL ESPN players (not just FanDuel slate)
 * Includes: projections, TD odds, game lines, consensus, P90 values
 *
 * This file is used by espn_lineup_optimizer.py for weekly roster decisions.
 */

import neatCsv from "neat-csv";
import fsextra from "fs-extra";
const { readFile, writeFile, pathExists } = fsextra;
import { createObjectCsvWriter } from "csv-writer";
import fetch from "node-fetch";

// ============================================================================
// CONFIGURATION
// ============================================================================

const ESPN_BASE_URL = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2025/segments/0/leaguedefaults/3?view=kona_player_info&playerId={}';
const ESPN_PLAYER_LIST_URL = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2025/players?scoringPeriodId=0&view=players_wl&platformVersion=6c0d90bbc8abfb789ccf5f8728b6459da4b18c82';

const POSITION_MAP = {
  1: 'QB',
  2: 'RB',
  3: 'WR',
  4: 'TE',
  16: 'D',
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function removePunc(name) {
  return name
    .toLowerCase()
    .replace(/\./g, "")
    .replace(/'/g, "")
    .replace(/-/g, " ")
    .replace(/jr/g, "")
    .replace(/sr/g, "")
    .replace(/iii/g, "")
    .replace(/ii/g, "")
    .trim();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// STEP 1: FETCH ESPN PLAYER LIST
// ============================================================================

async function fetchEspnPlayerList() {
  console.log('='.repeat(80));
  console.log('ESPN PLAYER DATA PIPELINE');
  console.log('='.repeat(80));
  console.log('\n=== STEP 1: Fetching ESPN Player List ===\n');

  const headers = {
    'X-Fantasy-Filter': '{"filterActive":{"value":true}}',
    'sec-ch-ua-platform': '"macOS"',
    'Referer': 'https://fantasy.espn.com/',
    'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    'X-Fantasy-Platform': 'espn-fantasy-web',
    'sec-ch-ua-mobile': '?0',
    'X-Fantasy-Source': 'kona',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'DNT': '1',
  };

  try {
    const response = await fetch(ESPN_PLAYER_LIST_URL, { headers });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(`  ✓ Fetched ${data.length} players from ESPN API`);

    // Filter to only relevant positions (QB, RB, WR, TE, D/ST)
    const relevantPlayers = data.filter(p => {
      const pos = POSITION_MAP[p.defaultPositionId];
      return pos !== undefined;
    });

    console.log(`  ✓ Filtered to ${relevantPlayers.length} players (QB/RB/WR/TE/D)`);

    return relevantPlayers.map(p => ({
      id: p.id,
      name: removePunc(p.fullName),
      fullName: p.fullName,
      position: POSITION_MAP[p.defaultPositionId],
      team: p.proTeamId || '', // ESPN team ID (needs mapping)
    }));
  } catch (error) {
    console.error(`  ❌ Error fetching ESPN player list: ${error.message}`);
    throw error;
  }
}

// ============================================================================
// STEP 2: FETCH ESPN PROJECTIONS
// ============================================================================

async function fetchEspnProjection(espnPlayerId) {
  const url = ESPN_BASE_URL.replace('{}', espnPlayerId);

  try {
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    };

    const response = await fetch(url, { headers, timeout: 10000 });

    if (!response.ok) {
      return null;
    }

    const projections = await response.json();

    if (!projections || projections.length === 0) {
      return null;
    }

    // Sort by DATA_TIMESTAMP to get most recent
    const sortedProjections = projections.sort((a, b) => {
      const dateA = a.DATA_TIMESTAMP || '';
      const dateB = b.DATA_TIMESTAMP || '';
      return dateB.localeCompare(dateA);
    });

    const mostRecent = sortedProjections[0];

    return {
      espnScoreProjection: mostRecent.SCORE_PROJECTION || 0,
      espnLowScore: mostRecent.LOW_SCORE || 0,
      espnHighScore: mostRecent.HIGH_SCORE || 0,
      espnOutsideProjection: mostRecent.OUTSIDE_PROJECTION || 0,
      espnSimulationProjection: mostRecent.SIMULATION_PROJECTION || 0,
    };
  } catch (error) {
    return null;
  }
}

async function addEspnProjections(players) {
  console.log('\n=== STEP 2: Fetching ESPN Projections ===\n');

  const cacheFile = './espn_projections.json';
  let projectionCache = {};

  // Load cache if exists
  if (await pathExists(cacheFile)) {
    console.log('  Loading cached projections from espn_projections.json...');
    const data = await readFile(cacheFile, 'utf-8');
    projectionCache = JSON.parse(data);
    console.log(`  ✓ Loaded ${Object.keys(projectionCache).length} cached projections`);
  }

  // Filter players with projections
  const playersWithProjections = [];
  let fetchCount = 0;
  let cacheHits = 0;

  for (let i = 0; i < players.length; i++) {
    const player = players[i];
    const espnId = player.id.toString();

    // Progress indicator
    if ((i + 1) % 50 === 0 || (i + 1) === players.length) {
      process.stdout.write(`  Processing player ${i + 1}/${players.length}...\r`);
    }

    // Check cache first
    let projection = projectionCache[espnId];

    if (!projection) {
      // Fetch from API
      projection = await fetchEspnProjection(espnId);
      if (projection) {
        projectionCache[espnId] = projection;
        fetchCount++;
        await sleep(50); // Rate limiting
      }
    } else {
      cacheHits++;
    }

    if (projection) {
      playersWithProjections.push({
        ...player,
        ...projection,
      });
    }
  }

  console.log(`\n  ✓ ${cacheHits} cache hits, ${fetchCount} API calls`);
  console.log(`  ✓ ${playersWithProjections.length} players have projections`);

  // Save updated cache
  await writeFile(cacheFile, JSON.stringify(projectionCache, null, 2));
  console.log('  ✓ Updated espn_projections.json cache');

  return playersWithProjections;
}

// ============================================================================
// STEP 3: LOAD GAME LINES & TD ODDS FROM KNAPSACK
// ============================================================================

async function loadGameData() {
  console.log('\n=== STEP 3: Loading Game Lines & TD Odds ===\n');

  // Load game lines
  let gameLines = {};
  if (await pathExists('./game_lines.csv')) {
    const data = await readFile('./game_lines.csv');
    const lines = await neatCsv(data);
    lines.forEach(line => {
      gameLines[line.team_abbr] = {
        opponent: line.opponent_abbr,
        spread: parseFloat(line.spread),
        total: parseFloat(line.total),
        projectedPts: parseFloat(line.projected_pts),
        oppProjPts: parseFloat(line.projected_pts) - parseFloat(line.spread),
      };
    });
    console.log(`  ✓ Loaded game lines for ${Object.keys(gameLines).length} teams`);
  }

  // Load TD odds
  let tdOdds = {};
  if (await pathExists('./td_odds.json')) {
    const data = await readFile('./td_odds.json', 'utf-8');
    tdOdds = JSON.parse(data);
    console.log(`  ✓ Loaded TD odds for ${Object.keys(tdOdds).length} players`);
  }

  // Load knapsack for team mapping
  let teamMap = {};
  if (await pathExists('./knapsack.csv')) {
    const data = await readFile('./knapsack.csv');
    const players = await neatCsv(data);
    players.forEach(p => {
      const nameKey = removePunc(p.name);
      teamMap[nameKey] = p.team;
    });
    console.log(`  ✓ Loaded team mapping from knapsack.csv`);
  }

  return { gameLines, tdOdds, teamMap };
}

// ============================================================================
// STEP 4: BUILD CONSENSUS & CALCULATE P90
// ============================================================================

function calculateConsensusAndP90(player, gameData) {
  const { gameLines, tdOdds, teamMap } = gameData;

  // Get team from mapping
  const team = teamMap[player.name] || '';
  const gameInfo = gameLines[team] || {};

  // Build consensus from ESPN projections
  const projections = [
    player.espnScoreProjection,
    player.espnHighScore,
    player.espnOutsideProjection,
    player.espnSimulationProjection,
  ].filter(v => v && v > 0);

  if (projections.length === 0) {
    return null; // Skip players with no projections
  }

  let consensus = projections.reduce((sum, v) => sum + v, 0) / projections.length;
  let uncertainty = 0;

  if (projections.length > 1) {
    const variance = projections.reduce((sum, v) => sum + Math.pow(v - consensus, 2), 0) / projections.length;
    uncertainty = Math.sqrt(variance);
  } else {
    uncertainty = consensus * 0.3; // Default 30% uncertainty
  }

  // Apply TD odds boost (RB/WR/TE only)
  let adjustedConsensus = consensus;
  const tdProbability = tdOdds[player.name] ? parseFloat(tdOdds[player.name].probability) : 0;

  if (tdProbability > 0 && ['RB', 'WR', 'TE'].includes(player.position)) {
    const tdBoost = (tdProbability / 100) * 0.5;
    adjustedConsensus = consensus * (1 + tdBoost);
    uncertainty = uncertainty * (1 + tdBoost * 0.5);
  }

  // Apply game script adjustments
  if (adjustedConsensus && gameInfo.projectedPts && gameInfo.oppProjPts) {
    const spread = gameInfo.projectedPts - gameInfo.oppProjPts;
    const total = gameInfo.projectedPts + gameInfo.oppProjPts;

    if (player.position === 'RB') {
      if (spread > 3) {
        adjustedConsensus *= 1.08; // +8% for favored teams
      } else if (spread < -3) {
        adjustedConsensus *= 0.94; // -6% for trailing teams
      }
    } else if (player.position === 'WR' || player.position === 'TE') {
      if (spread < -3) {
        adjustedConsensus *= 1.10; // +10% for trailing teams
      } else if (spread > 7) {
        adjustedConsensus *= 0.96; // -4% when heavily favored
      }
    } else if (player.position === 'QB') {
      if (total > 50) {
        adjustedConsensus *= 1.08; // +8% for shootouts
      } else if (total < 42) {
        adjustedConsensus *= 0.95; // -5% for low-scoring games
      }
    } else if (player.position === 'D') {
      if (gameInfo.oppProjPts < 18) {
        adjustedConsensus *= 1.25; // +25% vs weak offenses
      } else if (gameInfo.oppProjPts > 26) {
        adjustedConsensus *= 0.80; // -20% vs strong offenses
      }
    }
  }

  consensus = parseFloat(adjustedConsensus.toFixed(2));
  uncertainty = parseFloat(uncertainty.toFixed(2));

  // Calculate P90 using exact log-normal formula
  let p90 = 0;
  if (consensus > 0) {
    const mean = consensus;
    const std = uncertainty;
    const variance = std * std;
    const sigmaSquared = Math.log(1 + variance / (mean * mean));
    const mu = Math.log(mean) - sigmaSquared / 2;
    const sigma = Math.sqrt(sigmaSquared);
    const z90 = 1.2815515655446004;
    p90 = parseFloat(Math.exp(mu + sigma * z90).toFixed(2));
  }

  return {
    name: player.name,
    fullName: player.fullName,
    position: player.position,
    team,
    game: gameInfo.opponent ? `${team}@${gameInfo.opponent}` : '',
    consensus,
    uncertainty,
    p90,
    espnScoreProjection: player.espnScoreProjection,
    espnLowScore: player.espnLowScore,
    espnHighScore: player.espnHighScore,
    espnOutsideProjection: player.espnOutsideProjection,
    espnSimulationProjection: player.espnSimulationProjection,
    projTeamPts: gameInfo.projectedPts || '',
    projOppPts: gameInfo.oppProjPts || '',
    tdProbability: tdProbability || '',
    espnId: player.id,
  };
}

// ============================================================================
// STEP 5: WRITE ESPN CSV
// ============================================================================

async function writeEspnCsv(players) {
  console.log('\n=== STEP 4: Writing ESPN Players CSV ===\n');

  // Sort by P90 descending
  players.sort((a, b) => (b.p90 || 0) - (a.p90 || 0));

  const writer = createObjectCsvWriter({
    path: './espn_players_full.csv',
    header: [
      { id: 'name', title: 'name' },
      { id: 'fullName', title: 'fullName' },
      { id: 'position', title: 'position' },
      { id: 'team', title: 'team' },
      { id: 'game', title: 'game' },
      { id: 'consensus', title: 'consensus' },
      { id: 'p90', title: 'p90' },
      { id: 'espnScoreProjection', title: 'espnScoreProjection' },
      { id: 'espnLowScore', title: 'espnLowScore' },
      { id: 'espnHighScore', title: 'espnHighScore' },
      { id: 'espnOutsideProjection', title: 'espnOutsideProjection' },
      { id: 'espnSimulationProjection', title: 'espnSimulationProjection' },
      { id: 'uncertainty', title: 'uncertainty' },
      { id: 'projTeamPts', title: 'projTeamPts' },
      { id: 'projOppPts', title: 'projOppPts' },
      { id: 'tdProbability', title: 'tdProbability' },
      { id: 'espnId', title: 'espnId' },
    ],
  });

  await writer.writeRecords(players);
  console.log(`  ✓ Wrote ${players.length} players to espn_players_full.csv`);
  console.log('\n  Position breakdown:');

  const posCounts = {};
  players.forEach(p => {
    posCounts[p.position] = (posCounts[p.position] || 0) + 1;
  });

  Object.keys(posCounts).sort().forEach(pos => {
    console.log(`    ${pos}: ${posCounts[pos]}`);
  });
}

// ============================================================================
// MAIN PIPELINE
// ============================================================================

async function main() {
  try {
    // Step 1: Fetch ESPN player list
    const espnPlayers = await fetchEspnPlayerList();

    // Step 2: Add ESPN projections
    const playersWithProjections = await addEspnProjections(espnPlayers);

    // Step 3: Load game data
    const gameData = await loadGameData();

    // Step 4: Calculate consensus & P90
    console.log('\n=== STEP 4: Calculating Consensus & P90 ===\n');
    const finalPlayers = playersWithProjections
      .map(p => calculateConsensusAndP90(p, gameData))
      .filter(p => p !== null && p.consensus > 0);

    console.log(`  ✓ ${finalPlayers.length} players with valid projections`);

    // Step 5: Write CSV
    await writeEspnCsv(finalPlayers);

    console.log('\n' + '='.repeat(80));
    console.log('COMPLETE!');
    console.log('='.repeat(80));
    console.log('\nOutput: espn_players_full.csv');
    console.log('Use this file with espn_lineup_optimizer.py\n');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if called directly
main();
