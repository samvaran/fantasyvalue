import { load } from "cheerio";
import { launch } from "puppeteer";
import neatCsv from "neat-csv";
import fsextra from "fs-extra";
const { readFile, pathExists } = fsextra;
import { createObjectCsvWriter } from "csv-writer";
import fetch from "node-fetch";

// ============================================================================
// CONFIGURATION
// ============================================================================

const MINIMAL_ARGS = [
  "--autoplay-policy=user-gesture-required",
  "--disable-background-networking",
  "--disable-background-timer-throttling",
  "--disable-backgrounding-occluded-windows",
  "--disable-breakpad",
  "--disable-client-side-phishing-detection",
  "--disable-component-update",
  "--disable-default-apps",
  "--disable-dev-shm-usage",
  "--disable-domain-reliability",
  "--disable-extensions",
  "--disable-features=AudioServiceOutOfProcess",
  "--disable-hang-monitor",
  "--disable-ipc-flooding-protection",
  "--disable-notifications",
  "--disable-offer-store-unmasked-wallet-cards",
  "--disable-popup-blocking",
  "--disable-print-preview",
  "--disable-prompt-on-repost",
  "--disable-renderer-backgrounding",
  "--disable-setuid-sandbox",
  "--disable-speech-api",
  "--disable-sync",
  "--hide-scrollbars",
  "--ignore-gpu-blacklist",
  "--metrics-recording-only",
  "--mute-audio",
  "--no-default-browser-check",
  "--no-first-run",
  "--no-pings",
  "--no-sandbox",
  "--no-zygote",
  "--password-store=basic",
  "--use-gl=swiftshader",
  "--use-mock-keychain",
];

const ESPN_BASE_URL =
  "https://watsonfantasyfootball.espn.com/espnpartner/dallas/projections/projections_{}_ESPNFantasyFootball_2025.json";

// Rate limiting for ESPN
const ESPN_MIN_DELAY = 0.3;
const ESPN_MAX_DELAY = 0.8;
const ESPN_BATCH_SIZE = 100;
const ESPN_BATCH_BREAK = 5;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function delay(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

function sigmoid(x) {
  const scalingFactor = 10;
  return 1 / (1 + Math.exp(-1 * scalingFactor * (x - 0.45)));
}

const removePunc = (s) => {
  if (s.includes("(")) {
    s = s.split("(")[0].trim();
  }
  s = s.replace(/[^\w\s]/gi, "");
  s = s.replace(/Jr\.|Jr|III|II|IV/g, "");
  s = s.trim();
  s = s.toLowerCase();
  return s;
};

const oddsToProbability = (odds) => {
  let prob = 0;
  const oddsNumber = Number(odds.slice(1));
  if (odds[0] == "+") {
    prob = 100 / (oddsNumber + 100);
  } else {
    prob = oddsNumber / (oddsNumber + 100);
  }
  return (prob * 100).toFixed(2);
};

// ============================================================================
// DATA LOADING FUNCTIONS
// ============================================================================

async function loadExistingData(position) {
  /**
   * Load existing player data from position-specific CSV if it exists
   */
  const filepath = `./fanduel-${position}.csv`;
  if (await pathExists(filepath)) {
    let data = await readFile(filepath);
    return await neatCsv(data);
  }
  return null;
}

async function loadKnapsackData() {
  /**
   * Load existing knapsack.csv data
   */
  if (await pathExists("./knapsack.csv")) {
    let data = await readFile("./knapsack.csv");
    return await neatCsv(data);
  }
  return null;
}

// ============================================================================
// STEP 1: FETCH FANTASYPROS PROJECTIONS
// ============================================================================

async function scrapeFantasyProsTable(url, pos) {
  const browser = await launch({
    headless: true,
    args: MINIMAL_ARGS,
  });
  console.log("  LAUNCHED", url);

  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 0 });
  await page.$eval(
    ".everything-but-phone .select-advanced-content__text",
    (el) => el.click()
  );
  await delay(500);

  // Scroll to load all content
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });
  await delay(500);
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });

  const content = await page.content();
  const $ = load(content);
  console.log("  CLOSED", url);
  await browser.close();

  const headers = url.includes("dst.php")
    ? ["rank", "wsis", "name", "opp", "matchup", "grade", "fpProjPts"]
    : ["rank", "wsis", "name", "opp", "upside", "bust", "matchup", "grade", "fpProjPts"];

  let rows = $("#ranking-table > tbody > tr.player-row")
    .toArray()
    .map((row) => {
      let rowData = {};
      $(row)
        .find("td")
        .toArray()
        .forEach((d, i) => {
          if (headers[i] == "name") {
            let name = $(d).find(".player-cell a").text().trim();
            name = removePunc(name);
            if (pos === "D") {
              name = name.split(" ").pop();
            }
            rowData[headers[i]] = name;
          } else if (["wsis", "opp", "matchup", "grade"].includes(headers[i])) {
            // skip
          } else if (headers[i] == "fpProjPts") {
            let ppTemp = $(d).text().trim();
            if (ppTemp == "-") {
              ppTemp = 3;
            }
            rowData[headers[i]] = ppTemp;
          } else {
            rowData[headers[i]] = $(d).text().trim();
          }
        });
      return rowData;
    });

  return rows;
}

async function fetchFantasyProsProjections() {
  console.log("\n=== STEP 1: Fetching FantasyPros Projections ===\n");

  const positions = {
    QB: { url: "https://www.fantasypros.com/nfl/rankings/qb.php" },
    RB: { url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-rb.php" },
    WR: { url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-wr.php" },
    TE: { url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-te.php" },
    D: { url: "https://www.fantasypros.com/nfl/rankings/dst.php" },
  };

  const players = {};

  for (const [pos, config] of Object.entries(positions)) {
    await delay(500);
    console.log(`Fetching ${pos}...`);
    const data = await scrapeFantasyProsTable(config.url, pos);
    players[pos] = { data };
  }

  return players;
}

// ============================================================================
// STEP 2: ADD FANDUEL SALARIES
// ============================================================================

async function addFanduelSalaries(players) {
  console.log("\n=== STEP 2: Adding FanDuel Salaries ===\n");

  // Find the most recent FanDuel CSV
  const fs = await import("fs/promises");
  const files = await fs.readdir("./");
  const fanduelFiles = files
    .filter((file) => file.startsWith("FanDuel-NFL") && file.endsWith(".csv"))
    .sort();

  if (fanduelFiles.length === 0) {
    throw new Error('No CSV files starting with "FanDuel-NFL" found');
  }

  const selectedFile = fanduelFiles[fanduelFiles.length - 1];
  console.log(`Reading from: ${selectedFile}`);

  let data = await readFile(`./${selectedFile}`);
  data = await neatCsv(data);

  let dataByPlayer = {};
  data.forEach((d) => {
    let name = d.Position === "D" ? d["Last Name"] : d.Nickname;
    name = removePunc(name);
    dataByPlayer[name] = {
      name,
      fppg: d.FPPG,
      played: d.Played,
      salary: d.Salary,
      game: d.Game,
      team: d.Team,
      position: d.Position,
      opponent: d.Opponent,
      injury: d["Injury Indicator"],
      injuryDetail: d["Injury Details"],
    };
  });

  console.log("EXCLUDED SALARIES (no match in FanDuel data):");
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data
      .map((d) => {
        let salaryData = dataByPlayer[d.name] || {};
        if (!dataByPlayer[d.name]) {
          console.log(`  ${d.name}`);
        }
        return { ...d, ...salaryData };
      })
      .filter((d) => {
        // Handle both fpProjPts (new) and projPts (old cached files)
        const pts = d.fpProjPts || d.projPts;
        return !!d.salary && d.injury != "IR" && pts > 0;
      });
    players[pos].data = augmentedData;
  });

  return players;
}

// ============================================================================
// STEP 3: FETCH/LOAD AND ADD DRAFTKINGS GAME LINES
// ============================================================================

function getTeamAbbreviation(teamName) {
  // Map normalized team names to NFL abbreviations
  const teamMap = {
    'cardinals': 'ARI',
    'falcons': 'ATL',
    'ravens': 'BAL',
    'bills': 'BUF',
    'panthers': 'CAR',
    'bears': 'CHI',
    'bengals': 'CIN',
    'browns': 'CLE',
    'cowboys': 'DAL',
    'broncos': 'DEN',
    'lions': 'DET',
    'packers': 'GB',
    'texans': 'HOU',
    'colts': 'IND',
    'jaguars': 'JAX',
    'chiefs': 'KC',
    'raiders': 'LV',
    'chargers': 'LAC',
    'rams': 'LAR',
    'dolphins': 'MIA',
    'vikings': 'MIN',
    'patriots': 'NE',
    'saints': 'NO',
    'giants': 'NYG',
    'jets': 'NYJ',
    'eagles': 'PHI',
    'steelers': 'PIT',
    'seahawks': 'SEA',
    'ers': 'SF', // 49ers -> ers
    'buccaneers': 'TB',
    'titans': 'TEN',
    'commanders': 'WAS',
  };

  return teamMap[teamName] || '';
}

function parseTeamName(line) {
  // Remove -logo suffix
  let name = line.replace(/-logo$/, "").trim();

  // Extract just the team name (e.g., "TB Buccaneers" -> "buccaneers")
  // Split by space and take the last word (team name)
  const parts = name.split(/\s+/);
  if (parts.length > 1) {
    name = parts[parts.length - 1];
  }

  // Apply removePunc normalization
  name = removePunc(name);

  return name;
}

function parseGameLinesText(text) {
  const lines = text.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);

  const games = [];
  let i = 0;
  let gamesFound = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line === "Today" || line.match(/^\d{1,2}:\d{2}\s+(AM|PM)/)) {
      // Track we're in the current week section
    }

    // Stop conditions
    if (gamesFound > 0) {
      if (line === "Tomorrow") {
        console.log(`  Stopping at "Tomorrow" section after finding ${gamesFound} games`);
        break;
      }
      if (gamesFound >= 17) {
        console.log(`  Stopping after finding ${gamesFound} games (max week limit)`);
        break;
      }
      if (line.startsWith("GAME LINES") || line === "LIVE BLITZ ⚡") {
        console.log(`  Stopping at section "${line}" after finding ${gamesFound} games`);
        break;
      }
    }

    // Skip headers
    if (
      line === "Today" ||
      line === "Tomorrow" ||
      line === "Spread" ||
      line === "Total" ||
      line === "Moneyline" ||
      line === "AT" ||
      line.startsWith("Get A Profit") ||
      line === "Opt In" ||
      line.startsWith("GAME LINES")
    ) {
      i++;
      continue;
    }

    // Look for team names
    const teamPattern = /^[A-Z]{2,3}\s+[A-Za-z]+/;

    if (teamPattern.test(lines[i])) {
      const team1 = parseTeamName(lines[i]);
      i++;

      // Skip score if present
      if (i < lines.length && lines[i].match(/^\d{1,2}$/)) {
        i++;
      }

      // Look for "AT" separator
      if (i < lines.length && lines[i] === "AT") {
        i++;
      }

      // Get team2
      if (i >= lines.length || !teamPattern.test(lines[i])) {
        continue;
      }

      const team2 = parseTeamName(lines[i]);
      i++;

      // Skip score if present
      if (i < lines.length && lines[i].match(/^\d{1,2}$/)) {
        i++;
      }

      // Look for spreads and totals
      let team1Spread = null;
      let team2Spread = null;
      let total = null;

      let lookAhead = 0;
      while (i + lookAhead < lines.length && lookAhead < 30) {
        const line = lines[i + lookAhead];

        if (line.match(/^[+\-]\d+(\.\d+)?$/) && !line.match(/^[+\-]\d{3,}$/)) {
          const value = parseFloat(line);
          if (Math.abs(value) <= 20) {
            if (team1Spread === null) {
              team1Spread = value;
            } else if (team2Spread === null) {
              team2Spread = value;
            }
          }
        }

        if (line === "O" && i + lookAhead + 1 < lines.length) {
          const nextLine = lines[i + lookAhead + 1];
          if (nextLine.match(/^\d+(\.\d+)?$/) && parseFloat(nextLine) > 25) {
            total = parseFloat(nextLine);
          }
        }

        if (line === "More Bets" || teamPattern.test(line)) {
          break;
        }

        lookAhead++;
      }

      // Advance past this game
      while (i < lines.length && lines[i] !== "More Bets") {
        i++;
      }
      if (i < lines.length && lines[i] === "More Bets") {
        i++;
      }

      // Create game entries
      if (team1Spread !== null && team2Spread !== null && total !== null) {
        const calculateTeamPoints = (total, spread, isFavorite) => {
          const absSpread = Math.abs(spread);
          if (isFavorite) {
            return ((total + absSpread) / 2).toFixed(2);
          } else {
            return ((total - absSpread) / 2).toFixed(2);
          }
        };

        const team1IsFavorite = team1Spread < 0;
        const team1Abbr = getTeamAbbreviation(team1);
        const team2Abbr = getTeamAbbreviation(team2);

        games.push({
          team: team1,
          team_abbr: team1Abbr,
          opponent: team2,
          opponent_abbr: team2Abbr,
          spread: team1Spread,
          total: total,
          projected_pts: calculateTeamPoints(total, team1Spread, team1IsFavorite),
        });

        const team2IsFavorite = team2Spread < 0;
        games.push({
          team: team2,
          team_abbr: team2Abbr,
          opponent: team1,
          opponent_abbr: team1Abbr,
          spread: team2Spread,
          total: total,
          projected_pts: calculateTeamPoints(total, team2Spread, team2IsFavorite),
        });

        gamesFound++;
      }
    } else {
      i++;
    }
  }

  return games;
}

async function fetchGameLines(shouldFetch = true) {
  console.log("\n=== STEP 3: DraftKings Game Lines ===\n");

  if (shouldFetch) {
    console.log("  Fetching from DraftKings...");

    const url = "https://sportsbook.draftkings.com/leagues/football/nfl?category=game-lines&subcategory=game";

    const browser = await launch({
      headless: true,
      args: MINIMAL_ARGS,
    });
    console.log("  LAUNCHED", url);

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "load", timeout: 0 });
    await delay(3000);

    console.log("  Extracting text...");

    const copiedText = await page.evaluate(() => {
      const selectors = [
        '[class*="parlay-card"]',
        '[class*="sportsbook-content"]',
        'main [class*="sportsbook"]',
        '.sportsbook-wrapper__body',
        '[data-testid="event-list"]',
      ];

      for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element && element.innerText.length > 1000) {
          return element.innerText;
        }
      }

      return document.body.innerText;
    });

    await browser.close();
    console.log("  CLOSED", url);

    const games = parseGameLinesText(copiedText);
    console.log(`  Found ${games.length / 2} games`);

    // Write to CSV (cache)
    const writer = createObjectCsvWriter({
      path: "./game_lines.csv",
      header: [
        { id: "team", title: "team" },
        { id: "team_abbr", title: "team_abbr" },
        { id: "opponent", title: "opponent" },
        { id: "opponent_abbr", title: "opponent_abbr" },
        { id: "spread", title: "spread" },
        { id: "total", title: "total" },
        { id: "projected_pts", title: "projected_pts" },
      ],
    });

    await writer.writeRecords(games);
    console.log("  Cached to game_lines.csv");
  } else {
    console.log("  Using cached game_lines.csv");
  }
}

// ============================================================================
// STEP 4: FETCH/LOAD AND ADD DRAFTKINGS TD ODDS
// ============================================================================

async function fetchAndCacheTdOdds(shouldFetch = true) {
  /**
   * Fetch TD odds from DraftKings and cache to td_odds.json
   * Or load from cached td_odds.json
   */

  if (shouldFetch) {
    console.log("  Fetching from DraftKings...");

    const url = "https://sportsbook.draftkings.com/leagues/football/nfl?category=td-scorers&subcategory=td-scorer";

    const browser = await launch({
      headless: true,
      args: MINIMAL_ARGS,
    });
    console.log("  LAUNCHED", url);

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "load", timeout: 0 });
    await delay(500);
    const content = await page.content();

    const dataByPlayer = {};
    const $ = load(content);
    console.log("  CLOSED", url);
    await browser.close();

    $(".cb-market__template--4-columns").each((gameIndex, game) => {
      const cells = $(game).children().toArray().slice(3);
      const columnCount = 4;

      const players = cells
        .filter((c, index) => index % columnCount === 0)
        .map((c) => $(c).text().trim())
        .filter((text) => !text.toLowerCase().includes("no touchdown"));

      const tdOddsList = cells
        .filter((c, index) => index % columnCount === 2)
        .map((c) => $(c).text().trim())
        .filter(Boolean);

      players.forEach((p, i) => {
        if (p.includes("D/ST")) {
          p = p.split(" ")[1];
        }
        const name = removePunc(p);
        const tdOdds = tdOddsList[i];
        const tdProbability = oddsToProbability(tdOdds);
        dataByPlayer[name] = { tdOdds, tdProbability };
      });
    });

    // Cache to JSON file
    const fs = await import("fs/promises");
    await fs.writeFile("./td_odds.json", JSON.stringify(dataByPlayer, null, 2));
    console.log("  Cached to td_odds.json");

    return dataByPlayer;
  } else {
    console.log("  Using cached td_odds.json");
    const fs = await import("fs/promises");
    const data = await fs.readFile("./td_odds.json", "utf-8");
    return JSON.parse(data);
  }
}

async function addDraftKingsTdOdds(players, shouldFetch = true) {
  console.log("\n=== STEP 4: DraftKings TD Odds ===\n");

  const dataByPlayer = await fetchAndCacheTdOdds(shouldFetch);

  console.log("\nEXCLUDED ODDS (no match in DraftKings data):");
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data.map((d) => {
      let oddsData = dataByPlayer[d.name] || { tdOdds: 0, tdProbability: "0" };
      if (!dataByPlayer[d.name]) {
        console.log(`  ${d.name}`);
      }
      return { ...d, ...oddsData };
    });
    players[pos].data = augmentedData;
  });

  return players;
}

// ============================================================================
// STEP 5: FETCH AND ADD ESPN IDS
// ============================================================================

async function fetchEspnPlayerList() {
  /**
   * Fetch the full ESPN player list from their API
   */
  console.log("  Fetching ESPN player list from API...");

  const url = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2025/players?scoringPeriodId=0&view=players_wl&platformVersion=6c0d90bbc8abfb789ccf5f8728b6459da4b18c82";

  const headers = {
    "X-Fantasy-Filter": '{"filterActive":{"value":true}}',
    "sec-ch-ua-platform": '"macOS"',
    "Referer": "https://fantasy.espn.com/",
    "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    "X-Fantasy-Platform": "espn-fantasy-web",
    "sec-ch-ua-mobile": "?0",
    "X-Fantasy-Source": "kona",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "DNT": "1",
  };

  try {
    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(`  Fetched ${data.length} players from ESPN API`);

    // Save to CSV for reference
    const csvData = data.map((player) => ({
      id: player.id,
      fullName: player.fullName,
    }));

    const writer = createObjectCsvWriter({
      path: "./espn_players.csv",
      header: [
        { id: "fullName", title: "fullName" },
        { id: "id", title: "id" },
      ],
    });

    await writer.writeRecords(csvData);
    console.log("  Saved to espn_players.csv");

    return data;
  } catch (error) {
    console.error(`  Error fetching ESPN player list: ${error.message}`);
    throw error;
  }
}

async function addEspnIds(players, fetchFromApi = false) {
  console.log("\n=== STEP 5: Adding ESPN Player IDs ===\n");

  let espnPlayers;

  if (fetchFromApi) {
    // Fetch from API
    espnPlayers = await fetchEspnPlayerList();
  } else {
    // Load from existing CSV
    console.log("  Loading from espn_players.csv...");
    if (!(await pathExists("./espn_players.csv"))) {
      throw new Error(
        "espn_players.csv not found. Run with --fetch-espn-ids to download from API."
      );
    }
    let data = await readFile(`./espn_players.csv`);
    const csvData = await neatCsv(data);
    espnPlayers = csvData.map((d) => ({
      id: d.id,
      fullName: d.fullName,
    }));
    console.log(`  Loaded ${espnPlayers.length} players from CSV`);
  }

  // Build lookup map
  let espnByPlayer = {};
  espnPlayers.forEach((player) => {
    let name = removePunc(player.fullName);
    espnByPlayer[name] = player.id;
  });

  console.log("\nEXCLUDED ESPN IDs (no match):");
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data.map((d) => {
      let espnId = espnByPlayer[d.name] || "";
      if (!espnByPlayer[d.name]) {
        console.log(`  ${d.name}`);
      }
      return { ...d, espnId };
    });
    players[pos].data = augmentedData;
  });

  return players;
}

// ============================================================================
// STEP 6: ADD ESPN PROJECTIONS
// ============================================================================

async function fetchEspnProjection(espnPlayerId) {
  if (!espnPlayerId || espnPlayerId === "") {
    return null;
  }

  const url = ESPN_BASE_URL.replace("{}", espnPlayerId);

  try {
    const headers = {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    };

    const response = await fetch(url, { headers, timeout: 10000 });

    if (!response.ok) {
      return null;
    }

    const projections = await response.json();

    if (!projections || projections.length === 0) {
      return null;
    }

    // Sort by DATA_TIMESTAMP to get the most recent projection
    const sortedProjections = projections.sort((a, b) => {
      const dateA = a.DATA_TIMESTAMP || "";
      const dateB = b.DATA_TIMESTAMP || "";
      return dateB.localeCompare(dateA);
    });

    const mostRecent = sortedProjections[0];

    return {
      espnScoreProjection: mostRecent.SCORE_PROJECTION || "",
      espnLowScore: mostRecent.LOW_SCORE || "",
      espnHighScore: mostRecent.HIGH_SCORE || "",
      espnOutsideProjection: mostRecent.OUTSIDE_PROJECTION || "",
      espnSimulationProjection: mostRecent.SIMULATION_PROJECTION || "",
    };
  } catch (error) {
    return null;
  }
}

async function fetchAndCacheEspnProjections(players, shouldFetch = true) {
  /**
   * Fetch ESPN projections and cache to espn_projections.json
   * Or load from cached espn_projections.json
   */

  const cacheFile = "./espn_projections.json";

  if (shouldFetch) {
    console.log("  Fetching ESPN projections...");

    // Flatten all players and filter by fpProjPts >= 2.5
    let allPlayers = [];
    Object.keys(players).forEach((pos) => {
      players[pos].data.forEach((p) => {
        allPlayers.push({ ...p, _pos: pos });
      });
    });

    // Filter and sort
    const originalCount = allPlayers.length;
    allPlayers = allPlayers.filter((p) => {
      const fpProjPts = parseFloat(p.fpProjPts);
      return !isNaN(fpProjPts) && fpProjPts >= 2.5;
    });

    console.log(`  Filtered out ${originalCount - allPlayers.length} players with < 2.5 fpProjPts`);
    console.log(`  Fetching for ${allPlayers.length} players\n`);

    allPlayers.sort((a, b) => {
      const fpProjPtsA = parseFloat(a.fpProjPts) || 0;
      const fpProjPtsB = parseFloat(b.fpProjPts) || 0;
      return fpProjPtsB - fpProjPtsA;
    });

    // Fetch projections
    const projectionsByPlayer = {};
    let playersWithEspn = 0;
    let playersSkipped = 0;

    for (let i = 0; i < allPlayers.length; i++) {
      const player = allPlayers[i];
      const espnId = player.espnId || "";

      if (!espnId) {
        playersSkipped++;
        continue;
      }

      process.stdout.write(`  [${i + 1}/${allPlayers.length}] ${player.name}`);

      const projection = await fetchEspnProjection(espnId);

      if (projection) {
        projectionsByPlayer[player.name] = projection;
        console.log(` ✓ ${projection.espnScoreProjection}`);
        playersWithEspn++;
      } else {
        console.log(" ✗");
      }

      // Rate limiting
      if (i < allPlayers.length - 1) {
        const delayTime = ESPN_MIN_DELAY + Math.random() * (ESPN_MAX_DELAY - ESPN_MIN_DELAY);
        if ((i + 1) % ESPN_BATCH_SIZE === 0) {
          console.log(`\n  --- Batch complete. Taking ${ESPN_BATCH_BREAK}s break ---\n`);
          await delay(ESPN_BATCH_BREAK * 1000);
        } else {
          await delay(delayTime * 1000);
        }
      }
    }

    console.log(`\n  Fetched ESPN projections: ${playersWithEspn}/${allPlayers.length}`);

    // Cache to JSON
    const fs = await import("fs/promises");
    await fs.writeFile(cacheFile, JSON.stringify(projectionsByPlayer, null, 2));
    console.log("  Cached to espn_projections.json");

    return projectionsByPlayer;
  } else {
    console.log("  Using cached espn_projections.json");
    const fs = await import("fs/promises");
    const data = await fs.readFile(cacheFile, "utf-8");
    return JSON.parse(data);
  }
}

async function addEspnProjections(players, shouldFetch = true) {
  console.log("\n=== STEP 6: ESPN Projections ===\n");

  const projectionsByPlayer = await fetchAndCacheEspnProjections(players, shouldFetch);

  // Add projections to players
  Object.keys(players).forEach((pos) => {
    players[pos].data = players[pos].data.map((p) => {
      const projection = projectionsByPlayer[p.name] || {
        espnScoreProjection: "",
        espnLowScore: "",
        espnHighScore: "",
        espnOutsideProjection: "",
        espnSimulationProjection: "",
      };
      return { ...p, ...projection };
    });
  });

  return players;
}

// ============================================================================
// STEP 7: ADD GAME LINES DATA
// ============================================================================

async function addGameLinesData(players) {
  console.log("\n=== STEP 7: Adding Game Lines Data ===\n");

  // Try to load game_lines.csv
  if (!(await pathExists("./game_lines.csv"))) {
    console.log("  game_lines.csv not found, skipping game lines data");
    console.log("  (Run with --lines to fetch game lines first)");

    // Add empty fields
    Object.keys(players).forEach((pos) => {
      players[pos].data = players[pos].data.map((p) => ({
        ...p,
        projTeamPts: "",
        projOppPts: "",
      }));
    });

    return players;
  }

  let data = await readFile("./game_lines.csv");
  const gameLines = await neatCsv(data);

  // Build lookup by team abbreviation
  const gameLinesByTeamAbbr = {};
  gameLines.forEach((line) => {
    const teamAbbr = line.team_abbr;
    gameLinesByTeamAbbr[teamAbbr] = {
      projTeamPts: line.projected_pts,
      projOppPts: line.projected_pts, // Will be set properly below
    };
  });

  // Set opponent projected points
  gameLines.forEach((line) => {
    const teamAbbr = line.team_abbr;
    const oppAbbr = line.opponent_abbr;
    if (gameLinesByTeamAbbr[teamAbbr] && gameLinesByTeamAbbr[oppAbbr]) {
      gameLinesByTeamAbbr[teamAbbr].projOppPts = gameLinesByTeamAbbr[oppAbbr].projTeamPts;
    }
  });

  console.log("  Joining game lines data...");

  Object.keys(players).forEach((pos) => {
    players[pos].data = players[pos].data.map((p) => {
      // Match on team abbreviation
      const teamAbbr = p.team || "";
      const gameLineData = gameLinesByTeamAbbr[teamAbbr] || {
        projTeamPts: "",
        projOppPts: "",
      };

      return {
        ...p,
        projTeamPts: gameLineData.projTeamPts,
        projOppPts: gameLineData.projOppPts,
      };
    });
  });

  console.log("  Game lines data added");
  return players;
}

// ============================================================================
// STEP 8: ADD ANALYSIS (VALUE CALCULATIONS)
// ============================================================================

function addAnalysis(players) {
  console.log("\n=== STEP 8: Adding Value Analysis ===\n");

  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data.map((d) => {
      // Handle both fpProjPts (new) and projPts (old cached files)
      const projPts = Number(d.fpProjPts) || Number(d.projPts) || 0;
      const salary = Number(d.salary) || 0;

      let value = salary > 0 ? (projPts / (salary / 1000)).toFixed(2) : "0";
      let tdProbSigmoid = (sigmoid(d.tdProbability / 100) * 100).toFixed(2);
      let tdValue = salary > 0 ? (tdProbSigmoid / (salary / 1000)).toFixed(2) : "0";

      return {
        name: d.name,
        position: d.position,
        team: d.team,
        game: d.game,
        fpProjPts: projPts,
        salary: salary,
        value: Number(value),
        tdOdds: d.tdOdds,
        tdProbability: d.tdProbability,
        tdProbSigmoid: tdProbSigmoid,
        tdValue: tdValue,
        injury: d.injury + ": " + d.injuryDetail,
        espnId: d.espnId || "",
        espnScoreProjection: Number(d.espnScoreProjection) || "",
        espnLowScore: Number(d.espnLowScore) || "",
        espnHighScore: Number(d.espnHighScore) || "",
        espnOutsideProjection: Number(d.espnOutsideProjection) || "",
        espnSimulationProjection: Number(d.espnSimulationProjection) || "",
        projTeamPts: d.projTeamPts || "",
        projOppPts: d.projOppPts || "",
      };
    });

    augmentedData.sort((a, b) => (a.value > b.value ? -1 : 1));
    players[pos].data = augmentedData;
  });

  return players;
}

// ============================================================================
// STEP 9: REGRESSION-BASED PROJECTION CONVERSION
// ============================================================================

function linearRegression(xValues, yValues) {
  /**
   * Simple linear regression: y = a + b*x
   * Returns: { intercept, slope, r2 }
   */
  const n = xValues.length;

  const sumX = xValues.reduce((sum, x) => sum + x, 0);
  const sumY = yValues.reduce((sum, y) => sum + y, 0);
  const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
  const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);
  const sumY2 = yValues.reduce((sum, y) => sum + y * y, 0);

  const meanX = sumX / n;
  const meanY = sumY / n;

  // Calculate slope (b)
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

  // Calculate intercept (a)
  const intercept = meanY - slope * meanX;

  // Calculate R²
  const ssTotal = sumY2 - n * meanY * meanY;
  const ssResidual = yValues.reduce((sum, y, i) => {
    const predicted = intercept + slope * xValues[i];
    return sum + Math.pow(y - predicted, 2);
  }, 0);
  const r2 = 1 - (ssResidual / ssTotal);

  return { intercept, slope, r2 };
}

async function loadOrBuildModels(players, shouldBuild = false) {
  if (!shouldBuild && await pathExists("./regression_models.csv")) {
    // Load existing models
    console.log("  Loading models from regression_models.csv...");
    const data = await readFile("./regression_models.csv");
    const modelData = await neatCsv(data);

    const models = {};
    modelData.forEach(row => {
      models[row.model] = {
        intercept: parseFloat(row.intercept),
        slope: parseFloat(row.slope),
        r2: parseFloat(row.r2),
      };
    });

    console.log(`  Loaded ${Object.keys(models).length} models\n`);
    return models;
  }

  // Build new models
  console.log("  Building new regression models...\n");

  // Flatten all players with both FP and ESPN data
  let trainingData = [];
  Object.keys(players).forEach((pos) => {
    if (pos !== 'FLEX') {
      players[pos].data.forEach(p => {
        if (p.fpProjPts && p.espnOutsideProjection &&
            p.espnScoreProjection && p.espnSimulationProjection) {
          trainingData.push(p);
        }
      });
    }
  });

  console.log(`  Training data: ${trainingData.length} players with both FP and ESPN projections\n`);

  // Build regression models (apples-to-apples: primary projections only)
  const models = {};

  // Model 1: espnOutsideProjection → fpProjPts
  models.espnOutside = linearRegression(
    trainingData.map(p => p.espnOutsideProjection),
    trainingData.map(p => p.fpProjPts)
  );
  console.log(`  Model: espnOutsideProjection → fpProjPts`);
  console.log(`    fpProjPts = ${models.espnOutside.intercept.toFixed(3)} + ${models.espnOutside.slope.toFixed(3)} * espnOutside`);
  console.log(`    R² = ${models.espnOutside.r2.toFixed(3)}\n`);

  // Model 2: espnScoreProjection → fpProjPts
  models.espnScore = linearRegression(
    trainingData.map(p => p.espnScoreProjection),
    trainingData.map(p => p.fpProjPts)
  );
  console.log(`  Model: espnScoreProjection → fpProjPts`);
  console.log(`    fpProjPts = ${models.espnScore.intercept.toFixed(3)} + ${models.espnScore.slope.toFixed(3)} * espnScore`);
  console.log(`    R² = ${models.espnScore.r2.toFixed(3)}\n`);

  // Model 3: espnSimulationProjection → fpProjPts
  models.espnSim = linearRegression(
    trainingData.map(p => p.espnSimulationProjection),
    trainingData.map(p => p.fpProjPts)
  );
  console.log(`  Model: espnSimulationProjection → fpProjPts`);
  console.log(`    fpProjPts = ${models.espnSim.intercept.toFixed(3)} + ${models.espnSim.slope.toFixed(3)} * espnSim`);
  console.log(`    R² = ${models.espnSim.r2.toFixed(3)}\n`);

  return models;
}

async function convertProjectionsToFpScale(players, shouldBuildModels = false) {
  console.log("\n=== STEP 9: Converting Projections to fpProjPts Scale ===\n");

  const models = await loadOrBuildModels(players, shouldBuildModels);

  // Apply models to convert all ESPN projections to fpProjPts scale
  // REPLACE original ESPN columns with converted versions
  Object.keys(players).forEach((pos) => {
    if (pos === 'FLEX') return;

    players[pos].data = players[pos].data.map((p) => {
      // Convert ESPN projections using learned models, REPLACING originals
      let espnOutsideProjection = p.espnOutsideProjection;
      let espnLowScore = p.espnLowScore;
      let espnHighScore = p.espnHighScore;
      let espnScoreProjection = p.espnScoreProjection;
      let espnSimulationProjection = p.espnSimulationProjection;

      if (p.espnOutsideProjection) {
        // Use espnOutside model for Outside, Low, High (same underlying model)
        espnOutsideProjection = parseFloat((models.espnOutside.intercept + models.espnOutside.slope * p.espnOutsideProjection).toFixed(2));
        espnLowScore = parseFloat((models.espnOutside.intercept + models.espnOutside.slope * p.espnLowScore).toFixed(2));
        espnHighScore = parseFloat((models.espnOutside.intercept + models.espnOutside.slope * p.espnHighScore).toFixed(2));
      }

      if (p.espnScoreProjection) {
        espnScoreProjection = parseFloat((models.espnScore.intercept + models.espnScore.slope * p.espnScoreProjection).toFixed(2));
      }

      if (p.espnSimulationProjection) {
        espnSimulationProjection = parseFloat((models.espnSim.intercept + models.espnSim.slope * p.espnSimulationProjection).toFixed(2));
      }

      // Calculate consensus and uncertainty (all in fpProjPts scale)
      const projections_fp_scale = [
        p.fpProjPts,
        espnOutsideProjection,
        espnLowScore,
        espnHighScore,
        espnScoreProjection,
        espnSimulationProjection,
      ].filter(v => v !== null && v !== "" && !isNaN(v));

      let consensus = null;
      let uncertainty = null;

      if (projections_fp_scale.length > 0) {
        consensus = parseFloat((projections_fp_scale.reduce((sum, v) => sum + v, 0) / projections_fp_scale.length).toFixed(2));

        if (projections_fp_scale.length > 1) {
          const variance = projections_fp_scale.reduce((sum, v) => sum + Math.pow(v - consensus, 2), 0) / projections_fp_scale.length;
          uncertainty = parseFloat(Math.sqrt(variance).toFixed(2));
        } else {
          uncertainty = 0;
        }
      }

      // Apply TD odds and game script adjustments to consensus
      let adjustedConsensus = consensus;

      // TD Odds boost - ONLY for skill positions (RB, WR, TE)
      // Exclude QBs and Defenses
      if (consensus && p.tdProbability && p.tdProbability > 0) {
        if (p.position === 'RB' || p.position === 'WR' || p.position === 'TE') {
          // TD Odds boost: Higher TD probability = higher boost
          // Scale: 20% TD prob = +10% boost, 40% = +20% boost, 60% = +30% boost
          const tdBoost = (p.tdProbability / 100) * 0.5; // Max 50% boost at 100% TD prob
          adjustedConsensus = consensus * (1 + tdBoost);

          // Also increase uncertainty for high TD probability players (more variance)
          if (uncertainty) {
            uncertainty = uncertainty * (1 + tdBoost * 0.5);
          }
        }
      }

      // Game script adjustments
      if (adjustedConsensus && p.projTeamPts && p.projOppPts) {
        const spread = p.projTeamPts - p.projOppPts;
        const total = p.projTeamPts + p.projOppPts;

        if (p.position === 'RB') {
          // RBs benefit from positive game script (team winning = more rush attempts)
          if (spread > 3) {
            adjustedConsensus *= 1.08; // +8% boost for RBs on favored teams
          } else if (spread < -3) {
            adjustedConsensus *= 0.94; // -6% penalty for RBs on trailing teams
          }
        } else if (p.position === 'WR' || p.position === 'TE') {
          // Pass catchers benefit from negative game script (team trailing = more pass attempts)
          if (spread < -3) {
            adjustedConsensus *= 1.10; // +10% boost for pass catchers on trailing teams
          } else if (spread > 7) {
            adjustedConsensus *= 0.96; // -4% penalty when team is heavily favored (run-heavy)
          }
        } else if (p.position === 'QB') {
          // QBs benefit from high-scoring games
          if (total > 50) {
            adjustedConsensus *= 1.08; // +8% boost for QBs in shootouts
          } else if (total < 42) {
            adjustedConsensus *= 0.95; // -5% penalty for QBs in low-scoring games
          }
        } else if (p.position === 'D') {
          // Defenses benefit from facing weak offenses
          if (p.projOppPts < 18) {
            adjustedConsensus *= 1.25; // +25% boost for defenses vs weak offenses
          } else if (p.projOppPts > 26) {
            adjustedConsensus *= 0.80; // -20% penalty vs strong offenses
          }
        }
      }

      consensus = parseFloat(adjustedConsensus.toFixed(2));
      uncertainty = parseFloat(uncertainty.toFixed(2));

      // Calculate P90 ceiling value using log-normal distribution (exact, no simulation needed)
      let p90 = 0;
      let ceilingValue = 0;

      if (consensus > 0 && p.salary > 0) {
        const mean = consensus;
        const std = uncertainty;
        const variance = std * std;
        const sigmaSquared = Math.log(1 + variance / (mean * mean));
        const mu = Math.log(mean) - sigmaSquared / 2;
        const sigma = Math.sqrt(sigmaSquared);

        // Direct calculation: P90 = exp(mu + sigma * z_0.90)
        // where z_0.90 ≈ 1.2816 (90th percentile of standard normal)
        const z90 = 1.2815515655446004; // More precise value
        p90 = parseFloat(Math.exp(mu + sigma * z90).toFixed(2));
        ceilingValue = parseFloat((p90 / (p.salary / 1000)).toFixed(3));
      }

      // Reorder fields to match desired column order
      return {
        name: p.name,
        position: p.position,
        team: p.team,
        game: p.game,
        salary: p.salary,
        value: p.value,
        ceilingValue,
        consensus,
        p90,
        fpProjPts: p.fpProjPts,
        espnScoreProjection,
        espnLowScore,
        espnHighScore,
        espnOutsideProjection,
        espnSimulationProjection,
        uncertainty,
        projTeamPts: p.projTeamPts,
        projOppPts: p.projOppPts,
        tdOdds: p.tdOdds,
        tdProbability: p.tdProbability,
        espnId: p.espnId,
        injury: p.injury,
      };
    });
  });

  return { players, models: shouldBuildModels ? models : null };
}

// ============================================================================
// STEP 10: ADD FLEX POSITION
// ============================================================================

function addFlex(players) {
  const rbs = players["RB"]["data"];
  const wrs = players["WR"]["data"];
  const tes = players["TE"]["data"];
  const flexs = [...rbs, ...wrs, ...tes];
  flexs.sort((a, b) => (a.value > b.value ? -1 : 1));
  players["FLEX"] = { data: flexs };
  return players;
}

// ============================================================================
// STEP 11: WRITE CSV FILES
// ============================================================================

async function writeRegressionModels(models) {
  console.log("\n=== Writing Regression Models ===\n");

  const modelData = Object.keys(models).map((name) => ({
    model: name,
    intercept: models[name].intercept,
    slope: models[name].slope,
    r2: models[name].r2,
  }));

  const writer = createObjectCsvWriter({
    path: "./regression_models.csv",
    header: [
      { id: "model", title: "model" },
      { id: "intercept", title: "intercept" },
      { id: "slope", title: "slope" },
      { id: "r2", title: "r2" },
    ],
  });

  await writer.writeRecords(modelData);
  console.log("  Wrote regression_models.csv");
}

async function writeCsvs(players) {
  console.log("\n=== STEP 11: Writing CSV Files ===\n");

  let allData = [];

  let allPromises = Object.keys(players).map(async (pos) => {
    const data = players[pos].data;
    if (pos !== "FLEX") {
      allData = [...allData, ...data];
    }

    // Sort by ceiling value (descending)
    data.sort((a, b) => (b.ceilingValue || 0) - (a.ceilingValue || 0));

    console.log(`Writing ${data.length} ${pos}s to fanduel-${pos}.csv (sorted by ceiling value)`);

    const header = Object.keys(data[0]).map((k) => {
      return { id: k, title: k };
    });
    const writer = createObjectCsvWriter({
      path: `./fanduel-${pos}.csv`,
      header,
    });
    await writer.writeRecords(data);
  });

  const header = Object.keys(allData[0]).map((k) => {
    return { id: k, title: k };
  });
  const writer = createObjectCsvWriter({
    path: `./knapsack.csv`,
    header,
  });
  allPromises = [...allPromises, writer.writeRecords(allData)];

  await Promise.all(allPromises);
  console.log(`\nWrote knapsack.csv with ${allData.length} total players`);
}

// ============================================================================
// MAIN ORCHESTRATION
// ============================================================================

async function main() {
  const args = process.argv.slice(2);

  // Show help if help flag is present
  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    return;
  }

  // Parse command line arguments - these control whether to FETCH or use CACHE
  // If no args provided, all flags default to false (use cached data)
  const shouldFetch = {
    fantasypros: args.includes("--fp") || args.includes("--all"),
    odds: args.includes("--dk") || args.includes("--all"),
    gamelines: args.includes("--lines") || args.includes("--all"),
    espn: args.includes("--espn") || args.includes("--all"),
  };

  // Parse --build-models flag (independent of fetch flags)
  const shouldBuildModels = args.includes("--build-models");

  console.log("\n" + "=".repeat(80));
  console.log("FANTASY VALUE DATA PIPELINE");
  console.log("=".repeat(80));

  console.log("\nData Sources:");
  console.log(`  FantasyPros Projections: ${shouldFetch.fantasypros ? "FETCH" : "CACHE"}`);
  console.log(`  FanDuel Salaries: ALWAYS (local join)`);
  console.log(`  DraftKings Game Lines: ${shouldFetch.gamelines ? "FETCH" : "CACHE"}`);
  console.log(`  DraftKings TD Odds: ${shouldFetch.odds ? "FETCH" : "CACHE"}`);
  console.log(`  ESPN Player IDs: ${shouldFetch.espn ? "FETCH from API" : "CACHE"}`);
  console.log(`  ESPN Projections: ${shouldFetch.espn ? "FETCH" : "CACHE"}`);
  console.log(`  Regression Models: ${shouldBuildModels ? "BUILD from data" : "LOAD from cache"}`);

  let players;

  // Step 1: FantasyPros or load existing
  if (shouldFetch.fantasypros) {
    players = await fetchFantasyProsProjections();
  } else {
    console.log("\n=== STEP 1: Loading Cached FantasyPros Data ===\n");
    players = {};
    for (const pos of ["QB", "RB", "WR", "TE", "D"]) {
      const data = await loadExistingData(pos);
      if (!data) {
        throw new Error(`No existing data found for ${pos}. Run with --fp first.`);
      }
      players[pos] = { data };
      console.log(`  Loaded ${data.length} ${pos}s from fanduel-${pos}.csv`);
    }
  }

  // Step 2: FanDuel Salaries (always run - local data)
  players = await addFanduelSalaries(players);

  // Step 3: DraftKings Game Lines (ALWAYS runs - fetch or use cache)
  await fetchGameLines(shouldFetch.gamelines);

  // Step 4: DraftKings TD Odds (ALWAYS runs - fetch or use cache)
  players = await addDraftKingsTdOdds(players, shouldFetch.odds);

  // Step 5: ESPN IDs (always run - fetch from API if running ESPN projections)
  players = await addEspnIds(players, shouldFetch.espn);

  // Step 6: ESPN Projections (ALWAYS runs - fetch or use cache)
  players = await addEspnProjections(players, shouldFetch.espn);

  // Step 7: Game Lines Data (always run - joins if available)
  players = await addGameLinesData(players);

  // Step 8: Analysis (always run)
  players = addAnalysis(players);

  // Step 9: Convert Projections to fpProjPts Scale (always run)
  const { players: convertedPlayers, models } = await convertProjectionsToFpScale(players, shouldBuildModels);
  players = convertedPlayers;

  // Step 10: Add FLEX (always run)
  players = addFlex(players);

  // Step 11: Write CSVs (always run)
  await writeCsvs(players);

  // Only write regression models if we just built them
  if (shouldBuildModels) {
    await writeRegressionModels(models);
  }

  console.log("\n" + "=".repeat(80));
  console.log("COMPLETE!");
  console.log("=".repeat(80));
  console.log("\nOutput files:");
  console.log("  - knapsack.csv (all players with fpProjPts-scaled projections, consensus, and uncertainty)");
  console.log("  - fanduel-QB.csv, fanduel-RB.csv, fanduel-WR.csv, fanduel-TE.csv, fanduel-D.csv, fanduel-FLEX.csv");
  if (shouldBuildModels) {
    console.log("  - regression_models.csv (ESPN→FP conversion models with R² values) [REBUILT]");
  }
  console.log("  - game_lines.csv (spreads, totals, projected points by team)");
  console.log("  - td_odds.json (cached TD odds data)");
  console.log("  - espn_projections.json (cached ESPN projections)");
}

function printUsage() {
  console.log(`
Usage: node fetch_data.js [options]

Options:
  --all            Fetch all data sources (full refresh)
  --fp             Fetch FantasyPros projections
  --lines          Fetch DraftKings game lines (spreads, totals)
  --dk             Fetch DraftKings TD odds
  --espn           Fetch ESPN projections (also refreshes ESPN player list from API)
  --build-models   Rebuild regression models from data (default: load from regression_models.csv)
  --help, -h       Show this help message

How it works:
  - Flags control whether to FETCH fresh data or use CACHED data
  - All processing steps ALWAYS run (analysis, joins, CSV generation)
  - Cached data is stored in: game_lines.csv, td_odds.json, espn_projections.json, regression_models.csv
  - Regression models convert ESPN projections to fpProjPts scale for apples-to-apples comparison

Examples:
  node fetch_data.js                    # Use all cached data (regenerate CSVs only)
  node fetch_data.js --all              # Fetch everything fresh
  node fetch_data.js --dk               # Fetch TD odds, use cached data for rest
  node fetch_data.js --lines --dk       # Fetch game lines and TD odds, cache for rest
  node fetch_data.js --espn             # Fetch ESPN projections (+ player list), cache for rest
  node fetch_data.js --fp               # Fetch FantasyPros, use cached data for rest
  node fetch_data.js --build-models     # Rebuild regression models from current data
  node fetch_data.js --all --build-models  # Full refresh including model rebuild
`);
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error("\n❌ Error:", error.message);
    process.exit(1);
  });
}

export {
  fetchFantasyProsProjections,
  addFanduelSalaries,
  addDraftKingsTdOdds,
  addEspnIds,
  addEspnProjections,
  addAnalysis,
  addFlex,
  writeCsvs,
};
