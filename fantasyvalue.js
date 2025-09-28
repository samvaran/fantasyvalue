import { load } from "cheerio";
import { launch } from "puppeteer";
import neatCsv from "neat-csv";
import fsextra from "fs-extra";
const { readFile } = fsextra;
import { createObjectCsvWriter } from "csv-writer";

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

function sigmoid(x) {
  const scalingFactor = 10;
  return 1 / (1 + Math.exp(-1 * scalingFactor * (x - 0.45)));
}

const writeCsvs = async (players) => {
  let allData = [];

  let allPromises = Object.keys(players).map(async (pos) => {
    const data = players[pos].data;
    if (pos !== "FLEX") {
      allData = [...allData, ...data];
    }

    console.log(`Writing ${data.length} ${pos}s to CSV...`);

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
};

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

const addAnalysis = (players) => {
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data.map((d) => {
      let value = (d.projPts / (d.salary / 1000)).toFixed(2);
      let tdProbSigmoid = (sigmoid(d.tdProbability / 100) * 100).toFixed(2);
      let tdValue = (tdProbSigmoid / (d.salary / 1000)).toFixed(2);
      d = { ...d, value };

      return {
        name: removePunc(d.name),
        position: d.position,
        team: d.team,
        game: d.game,
        projPts: Number(d.projPts),
        salary: Number(d.salary),
        value: Number(d.value),
        tdOdds: d.tdOdds,
        tdProbability: d.tdProbability,
        tdProbSigmoid: tdProbSigmoid,
        tdValue: tdValue,
        injury: d.injury + ": " + d.injuryDetail,
      };
    });

    augmentedData.sort((a, b) => (a.value > b.value ? -1 : 1));

    players[pos].data = augmentedData;
  });
  return players;
};

const addSalaries = async (players) => {
  // Find all CSV files starting with "FanDuel-NFL"
  const fs = await import("fs/promises");
  const files = await fs.readdir("./");
  const fanduelFiles = files
    .filter((file) => file.startsWith("FanDuel-NFL") && file.endsWith(".csv"))
    .sort();

  if (fanduelFiles.length === 0) {
    throw new Error('No CSV files starting with "FanDuel-NFL" found');
  }

  // Pick the last file after sorting
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

  console.log("EXCLUDED SALARIES");
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data
      .map((d) => {
        let salaryData = dataByPlayer[d.name] || {};
        if (!dataByPlayer[d.name]) {
          console.log(d.name);
        }
        return { ...d, ...salaryData };
      })
      .filter((d) => {
        return !!d.salary && d.injury != "IR" && d.projPts > 0;
      });
    players[pos].data = augmentedData;
  });

  return players;
};

const scrapeTable = async (url, pos) => {
  // const result = await request.get(url);

  const browser = await launch({
    headless: true,
    args: MINIMAL_ARGS,
  });
  console.log("LAUNCHED", url);

  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 0 });
  await page.$eval(
    ".everything-but-phone .select-advanced-content__text",
    (el) => el.click()
  );
  await delay(500);
  // Scrolling to the bottom of the page
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });
  await delay(500);
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });
  const content = await page.content();

  const $ = load(content);
  console.log("CLOSED", url, content.length);
  await browser.close();

  const headers = url.includes("dst.php")
    ? ["rank", "wsis", "name", "opp", "matchup", "grade", "projPts"]
    : [
        "rank",
        "wsis",
        "name",
        "opp",
        "upside",
        "bust",
        "matchup",
        "grade",
        "projPts",
      ];

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
            //pass
          } else if (headers[i] == "projPts") {
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
};

const addPredictions = async (players) => {
  let allPromises = Object.keys(players).map(async (pos) => {
    await delay(500);
    const url = players[pos].url;
    let data = await scrapeTable(url, pos);
    // data = data.filter((p) => p.projPts >= 3.0);
    players[pos].data = data;
    return data;
  });

  await Promise.all(allPromises);
  return players;
};

function delay(time) {
  return new Promise(function (resolve) {
    setTimeout(resolve, time);
  });
}

const probabilityString = (prob) => {
  return String((prob * 100).toFixed(1)) + "%";
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

const addOdds = async (players) => {
  const url =
    "https://sportsbook.draftkings.com/leagues/football/nfl?category=td-scorers&subcategory=td-scorer";
  const browser = await launch({
    headless: true,
    args: MINIMAL_ARGS,
  });
  console.log("LAUNCHED", url);

  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 0 });
  await delay(500);
  const content = await page.content();

  const dataByPlayer = {};

  const $ = load(content);
  console.log("CLOSED", url, content.length);
  await browser.close();

  $(".cb-market__template--4-columns").each((gameIndex, game) => {
    // Get all cells, skip first 3 (slice(3))
    const cells = $(game).children().toArray().slice(3);
    const columnCount = 4;

    // Get players from column 1 (index 0)
    const players = cells
      .filter((c, index) => index % columnCount === 0)
      .map((c) => $(c).text().trim())
      .filter((text) => !text.toLowerCase().includes("no touchdown"));

    // Get TD odds from column 3 (index 2)
    const tdOddsList = cells
      .filter((c, index) => index % columnCount === 2)
      .map((c) => $(c).text().trim())
      .filter(Boolean);

    // Process each player
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

  console.log("EXCLUDED ODDS");
  Object.keys(players).forEach((pos) => {
    let augmentedData = players[pos].data.map((d) => {
      let oddsData = dataByPlayer[d.name] || { tdOdds: 0, tdProbability: "0" };
      if (!dataByPlayer[d.name]) {
        console.log(d.name);
      }
      return { ...d, ...oddsData };
    });
    players[pos].data = augmentedData;
  });

  return players;
};

const addFlex = (players) => {
  const rbs = players["RB"]["data"];
  const wrs = players["WR"]["data"];
  const tes = players["TE"]["data"];
  const flexs = [...rbs, ...wrs, ...tes];
  flexs.sort((a, b) => (a.value > b.value ? -1 : 1));
  players["FLEX"] = { data: [] };
  players["FLEX"].data = flexs;
  return players;
};

async function main() {
  let players = {
    QB: { url: "https://www.fantasypros.com/nfl/rankings/qb.php" },
    RB: {
      url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-rb.php",
    },
    WR: {
      url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-wr.php",
    },
    TE: {
      url: "https://www.fantasypros.com/nfl/rankings/half-point-ppr-te.php",
    },
    D: { url: "https://www.fantasypros.com/nfl/rankings/dst.php" },
  };
  players = await addPredictions(players);
  console.log("players", players);
  players = await addSalaries(players);
  console.log("players2", players);
  players = await addOdds(players);
  console.log("players3", players);
  players = addAnalysis(players);
  console.log("players4", players);
  players = addFlex(players);

  await writeCsvs(players);
}

async function test() {
  const data = await addOdds();
  console.log(data);
}

main();
// test();
