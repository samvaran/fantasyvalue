import { launch } from "puppeteer";
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

function delay(time) {
  return new Promise(function (resolve) {
    setTimeout(resolve, time);
  });
}

const scrapeGameLines = async () => {
  const url = "https://sportsbook.draftkings.com/leagues/football/nfl";
  const browser = await launch({
    headless: true,
    args: MINIMAL_ARGS,
  });
  console.log("LAUNCHED game lines", url);

  const page = await browser.newPage();

  // Enable request interception to capture API calls
  const apiResponses = [];
  const allApiUrls = [];

  page.on('response', async (response) => {
    const url = response.url();
    if (url.includes('api')) {
      allApiUrls.push(url);
      // Capture any API response that might have NFL data
      if (url.includes('88808') || url.includes('event') || url.includes('nfl')) {
        try {
          const data = await response.json();
          apiResponses.push({ url, data });
        } catch (e) {
          // Not JSON, ignore
        }
      }
    }
  });

  await page.goto(url, { waitUntil: "load", timeout: 0 });
  await delay(4000); // Wait for API calls to complete

  await browser.close();

  console.log(`\nAll API URLs (${allApiUrls.length}):`);
  allApiUrls.forEach(u => console.log(u));

  console.log(`\nCaptured ${apiResponses.length} API responses with data`);

  const games = [];
  const daysOfWeek = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  // Save first relevant API response for inspection
  if (apiResponses.length > 0) {
    const fs = await import("fs/promises");
    await fs.writeFile("api-response.json", JSON.stringify(apiResponses[0].data, null, 2));
    console.log("Saved first API response to api-response.json");
  }

  // Process captured API data
  apiResponses.forEach(({ url, data }) => {
    console.log(`Processing API response from: ${url.slice(0, 100)}...`);

    if (data && data.eventGroup && data.eventGroup.events) {
      data.eventGroup.events.forEach((event) => {
        try {
          const startDate = new Date(event.startDate);
          const dayOfWeek = daysOfWeek[startDate.getDay()];

          // Only include Sunday and Monday games
          if (dayOfWeek !== 'Sun' && dayOfWeek !== 'Mon') {
            return;
          }

          const teamOne = event.teamName1 || event.name1;
          const teamTwo = event.teamName2 || event.name2;

          if (!teamOne || !teamTwo) return;

          let awaySpread = null;
          let homeSpread = null;
          let gameTotal = null;

          if (event.displayGroups) {
            event.displayGroups.forEach((displayGroup) => {
              if (displayGroup.markets) {
                displayGroup.markets.forEach((market) => {
                  if (market.name && market.name.toLowerCase().includes('spread')) {
                    if (market.outcomes && market.outcomes.length >= 2) {
                      awaySpread = market.outcomes[0].line;
                      homeSpread = market.outcomes[1].line;
                    }
                  }

                  if (market.name && market.name.toLowerCase().includes('total')) {
                    if (market.outcomes && market.outcomes.length > 0) {
                      gameTotal = market.outcomes[0].line;
                    }
                  }
                });
              }
            });
          }

          games.push({
            team: teamOne,
            opponent: teamTwo,
            spread: awaySpread,
            gameTotal: gameTotal,
            gameDay: dayOfWeek,
            startTime: startDate.toLocaleString(),
            isHome: false
          });

          games.push({
            team: teamTwo,
            opponent: teamOne,
            spread: homeSpread,
            gameTotal: gameTotal,
            gameDay: dayOfWeek,
            startTime: startDate.toLocaleString(),
            isHome: true
          });

          console.log(`Found game: ${teamOne} @ ${teamTwo} on ${dayOfWeek}, spreads: [${awaySpread}, ${homeSpread}], total: ${gameTotal}`);

        } catch (err) {
          console.error('Error parsing event:', err);
        }
      });
    }
  });

  console.log(`\nTotal teams found: ${games.length}`);
  return games;
};

const writeGameLinesToCsv = async (gameLines) => {
  if (gameLines.length === 0) {
    console.log("No game lines to write");
    return;
  }

  console.log(`\nWriting ${gameLines.length} game lines to CSV...`);

  const header = Object.keys(gameLines[0]).map((k) => {
    return { id: k, title: k };
  });
  const writer = createObjectCsvWriter({
    path: `./game-lines.csv`,
    header,
  });
  await writer.writeRecords(gameLines);
  console.log("Game lines written to game-lines.csv");
};

async function main() {
  try {
    const gameLines = await scrapeGameLines();
    await writeGameLinesToCsv(gameLines);
  } catch (error) {
    console.error("Error:", error);
    console.error(error.stack);
  }
}

main();
