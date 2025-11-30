"""
Web scrapers for fantasy football data sources.

All scrapers inherit from BaseScraper and implement automatic caching.
"""
import json
import time
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils import (
    normalize_name,
    odds_to_probability,
    get_team_abbreviation,
    parse_team_name,
    get_cache_dir,
    find_latest_fanduel_csv,
)


# ============================================================================
# BASE SCRAPER
# ============================================================================

class BaseScraper(ABC):
    """
    Base class for all scrapers with built-in caching.
    """

    def __init__(self, cache_name: str):
        """
        Args:
            cache_name: Name for cache file (e.g., "fantasypros", "td_odds")
        """
        self.cache_name = cache_name
        self.cache_dir = get_cache_dir()

    @abstractmethod
    def _fetch(self) -> Any:
        """
        Subclass implements actual scraping logic.

        Returns:
            Scraped data (format depends on scraper)
        """
        pass

    @property
    def cache_file(self) -> Path:
        """Get cache file path (JSON)"""
        return self.cache_dir / f"{self.cache_name}.json"

    def get_data(self, use_cache: bool = True) -> Any:
        """
        Get data (from cache or by fetching).

        Args:
            use_cache: If True, load from cache if available

        Returns:
            Scraped data
        """
        if use_cache and self.cache_file.exists():
            print(f"  Loading from cache: {self.cache_file.name}")
            return self._load_cache()

        print(f"  Fetching fresh data...")
        data = self._fetch()
        self._save_cache(data)
        print(f"  Cached to {self.cache_file.name}")
        return data

    def _load_cache(self) -> Any:
        """Load data from cache"""
        with open(self.cache_file, 'r') as f:
            return json.load(f)

    def _save_cache(self, data: Any) -> None:
        """Save data to cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# SELENIUM DRIVER UTILITIES
# ============================================================================

def create_headless_driver() -> webdriver.Chrome:
    """
    Create a headless Chrome driver with minimal args (matching puppeteer MINIMAL_ARGS).
    """
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-background-networking')
    options.add_argument('--disable-sync')
    options.add_argument('--mute-audio')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    return webdriver.Chrome(options=options)


# ============================================================================
# FANTASYPROS SCRAPER
# ============================================================================

class FantasyProsScraper(BaseScraper):
    """
    Scrape FantasyPros consensus projections for all positions.
    """

    URLS = {
        'QB': 'https://www.fantasypros.com/nfl/rankings/qb.php',
        'RB': 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-rb.php',
        'WR': 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-wr.php',
        'TE': 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-te.php',
        'D': 'https://www.fantasypros.com/nfl/rankings/dst.php',
    }

    def __init__(self):
        super().__init__('fantasypros')

    def _fetch(self) -> Dict[str, List[Dict]]:
        """
        Fetch FantasyPros projections for all positions.

        Returns:
            Dict mapping position -> list of player dicts
        """
        all_data = {}

        for position, url in self.URLS.items():
            print(f"    Fetching {position}...")
            time.sleep(0.5)  # Rate limiting

            driver = create_headless_driver()
            try:
                driver.get(url)

                # Expand dropdown to show all players
                try:
                    dropdown = driver.find_element(
                        By.CSS_SELECTOR,
                        '.everything-but-phone .select-advanced-content__text'
                    )
                    dropdown.click()
                    time.sleep(0.5)
                except:
                    pass  # Dropdown not found, continue anyway

                # Scroll to load all content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)

                # Extract data using JavaScript (matching the JS implementation)
                players = driver.execute_script("""
                    const rows = [];
                    const tableRows = document.querySelectorAll('#ranking-table tbody tr.player-row');

                    tableRows.forEach(row => {
                        const cells = row.querySelectorAll('td');

                        // Find player name
                        let nameCell = row.querySelector('.player-cell a');
                        if (!nameCell) {
                            nameCell = row.querySelector('a[class*="player"]');
                        }
                        let playerName = nameCell ? nameCell.textContent.trim() : '';

                        // Find projection points (last numeric cell)
                        let projPts = '';
                        for (let i = cells.length - 1; i >= 0; i--) {
                            const text = cells[i].textContent.trim();
                            if (text && !isNaN(parseFloat(text)) && text !== '-') {
                                projPts = text;
                                break;
                            }
                        }

                        if (!projPts || projPts === '-') {
                            projPts = '3';
                        }

                        if (playerName) {
                            rows.push({
                                name: playerName,
                                fpProjPts: projPts
                            });
                        }
                    });

                    return rows;
                """)

                # Normalize names
                normalized_players = []
                for player in players:
                    name = normalize_name(player['name'], is_defense=(position == 'D'))
                    if name:
                        normalized_players.append({
                            'name': name,
                            'fpProjPts': float(player['fpProjPts'])
                        })

                all_data[position] = normalized_players
                print(f"      Parsed {len(normalized_players)} {position} players")

            finally:
                driver.quit()

        return all_data


# ============================================================================
# FANDUEL SALARY LOADER
# ============================================================================

class FanDuelLoader(BaseScraper):
    """
    Load FanDuel salaries from local CSV file.

    Note: This loader ALWAYS reads from the CSV file and never caches,
    since the CSV is updated weekly and is the source of truth.
    """

    def __init__(self):
        super().__init__('fanduel_salaries')

    def get_data(self, use_cache: bool = True) -> Dict[str, Dict]:
        """
        Override get_data to always read from CSV (no caching).

        Args:
            use_cache: Ignored - always reads from CSV

        Returns:
            Player data from CSV
        """
        # Always fetch fresh from CSV, never cache
        return self._fetch()

    def _fetch(self) -> Dict[str, Dict]:
        """
        Load FanDuel salary CSV.

        Returns:
            Dict mapping normalized player name -> player data
        """
        csv_file = find_latest_fanduel_csv()
        print(f"    Reading from: {csv_file.name}")

        df = pd.read_csv(csv_file)

        players_by_name = {}
        for _, row in df.iterrows():
            # Extract name (Nickname for players, Last Name for defenses)
            if row['Position'] == 'D':
                name = row['Last Name']
            else:
                name = row['Nickname']

            name = normalize_name(name, is_defense=(row['Position'] == 'D'))

            players_by_name[name] = {
                'name': name,
                'fppg': float(row.get('FPPG', 0)),
                'played': int(row.get('Played', 0)),
                'salary': int(row['Salary']),
                'game': row['Game'],
                'team': row['Team'],
                'position': row['Position'],
                'opponent': row['Opponent'],
                'injury_status': row.get('Injury Indicator', ''),
                'injury_detail': row.get('Injury Details', ''),
            }

        return players_by_name


# ============================================================================
# DRAFTKINGS GAME LINES SCRAPER
# ============================================================================

class DraftKingsGameLinesScraper(BaseScraper):
    """
    Scrape DraftKings game lines (spreads, totals, projected points).
    """

    URL = 'https://sportsbook.draftkings.com/leagues/football/nfl?category=game-lines&subcategory=game'

    def __init__(self):
        super().__init__('game_lines')

    def _fetch(self) -> List[Dict]:
        """
        Scrape game lines from DraftKings.

        Returns:
            List of game line dicts (one per team)
        """
        driver = create_headless_driver()
        try:
            driver.get(self.URL)
            time.sleep(3)  # Wait for page load

            # Scroll to load all games
            driver.execute_script("""
                return new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 500;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;

                        if (totalHeight >= scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 200);
                });
            """)
            time.sleep(2)

            # Extract page text
            element = driver.find_element(By.CSS_SELECTOR, '.cms-market-selector-content')
            page_text = element.text

            # Save raw text for debugging
            output_dir = Path('data/intermediate')
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_file = output_dir / 'game_lines_raw.txt'
            raw_file.write_text(page_text)
            print(f"      Saved raw text to {raw_file}")

            # Parse game lines
            games = self._parse_game_lines_text(page_text)
            print(f"      Found {len(games) // 2} games")

            return games

        finally:
            driver.quit()

    def _parse_game_lines_text(self, text: str) -> List[Dict]:
        """
        Parse game lines from DraftKings page text.

        This is a direct port of parseGameLinesText() from fetch_data.js.
        """
        from datetime import datetime, timedelta

        # Calculate this week's Monday cutoff
        # NFL week runs Thursday-Monday, so we include games up to and including next Monday
        now = datetime.now()
        days_until_monday = (7 - now.weekday()) % 7  # 0 = Monday, 6 = Sunday
        if days_until_monday == 0 and now.weekday() == 0:
            # If today is Monday, include today
            cutoff_date = now
        else:
            # Get next Monday
            cutoff_date = now + timedelta(days=days_until_monday if days_until_monday > 0 else 7)

        print(f"        Today: {now.strftime('%A %b %d')}")
        print(f"        Cutoff: {cutoff_date.strftime('%A %b %d')} (this week's Monday)")

        lines = [l.strip() for l in text.split('\n') if l.strip()]

        games = []
        i = 0
        games_found = 0
        in_today_section = False
        in_tomorrow_section = False
        current_game_time = None

        def is_valid_game_time(time_str: str) -> bool:
            """Check if game is at or after 10:00 AM PT"""
            match = re.match(r'^(\d{1,2}):(\d{2})\s+(AM|PM)', time_str)
            if not match:
                return True  # No time, assume valid

            hour, minute, meridiem = match.groups()
            hour = int(hour)
            minute = int(minute)

            # Convert to 24-hour
            if meridiem == 'PM' and hour != 12:
                hour += 12
            if meridiem == 'AM' and hour == 12:
                hour = 0

            # Get local timezone offset
            now = datetime.now(timezone.utc).astimezone()
            local_offset_min = int(now.utcoffset().total_seconds() / 60)

            # PT offset (PST = UTC-8 = 480, PDT = UTC-7 = 420)
            # Simple DST check
            jan = datetime(now.year, 1, 1, tzinfo=timezone.utc).astimezone()
            jul = datetime(now.year, 7, 1, tzinfo=timezone.utc).astimezone()
            is_dst = now.utcoffset() < max(jan.utcoffset(), jul.utcoffset())
            pt_offset_min = 420 if is_dst else 480  # PDT or PST

            # Convert local to PT
            local_minutes = hour * 60 + minute
            offset_diff = pt_offset_min - local_offset_min
            pt_minutes = local_minutes + offset_diff
            pt_hour = (pt_minutes % 1440) // 60
            pt_min = pt_minutes % 60

            # Check if at or after 10:00 AM PT
            after_cutoff = pt_hour > 10 or (pt_hour == 10 and pt_min >= 0)

            print(f"        Time check: {time_str} → {pt_hour}:{pt_min:02d} PT → {'INCLUDE' if after_cutoff else 'SKIP (before 10 AM PT)'}")
            return after_cutoff

        while i < len(lines):
            line = lines[i]

            # Track section markers
            if line == "Today":
                in_today_section = True
                in_tomorrow_section = False
                print("        Entering TODAY section")
                i += 1
                continue

            if line == "Tomorrow":
                in_today_section = False
                in_tomorrow_section = True
                print("        Entering TOMORROW section")
                i += 1
                continue

            # Capture game time
            time_match = re.match(r'^(Today|Tomorrow)\s+(\d{1,2}:\d{2}\s+(?:AM|PM))', line)
            if time_match:
                current_game_time = time_match.group(2)
                section = time_match.group(1)
                if section == "Today":
                    in_today_section = True
                    in_tomorrow_section = False
                elif section == "Tomorrow":
                    in_today_section = False
                    in_tomorrow_section = True
                i += 1
                continue

            # Standalone time
            if re.match(r'^\d{1,2}:\d{2}\s+(AM|PM)$', line):
                current_game_time = line
                i += 1
                continue

            # Stop at future dates (beyond this week's Monday)
            # Format: "THU DEC 4th" or "SUN DEC 7th"
            date_match = re.match(r'^[A-Z]{3}\s+([A-Z]{3})\s+(\d{1,2})(st|nd|rd|th)', line)
            if date_match:
                month_abbr = date_match.group(1)
                day = int(date_match.group(2))

                # Parse the date
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_abbr, now.month)

                # Assume current year, but handle year rollover
                year = now.year
                if month < now.month:
                    year += 1

                game_date = datetime(year, month, day)

                # Check if this game is after this week's Monday
                if game_date.date() > cutoff_date.date():
                    print(f"        Stopping at future date \"{line}\" ({game_date.strftime('%b %d')}) after cutoff ({cutoff_date.strftime('%b %d')})")
                    print(f"        Found {games_found} games for this week")
                    break

            # Stop conditions
            if games_found > 0:
                if line.startswith("GAME LINES") or line == "LIVE BLITZ ⚡":
                    print(f"        Stopping at section \"{line}\" after finding {games_found} games")
                    break

            # Only process in Today/Tomorrow sections
            if not in_today_section and not in_tomorrow_section:
                i += 1
                continue

            # Skip headers
            if line in ["Today", "Tomorrow", "Spread", "Total", "Moneyline", "AT", "Opt In"] or line.startswith("Get A Profit") or line.startswith("GAME LINES"):
                i += 1
                continue

            # Look for team names
            team_pattern = re.compile(r'^[A-Z]{2,3}\s+[A-Za-z0-9]+')

            if team_pattern.match(line):
                team1 = parse_team_name(line)
                i += 1

                # Skip score if present
                if i < len(lines) and re.match(r'^\d{1,2}$', lines[i]):
                    i += 1

                # Look for "AT" separator
                if i < len(lines) and lines[i] == "AT":
                    i += 1

                # Get team2
                if i >= len(lines) or not team_pattern.match(lines[i]):
                    continue

                team2 = parse_team_name(lines[i])
                i += 1

                # Skip score if present
                if i < len(lines) and re.match(r'^\d{1,2}$', lines[i]):
                    i += 1

                # Look for spreads, totals, and odds
                # Pattern for each team:
                # 1. Spread (e.g., -7.5)
                # 2. Spread odds (e.g., −108)
                # 3. O (over indicator)
                # 4. Total (e.g., 50.5)
                # 5. Over odds (e.g., −118)
                # 6. Moneyline (e.g., −395) ← First 3-digit number AFTER over odds
                # Then repeat for team 2 with U instead of O

                team1_spread = None
                team1_spread_odds = None
                team2_spread = None
                team2_spread_odds = None
                total = None
                total_over_odds = None
                total_under_odds = None
                team1_moneyline = None
                team2_moneyline = None

                look_ahead = 0
                over_odds_line_index = None
                under_odds_line_index = None

                while i + look_ahead < len(lines) and look_ahead < 30:
                    check_line = lines[i + look_ahead]

                    # Spread (e.g., "+3.5", "-7")
                    if re.match(r'^[+\-]\d+(\.\d+)?$', check_line) and not re.match(r'^[+\-]\d{3,}$', check_line):
                        value = float(check_line)
                        if abs(value) <= 20:
                            # Next line should be the spread odds
                            if i + look_ahead + 1 < len(lines):
                                next_line = lines[i + look_ahead + 1]
                                # Spread odds format: −110, +100, etc. (use minus sign or hyphen)
                                if re.match(r'^[+\-−]\d+$', next_line):
                                    odds = int(next_line.replace('−', '-'))  # Replace unicode minus
                                    if team1_spread is None:
                                        team1_spread = value
                                        team1_spread_odds = odds
                                    elif team2_spread is None:
                                        team2_spread = value
                                        team2_spread_odds = odds

                    # Total (look for "O" followed by number, then odds)
                    if check_line == "O" and i + look_ahead + 1 < len(lines):
                        next_line = lines[i + look_ahead + 1]
                        if re.match(r'^\d+(\.\d+)?$', next_line) and float(next_line) > 25:
                            total = float(next_line)
                            # Over odds are 2 lines ahead (after the total)
                            if i + look_ahead + 2 < len(lines):
                                odds_line = lines[i + look_ahead + 2]
                                if re.match(r'^[+\-−]\d+$', odds_line):
                                    total_over_odds = int(odds_line.replace('−', '-'))
                                    # Remember where we found over odds
                                    over_odds_line_index = look_ahead + 2

                    # Total under odds (look for "U" followed by number, then odds)
                    if check_line == "U" and i + look_ahead + 1 < len(lines):
                        next_line = lines[i + look_ahead + 1]
                        if re.match(r'^\d+(\.\d+)?$', next_line) and float(next_line) > 25:
                            # Under odds are 2 lines ahead
                            if i + look_ahead + 2 < len(lines):
                                odds_line = lines[i + look_ahead + 2]
                                if re.match(r'^[+\-−]\d+$', odds_line):
                                    total_under_odds = int(odds_line.replace('−', '-'))
                                    # Remember where we found under odds
                                    under_odds_line_index = look_ahead + 2

                    # Moneyline: First 3+ digit number AFTER the over/under odds line
                    if re.match(r'^[+\-−]\d{3,}$', check_line):
                        ml = int(check_line.replace('−', '-'))
                        # Team 1 ML: comes after over odds
                        if (over_odds_line_index is not None and
                            look_ahead > over_odds_line_index and
                            team1_moneyline is None):
                            team1_moneyline = ml
                        # Team 2 ML: comes after under odds
                        elif (under_odds_line_index is not None and
                              look_ahead > under_odds_line_index and
                              team2_moneyline is None):
                            team2_moneyline = ml

                    # Stop at next game
                    if check_line == "More Bets" or team_pattern.match(check_line):
                        break

                    look_ahead += 1

                # Advance past this game
                while i < len(lines) and lines[i] != "More Bets":
                    i += 1
                if i < len(lines) and lines[i] == "More Bets":
                    i += 1

                # Create game entries if valid
                if team1_spread is not None and team2_spread is not None and total is not None:
                    # Skip games before 10 AM PT if in Today section
                    if in_today_section and current_game_time and not is_valid_game_time(current_game_time):
                        print(f"        Skipping {team1} vs {team2} (before 10 AM PT cutoff)")
                        continue

                    # Calculate projected points
                    def calc_team_points(total_pts, spread, is_favorite):
                        abs_spread = abs(spread)
                        if is_favorite:
                            return round((total_pts + abs_spread) / 2, 2)
                        else:
                            return round((total_pts - abs_spread) / 2, 2)

                    team1_is_favorite = team1_spread < 0
                    team1_abbr = get_team_abbreviation(team1)
                    team2_abbr = get_team_abbreviation(team2)

                    games.append({
                        'team': team1,
                        'team_abbr': team1_abbr,
                        'opponent': team2,
                        'opponent_abbr': team2_abbr,
                        'spread': team1_spread,
                        'spread_odds': team1_spread_odds,
                        'total': total,
                        'total_over_odds': total_over_odds,
                        'total_under_odds': total_under_odds,
                        'moneyline': team1_moneyline,
                        'projected_pts': calc_team_points(total, team1_spread, team1_is_favorite),
                    })

                    team2_is_favorite = team2_spread < 0
                    games.append({
                        'team': team2,
                        'team_abbr': team2_abbr,
                        'opponent': team1,
                        'opponent_abbr': team1_abbr,
                        'spread': team2_spread,
                        'spread_odds': team2_spread_odds,
                        'total': total,
                        'total_over_odds': total_over_odds,
                        'total_under_odds': total_under_odds,
                        'moneyline': team2_moneyline,
                        'projected_pts': calc_team_points(total, team2_spread, team2_is_favorite),
                    })

                    games_found += 1
            else:
                i += 1

        return games

    @property
    def cache_file(self) -> Path:
        """Override to use CSV cache"""
        output_dir = Path('data/intermediate')
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / 'game_lines.csv'

    def _load_cache(self) -> List[Dict]:
        """Load from CSV"""
        df = pd.read_csv(self.cache_file)
        return df.to_dict('records')

    def _save_cache(self, data: List[Dict]) -> None:
        """Save to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(self.cache_file, index=False)


# ============================================================================
# DRAFTKINGS TD ODDS SCRAPER
# ============================================================================

class DraftKingsTdOddsScraper(BaseScraper):
    """
    Scrape DraftKings anytime TD scorer odds.
    """

    URL = 'https://sportsbook.draftkings.com/leagues/football/nfl?category=td-scorers&subcategory=td-scorer'

    def __init__(self):
        super().__init__('td_odds')

    def _fetch(self) -> Dict[str, Dict]:
        """
        Scrape TD odds from DraftKings.

        Returns:
            Dict mapping normalized player name -> {tdOdds, tdProbability}
        """
        driver = create_headless_driver()
        try:
            driver.get(self.URL)
            time.sleep(0.5)

            # Get page HTML
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            data_by_player = {}

            # Find all TD scorer tables (4-column layout)
            tables = soup.select('.cb-market__template--4-columns')

            for table in tables:
                cells = table.find_all(recursive=False)[3:]  # Skip first 3 headers
                column_count = 4

                # Extract players (every 4th cell starting from 0)
                players = []
                for i in range(0, len(cells), column_count):
                    cell_text = cells[i].get_text(strip=True)
                    if cell_text and 'no touchdown' not in cell_text.lower():
                        players.append(cell_text)

                # Extract odds (every 4th cell starting from 2)
                td_odds_list = []
                for i in range(2, len(cells), column_count):
                    cell_text = cells[i].get_text(strip=True)
                    if cell_text:
                        td_odds_list.append(cell_text)

                # Match players with odds
                for player, odds in zip(players, td_odds_list):
                    # Handle defenses (e.g., "D/ST Patriots" -> "patriots")
                    if 'D/ST' in player:
                        player = player.split()[1]

                    name = normalize_name(player)
                    td_probability = odds_to_probability(odds)

                    data_by_player[name] = {
                        'tdOdds': odds,
                        'tdProbability': td_probability,
                    }

            print(f"      Found TD odds for {len(data_by_player)} players")
            return data_by_player

        finally:
            driver.quit()


# ============================================================================
# ESPN SCRAPERS
# ============================================================================

class EspnPlayerListScraper(BaseScraper):
    """
    Fetch ESPN player list (IDs) from API.
    """

    URL = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2025/players?scoringPeriodId=0&view=players_wl&platformVersion=6c0d90bbc8abfb789ccf5f8728b6459da4b18c82'

    HEADERS = {
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
    }

    def __init__(self):
        super().__init__('espn_players')

    def _fetch(self) -> Dict[str, str]:
        """
        Fetch ESPN player list.

        Returns:
            Dict mapping normalized player name -> ESPN ID
        """
        response = requests.get(self.URL, headers=self.HEADERS)
        response.raise_for_status()

        players = response.json()
        print(f"      Fetched {len(players)} players from ESPN API")

        # Build lookup map (use FIRST match to avoid duplicates)
        espn_by_player = {}
        for player in players:
            name = normalize_name(player['fullName'])
            # Only set if not already set (prefer first match)
            if name not in espn_by_player:
                espn_by_player[name] = str(player['id'])

        return espn_by_player

    @property
    def cache_file(self) -> Path:
        """Override to use CSV cache"""
        output_dir = Path('data/intermediate')
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / 'espn_players.csv'

    def _load_cache(self) -> Dict[str, str]:
        """Load from CSV"""
        df = pd.read_csv(self.cache_file)
        return {row['fullName']: str(row['id']) for _, row in df.iterrows()}

    def _save_cache(self, data: Dict[str, str]) -> None:
        """Save to CSV"""
        rows = [{'fullName': name, 'id': espn_id} for name, espn_id in data.items()]
        df = pd.DataFrame(rows)
        df.to_csv(self.cache_file, index=False)


class EspnProjectionsScraper(BaseScraper):
    """
    Fetch ESPN Watson projections for individual players.
    """

    BASE_URL = 'https://watsonfantasyfootball.espn.com/espnpartner/dallas/projections/projections_{}_ESPNFantasyFootball_2025.json'

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    # Rate limiting
    MIN_DELAY = 0.3
    MAX_DELAY = 0.8
    BATCH_SIZE = 100
    BATCH_BREAK = 5

    def __init__(self):
        super().__init__('espn_projections')

    def fetch_all(self, players_with_ids: List[Dict], min_projection: float = 2.5) -> Dict[str, Dict]:
        """
        Fetch ESPN projections for list of players.

        Args:
            players_with_ids: List of player dicts with 'name' and 'espnId' keys
            min_projection: Only fetch for players with FP projection >= this value

        Returns:
            Dict mapping player name -> projection data
        """
        # Filter players
        filtered_players = [
            p for p in players_with_ids
            if p.get('fpProjPts', 0) >= min_projection and p.get('espnId')
        ]

        print(f"      Fetching for {len(filtered_players)} players (filtered from {len(players_with_ids)})")

        # Sort by projection (descending)
        filtered_players.sort(key=lambda p: p.get('fpProjPts', 0), reverse=True)

        projections_by_player = {}
        players_with_espn = 0

        for i, player in enumerate(filtered_players):
            espn_id = player['espnId']
            name = player['name']

            print(f"      [{i+1}/{len(filtered_players)}] {name}", end='')

            projection = self._fetch_single_player(espn_id)

            if projection:
                projections_by_player[name] = projection
                players_with_espn += 1
                print(f" ✓ {projection.get('espnScoreProjection', 'N/A')}")
            else:
                print(" ✗")

            # Rate limiting
            if i < len(filtered_players) - 1:
                import random
                delay_time = self.MIN_DELAY + random.random() * (self.MAX_DELAY - self.MIN_DELAY)

                if (i + 1) % self.BATCH_SIZE == 0:
                    print(f"\n      --- Batch complete. Taking {self.BATCH_BREAK}s break ---\n")
                    time.sleep(self.BATCH_BREAK)
                else:
                    time.sleep(delay_time)

        print(f"\n      Fetched ESPN projections: {players_with_espn}/{len(filtered_players)}")
        return projections_by_player

    def _fetch_single_player(self, espn_id: str) -> Optional[Dict]:
        """
        Fetch projection for a single player.

        Args:
            espn_id: ESPN player ID

        Returns:
            Projection dict or None if not found
        """
        url = self.BASE_URL.format(espn_id)

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=10)

            if response.status_code == 404:
                return None

            if not response.ok:
                return None

            projections = response.json()

            if not projections or len(projections) == 0:
                return None

            # Sort by timestamp, get most recent
            sorted_projs = sorted(
                projections,
                key=lambda p: p.get('DATA_TIMESTAMP', ''),
                reverse=True
            )
            most_recent = sorted_projs[0]

            return {
                'espnScoreProjection': most_recent.get('SCORE_PROJECTION', ''),
                'espnLowScore': most_recent.get('LOW_SCORE', ''),
                'espnHighScore': most_recent.get('HIGH_SCORE', ''),
                'espnOutsideProjection': most_recent.get('OUTSIDE_PROJECTION', ''),
                'espnSimulationProjection': most_recent.get('SIMULATION_PROJECTION', ''),
            }

        except Exception:
            return None

    def _fetch(self) -> Dict:
        """Not used - call fetch_all() instead"""
        raise NotImplementedError("Use fetch_all() instead")
