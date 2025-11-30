"""
Utility functions for data processing.
"""
import re
from pathlib import Path


# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def normalize_name(name: str, is_defense: bool = False) -> str:
    """
    Normalize player/team name for matching across data sources.

    Replicates the removePunc function from fetch_data.js:
    - Removes parentheticals
    - Removes punctuation
    - Removes Jr./III/II/IV suffixes
    - Converts to lowercase
    - For defenses, extracts just the team name
    """
    # Remove parentheticals
    if "(" in name:
        name = name.split("(")[0].strip()

    # Remove punctuation (keep only word characters and spaces)
    name = re.sub(r'[^\w\s]', '', name)

    # Remove suffixes
    name = re.sub(r'\b(Jr\.?|III|II|IV)\b', '', name)

    # Trim and lowercase
    name = name.strip().lower()

    # For defenses, extract last word (team name)
    if is_defense and ' ' in name:
        name = name.split()[-1]

    return name


# ============================================================================
# ODDS CONVERSION
# ============================================================================

def odds_to_probability(odds: str) -> float:
    """
    Convert American odds to probability percentage.

    Args:
        odds: American odds string (e.g., "+150", "-200")

    Returns:
        Probability as percentage (0-100)

    Examples:
        "+150" -> 40.0  (underdog)
        "-200" -> 66.67 (favorite)
    """
    if not odds or len(odds) < 2:
        return 0.0

    try:
        odds_number = float(odds[1:])

        if odds[0] == '+':
            # Positive odds (underdog)
            prob = 100 / (odds_number + 100)
        else:
            # Negative odds (favorite)
            prob = odds_number / (odds_number + 100)

        return round(prob * 100, 2)

    except (ValueError, IndexError):
        return 0.0


# ============================================================================
# TEAM ABBREVIATIONS
# ============================================================================

# NFL team name to abbreviation mapping
TEAM_ABBR_MAP = {
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
    '49ers': 'SF',
    'ers': 'SF',  # Fallback for "49ers"
    'buccaneers': 'TB',
    'titans': 'TEN',
    'commanders': 'WAS',
}


def get_team_abbreviation(team_name: str) -> str:
    """
    Convert team name to NFL abbreviation.

    Args:
        team_name: Team name (normalized)

    Returns:
        Team abbreviation (e.g., "KC", "SF")
    """
    return TEAM_ABBR_MAP.get(team_name.lower(), "")


def parse_team_name(text: str) -> str:
    """
    Parse team name from DraftKings text.

    Handles formats like:
    - "TB Buccaneers" -> "buccaneers"
    - "SF 49ers-logo" -> "49ers"
    """
    # Remove -logo suffix
    text = text.replace('-logo', '').strip()

    # Split by whitespace and take last word
    parts = text.split()
    if len(parts) > 1:
        team_name = parts[-1]
    else:
        team_name = text

    # Normalize
    return normalize_name(team_name)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def get_cache_dir() -> Path:
    """Get cache directory path"""
    cache_dir = Path(__file__).parent / 'cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def find_latest_fanduel_csv() -> Path:
    """
    Find the most recent FanDuel salary CSV file.

    Returns:
        Path to latest FanDuel CSV

    Raises:
        FileNotFoundError: If no FanDuel CSV found
    """
    cwd = Path.cwd()
    fanduel_files = sorted(cwd.glob('FanDuel-NFL*.csv'))

    if not fanduel_files:
        raise FileNotFoundError('No FanDuel-NFL*.csv files found in current directory')

    return fanduel_files[-1]  # Most recent (alphabetically sorted)
