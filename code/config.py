"""
Configuration file for Fantasy DFS Optimizer (CVaR-MILP Version)

Edit this file to customize optimizer behavior without changing code.
"""

# ============================================================================
# DATA PATHS
# ============================================================================

# Input data directories
DATA_INPUT_DIR = 'data/input'
DATA_INTERMEDIATE_DIR = 'data/intermediate'
CACHE_DIR = 'cache'
OUTPUTS_DIR = 'outputs'

# Specific data files (legacy paths for backward compatibility)
PLAYERS_INTEGRATED = 'data/intermediate/players_integrated.csv'
GAME_SCRIPTS = 'data/intermediate/game_script.csv'

# ============================================================================
# CVaR OPTIMIZER SETTINGS
# ============================================================================

# Number of lineups to generate
DEFAULT_N_LINEUPS = 20

# Number of scenarios for CVaR optimization
# More scenarios = more precise optimization but slower
DEFAULT_N_SCENARIOS = 200

# CVaR alpha: tail probability to optimize
# 0.20 = optimize for top 20% (p80)
# 0.10 = optimize for top 10% (p90) - more aggressive
DEFAULT_CVAR_ALPHA = 0.20

# Solver time limit per lineup (seconds)
DEFAULT_SOLVER_TIME_LIMIT = 60

# If True, skip anchored strategies and generate all lineups as "general"
# (no forced players, just exclude previous lineups for diversity)
GENERAL_ONLY = True

# ============================================================================
# PLAYER CONSTRAINTS
# ============================================================================

# Force-include players (always in lineup)
# Use player IDs or exact names as they appear in data
FORCE_INCLUDE = [
    # Example: '123456-789012',  # FanDuel player ID
    # Example: 'Patrick Mahomes',
]

# Exclude players (never in lineup)
# Useful for injuries, personal preferences, etc.
EXCLUDE_PLAYERS = [
    # Example: 'Tua Tagovailoa',
    # Example: 'Joe Burrow',
]

# ============================================================================
# LINEUP CONSTRAINTS (FanDuel)
# ============================================================================

SALARY_CAP = 60000                 # FanDuel salary cap
LINEUP_SIZE = 9                    # Total players in lineup

# Position requirements
MIN_QB = 1
MAX_QB = 1

MIN_RB = 2
MAX_RB = 3

MIN_WR = 3
MAX_WR = 4

MIN_TE = 1
MAX_TE = 2

MIN_FLEX = 1  # RB/WR/TE
MAX_FLEX = 1

MIN_DEF = 1
MAX_DEF = 1

# ============================================================================
# WEB SCRAPING
# ============================================================================

# Cache expiration (seconds)
CACHE_EXPIRATION = {
    'fantasypros': 3600,      # 1 hour
    'draftkings_lines': 1800,  # 30 minutes
    'draftkings_td_odds': 1800,  # 30 minutes
    'espn_players': 86400,    # 24 hours
    'espn_projections': 3600,  # 1 hour
}

# Selenium settings
SELENIUM_HEADLESS = True          # Run browser in background
SELENIUM_TIMEOUT = 30             # Page load timeout (seconds)

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

# Verbosity
VERBOSE = True                    # Print detailed progress
SUPPRESS_DISTRIBUTION_WARNINGS = True  # Suppress distribution fitting warnings

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_excluded_players():
    """Get list of all excluded players."""
    return list(EXCLUDE_PLAYERS)


def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check salary cap
    if SALARY_CAP <= 0:
        errors.append("SALARY_CAP must be positive")

    # Check lineup size
    min_lineup = MIN_QB + MIN_RB + MIN_WR + MIN_TE + MIN_FLEX + MIN_DEF
    max_lineup = MAX_QB + MAX_RB + MAX_WR + MAX_TE + MAX_FLEX + MAX_DEF

    if min_lineup > LINEUP_SIZE:
        errors.append(f"Minimum position requirements ({min_lineup}) exceed lineup size ({LINEUP_SIZE})")

    if max_lineup < LINEUP_SIZE:
        errors.append(f"Maximum position requirements ({max_lineup}) below lineup size ({LINEUP_SIZE})")

    # Check CVaR settings
    if DEFAULT_N_LINEUPS <= 0:
        errors.append("DEFAULT_N_LINEUPS must be positive")

    if DEFAULT_N_SCENARIOS <= 0:
        errors.append("DEFAULT_N_SCENARIOS must be positive")

    if not 0 < DEFAULT_CVAR_ALPHA < 1:
        errors.append("DEFAULT_CVAR_ALPHA must be between 0 and 1")

    # Check force include vs exclude
    force_and_exclude = set(FORCE_INCLUDE) & set(get_all_excluded_players())
    if force_and_exclude:
        errors.append(f"Players cannot be both forced and excluded: {force_and_exclude}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# Validate on import
if __name__ != '__main__':
    validate_config()
