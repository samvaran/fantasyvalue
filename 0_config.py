"""
Configuration file for Fantasy DFS Optimizer

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

# Specific data files
PLAYERS_INTEGRATED = 'data/intermediate/players_integrated.csv'
GAME_SCRIPTS = 'data/intermediate/game_script_continuous.csv'

# ============================================================================
# OPTIMIZER DEFAULTS
# ============================================================================

# Phase 1: Candidate Generation
DEFAULT_CANDIDATES = 200          # Number of lineups to generate
QUICK_TEST_CANDIDATES = 50         # For quick testing

# Phase 2 & 3: Monte Carlo Simulations
DEFAULT_SIMS = 5000               # Simulations per lineup (production)
QUICK_TEST_SIMS = 1000             # Simulations for quick test

# Phase 3: Genetic Algorithm
DEFAULT_MAX_GENERATIONS = 50       # Max generations for genetic algorithm
QUICK_TEST_MAX_GENERATIONS = 30    # Max generations for quick test
DEFAULT_CONVERGENCE_PATIENCE = 5   # Stop if no improvement for N generations
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # 1% improvement threshold

# Fitness function
# Options: 'conservative', 'balanced', 'aggressive', 'tournament'
DEFAULT_FITNESS = 'balanced'

# Parallel processing
DEFAULT_PROCESSES = None           # None = auto-detect CPU cores

# ============================================================================
# PLAYER CONSTRAINTS
# ============================================================================

# Force-include players (always in lineup)
# Use exact player names as they appear in data
FORCE_INCLUDE = [
    # Example: 'Patrick Mahomes',
    # Example: 'Christian McCaffrey',
]

# Exclude players (never in lineup)
# Useful for injuries, personal preferences, etc.
EXCLUDE_PLAYERS = [
    # Example: 'Player Name',
]

# Position-specific exclusions
EXCLUDE_QBS = [
    # Example: 'Baker Mayfield',
]

EXCLUDE_RBS = [
    # Example: 'Ezekiel Elliott',
]

EXCLUDE_WRS = [
    # Example: 'Julio Jones',
]

EXCLUDE_TES = [
    # Example: 'Travis Kelce',
]

EXCLUDE_DEFS = [
    # Example: 'Patriots',
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
# MILP OPTIMIZATION (Phase 1)
# ============================================================================

# Diversity constraints
MAX_OVERLAP_CHALK = 7              # Max shared players for chalk lineups (1-20)
MAX_OVERLAP_MODERATE = 6           # Max shared players for moderate lineups (21-100)
MAX_OVERLAP_CONTRARIAN = 6         # Max shared players for contrarian lineups (101-1000)

# Temperature-based sampling
# Higher temperature = more randomness
TEMP_DETERMINISTIC = 0.0           # Lineups 1-20 (chalk)
TEMP_MODERATE_MIN = 0.3            # Lineups 21-100 (start)
TEMP_MODERATE_MAX = 1.1            # Lineups 21-100 (end)
TEMP_CONTRARIAN_MIN = 1.5          # Lineups 101-1000 (start)
TEMP_CONTRARIAN_MAX = 3.0          # Lineups 101-1000 (end)

# ============================================================================
# GAME SCRIPT ADJUSTMENTS
# ============================================================================

# Game script floor multipliers by position
GAME_SCRIPT_FLOOR = {
    'shootout': {
        'QB': 0.95,   # Slight floor reduction (more variance)
        'RB': 0.90,   # Reduced floor (less running in shootouts)
        'WR': 0.95,   # Slight floor reduction
        'TE': 0.95,   # Slight floor reduction
        'D': 0.85,    # Reduced floor (high scoring = fewer points for D)
    },
    'defensive': {
        'QB': 0.85,   # Reduced floor (low scoring)
        'RB': 1.05,   # Slight floor boost (more conservative)
        'WR': 0.85,   # Reduced floor
        'TE': 0.85,   # Reduced floor
        'D': 1.15,    # Floor boost (low scoring = more points for D)
    },
    'blowout_favorite': {
        'QB': 0.90,   # Reduced floor (may sit in 4th quarter)
        'RB': 1.20,   # Strong floor boost (clock management)
        'WR': 0.90,   # Reduced floor
        'TE': 0.90,   # Reduced floor
        'D': 1.10,    # Moderate boost (potential for turnovers/sacks)
    },
    'blowout_underdog': {
        'QB': 0.95,   # Slight floor reduction (garbage time variability)
        'RB': 0.85,   # Reduced floor (abandoned run game)
        'WR': 0.95,   # Slight floor reduction
        'TE': 0.95,   # Slight floor reduction
        'D': 0.90,    # Slight reduction (garbage time scores)
    },
    'competitive': {
        'QB': 1.00,   # No change (balanced)
        'RB': 1.00,   # No change
        'WR': 1.00,   # No change
        'TE': 1.00,   # No change
        'D': 1.00,    # No change
    }
}

# Game script ceiling multipliers by position
GAME_SCRIPT_CEILING = {
    'shootout': {
        'QB': 1.15,   # Boost ceiling (high scoring)
        'RB': 0.90,   # Reduced ceiling (less running)
        'WR': 1.20,   # Strong ceiling boost
        'TE': 1.15,   # Ceiling boost
        'D': 0.80,    # Reduced ceiling (high scoring = fewer D points)
    },
    'defensive': {
        'QB': 0.80,   # Reduced ceiling (low scoring)
        'RB': 0.90,   # Reduced ceiling
        'WR': 0.75,   # Strong ceiling reduction
        'TE': 0.80,   # Reduced ceiling
        'D': 1.25,    # Strong ceiling boost (sacks, turnovers, TDs)
    },
    'blowout_favorite': {
        'QB': 0.85,   # Reduced ceiling (conservative play)
        'RB': 1.10,   # Ceiling boost
        'WR': 0.85,   # Reduced ceiling
        'TE': 0.85,   # Reduced ceiling
        'D': 1.15,    # Ceiling boost
    },
    'blowout_underdog': {
        'QB': 1.05,   # Slight ceiling boost (garbage time)
        'RB': 0.85,   # Reduced ceiling
        'WR': 1.05,   # Slight ceiling boost (garbage time)
        'TE': 1.00,   # No change
        'D': 0.85,    # Reduced ceiling
    },
    'competitive': {
        'QB': 1.00,   # No change
        'RB': 1.00,   # No change
        'WR': 1.00,   # No change
        'TE': 1.00,   # No change
        'D': 1.00,    # No change
    }
}

# TD Odds Adjustments
TD_ODDS_FLOOR_IMPACT = 0.05    # 5% floor boost per 100% TD probability
TD_ODDS_CEILING_IMPACT = 0.15  # 15% ceiling boost per 100% TD probability

# Team offensive totals weight in game script calculation
TEAM_TOTAL_WEIGHT = 0.20  # 20% weight for team offensive projections signal

# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

# Custom fitness function weights (if you want to create your own)
# fitness = (median * W1) + (p90 * W2) - (std * W3) - (p10 * W4)

CUSTOM_FITNESS_WEIGHTS = {
    'median': 1.0,
    'p90': 0.0,
    'std': 0.0,
    'p10': 0.0,
}

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
SAVE_CHECKPOINTS = True           # Save state after each iteration
SUPPRESS_DISTRIBUTION_WARNINGS = True  # Suppress distribution fitting warnings (default: True)

# Output file settings
SAVE_TOP_N_LINEUPS = 10          # Number of best lineups to save
SAVE_ALL_CANDIDATES = True        # Save all 1000 candidates
SAVE_ALL_EVALUATIONS = True       # Save all Monte Carlo results

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Distribution fitting
DISTRIBUTION_TYPE = 'shifted_lognormal'  # Distribution for projections

# Genetic algorithm
GENETIC_TOURNAMENT_SIZE = 5       # Tournament selection size
GENETIC_MUTATION_RATE = 0.30      # Probability of mutation (30%)
GENETIC_ELITE_RATIO = 0.20        # Top 20% preserved as elite
GENETIC_NUM_PARENTS = 50          # Number of parents for crossover
GENETIC_NUM_OFFSPRING = 100       # Offspring per generation

# Salary constraint for mutation
MUTATION_SALARY_TOLERANCE = 500   # Allow ±$500 salary change during mutation

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_excluded_players():
    """Get combined list of all excluded players."""
    excluded = set(EXCLUDE_PLAYERS)
    excluded.update(EXCLUDE_QBS)
    excluded.update(EXCLUDE_RBS)
    excluded.update(EXCLUDE_WRS)
    excluded.update(EXCLUDE_TES)
    excluded.update(EXCLUDE_DEFS)
    return list(excluded)


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

    # Check candidates
    if DEFAULT_CANDIDATES <= 0:
        errors.append("DEFAULT_CANDIDATES must be positive")

    # Check simulations
    if DEFAULT_SIMS <= 0:
        errors.append("DEFAULT_SIMS must be positive")

    # Check fitness function
    valid_fitness = ['conservative', 'balanced', 'aggressive', 'tournament']
    if DEFAULT_FITNESS not in valid_fitness:
        errors.append(f"DEFAULT_FITNESS must be one of {valid_fitness}")

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
