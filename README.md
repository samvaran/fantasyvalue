# Fantasy DFS Lineup Optimizer - V4

A sophisticated DFS (Daily Fantasy Sports) lineup optimizer that uses Monte Carlo simulation, game script analysis, and genetic algorithms to generate optimal NFL lineups for FanDuel.

## Quick Start

```bash
# 1. Fetch latest data
python fetch_data.py --all

# 2. Analyze game scripts
python game_script_continuous.py

# 3. Integrate data
python data_integration.py

# 4. Run optimizer
python run_optimizer.py --quick-test

# 5. View results
python view_progress.py
```

## Project Structure

```
_V4/
├── docs/                          # Documentation
│   ├── README.md                 # Project overview
│   ├── QUICK_START.md            # 5-minute getting started guide
│   ├── WORKFLOW.md               # Data pipeline workflow
│   ├── OPTIMIZER_USAGE.md        # Complete optimizer usage guide
│   ├── OPTIMIZER_DESIGN.md       # Technical design document
│   ├── PIPELINE_REVIEW.md        # End-to-end pipeline review
│   └── PIPELINE_SUMMARY.txt      # ASCII art summary
│
├── data/                          # Data files (not in git)
│   ├── input/                    # Input CSVs (FanDuel salaries)
│   └── intermediate/             # Generated intermediate files
│       ├── players_raw.csv       # Merged player data
│       ├── players_integrated.csv # Final integrated data (38 columns)
│       ├── game_lines.csv        # Betting lines
│       ├── game_script_continuous.csv # Game script analysis
│       └── ...
│
├── optimizer/                     # Optimizer modules
│   ├── generate_candidates.py   # Phase 1: MILP candidate generation
│   ├── evaluate_lineups.py      # Phase 2: Monte Carlo evaluation
│   ├── optimize_genetic.py      # Phase 3: Genetic refinement
│   └── utils/
│       ├── milp_solver.py       # MILP formulation
│       ├── monte_carlo.py       # Parallel simulation engine
│       ├── distribution_fit.py  # Distribution fitting
│       └── genetic_operators.py # Genetic algorithm
│
├── outputs/                       # Optimizer results (not in git)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── BEST_LINEUPS.csv     # Top 10 lineups
│       ├── final_summary.json   # Run statistics
│       └── ...
│
├── cache/                         # Web scrape cache (not in git)
│
├── Core Scripts
│   ├── fetch_data.py             # Data fetching orchestration
│   ├── game_script_continuous.py # Game script analysis
│   ├── data_integration.py       # Data integration pipeline
│   ├── run_optimizer.py          # Main optimizer orchestrator
│   ├── view_progress.py          # Progress monitoring
│   ├── scrapers.py               # Web scrapers
│   ├── utils.py                  # Helper functions
│   └── models.py                 # Data classes
│
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore rules
```

## Core Pipeline

### Phase 1: Data Fetching (2-5 minutes)

Fetches and merges data from multiple sources:
- FanDuel salaries (local CSV)
- FantasyPros projections
- DraftKings game lines (spread, total, odds)
- DraftKings TD odds
- ESPN projections (low/high scores)

**Output:** `data/intermediate/players_raw.csv` (378 players × 19 columns)

### Phase 2A: Game Script Analysis (1 second)

Calculates continuous probabilities for each game:
- Shootout probability
- Defensive probability
- Blowout probability
- Competitive probability

Uses sigmoid functions for smooth transitions instead of discrete buckets.

**Output:** `data/intermediate/game_script_continuous.csv` (14 games × 11 columns)

### Phase 2B: Data Integration (1 second)

Calculates floor/ceiling for 5 game script scenarios:
- Shootout, Defensive, Competitive
- Blowout (favorite), Blowout (underdog)

Position-specific adjustments (e.g., RBs boosted in blowouts, WRs boosted in shootouts).

**Output:** `data/intermediate/players_integrated.csv` (378 players × 38 columns)

### Phase 3: Optimization (70-90 minutes)

Three-phase hybrid optimization:

**Phase 3.1: Candidate Generation (3 min)**
- MILP-based lineup generation
- Tiered sampling (chalk → moderate → contrarian)
- 1000 diverse candidates

**Phase 3.2: Monte Carlo Evaluation (25-30 min)**
- 10,000 simulations per lineup
- Parallel processing (12x speedup)
- Shifted log-normal distributions

**Phase 3.3: Genetic Refinement (40-60 min/iteration)**
- Tournament selection
- Position-aware crossover
- Salary-preserving mutation
- Converges in 3-5 iterations typically

**Output:** `outputs/run_YYYYMMDD_HHMMSS/BEST_LINEUPS.csv` (Top 10 lineups)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Add FanDuel CSV to data/input/
# File format: FanDuel-NFL-*.csv
```

## Usage Examples

### Full Weekly Workflow

```bash
# Monday/Tuesday - Fetch fresh data when salaries drop
python fetch_data.py --all
python game_script_continuous.py
python data_integration.py

# Tuesday/Wednesday - Run optimizer overnight
python run_optimizer.py --candidates 1000 --sims 10000

# Monitor progress in separate terminal
python view_progress.py
```

### Quick Test

```bash
# Test optimizer with small dataset (2-3 minutes)
python run_optimizer.py --quick-test
```

### Different Strategies

```bash
# Conservative: High floor, low risk
python run_optimizer.py --fitness conservative

# Balanced: Expected value (default)
python run_optimizer.py --fitness balanced

# Aggressive: Boom/bust potential
python run_optimizer.py --fitness aggressive

# Tournament: Pure upside
python run_optimizer.py --fitness tournament
```

### Resume Interrupted Run

```bash
python run_optimizer.py --run-name 20241130_143022
```

## Key Features

✅ **Continuous Game Scripts** - Uses probabilities (0-1) instead of discrete buckets to capture mixed signals

✅ **Pre-calculated Scenarios** - All 5 game script floor/ceiling values calculated upfront for fast simulation

✅ **Parallel Processing** - 12x speedup using multiprocessing for Monte Carlo simulations

✅ **Checkpointing** - Resume interrupted runs from any iteration, view progress in real-time

✅ **Multiple Fitness Functions** - Choose strategy: conservative, balanced, aggressive, or tournament

✅ **Robust Error Handling** - Handles missing data, column name variations, auto-fills defaults

## Performance (8-core machine)

| Phase | Operation | Runtime |
|-------|-----------|---------|
| Data Fetching | Web scraping + merge | 2-5 min |
| Game Scripts | Continuous analysis | 1 sec |
| Integration | Floor/ceiling calc | 1 sec |
| Phase 1 | 1000 candidates (MILP) | 3 min |
| Phase 2 | 1000 × 10k sims (MC) | 25-30 min |
| Phase 3 | Genetic refinement | 40-60 min/iter |
| **Total** | **1-2 iterations** | **70-90 min** |
| **Typical** | **3-5 iterations** | **3-4 hours** |

## Documentation

- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Dead-simple guide to running the pipeline
- **[docs/PIPELINE_REVIEW.md](docs/PIPELINE_REVIEW.md)** - Complete end-to-end technical overview

Additional documentation available in `_ARCHIVE/docs/`

## Troubleshooting

**"No DEFs available"**
- Already fixed: Auto-fills missing floor/ceiling for 201 players (including all defenses)

**"KeyError: 'id' or 'fdSalary'"**
- Already fixed: Column normalization handles both naming conventions

**"Out of memory"**
- Reduce `--processes` or `--candidates`

**"Optimizer converged too early"**
- Increase `--patience` (default: 3) or decrease `--threshold` (default: 0.01)

## License

MIT
