# Fantasy DFS Lineup Optimizer

A sophisticated DFS (Daily Fantasy Sports) lineup optimizer that uses Monte Carlo simulation, game script analysis, and genetic algorithms to generate optimal NFL lineups for FanDuel.

## Features

- **Self-Contained Week Architecture** - Each week's data lives in its own directory for easy historical analysis
- **4-Step Modular Pipeline** - Fetch → Integrate → Optimize → Backtest
- **Monte Carlo Simulation** - 10,000+ simulations per lineup for accurate distribution estimates
- **Game Script Analysis** - Continuous probability distributions for shootout/defensive/blowout scenarios
- **Genetic Algorithm** - Tournament selection with position-aware crossover and mutation
- **Multiple Fitness Functions** - Optimized strategies for cash games, GPPs, or balanced play
- **Parallel Processing** - 12x speedup using multiprocessing
- **Automatic Checkpointing** - Resume interrupted runs from any point

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FanDuel salaries CSV and place in:
mkdir -p data/2025_12_08/inputs
cp ~/Downloads/FanDuel-NFL-*.csv data/2025_12_08/inputs/fanduel_salaries.csv

# 3. Run complete pipeline
python run_week.py

# 4. Get best lineups from:
# data/2025_12_08/outputs/run_*/3_lineups.csv
```

For detailed setup instructions, see [docs/QUICK_START.md](docs/QUICK_START.md)

## Architecture

### Self-Contained Week Directories

Each week's data is completely independent, enabling historical replay and backtesting:

```
data/
└── YYYY_MM_DD/              # Week directory (Sunday date)
    ├── inputs/              # Raw data sources (CSV)
    │   ├── fanduel_salaries.csv
    │   ├── fantasypros_projections.csv
    │   ├── game_lines.csv
    │   ├── td_odds.csv
    │   └── fanduel_results.json  (for backtest)
    │
    ├── intermediate/        # Integrated data
    │   ├── 1_players.csv         (merged player data)
    │   └── 2_game_scripts.csv    (game probabilities)
    │
    └── outputs/             # Optimization results
        └── run_YYYYMMDD_HHMMSS/
            ├── 0_config.json         # Run configuration
            ├── 1_candidates.csv      # Initial lineups
            ├── 2_simulations.csv     # MC evaluations
            ├── 3_lineups.csv         # ⭐ BEST LINEUPS
            ├── 4_summary.json        # Stats
            ├── 5_scored_lineups.csv  # Backtest scores
            ├── 6_backtest_games.csv  # Game analysis
            └── 7_backtest_summary.json # Performance metrics
```

### 4-Step Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  1. FETCH    │ → │ 2. INTEGRATE │ → │ 3. OPTIMIZE  │ → │ 4. BACKTEST  │
│              │   │              │   │              │   │              │
│ • FanDuel    │   │ • Merge      │   │ • Generate   │   │ • Score      │
│ • FantasyPros│   │   players    │   │   candidates │   │   lineups    │
│ • DraftKings │   │ • Calculate  │   │ • Monte Carlo│   │ • Analyze    │
│   lines/odds │   │   game       │   │   simulation │   │   accuracy   │
│              │   │   scripts    │   │ • Genetic    │   │ • Calibrate  │
│              │   │              │   │   algorithm  │   │   ranges     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

## Usage

### Orchestrated Pipeline (Recommended)

```bash
# Run current week (auto-detects next Sunday)
python run_week.py

# Run specific week
python run_week.py --week 2025-12-08

# Quick test (~2 minutes)
python run_week.py --quick-test

# Historical backtest
python run_week.py --week 2025-11-30 --backtest-only
```

### Individual Steps

```bash
# Step 1: Fetch data from external sources
python code/1_fetch_data.py --week-dir data/2025_12_08

# Step 2: Integrate and calculate game scripts
python code/2_data_integration.py --week-dir data/2025_12_08

# Step 3: Optimize lineups (full run ~1-2 hours)
python code/3_run_optimizer.py --week-dir data/2025_12_08

# Step 3 (quick test ~2 minutes)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test

# Step 4: Backtest against actual results
python code/4_backtest.py --week-dir data/2025_12_08 --run-dir run_YYYYMMDD_HHMMSS
```

### Contest Strategies

```bash
# Cash games (50/50, Double-Up)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --fitness median

# Balanced (default - good for small GPPs)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --fitness balanced

# Large GPPs (Milly Maker)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --fitness upside

# Conservative / safe cash
python code/3_run_optimizer.py --week-dir data/2025_12_08 --fitness safe
```

## Optimizer Details

### Phase 1: Candidate Generation (MILP)

Uses mixed-integer linear programming to generate diverse lineups:

- **Tier 1 (1-20)**: Deterministic chalk - max consensus projections
- **Tier 2 (21-100)**: Temperature-based variation
- **Tier 3 (101-1000)**: Random weighted sampling for contrarian plays

### Phase 2: Monte Carlo Evaluation

Simulates each lineup 10,000 times to estimate score distribution:

- Samples from player-specific distributions
- Applies game script adjustments (shootout/defensive/blowout)
- Models negative correlation (QB/DEF on same team, RB/QB on same team)
- Calculates P10, median, P90, mean, std, skewness

### Phase 3: Genetic Algorithm Refinement

Evolves population toward optimal fitness:

- **Selection**: Tournament selection (best of 3)
- **Crossover**: Position-aware (preserves valid lineups)
- **Mutation**: Swap players while maintaining salary cap
- **Elitism**: Keep best 100 lineups each generation
- **Convergence**: Stops after 5 generations without improvement

## Performance

**Default Settings** (1000 candidates, 10k sims, 8-core machine):
- Phase 1: 1-2 minutes
- Phase 2: 20-40 minutes (parallelized)
- Phase 3: 30-60 minutes
- **Total**: ~1-2 hours

**Quick Test** (50 candidates, 1k sims):
- **Total**: ~2 minutes

## Project Structure

```
fantasyvalue/
├── run_week.py                # Main entry point - orchestrates full pipeline
│
├── code/                      # All Python code
│   ├── 1_fetch_data.py        # Step 1: Data fetching
│   ├── 2_data_integration.py # Step 2: Data merging & game scripts
│   ├── 3_run_optimizer.py    # Step 3: Optimization
│   ├── 4_backtest.py          # Step 4: Performance analysis
│   │
│   ├── models.py              # Data models
│   ├── scrapers.py            # Web scrapers
│   ├── utils.py               # Helper functions
│   │
│   └── optimizer/             # Optimizer modules
│       ├── generate_candidates.py # Phase 1: MILP generation
│       ├── evaluate_lineups.py    # Phase 2: Monte Carlo
│       ├── optimize_genetic.py    # Phase 3: Genetic algorithm
│       └── utils/
│           ├── milp_solver.py
│           ├── monte_carlo.py
│           └── genetic_operators.py
│
├── data/                      # Week directories (not in git)
│   └── YYYY_MM_DD/
│       ├── inputs/
│       ├── intermediate/
│       └── outputs/
│
├── docs/                      # Documentation
│   ├── QUICK_START.md         # 5-minute setup guide
│   ├── WORKFLOW.md            # Complete pipeline walkthrough
│   ├── OPTIMIZER_USAGE.md     # Advanced usage & tuning
│   └── ARCHITECTURE_REFACTOR.md # Technical architecture
│
├── requirements.txt
└── README.md
```

## Migration from Old Structure

If you have data in the old structure (data/input/, cache/, outputs/), use the migration script:

```bash
# Auto-detect week and migrate
python migrate_to_new_structure.py --auto

# Specify week explicitly
python migrate_to_new_structure.py --week 2025-11-30

# Dry run (preview without changes)
python migrate_to_new_structure.py --week 2025-11-30 --dry-run
```

## Documentation

- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Get up and running in 5 minutes
- **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Detailed pipeline walkthrough with examples
- **[docs/OPTIMIZER_USAGE.md](docs/OPTIMIZER_USAGE.md)** - Advanced usage, parameter tuning, strategies
- **[docs/ARCHITECTURE_REFACTOR.md](docs/ARCHITECTURE_REFACTOR.md)** - Technical architecture and design decisions

## Troubleshooting

### No lineups generated
```
Generated 0 chalk lineups
```
**Fix**: Check that DEF players exist in integrated data:
```bash
python -c "import pandas as pd; print(pd.read_csv('data/WEEK/intermediate/1_players.csv')['position'].value_counts())"
```
Should show 'D' or 'DEF' position. If not, re-run data integration.

### Slow performance
**Fix**: Reduce workload or increase parallelization:
```bash
python code/3_run_optimizer.py --week-dir data/WEEK --candidates 500 --simulations 5000
# or
python code/3_run_optimizer.py --week-dir data/WEEK --processes 16
```

### Out of memory
**Fix**: Reduce parallel processes:
```bash
python code/3_run_optimizer.py --week-dir data/WEEK --processes 4
```

See [docs/OPTIMIZER_USAGE.md](docs/OPTIMIZER_USAGE.md) for more troubleshooting tips.

## License

MIT
