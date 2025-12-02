# Data Pipeline Workflow

Complete guide to the 4-step DFS optimization pipeline.

## Overview

The optimizer uses a modular pipeline where each step is independent and data-source agnostic:

```
1. FETCH          2. INTEGRATE       3. OPTIMIZE        4. BACKTEST
   │                  │                  │                  │
   ├─ FanDuel         ├─ Merge          ├─ Generate        ├─ Score
   ├─ FantasyPros     │  players         │  candidates      │  lineups
   ├─ DraftKings      ├─ Calculate      ├─ Monte Carlo    ├─ Analyze
   │  lines           │  game scripts    │  simulation      │  performance
   └─ TD odds         └─ Integrate      └─ Genetic algo   └─ Calibrate
                         matchups                              ranges
```

All data is organized in self-contained week directories, enabling historical replay and analysis.

## Directory Structure

```
data/
└── YYYY_MM_DD/              # Week directory (Sunday date)
    ├── inputs/              # Step 1 output (raw CSV data)
    │   ├── fanduel_salaries.csv
    │   ├── fantasypros_projections.csv
    │   ├── game_lines.csv
    │   ├── td_odds.csv
    │   └── fanduel_results.json    (for backtest)
    │
    ├── intermediate/        # Step 2 output (integrated data)
    │   ├── 1_players.csv
    │   └── 2_game_scripts.csv
    │
    └── outputs/             # Step 3 & 4 output
        └── run_YYYYMMDD_HHMMSS/
            ├── 0_config.json         # Run configuration
            ├── 1_candidates.csv      # Initial lineups
            ├── 2_simulations.csv     # MC evaluations
            ├── 3_lineups.csv         # Best lineups
            ├── 4_summary.json        # Optimizer stats
            ├── 5_scored_lineups.csv  # Backtest scores
            ├── 6_backtest_games.csv  # Game analysis
            └── 7_backtest_summary.json # Backtest stats
```

## Step 1: Fetch Data

**Script**: `1_fetch_data.py`

Downloads raw data from external sources and converts to CSV format.

### Data Sources

1. **FanDuel Salaries** (manual download)
   - Player salaries and positions
   - Required input for lineup construction

2. **FantasyPros Projections** (API)
   - Expert consensus projections
   - Primary projection source

3. **DraftKings Game Lines** (scraper)
   - Point totals and spreads
   - Used for game script analysis

4. **DraftKings TD Odds** (scraper)
   - Touchdown probability odds
   - Used for upside estimation

### Usage

```bash
# Fetch all sources to current week
python code/1_fetch_data.py

# Fetch to specific week
python code/1_fetch_data.py --week-dir data/2025_12_08

# Force re-fetch (ignore cache)
python code/1_fetch_data.py --force

# Fetch specific sources only
python code/1_fetch_data.py --fp --dk
```

### Smart Caching

The script automatically caches downloaded data and skips re-downloads unless:
- `--force` flag is used
- Files don't exist
- More than 24 hours have passed (for current week)

### Output Files

All outputs are CSV format for readability:

```
inputs/
├── fanduel_salaries.csv       # Player pool (623 players typical)
├── fantasypros_projections.csv # Projections (507 players typical)
├── game_lines.csv             # Game lines (~16 games/week)
└── td_odds.csv                # TD odds (339 players typical)
```

## Step 2: Data Integration

**Script**: `2_data_integration.py`

Merges all data sources and enriches with game script analysis.

### Process Flow

```
┌─────────────┐
│   FanDuel   │ (primary - defines player pool)
└──────┬──────┘
       │
       ├─ JOIN FantasyPros (on normalized name) → fpProjPts
       ├─ JOIN TD Odds (on normalized name) → tdOdds
       └─ JOIN Game Scripts (on team) → game context
              │
              ▼
       ┌─────────────┐
       │  1_players  │ (integrated player data)
       └─────────────┘
```

### Player Name Matching

Uses intelligent name normalization to match players across sources:

**For Players**:
```python
"Patrick Mahomes II" → "patrick mahomes ii"
"D.K. Metcalf" → "dk metcalf"
"Gabe Davis" → "gabe davis"
```

**For Defense Teams**:
```python
"Seattle Seahawks" → "seahawks"      # Extract team name
"Los Angeles Rams" → "rams"          # Handle multi-word cities
"San Francisco 49ers" → "49ers"      # Preserve numbers
```

### Game Script Calculation

Analyzes betting lines to calculate continuous probability distributions:

**Script Types**:
- **Shootout**: High total, tight spread → pass-heavy, high scoring
- **Defensive**: Low total → run-heavy, low scoring
- **Blowout**: Large spread → winning team runs, losing team passes
- **Competitive**: Tight spread, mid total → balanced game flow

**Example**:
```
CAR @ LAR  (Total: 48.5, Spread: -7.5)
  Shootout:     0.532 (53.2%)
  Defensive:    0.186 (18.6%)
  Blowout:      0.194 (19.4%)
  Competitive:  0.088 ( 8.8%)
```

### Usage

```bash
# Integrate data for specific week
python code/2_data_integration.py --week-dir data/2025_12_08

# Current week (auto-detect)
python code/2_data_integration.py
```

### Output Files

```
intermediate/
├── 1_players.csv       # ~345 matched players
│   ├── id, name, position, salary
│   ├── fpProjPts (consensus projection)
│   ├── tdOdds, team, opponent
│   └── game_* (script probabilities)
│
└── 2_game_scripts.csv  # ~16 games
    ├── game_id, home/away teams
    ├── total, spread, moneylines
    └── script probabilities
```

## Step 3: Optimize Lineups

**Script**: `3_run_optimizer.py`

Generates and evaluates lineups using a 3-phase optimization strategy.

### Three-Phase Approach

#### Phase 1: Candidate Generation (MILP)

Uses mixed-integer linear programming to generate diverse candidates:

**Tier 1 (1-20)**: Deterministic chalk
- Maximize consensus projections
- Strict max-overlap (7 players) for diversity

**Tier 2 (21-100)**: Temperature-based variation
- Temperature ramps from 0.3 → 1.1
- Maintains diversity via max-overlap constraint

**Tier 3 (101-1000)**: Random weighted sampling
- High temperature (1.5-3.0)
- Stricter max-overlap (6 players) for contrarian plays

#### Phase 2: Monte Carlo Evaluation

Simulates each lineup 10,000+ times to estimate distribution:

```
For each lineup:
  For each simulation:
    For each player:
      Sample from distribution:
        - Base: Consensus projection
        - Game script adjustment
        - TD probability boost
        - Negative correlation (RB/QB, WR/DEF)

    Sum player scores → lineup score

  Calculate percentiles: P10, P50 (median), P90, mean, std
```

**Key Metrics**:
- **Median**: Most likely score (50th percentile)
- **P90**: Upside scenario (90th percentile)
- **P10**: Downside scenario (10th percentile)
- **Std**: Volatility/variance

#### Phase 3: Genetic Algorithm Refinement

Evolves the population toward optimal fitness:

```
Generation loop:
  1. Select parents (tournament selection)
  2. Create offspring (crossover)
  3. Mutate some offspring (swap players)
  4. Evaluate new lineups (Monte Carlo)
  5. Keep best 100 (elitism)
  6. Check convergence

Repeat until:
  - No improvement for 5 generations, OR
  - Max generations reached (30)
```

**Fitness Functions**:
- `balanced`: Median + 0.3×P90 - 0.1×P10 (default)
- `median`: Pure median maximization
- `upside`: Median + P90 (tournament focus)
- `safe`: Median - 0.5×Std (cash game focus)

### Usage

```bash
# Full optimization (~1-2 hours)
python code/3_run_optimizer.py --week-dir data/2025_12_08

# Quick test (~2 minutes)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test

# Custom parameters
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --candidates 500 \
  --simulations 5000 \
  --fitness upside \
  --iterations 3

# Resume interrupted run
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --run-name run_20251208_143022
```

### Configuration Tracking

All run parameters are saved to `0_config.json`:

```json
{
  "timestamp": "2025-12-08T14:30:22",
  "week": "2025_12_08",
  "optimizer": {
    "candidates": 1000,
    "simulations": 10000,
    "fitness": "balanced",
    "max_generations": 30
  },
  "data_sources": {
    "players": "intermediate/1_players.csv",
    "game_scripts": "intermediate/2_game_scripts.csv"
  },
  "completed": true
}
```

This enables exact reproduction of historical runs.

### Output Files

```
outputs/run_YYYYMMDD_HHMMSS/
├── 0_config.json          # Run configuration
├── 1_candidates.csv       # All candidates (1000 lineups)
├── 2_simulations.csv      # MC results (10k sims each)
├── 3_lineups.csv          # Best lineups (top 100)
├── 4_summary.json         # Stats and timing
└── iteration_N/           # Per-iteration details
    ├── optimal_lineups.csv
    └── generation_N.json
```

**Best Lineups (`3_lineups.csv`)** contains:
```
lineup_id, QB, RB, RB, WR, WR, WR, TE, FLEX, DEF,
total_salary, projected_mean, projected_median,
p10, p90, std, skewness, fitness
```

## Step 4: Backtest Performance

**Script**: `4_backtest.py`

Scores lineups against actual results and evaluates projection accuracy.

### Process

1. **Load Actual Results**
   - FanDuel results JSON (downloaded after slate)
   - Parses player scores and game info

2. **Score Lineups**
   - Match players by normalized name
   - Sum actual fantasy points
   - Fall back to consensus for missing players

3. **Analyze Performance**
   - Compare actual vs projected
   - Check calibration (P10-P90 range)
   - Identify best/worst lineups

4. **Game Script Analysis**
   - Calculate actual game scripts
   - Compare to predictions

### Usage

```bash
# Backtest specific run
python code/4_backtest.py \
  --week-dir data/2025_12_08 \
  --run-dir run_20251208_143022

# Auto-detect latest run
python code/4_backtest.py --week-dir data/2025_12_08

# Custom consensus projections
python code/4_backtest.py \
  --week-dir data/2025_12_08 \
  --run-dir run_20251208_143022 \
  --consensus custom_projections.csv
```

### Output Files

```
outputs/run_YYYYMMDD_HHMMSS/
├── 5_scored_lineups.csv    # All lineups with actual scores
├── 6_backtest_games.csv    # Actual game scripts
└── 7_backtest_summary.json # Performance metrics
```

### Metrics

**Projection Accuracy**:
- Average error (actual - projected)
- RMSE (root mean squared error)
- Correlation coefficient

**Calibration**:
- Within P10-P90: Should be ~80%
- Above P90: Should be ~10%
- Below P10: Should be ~10%

**Example Summary**:
```json
{
  "lineups": {
    "total": 100,
    "avg_actual": 119.67,
    "max_actual": 151.34,
    "min_actual": 99.42
  },
  "projections": {
    "avg_projected_mean": 131.95,
    "avg_error": -12.28,
    "rmse": 15.35,
    "correlation_mean": 0.08
  },
  "calibration": {
    "within_p10_p90_pct": 93.0,
    "above_p90_pct": 0.0,
    "below_p10_pct": 7.0
  }
}
```

## Orchestration Script

**Script**: `run_week.py`

Runs all 4 steps in sequence with smart defaults.

### Features

- Auto-detects current week (next Sunday)
- Skips data fetch for historical weeks
- Validates data between steps
- Continues from last successful step on errors

### Usage

```bash
# Run current week (auto-detect)
python run_week.py

# Run specific week
python run_week.py --week 2025-12-08

# Quick test mode
python run_week.py --quick-test

# Skip certain steps
python run_week.py --skip-fetch --skip-integrate

# Run backtest only
python run_week.py --week 2025-11-30 --backtest-only
```

### Workflow

```
1. Determine week (current or specified)
2. Create week directory structure
3. Step 1: Fetch data (skip if historical)
4. Step 2: Integrate data
5. Step 3: Optimize lineups
6. Step 4: Backtest (if results available)
7. Print summary and file locations
```

## Data Flow Summary

```
External APIs
    │
    ├─ FanDuel (manual) ────────┐
    ├─ FantasyPros ─────────────┤
    ├─ DraftKings (lines) ──────┤
    └─ DraftKings (odds) ───────┤
                                 │
                          ┌──────▼──────┐
                          │ 1_fetch.py  │
                          └──────┬──────┘
                                 │
                          inputs/*.csv
                                 │
                          ┌──────▼──────────┐
                          │ 2_integrate.py  │
                          └──────┬──────────┘
                                 │
                      intermediate/*.csv
                                 │
                          ┌──────▼──────────┐
                          │ 3_optimizer.py  │
                          └──────┬──────────┘
                                 │
                        outputs/run_*/*.csv
                                 │
                          ┌──────▼──────────┐
                          │ 4_backtest.py   │
                          └──────┬──────────┘
                                 │
                    outputs/run_*/*_summary.json
```

## Migration from Old Structure

If you have data in the old structure, use the migration script:

```bash
# Auto-detect week and migrate
python migrate_to_new_structure.py --auto

# Specify week explicitly
python migrate_to_new_structure.py --week 2025-11-30

# Dry run (preview without changes)
python migrate_to_new_structure.py --week 2025-11-30 --dry-run
```

This copies data from:
- `data/input/` → `data/YYYY_MM_DD/inputs/`
- `data/intermediate/` → `data/YYYY_MM_DD/intermediate/`
- `cache/` → `data/YYYY_MM_DD/inputs/` (converted to CSV)
- `outputs/` → `data/YYYY_MM_DD/outputs/` (files renamed)

---

**Next**: See OPTIMIZER_USAGE.md for advanced parameter tuning and optimization strategies.
