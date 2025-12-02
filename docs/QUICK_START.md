# Quick Start Guide

Get your DFS lineup optimizer running in minutes!

## Prerequisites

- Python 3.8+
- pip package manager

## 1. Installation

```bash
# Clone the repository
cd fantasyvalue

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Get FanDuel Salaries

Download the current week's player salaries from FanDuel:

1. Go to FanDuel NFL contests
2. Export the player list (usually named `FanDuel-NFL-YYYY-MM-DD-players-list.csv`)
3. Save it to your current week directory:

```bash
# Create week directory (Sunday's date in YYYY_MM_DD format)
mkdir -p data/2025_12_08/inputs

# Copy FanDuel CSV
cp ~/Downloads/FanDuel-NFL-*.csv data/2025_12_08/inputs/fanduel_salaries.csv
```

## 3. Run the Full Pipeline

The easiest way is to use the orchestration script:

```bash
# Run current week (auto-detects this Sunday)
python run_week.py

# Or specify a week explicitly
python run_week.py --week 2025-12-08
```

This runs all 4 steps automatically:
1. **Fetch** - Download projections and odds from FantasyPros and DraftKings
2. **Integrate** - Merge all data sources and calculate game scripts
3. **Optimize** - Generate and evaluate lineups using genetic algorithm
4. **Backtest** (optional) - Score lineups against actual results if available

### Quick Test Mode

To verify everything works before running the full optimization:

```bash
python run_week.py --quick-test
```

This runs a faster version (50 candidates, 1000 simulations) in ~2 minutes.

## 4. Run Individual Steps

You can also run each step independently:

### Step 1: Fetch Data

```bash
python code/1_fetch_data.py --week-dir data/2025_12_08
```

Downloads and converts to CSV:
- FantasyPros projections → `fantasypros_projections.csv`
- DraftKings game lines → `game_lines.csv`
- DraftKings TD odds → `td_odds.csv`

### Step 2: Data Integration

```bash
python code/2_data_integration.py --week-dir data/2025_12_08
```

Merges all data sources and outputs:
- `1_players.csv` - All players with integrated projections
- `2_game_scripts.csv` - Game script probabilities (shootout, defensive, etc.)

### Step 3: Run Optimizer

```bash
# Full optimization (takes ~1-2 hours)
python code/3_run_optimizer.py --week-dir data/2025_12_08

# Quick test (~2 minutes)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test
```

Outputs to `data/2025_12_08/outputs/run_YYYYMMDD_HHMMSS/`:
- `0_config.json` - Run configuration
- `1_candidates.csv` - Initial candidate lineups
- `2_simulations.csv` - Monte Carlo evaluation results
- `3_lineups.csv` - **Best lineups** (this is what you want!)
- `4_summary.json` - Run statistics

### Step 4: Backtest (Optional)

After the slate finishes, download actual results from FanDuel and run:

```bash
python code/4_backtest.py --week-dir data/2025_12_08 --run-dir run_YYYYMMDD_HHMMSS
```

Outputs:
- `5_scored_lineups.csv` - All lineups scored with actual results
- `6_backtest_games.csv` - Game script analysis
- `7_backtest_summary.json` - Performance metrics

## 5. View Results

The best lineups are in:
```
data/YYYY_MM_DD/outputs/run_YYYYMMDD_HHMMSS/3_lineups.csv
```

This file contains the top lineups with:
- Player names and positions
- Salary usage
- Projected scores (mean, median, P10, P90)
- Risk metrics (std, skewness)
- Fitness scores

## 6. Resume Interrupted Run

If the optimizer is interrupted, resume with:

```bash
python code/3_run_optimizer.py --week-dir data/2025_12_08 --run-name run_YYYYMMDD_HHMMSS
```

## Common Workflows

### Current Week Optimization

```bash
# 1. Download FanDuel salaries to data/YYYY_MM_DD/inputs/

# 2. Run everything
python run_week.py

# 3. Get best lineups from:
#    data/YYYY_MM_DD/outputs/run_*/3_lineups.csv
```

### Historical Analysis

```bash
# 1. Ensure historical data exists in data/YYYY_MM_DD/

# 2. Run pipeline for that week
python run_week.py --week 2025-11-30

# 3. Backtest results
python code/4_backtest.py --week-dir data/2025_11_30 --run-dir run_*
```

### Multiple Optimization Runs

```bash
# Run multiple iterations to explore different parameter spaces
python code/3_run_optimizer.py --week-dir data/2025_12_08 --iterations 3
```

## Directory Structure

After running, you'll have:

```
data/
└── 2025_12_08/              # Week directory (Sunday date)
    ├── inputs/              # Raw data sources (CSV)
    │   ├── fanduel_salaries.csv
    │   ├── fantasypros_projections.csv
    │   ├── game_lines.csv
    │   └── td_odds.csv
    ├── intermediate/        # Processed data
    │   ├── 1_players.csv
    │   └── 2_game_scripts.csv
    └── outputs/             # Optimization results
        └── run_20251208_143022/
            ├── 0_config.json
            ├── 1_candidates.csv
            ├── 2_simulations.csv
            ├── 3_lineups.csv      ← YOUR BEST LINEUPS
            ├── 4_summary.json
            ├── 5_scored_lineups.csv   (after backtest)
            ├── 6_backtest_games.csv   (after backtest)
            └── 7_backtest_summary.json (after backtest)
```

## Troubleshooting

### Missing FanDuel CSV
```
Error: FanDuel salaries not found
```
Download from FanDuel and save to `data/YYYY_MM_DD/inputs/fanduel_salaries.csv`

### No Lineups Generated
```
Warning: Failed to generate lineup
```
Check that you have enough players at each position. May need to adjust `--max-overlap` parameter.

### Slow Performance
```
Evaluation taking too long
```
Use `--quick-test` flag for faster results, or reduce `--candidates` and `--simulations`.

## Next Steps

- **WORKFLOW.md** - Understand the complete data pipeline
- **OPTIMIZER_USAGE.md** - Advanced usage and parameter tuning
- **ARCHITECTURE_REFACTOR.md** - Technical architecture details

---

**Ready to go!** Start with `python run_week.py --quick-test` to verify everything works.
