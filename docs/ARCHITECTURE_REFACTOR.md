# Architecture Refactor: Self-Contained Week Folders

## Overview

Redesign the entire project structure so each week (current or historical) is a self-contained folder with identical structure. This enables:

- **Historical replay**: Run the full pipeline on any past week
- **Data-source agnostic scripts**: Pipeline doesn't care if data is current or historical
- **Reproducibility**: Each run tracks all configuration parameters
- **Readability**: CSV-first approach, natural file sorting

## Target Architecture

```
data/
├── 2025_12_01/                      # Week 13 (folder name = Sunday game date)
│   ├── inputs/                      # Raw data from 1_fetch_data
│   │   ├── fanduel_salaries.csv
│   │   ├── fantasypros_projections.csv
│   │   ├── game_lines.csv
│   │   ├── td_odds.csv
│   │   └── fanduel_results.csv      # Added post-week for backtest
│   │
│   ├── intermediate/                # Output from 2_data_integration
│   │   ├── 1_players.csv            # Single merged player file
│   │   └── 2_game_scripts.csv       # Game script probabilities
│   │
│   └── outputs/                     # Output from 3_run_optimizer + 4_backtest
│       └── run_20251201_143022/
│           ├── 0_config.json        # Run configuration
│           ├── 1_candidates.csv     # Phase 1 MILP
│           ├── 2_simulations.csv    # Phase 2 Monte Carlo
│           ├── 3_lineups.csv        # FINAL LINEUPS ⭐
│           ├── 4_summary.json       # Run statistics
│           ├── 5_scored_lineups.csv     # 4_backtest output
│           ├── 6_backtest_games.csv     # 4_backtest output
│           └── 7_backtest_summary.json  # 4_backtest output
```

## Data Flow

### Step 1: Fetch Data (`1_fetch_data.py`)
- **Purpose**: Download current week data ONLY
- **Inputs**: None (scrapes live APIs)
- **Outputs**: `data/CURRENT_WEEK/inputs/`
  - `fanduel_salaries.csv` - FanDuel contest export (user provides)
  - `fantasypros_projections.csv` - FantasyPros API (converted from JSON)
  - `game_lines.csv` - DraftKings spreads/totals
  - `td_odds.csv` - DraftKings anytime TD odds
- **Smart caching**: Skips re-download if files exist (unless `--force`)
- **No week argument**: Can ONLY fetch current data

### Step 2: Data Integration (`2_data_integration.py`)
- **Purpose**: Merge and normalize all data sources
- **Inputs**: `data/WEEK/inputs/*.csv`
- **Outputs**: `data/WEEK/intermediate/`
  - `1_players.csv` - Single merged player file with projections + game script adjustments
  - `2_game_scripts.csv` - Game-level probabilities (shootout, defensive, blowout, competitive)
- **Minimal output**: Only what optimizer needs

### Step 3: Run Optimizer (`3_run_optimizer.py`)
- **Purpose**: Generate and optimize lineups
- **Inputs**: `data/WEEK/intermediate/1_players.csv`
- **Outputs**: `data/WEEK/outputs/run_TIMESTAMP/`
  - `0_config.json` - All run parameters for reproducibility
  - `1_candidates.csv` - Phase 1 MILP candidates
  - `2_simulations.csv` - Phase 2 Monte Carlo evaluations
  - `3_lineups.csv` - **Final optimized lineups (top N only)**
  - `4_summary.json` - Run statistics and metadata

### Step 4: Backtest (`4_backtest.py`)
- **Purpose**: Score lineups against actual FanDuel results
- **Inputs**:
  - `data/WEEK/outputs/run_TIMESTAMP/3_lineups.csv`
  - `data/WEEK/inputs/fanduel_results.csv`
- **Outputs**: `data/WEEK/outputs/run_TIMESTAMP/` (same directory)
  - `5_scored_lineups.csv` - Lineups with actual scores
  - `6_backtest_games.csv` - Game-level accuracy analysis
  - `7_backtest_summary.json` - Validation metrics

### Orchestration Script (`run_week.py`)
- **Purpose**: Coordinate all steps for current or historical weeks
- **Usage**:
  ```bash
  # Analyze current week (fetches fresh data)
  python run_week.py

  # Analyze historical week (uses existing data)
  python run_week.py --week 2025-12-01

  # Re-fetch data for existing week
  python run_week.py --week 2025-12-01 --fetch

  # Run with optimizer options
  python run_week.py --candidates 1000 --sims 10000 --fitness balanced
  ```

## Key Design Decisions

### 1. CSV-First Philosophy
- Convert all JSON downloads to CSV immediately
- Humans can inspect/edit in Excel or text editors
- Version control friendly (git diffs work)
- Exception: config.json and summary.json (small metadata files)

### 2. Numeric Prefixes for Natural Sorting
- Files sort naturally in file browsers (0_, 1_, 2_, etc.)
- Clear execution order at a glance
- No need to remember "which file comes first?"

### 3. No Nested Subdirectories
- All run outputs in single `run_TIMESTAMP/` folder
- No `4_backtest/` subdirectory - just properly name CSVs
- Simpler structure, easier to navigate

### 4. Configuration Tracking
- Every run saves `0_config.json` with all parameters
- Enables exact reproduction of any historical run
- Track data sources, optimizer settings, timestamps

**Example `0_config.json`**:
```json
{
  "timestamp": "2025-12-01T14:30:22",
  "week": "2025_12_01",
  "optimizer": {
    "candidates": 1000,
    "simulations": 10000,
    "fitness": "balanced",
    "iterations": 5,
    "population_size": 200,
    "mutation_rate": 0.3,
    "enforce_diversity": true
  },
  "data_sources": {
    "fanduel": "inputs/fanduel_salaries.csv",
    "fantasypros": "inputs/fantasypros_projections.csv",
    "game_lines": "inputs/game_lines.csv",
    "td_odds": "inputs/td_odds.csv"
  },
  "completed": true,
  "phases_completed": ["candidates", "simulations", "genetic_1", "genetic_2"]
}
```

### 5. Week Folder Naming
- Format: `YYYY_MM_DD` (Sunday of game week)
- Example: `2025_12_01` for Week 13
- Sortable, unambiguous, no timezone issues

### 6. Self-Contained = Portable
- Each week folder contains everything needed
- Can zip and share entire week
- Can delete old weeks without breaking anything

## Implementation Plan

### ✅ Phase 0: Planning
- [x] Define architecture
- [x] Document data flow
- [x] Create implementation plan
- [ ] Review and approve plan

### 📝 Phase 1: Create Orchestration Script
**File**: `run_week.py`

- [ ] Implement week detection (YYYY_MM_DD format)
- [ ] Add conditional fetch logic (current vs historical)
- [ ] Add subprocess coordination for all 4 steps
- [ ] Add error handling and rollback
- [ ] Add progress reporting
- [ ] Test with current week
- [ ] Test with historical week

**Key Features**:
- Detect if week folder exists → skip fetch or warn user
- Pass `--week-dir` to all scripts
- Check for `fanduel_results.csv` before running backtest
- Pretty printing with step headers

### 📝 Phase 2: Refactor `1_fetch_data.py`
**Current behavior**: Mixed fetch + processing
**Target behavior**: Pure download + CSV conversion only

**Changes needed**:
- [ ] Remove `--week` argument (only fetches current)
- [ ] Add `--week-dir` argument (output location)
- [ ] Convert all JSON downloads to CSV immediately
- [ ] Remove all data processing (move to step 2)
- [ ] Add smart caching (skip if file exists unless `--force`)
- [ ] Save to `data/WEEK/inputs/` instead of `data/input/`
- [ ] Update FantasyPros scraper to save CSV
- [ ] Update DraftKings scraper to save CSVs
- [ ] Remove cache/ directory (use inputs/ directly)
- [ ] Add file validation (check CSVs are readable)

**Output CSVs**:
```csv
# fanduel_salaries.csv (already CSV, just move)
id,position,name,salary,team,opponent,game

# fantasypros_projections.csv (convert from JSON)
player_id,name,team,position,projection,floor,ceiling

# game_lines.csv (convert from DK API)
game_id,home_team,away_team,spread,total,home_moneyline,away_moneyline

# td_odds.csv (convert from DK API)
player_id,name,team,position,td_odds,anytime_td_price
```

### 📝 Phase 3: Refactor `2_data_integration.py`
**Current behavior**: Complex intermediate files
**Target behavior**: Minimal output (1_players.csv, 2_game_scripts.csv)

**Changes needed**:
- [ ] Add `--week-dir` argument
- [ ] Read from `data/WEEK/inputs/*.csv`
- [ ] Output to `data/WEEK/intermediate/`
- [ ] Simplify output to 2 files only
- [ ] Move game script analysis here (from optimizer)
- [ ] Merge player data with game script adjustments
- [ ] Remove players_raw.csv (just create players.csv directly)
- [ ] Remove players_integrated.csv (rename to players.csv)
- [ ] Add data validation and quality checks
- [ ] Pretty print data quality report

**Output CSVs**:
```csv
# 1_players.csv
id,name,position,team,opponent,salary,projection,floor,ceiling,
td_odds,game_script_shootout,game_script_defensive,game_script_blowout,
game_script_competitive,adjusted_projection,adjusted_floor,adjusted_ceiling

# 2_game_scripts.csv
game_id,home_team,away_team,spread,total,
prob_shootout,prob_defensive,prob_blowout,prob_competitive
```

### 📝 Phase 4: Refactor `3_run_optimizer.py`
**Current behavior**: Outputs to `outputs/run_TIMESTAMP/`
**Target behavior**: Outputs to `data/WEEK/outputs/run_TIMESTAMP/`

**Changes needed**:
- [ ] Add `--week-dir` argument
- [ ] Read from `data/WEEK/intermediate/1_players.csv`
- [ ] Output to `data/WEEK/outputs/run_TIMESTAMP/`
- [ ] Rename output files with numeric prefixes
- [ ] Create `0_config.json` at start of run
- [ ] Update config.json throughout run (track progress)
- [ ] Rename `BEST_LINEUPS.csv` → `3_lineups.csv`
- [ ] Rename `evaluations.csv` → `2_simulations.csv`
- [ ] Rename `final_summary.json` → `4_summary.json`
- [ ] Update resume logic to use new paths
- [ ] Update progress reporting

**File renames**:
- `candidates.csv` → `1_candidates.csv`
- `evaluations.csv` → `2_simulations.csv`
- `BEST_LINEUPS.csv` → `3_lineups.csv`
- `final_summary.json` → `4_summary.json`
- NEW: `0_config.json`

### 📝 Phase 5: Refactor `4_backtest.py` (rename from `5_backtest.py`)
**Current behavior**: Standalone script, manual paths
**Target behavior**: Integrated into pipeline, co-located outputs

**Changes needed**:
- [ ] Rename file: `5_backtest.py` → `4_backtest.py`
- [ ] Add `--week-dir` argument
- [ ] Add `--run-dir` argument (which run to backtest)
- [ ] Read lineups from `data/WEEK/outputs/RUN/3_lineups.csv`
- [ ] Read results from `data/WEEK/inputs/fanduel_results.csv`
- [ ] Output to same run directory (not separate folder)
- [ ] Rename outputs with numeric prefixes
- [ ] Skip gracefully if fanduel_results.csv missing
- [ ] Pretty print backtest summary to console

**File renames**:
- `scored_lineups.csv` → `5_scored_lineups.csv`
- `games.csv` → `6_backtest_games.csv`
- `backtest_summary.json` → `7_backtest_summary.json`

### 📝 Phase 6: Update Documentation
- [ ] Update `QUICK_START.md` with new commands
- [ ] Update `OPTIMIZER_USAGE.md` with new architecture
- [ ] Update `WORKFLOW.md` with new data flow
- [ ] Create `HISTORICAL_DATA.md` guide
- [ ] Update README.md with new structure
- [ ] Add migration guide for old data

### 📝 Phase 7: Migration & Testing
- [ ] Create migration script to move existing data
- [ ] Test full pipeline with current week
- [ ] Test full pipeline with historical week
- [ ] Test resume functionality
- [ ] Test backtest with/without results
- [ ] Test error handling (missing files, etc.)
- [ ] Clean up old files and directories

### 📝 Phase 8: Cleanup
- [ ] Remove `6_archive_week.py` (deprecated)
- [ ] Remove old `cache/` directory
- [ ] Remove old `data/input/` directory
- [ ] Remove old `data/intermediate/` directory
- [ ] Remove old `outputs/` directory (after migration)
- [ ] Update `.gitignore` for new structure

## Migration Strategy

### Moving Existing Data

Create a `migrate_to_new_structure.py` script that:

1. Finds current data in old locations
2. Prompts user for week date (YYYY_MM_DD)
3. Creates new week folder structure
4. Copies files to appropriate locations
5. Validates migration
6. Optionally removes old files

**Example**:
```bash
python migrate_to_new_structure.py --week 2025-12-01
```

This will:
- Move `data/input/FanDuel-NFL-*.csv` → `data/2025_12_01/inputs/fanduel_salaries.csv`
- Move `cache/fantasypros_*.json` → convert to `data/2025_12_01/inputs/fantasypros_projections.csv`
- Move `data/intermediate/players_integrated.csv` → `data/2025_12_01/intermediate/1_players.csv`
- Move `outputs/run_*/BEST_LINEUPS.csv` → `data/2025_12_01/outputs/run_*/3_lineups.csv`
- etc.

## Testing Plan

### Test Case 1: Current Week (Fresh Data)
```bash
python run_week.py --quick-test
```
Expected:
- Fetches fresh data to `data/YYYY_MM_DD/inputs/`
- Runs all 4 steps
- Creates numbered output files
- config.json has all parameters

### Test Case 2: Historical Week (Existing Data)
```bash
python run_week.py --week 2025-12-01 --quick-test
```
Expected:
- Skips fetch (data already exists)
- Runs steps 2-4
- Outputs to same week folder
- config.json shows correct week

### Test Case 3: Re-fetch Historical Week
```bash
python run_week.py --week 2025-12-01 --fetch --quick-test
```
Expected:
- Runs fetch even though folder exists
- Warns user about overwriting
- Continues with full pipeline

### Test Case 4: Backtest with Results
```bash
# First add results file
cp fanduel_results.json data/2025_12_01/inputs/fanduel_results.csv

# Then run
python run_week.py --week 2025-12-01
```
Expected:
- Runs all 4 steps including backtest
- Creates 5_scored_lineups.csv, 6_backtest_games.csv, 7_backtest_summary.json
- Prints backtest summary to console

### Test Case 5: Resume Interrupted Run
```bash
python run_week.py --week 2025-12-01 --resume run_20251201_143022
```
Expected:
- Loads existing config.json
- Detects completed phases
- Continues from last iteration
- Preserves all previous outputs

## Success Criteria

✅ Architecture is successfully refactored when:

1. **Self-contained weeks**: Each week folder contains all inputs, intermediates, and outputs
2. **Data-source agnostic**: Scripts don't distinguish current vs historical
3. **CSV-first**: All data files are CSV except small metadata JSONs
4. **Natural sorting**: Files sort correctly in all file browsers
5. **Reproducible**: Can replay any week with exact configuration
6. **No orphaned files**: All files have clear purpose and location
7. **Documentation updated**: All docs reflect new architecture
8. **Old structure removed**: No duplicate files in old locations
9. **Tests passing**: All test cases above pass successfully
10. **Migration complete**: Existing data moved to new structure

## Timeline Estimate

- **Phase 1** (Orchestration): 1-2 hours
- **Phase 2** (fetch_data refactor): 2-3 hours
- **Phase 3** (data_integration refactor): 2-3 hours
- **Phase 4** (optimizer refactor): 1-2 hours
- **Phase 5** (backtest refactor): 1 hour
- **Phase 6** (Documentation): 1-2 hours
- **Phase 7** (Migration & Testing): 2-3 hours
- **Phase 8** (Cleanup): 1 hour

**Total**: ~11-17 hours of focused work

## Notes

- Preserve backward compatibility during transition (support both old and new paths)
- Add deprecation warnings when old paths are used
- Create migration script before removing old structure
- Update .gitignore to ignore data/ folder (except .gitkeep files)
- Consider adding data/ folder to .gitignore with examples in docs/
