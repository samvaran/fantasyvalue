# Complete Pipeline Review - End to End

This document provides a comprehensive overview of the entire data pipeline from fetching raw data through to lineup optimization.

---

## Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA FETCHING                      │
│                        (fetch_data.py)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ├─> FanDuel Salaries (CSV)
                         ├─> FantasyPros Projections (web scrape)
                         ├─> DraftKings Game Lines (web scrape)
                         ├─> DraftKings TD Odds (web scrape)
                         ├─> ESPN Player IDs (API)
                         └─> ESPN Projections (API)
                         │
                         ↓
                    [players_raw.csv]
                    [game_lines.csv]
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                  PHASE 2A: GAME SCRIPT ANALYSIS                 │
│                  (game_script_continuous.py)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
               [game_script_continuous.csv]
                         │
                         │  Continuous probabilities:
                         │  - shootout_prob
                         │  - defensive_prob
                         │  - blowout_prob
                         │  - competitive_prob
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                 PHASE 2B: DATA INTEGRATION                      │
│                   (data_integration.py)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │  Combines:
                         │  - Player projections
                         │  - Game script probabilities
                         │  - TD odds adjustments
                         │  - Team offensive totals
                         │
                         ↓
                [players_integrated.csv]
                         │
                         │  38 columns including:
                         │  - Base projections (fpProjPts)
                         │  - 5 game scripts × (floor + ceiling)
                         │  - TD odds multipliers
                         │  - Game script probabilities
                         │  - Team context (is_favorite)
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    PHASE 3: OPTIMIZATION                        │
│                      (run_optimizer.py)                         │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ├─> PHASE 3.1: Generate Candidates
                         │   (optimizer/generate_candidates.py)
                         │   - 1000 MILP-optimized lineups
                         │   - Tiered sampling (chalk → random)
                         │
                         ├─> PHASE 3.2: Monte Carlo Evaluation
                         │   (optimizer/evaluate_lineups.py)
                         │   - 10,000 simulations per lineup
                         │   - Parallel processing (12x speedup)
                         │   - Distribution sampling
                         │
                         └─> PHASE 3.3: Genetic Refinement
                             (optimizer/optimize_genetic.py)
                             - Iterative improvement
                             - Convergence detection
                             - Elite preservation
                         │
                         ↓
                   [BEST_LINEUPS.csv]
```

---

## Phase 1: Data Fetching

**Script:** `fetch_data.py`
**Runtime:** ~2-5 minutes (depending on web scraping)
**Inputs:** Web sources + local FanDuel CSV
**Outputs:** `players_raw.csv`, `game_lines.csv`

### Data Sources

1. **FanDuel Salaries** (Local CSV)
   - Always fresh, never cached
   - Contains: player names, positions, teams, salaries

2. **FantasyPros Projections** (Web Scrape)
   - Consensus projections from experts
   - Contains: fpProjPts (main projection)
   - Cacheable (use `--fp` to refresh)

3. **DraftKings Game Lines** (Web Scrape)
   - Spread, total, moneylines
   - Over/under odds
   - Projected team points
   - Cacheable (use `--lines` to refresh)

4. **DraftKings TD Odds** (Web Scrape)
   - Anytime TD scorer odds for skill players
   - Converted to probability (tdProbability)
   - Cacheable (use `--dk` to refresh)

5. **ESPN Player IDs + Projections** (API)
   - espnScoreProjection (main)
   - espnLowScore, espnHighScore (floor/ceiling)
   - espnOutsideProjection, espnSimulationProjection
   - Cacheable (use `--espn` to refresh)

### Merge Strategy

All data sources merged on **normalized player names**:
- Lowercase conversion
- Suffix removal (Jr., Sr., III)
- Punctuation normalization
- Team abbreviation matching (JAC → JAX)

### Date Filtering

Games automatically filtered to "this week":
- Today through next Monday
- Uses actual date parsing (not hardcoded counts)
- Stops at future week games

### Usage

```bash
# Full refresh (all sources)
python fetch_data.py --all

# Just update game lines (if odds changed)
python fetch_data.py --lines

# Use all cached data
python fetch_data.py
```

### Output Schema: `players_raw.csv`

Key columns (19 total):
- `name`, `position`, `team`, `opponent`, `salary`
- `fpProjPts` - FantasyPros consensus
- `tdOdds`, `tdProbability` - DraftKings TD odds
- `proj_team_pts`, `proj_opp_pts` - Team totals from game lines
- `espnScoreProjection`, `espnLowScore`, `espnHighScore` - ESPN projections
- `espnOutsideProjection`, `espnSimulationProjection` - Alternative ESPN models

---

## Phase 2A: Game Script Analysis

**Script:** `game_script_continuous.py`
**Runtime:** ~1 second
**Inputs:** `game_lines.csv`
**Outputs:** `game_script_continuous.csv`

### Methodology

Instead of discrete buckets, calculates **continuous probabilities** (0-1) for each script type using sigmoid functions.

#### 1. Shootout Probability

Indicators:
- **High total** (>50 is strong, >48 is moderate)
- **Competitive spread** (closer to 0 is better)
- **Market favors over** (negative over odds)

Formula:
```
shootout_score = total_score × 0.50 + spread_score × 0.30 + market_score × 0.20
```

Sigmoid transitions:
- `total_score = sigmoid(total, midpoint=48, steepness=0.5)`
- `spread_score = 1 - sigmoid(abs(spread), midpoint=5, steepness=0.4)`

#### 2. Defensive Probability

Indicators:
- **Low total** (<42 is strong, <45 is moderate)
- **Market favors under**

Formula:
```
defensive_score = inverted_total_score × 0.70 + market_score × 0.30
```

#### 3. Blowout Probability

Indicators:
- **Large spread** (>7 is strong)
- **Lopsided moneylines**
- **Market confidence** (balanced spread odds = high confidence)

Formula:
```
blowout_score = spread_magnitude × 0.60 + moneyline_magnitude × 0.40
blowout_confidence = 1 - abs(fav_spread_prob - 0.5)
```

#### 4. Competitive Probability

Indicator:
- **Small spread** (closer to 0 = higher score)

Formula:
```
competitive_score = 1 - sigmoid(abs(spread), midpoint=3, steepness=0.6)
```

### Normalization

All scores normalized to probability distribution (sum = 1):
```
shootout_prob = shootout_score / total_score
defensive_prob = defensive_score / total_score
blowout_prob = blowout_score / total_score
competitive_prob = competitive_score / total_score
```

### Output Schema: `game_script_continuous.csv`

Columns (11 total):
- `game_id` - "AWAY@HOME"
- `favorite`, `underdog` - Team abbreviations
- `shootout_prob`, `defensive_prob`, `blowout_prob`, `competitive_prob` - Normalized (0-1)
- `primary_script` - Highest probability script
- `script_strength` - How dominant the primary script is (max probability)
- `blowout_confidence` - Market agreement on spread (0-1)

### Usage

```bash
python game_script_continuous.py
```

---

## Phase 2B: Data Integration

**Script:** `data_integration.py`
**Runtime:** ~1 second
**Inputs:** `players_raw.csv`, `game_lines.csv`
**Outputs:** `players_integrated.csv`, `game_scripts_enhanced.csv`

### Pipeline Steps

#### Step 1: Load Raw Data
- `players_raw.csv` - Player projections
- `game_lines.csv` - Betting lines
- TD odds lookup dictionary

#### Step 2: Calculate Team Offensive Totals

For each team, aggregate:
- `total_projected_pts` - Sum of all player projections
- `total_td_prob` - Sum of all TD probabilities

This provides a bottom-up signal to enhance game script analysis.

#### Step 3: Enhanced Game Script Analysis

Combines:
- **Betting lines** (spreads, totals, odds) - 80% weight
- **Team offensive totals** (projections, TD odds) - 20% weight

Team total signals:
```python
team_shootout_signal = min(total_proj / 55.0, 1.0)
team_td_signal = min(total_td_odds / 2.5, 1.0)

# Enhance shootout probability
shootout_score = (
    shootout_base * 0.80 +  # From betting lines
    (team_shootout_signal + team_td_signal) / 2 * 0.20  # From projections
)
```

#### Step 4: Calculate Floor/Ceiling for ALL Game Scripts

For each player, calculate floor/ceiling under **5 different scenarios**:
1. `floor_shootout` / `ceiling_shootout`
2. `floor_defensive` / `ceiling_defensive`
3. `floor_blowout_favorite` / `ceiling_blowout_favorite`
4. `floor_blowout_underdog` / `ceiling_blowout_underdog`
5. `floor_competitive` / `ceiling_competitive`

**Base values:**
- `original_floor` = `espnLowScore` (or `fpProjPts * 0.5` if missing)
- `original_ceiling` = `espnHighScore` (or `fpProjPts * 1.5` if missing)

**Adjustment formula:**
```python
adjusted_floor = original_floor * GAME_SCRIPT_FLOOR[script][position]
adjusted_ceiling = original_ceiling * GAME_SCRIPT_CEILING[script][position]

# CRITICAL CONSTRAINTS:
adjusted_floor = min(adjusted_floor, consensus * 0.95)  # Never exceed consensus
adjusted_ceiling = max(adjusted_ceiling, consensus * 1.05)  # Never fall below consensus
```

**Game Script Multipliers (examples):**

| Script | Position | Floor Mult | Ceiling Mult | Reasoning |
|--------|----------|------------|--------------|-----------|
| Shootout | QB | 0.95 | 1.15 | High variance, high ceiling |
| Shootout | RB | 0.90 | 0.90 | Less running in shootouts |
| Shootout | WR | 0.95 | 1.20 | Strong ceiling boost |
| Defensive | D | 1.15 | 1.25 | Low scoring = more D points |
| Blowout (Fav) | RB | 1.20 | 1.10 | Clock management, safe floor |
| Blowout (Dog) | RB | 0.85 | 0.90 | Abandoned run game |

#### Step 5: Calculate TD Odds Multipliers

These are **stored but not applied** (application happens in Monte Carlo phase):

```python
td_prob_scaled = tdProbability / 100.0

floor_mult = 1.0 + (td_prob_scaled * 0.05)  # Up to +5% boost
ceiling_mult = 1.0 + (td_prob_scaled * 0.15)  # Up to +15% boost
```

Example: Player with 20% TD probability
- `td_odds_floor_mult = 1.01` (1% boost)
- `td_odds_ceiling_mult = 1.03` (3% boost)

#### Step 6: Add Game Context

For each player:
- `game_id` - Which game they're in
- `shootout_prob`, `defensive_prob`, `blowout_prob`, `competitive_prob` - From game script analysis
- `is_favorite` - Boolean, is player's team the favorite?

### Output Schema: `players_integrated.csv`

**38 columns total:**

**Base Data (19):**
- `name`, `position`, `team`, `opponent`, `salary`
- `fpProjPts` - Consensus projection (mean)
- `tdProbability` - TD odds as probability
- `espnLowScore`, `espnHighScore` - Original floor/ceiling
- `proj_team_pts`, `proj_opp_pts`
- Other ESPN projections

**Game Script Floors/Ceilings (10):**
- `floor_shootout`, `ceiling_shootout`
- `floor_defensive`, `ceiling_defensive`
- `floor_blowout_favorite`, `ceiling_blowout_favorite`
- `floor_blowout_underdog`, `ceiling_blowout_underdog`
- `floor_competitive`, `ceiling_competitive`

**TD Odds Multipliers (2):**
- `td_odds_floor_mult` (e.g., 1.01)
- `td_odds_ceiling_mult` (e.g., 1.03)

**Game Context (7):**
- `game_id`
- `shootout_prob`, `defensive_prob`, `blowout_prob`, `competitive_prob`
- `is_favorite`

### Usage

```bash
python data_integration.py
```

---

## Phase 3: Lineup Optimization

**Script:** `run_optimizer.py`
**Runtime:** ~70-90 minutes for full pipeline
**Inputs:** `players_integrated.csv`, `game_script_continuous.csv`
**Outputs:** `outputs/run_YYYYMMDD_HHMMSS/BEST_LINEUPS.csv`

### Three-Phase Architecture

#### Phase 3.1: Generate Candidates

**Script:** `optimizer/generate_candidates.py`
**Runtime:** ~3 minutes for 1000 lineups
**Method:** MILP (Mixed Integer Linear Programming)

**Tiered Sampling:**
1. **Lineups 1-20**: Deterministic chalk (temperature = 0)
   - Uses expected value across all game scripts
   - Max 7 overlapping players between lineups

2. **Lineups 21-100**: Temperature-based variation (temp 0.3 → 1.1)
   - Samples game scripts with temperature-adjusted probabilities
   - More diverse than chalk

3. **Lineups 101-1000**: Random weighted sampling (temp 1.5 → 3.0)
   - Highly random contrarian plays
   - Max 6 overlapping players (stricter diversity)

**Projection Calculation:**

For each player, calculate MILP projection based on temperature:

```python
if temperature == 0:
    # Deterministic: weighted average across all game scripts
    for script in ['shootout', 'defensive', 'blowout', 'competitive']:
        script_key = determine_script_key(script, player['is_favorite'])
        floor = player[f'floor_{script_key}'] * player['td_odds_floor_mult']
        ceiling = player[f'ceiling_{script_key}'] * player['td_odds_ceiling_mult']
        scenario_midpoint = (floor + ceiling) / 2
        weighted_avg += scenario_midpoint * game_script_probs[script]

    milp_projection = (weighted_avg + consensus) / 2  # 50/50 blend

else:
    # Stochastic: sample a game script with temperature
    sampled_script = sample_game_script(game_script_probs, temperature)
    script_key = determine_script_key(sampled_script, player['is_favorite'])
    floor = player[f'floor_{script_key}'] * player['td_odds_floor_mult']
    ceiling = player[f'ceiling_{script_key}'] * player['td_odds_ceiling_mult']
    scenario_midpoint = (floor + ceiling) / 2

    milp_projection = (scenario_midpoint + consensus) / 2  # 50/50 blend
```

**MILP Constraints:**
- Salary cap: $60,000
- Exactly 9 players
- Position requirements: 1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DEF
- FLEX position: 1 (RB/WR/TE)
- Diversity: Max overlap with previous lineups

**Output:** `candidates.csv` (1000 lineups)

#### Phase 3.2: Monte Carlo Evaluation

**Script:** `optimizer/evaluate_lineups.py`
**Runtime:** ~25-30 minutes for 1000 × 10k simulations
**Method:** Parallel Monte Carlo simulation

**Simulation Process (per lineup, per iteration):**

1. **Sample game script** for each game:
   ```python
   sampled_script = np.random.choice(
       ['shootout', 'defensive', 'blowout', 'competitive'],
       p=[shootout_prob, defensive_prob, blowout_prob, competitive_prob]
   )
   ```

2. **Determine script key** based on team role:
   ```python
   if sampled_script == 'blowout':
       script_key = 'blowout_favorite' if player['is_favorite'] else 'blowout_underdog'
   else:
       script_key = sampled_script
   ```

3. **Get floor/ceiling** for sampled script:
   ```python
   floor = player[f'floor_{script_key}'] * player['td_odds_floor_mult']
   ceiling = player[f'ceiling_{script_key}'] * player['td_odds_ceiling_mult']
   ```

4. **Fit shifted log-normal distribution**:
   - Mean = `fpProjPts` (consensus)
   - P10 = `floor`
   - P90 = `ceiling`
   - Solves for (mu, sigma, shift) parameters

5. **Sample from distribution**:
   ```python
   points = sample_shifted_lognormal(mu, sigma, shift)
   ```

6. **Sum across all 9 players** to get lineup total

**Parallelization:**
- Uses `multiprocessing.Pool` with auto-detected CPU cores
- Achieves ~12x speedup on 8-core machine
- Each worker evaluates subset of lineups independently

**Metrics Calculated:**
- `mean` - Average score across simulations
- `median` - 50th percentile
- `p10` - 10th percentile (floor/bust scenario)
- `p90` - 90th percentile (ceiling/boom scenario)
- `std` - Standard deviation (variance)
- `sharpe` - Risk-adjusted return: `(mean - risk_free_rate) / std`

**Output:** `evaluations.csv` (1000 lineups with metrics)

#### Phase 3.3: Genetic Refinement

**Script:** `optimizer/optimize_genetic.py`
**Runtime:** ~40-60 minutes per iteration
**Method:** Genetic algorithm with elitism

**Algorithm:**

1. **Initialize** with top 100 from Phase 2

2. **Repeat for N generations** (typically 20-30):

   a. **Tournament Selection** (50 parents)
      - Pick 5 random lineups
      - Select best by fitness
      - Repeat 50 times

   b. **Crossover** (100 offspring)
      - Pair parents randomly
      - Position-aware crossover (swap by position)
      - Salary-aware (must stay under cap)

   c. **Mutation** (~30% of offspring)
      - 30% chance per offspring
      - Swap 1-2 players
      - Maintain salary constraint (±$500)

   d. **Parallel Evaluation** (100 offspring × 10k sims)
      - Same Monte Carlo as Phase 2
      - Uses multiprocessing for speed

   e. **Selection** (next generation = 100)
      - Top 20 from current generation (elitism)
      - Top 80 from offspring

   f. **Save Checkpoint**
      - Save generation results
      - Update optimizer state

3. **Convergence Check**
   - Track best fitness over iterations
   - Stop if no improvement for N iterations (patience)
   - Or if improvement < threshold

**Fitness Functions:**

User can choose strategy:

```python
# Conservative: High floor, low risk
fitness = median - 0.5 * std

# Balanced: Expected value (DEFAULT)
fitness = median

# Aggressive: Boom/bust potential
fitness = p90 - p10

# Tournament: Pure upside
fitness = p90
```

**Output per iteration:**
- `iteration_N/optimal_lineups.csv` - All 100 lineups
- `iteration_N/top_10_iteration_N.csv` - Best 10 from this iteration

### Orchestration & Checkpointing

**State Management:**

`optimizer_state.json` tracks:
```json
{
  "phase_1_complete": true,
  "phase_2_complete": true,
  "iterations": [
    {
      "iteration": 1,
      "best_fitness": 125.43,
      "best_median": 142.67,
      "n_generations": 25,
      "time": 2847.2
    }
  ],
  "best_fitness_history": [120.1, 123.5, 125.43],
  "total_time": 5423.8
}
```

**Resumability:**
- Automatically detects completed phases
- Continues from last iteration
- Preserves all previous results

**Final Outputs:**

Main file: `BEST_LINEUPS.csv`
- Top 10 lineups across ALL iterations
- Columns: lineup_id, player_ids, total_salary, mean, median, p10, p90, std, sharpe, fitness

Supporting files:
- `final_summary.json` - Complete run statistics
- `candidates.csv` - All 1000 initial candidates
- `evaluations.csv` - Initial Monte Carlo results
- `iteration_N/` - Results from each genetic iteration

### Usage

```bash
# Quick test (2-3 minutes)
python run_optimizer.py --quick-test

# Full production run (~90 minutes)
python run_optimizer.py --candidates 1000 --sims 10000

# Custom fitness function
python run_optimizer.py --fitness tournament

# Resume interrupted run
python run_optimizer.py --run-name 20241130_143022

# Monitor progress (separate terminal)
python view_progress.py
```

---

## Complete Pipeline Execution

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add FanDuel CSV to _V4 directory
# File format: "FanDuel-NFL-*.csv"
```

### Weekly Workflow

```bash
# STEP 1: Fetch fresh data (Monday/Tuesday when salaries drop)
python fetch_data.py --all

# STEP 2: Analyze game scripts
python game_script_continuous.py

# STEP 3: Integrate data
python data_integration.py

# STEP 4: Run optimizer (let it run overnight)
python run_optimizer.py --candidates 1000 --sims 10000

# STEP 5: Monitor progress (separate terminal)
python view_progress.py

# STEP 6: View results
head -20 outputs/run_YYYYMMDD_HHMMSS/BEST_LINEUPS.csv
```

### Quick Updates

```bash
# If only odds changed (no new salaries)
python fetch_data.py --lines
python game_script_continuous.py
python data_integration.py

# Re-run optimizer with new data
python run_optimizer.py --force-restart
```

---

## Data Flow Summary

| Step | Input | Process | Output | Runtime |
|------|-------|---------|--------|---------|
| 1. Fetch | Web sources + FD CSV | Scrape & merge | `players_raw.csv`, `game_lines.csv` | 2-5 min |
| 2A. Scripts | `game_lines.csv` | Continuous probabilities | `game_script_continuous.csv` | 1 sec |
| 2B. Integration | `players_raw.csv`, `game_lines.csv` | Floor/ceiling × 5 scripts | `players_integrated.csv` (38 cols) | 1 sec |
| 3.1. Candidates | `players_integrated.csv` | MILP + tiered sampling | `candidates.csv` (1000 lineups) | 3 min |
| 3.2. Evaluation | `candidates.csv` | Monte Carlo (10k sims) | `evaluations.csv` (w/ metrics) | 25-30 min |
| 3.3. Refinement | `evaluations.csv` | Genetic algorithm | `BEST_LINEUPS.csv` | 40-60 min/iter |

**Total: ~70-90 minutes** for complete pipeline (1-2 iterations)

---

## Key Design Decisions

### 1. Why Continuous Game Scripts?

**Instead of discrete buckets:**
- ❌ Classify as "Shootout" OR "Competitive" (loses information)

**We use continuous probabilities:**
- ✅ 60% Shootout, 30% Competitive, 10% Defensive (captures mixed signals)
- ✅ Allows weighted adjustments in simulation
- ✅ More robust to edge cases

### 2. Why Pre-calculate All Game Script Floors/Ceilings?

**Instead of calculating on-the-fly during simulation:**
- ❌ Slow (1M+ calculations per optimization)
- ❌ Harder to debug/validate

**We pre-calculate 5 scenarios:**
- ✅ Fast lookups during simulation
- ✅ Easy to validate adjustments
- ✅ Can inspect data between phases

### 3. Why Separate TD Odds Multipliers?

**Instead of baking into floor/ceiling:**
- ❌ Can't distinguish game script vs TD odds effects
- ❌ Hard to tune independently

**We store multipliers separately:**
- ✅ Apply during simulation: `floor * td_odds_floor_mult`
- ✅ Easy to tune TD impact (currently 5% floor, 15% ceiling)
- ✅ Can disable TD adjustments without regenerating data

### 4. Why Three-Phase Optimization?

**Instead of single MILP or pure Monte Carlo:**
- ❌ MILP alone: fast but deterministic, no variance modeling
- ❌ Pure MC: accurate but too slow for 1000+ lineups

**We use hybrid approach:**
- ✅ Phase 1 (MILP): Fast candidate generation (1000 in 3 min)
- ✅ Phase 2 (MC): Accurate evaluation with distributions (10k sims)
- ✅ Phase 3 (Genetic): Iterative refinement to local optimum

### 5. Why Parallelization?

**Sequential processing:**
- ❌ 1000 lineups × 10k sims = ~6-8 hours

**Multiprocessing:**
- ✅ 12x speedup on 8-core machine
- ✅ 1000 lineups × 10k sims = ~25-30 minutes
- ✅ Essential for practical use

### 6. Why Checkpointing?

**Without state persistence:**
- ❌ Can't resume interrupted runs
- ❌ Can't see progress during execution
- ❌ Hard to debug failures

**With state files:**
- ✅ Resume from any iteration
- ✅ View progress in real-time (`view_progress.py`)
- ✅ Track fitness history across iterations
- ✅ Compare multiple runs

---

## Performance Metrics

### Expected Runtimes (8-core machine)

| Component | Operation | Runtime |
|-----------|-----------|---------|
| **Data Fetching** | | |
| FanDuel load | Local CSV | <1 sec |
| FantasyPros scrape | Web scrape | 30-60 sec |
| DK game lines | Web scrape | 15-30 sec |
| DK TD odds | Web scrape | 30-60 sec |
| ESPN players + projections | API calls | 30-60 sec |
| **Total Fetch** | | **2-5 min** |
| | | |
| **Processing** | | |
| Game script analysis | Calculations | 1 sec |
| Data integration | Join + calculate | 1 sec |
| **Total Processing** | | **2 sec** |
| | | |
| **Optimization** | | |
| Phase 1 (1000 lineups) | MILP | 3 min |
| Phase 2 (1000 × 10k sims) | Parallel MC | 25-30 min |
| Phase 3 (per iteration) | Genetic + MC | 40-60 min |
| **Total Optimization** | 1-2 iterations | **70-90 min** |

### Parallelization Speedup

| Metric | Sequential | Parallel (8 cores) | Speedup |
|--------|------------|-------------------|---------|
| Time per lineup (10k sims) | ~2.0 sec | ~0.16 sec | 12.5x |
| 1000 lineups | ~33 min | ~2.7 min | 12.2x |
| 100 offspring | ~3.3 min | ~0.27 min | 12.2x |

---

## File Reference

### Core Pipeline Scripts

| File | Purpose | Runtime | Inputs | Outputs |
|------|---------|---------|--------|---------|
| `fetch_data.py` | Data collection | 2-5 min | Web + CSV | `players_raw.csv`, `game_lines.csv` |
| `game_script_continuous.py` | Game analysis | 1 sec | `game_lines.csv` | `game_script_continuous.csv` |
| `data_integration.py` | Floor/ceiling calc | 1 sec | Raw + scripts | `players_integrated.csv` |
| `run_optimizer.py` | Orchestration | 70-90 min | Integrated | `BEST_LINEUPS.csv` |
| `view_progress.py` | Monitoring | - | State files | Console output |

### Optimizer Modules

| File | Purpose |
|------|---------|
| `optimizer/generate_candidates.py` | Phase 1: MILP candidate generation |
| `optimizer/evaluate_lineups.py` | Phase 2: Parallel Monte Carlo |
| `optimizer/optimize_genetic.py` | Phase 3: Genetic refinement |
| `optimizer/utils/milp_solver.py` | MILP formulation with PuLP |
| `optimizer/utils/monte_carlo.py` | Parallel simulation engine |
| `optimizer/utils/distribution_fit.py` | Shifted log-normal fitting |
| `optimizer/utils/genetic_operators.py` | Crossover, mutation, selection |

### Data Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `players_raw.csv` | ~378 | 19 | Merged data from all sources |
| `game_lines.csv` | ~28 | 13 | Betting lines (2 rows per game) |
| `game_script_continuous.csv` | ~14 | 11 | Game script probabilities |
| `players_integrated.csv` | ~378 | 38 | Final integrated data |
| `candidates.csv` | 1000 | 6 | Generated lineup candidates |
| `evaluations.csv` | 1000 | 14 | Monte Carlo results |
| `BEST_LINEUPS.csv` | 10 | 11 | Top 10 optimal lineups |

### Documentation

| File | Purpose |
|------|---------|
| `PIPELINE_REVIEW.md` | This document - complete end-to-end overview |
| `OPTIMIZER_DESIGN.md` | Technical design and architecture |
| `OPTIMIZER_USAGE.md` | User guide with examples |
| `QUICK_START.md` | Get started in 5 minutes |
| `WORKFLOW.md` | Data pipeline sequence |
| `README.md` | Project overview |

---

## Troubleshooting

### Data Fetching Issues

**Problem:** "No games found"
- **Cause:** Week cutoff date incorrect or no games this week
- **Fix:** Check `game_lines_raw.txt` for scraped dates

**Problem:** "Player not found in merge"
- **Cause:** Name normalization mismatch
- **Fix:** Add name mapping to `utils.py`

**Problem:** "ESPN API timeout"
- **Cause:** Rate limiting or API downtime
- **Fix:** Retry or use cached data

### Data Integration Issues

**Problem:** "No game found for player X"
- **Cause:** Team abbreviation mismatch (JAC vs JAX)
- **Fix:** Add to `TEAM_ABBR_MAP` in `data_integration.py`

**Problem:** "Floor > Ceiling"
- **Cause:** Bad ESPN data or extreme game script multipliers
- **Fix:** Check constraints in `calculate_floor_ceiling_for_all_game_scripts()`

### Optimizer Issues

**Problem:** "No DEFs available"
- **Cause:** Missing floor/ceiling data for defenses
- **Fix:** Already auto-filled in `generate_candidates.py` (201 players)

**Problem:** "KeyError: 'id'"
- **Cause:** Column name mismatch in multiprocessing workers
- **Fix:** Already fixed with column normalization

**Problem:** "Out of memory"
- **Cause:** Too many parallel processes
- **Fix:** Reduce `--processes` or `--candidates`

**Problem:** "Optimizer converged too early"
- **Cause:** Low patience or high threshold
- **Fix:** Increase `--patience` or decrease `--threshold`

---

## Next Steps & Future Enhancements

### Potential Improvements

1. **Correlation Modeling**
   - Currently: Players evaluated independently
   - Enhancement: Model QB-WR stacks, game stacks
   - Impact: More realistic lineup variance

2. **Weather Integration**
   - Currently: No weather data
   - Enhancement: Scrape weather forecasts
   - Impact: Adjust totals, passing games

3. **Ownership Projections**
   - Currently: No ownership consideration
   - Enhancement: Add ownership data, optimize for GPP uniqueness
   - Impact: Better tournament lineups

4. **Player Props**
   - Currently: Only anytime TD odds
   - Enhancement: Passing yards, rushing yards, receptions
   - Impact: More granular projection adjustments

5. **Advanced Distributions**
   - Currently: Shifted log-normal (3 parameters)
   - Enhancement: More flexible distributions (GMM, kernel density)
   - Impact: Better tail modeling

6. **Live Optimizer**
   - Currently: Run once per week
   - Enhancement: Re-run as odds change throughout week
   - Impact: React to line movements, injuries

7. **Backtesting Framework**
   - Currently: No historical validation
   - Enhancement: Historical data, backtest results
   - Impact: Validate adjustments, tune parameters

8. **Multi-Contest Optimization**
   - Currently: Single lineup focus
   - Enhancement: Optimize lineup pool for multiple contests
   - Impact: Better diversification strategy

---

## Appendix: Column Reference

### players_integrated.csv Full Schema

| # | Column | Type | Description | Example |
|---|--------|------|-------------|---------|
| 1 | `name` | str | Player name | "Josh Allen" |
| 2 | `position` | str | Position | "QB" |
| 3 | `team` | str | Team abbr | "BUF" |
| 4 | `game` | str | Game matchup | "BUF@PIT" |
| 5 | `opponent` | str | Opponent abbr | "PIT" |
| 6 | `salary` | int | FanDuel salary | 9400 |
| 7 | `fppg` | float | Avg FD pts/game | 24.86 |
| 8 | `injury_status` | str | Injury designation | "Q" or "" |
| 9 | `injury_detail` | str | Injury description | "Ankle" or "" |
| 10 | `fpProjPts` | float | FP consensus projection | 21.9 |
| 11 | `tdOdds` | float | DK anytime TD odds | 750.0 |
| 12 | `tdProbability` | float | TD probability (%) | 11.76 |
| 13 | `espnId` | float | ESPN player ID | 3918298.0 |
| 14 | `proj_team_pts` | float | Team projected points | 23.75 |
| 15 | `proj_opp_pts` | float | Opponent projected pts | 20.75 |
| 16 | `espnScoreProjection` | float | ESPN main projection | 20.98 |
| 17 | `espnLowScore` | float | ESPN floor | 14.25 |
| 18 | `espnHighScore` | float | ESPN ceiling | 27.27 |
| 19 | `espnOutsideProjection` | float | ESPN alt model 1 | 20.79 |
| 20 | `espnSimulationProjection` | float | ESPN alt model 2 | 20.96 |
| 21 | `floor_shootout` | float | Floor in shootout | 13.54 |
| 22 | `ceiling_shootout` | float | Ceiling in shootout | 31.36 |
| 23 | `floor_defensive` | float | Floor in defensive | 12.11 |
| 24 | `ceiling_defensive` | float | Ceiling in defensive | 22.99 |
| 25 | `floor_blowout_favorite` | float | Floor as favorite blowout | 12.83 |
| 26 | `ceiling_blowout_favorite` | float | Ceiling as favorite blowout | 23.17 |
| 27 | `floor_blowout_underdog` | float | Floor as underdog blowout | 13.54 |
| 28 | `ceiling_blowout_underdog` | float | Ceiling as underdog blowout | 25.91 |
| 29 | `floor_competitive` | float | Floor in competitive game | 14.25 |
| 30 | `ceiling_competitive` | float | Ceiling in competitive game | 27.27 |
| 31 | `td_odds_floor_mult` | float | TD odds floor multiplier | 1.006 |
| 32 | `td_odds_ceiling_mult` | float | TD odds ceiling multiplier | 1.018 |
| 33 | `game_id` | str | Game identifier | "BUF@PIT" |
| 34 | `shootout_prob` | float | Shootout probability | 0.31 |
| 35 | `defensive_prob` | float | Defensive probability | 0.23 |
| 36 | `blowout_prob` | float | Blowout probability | 0.28 |
| 37 | `competitive_prob` | float | Competitive probability | 0.18 |
| 38 | `is_favorite` | bool | Is player on favorite? | False |

---

**End of Pipeline Review**
