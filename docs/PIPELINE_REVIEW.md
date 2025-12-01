# DFS Lineup Optimizer - Complete Pipeline Review (V2)

**Date:** November 30, 2024
**Version:** V2 (Major Update)

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Phase 1: Data Fetching](#phase-1-data-fetching)
3. [Phase 2: Data Integration & Game Scripts](#phase-2-data-integration--game-scripts)
4. [Phase 3: Optimization](#phase-3-optimization)
5. [Recent Major Improvements](#recent-major-improvements)
6. [Output Files](#output-files)
7. [Performance Metrics](#performance-metrics)

---

## Pipeline Overview

The optimizer uses a 3-phase pipeline to generate optimal DFS lineups:

```
1_fetch_data.py → 2_data_integration.py → 3_run_optimizer.py
    (2-5 min)            (1-2 sec)              (70-90 min)
```

### Data Flow

```
FanDuel CSV
FantasyPros API    →  players_raw.csv  →  players_integrated.csv  →  BEST_LINEUPS.csv
DraftKings Lines                ↓                    ↓                       ↑
ESPN Projections          game_lines.csv    game_scripts.csv         Optimization
TD Odds                                                               (3 sub-phases)
```

---

## Phase 1: Data Fetching

**Script:** `1_fetch_data.py`
**Runtime:** 2-5 minutes
**Output:** `data/intermediate/players_raw.csv` (378 players × 19 columns)

### Data Sources

1. **FanDuel Salaries** (local CSV)
   - `data/input/FanDuel-NFL-*.csv`
   - Columns: name, position, salary, fppg, game, team, opponent

2. **FantasyPros Projections** (API)
   - Consensus fantasy projections
   - URL: `https://api.fantasypros.com/...`

3. **DraftKings Game Lines** (web scrape)
   - Spread, total, moneyline odds
   - Determines favorite/underdog

4. **DraftKings TD Odds** (web scrape)
   - Anytime TD scorer probabilities
   - Used for floor/ceiling adjustments

5. **ESPN Projections** (web scrape)
   - Low/high score ranges
   - Alternative projections for validation

### Merging Strategy

- Primary key: player `name` (lowercase, normalized)
- Fuzzy matching for name variations
- Handles missing data gracefully (defaults filled)

### Key Columns Generated

| Column | Description | Source |
|--------|-------------|--------|
| `name` | Player name (lowercase) | FanDuel |
| `position` | QB/RB/WR/TE/DEF | FanDuel |
| `team` | Player team | FanDuel |
| `opponent` | Opponent team | FanDuel |
| `salary` | FanDuel salary | FanDuel |
| `fppg` | Fantasy points per game | FanDuel |
| `fpProjPts` | Consensus projection | FantasyPros |
| `spread` | Point spread | DraftKings |
| `total` | Over/under total | DraftKings |
| `tdProbability` | Anytime TD probability | DraftKings |
| `espnScoreProjection` | ESPN projection | ESPN |
| `espnLowScore` | ESPN floor | ESPN |
| `espnHighScore` | ESPN ceiling | ESPN |

---

## Phase 2: Data Integration & Game Scripts

**Scripts:**
- `game_script_continuous.py` (standalone, optional)
- `2_data_integration.py` (main integration)

**Runtime:** 1-2 seconds
**Outputs:**
- `data/intermediate/game_scripts_enhanced.csv` (12 games × 11 columns)
- `data/intermediate/players_integrated.csv` (378 players × 88 columns!)

### Step 2A: Game Script Analysis

Calculates **continuous probabilities** for each game scenario using sigmoid functions:

```python
shootout_prob = sigmoid(total - 48)      # High total → more passing
defensive_prob = sigmoid(43 - total)     # Low total → less scoring
blowout_prob = sigmoid(abs(spread) - 7)  # Large spread → blowout
competitive_prob = 1 - blowout_prob       # Close game
```

**Scenarios:**
- `shootout`: High-scoring, pass-heavy game
- `defensive`: Low-scoring, grind-it-out game
- `blowout_favorite`: Leading team runs clock
- `blowout_underdog`: Trailing team passes to catch up
- `competitive`: Close game, balanced script

**Output columns:**
```
team, opponent, spread, total, is_favorite,
shootout_prob, defensive_prob, blowout_prob, competitive_prob,
dominant_scenario, scenario_confidence
```

### Step 2B: Floor/Ceiling Calculation

For each player and each scenario, calculates:
1. **Base floor/ceiling** from ESPN projections
2. **Game script multipliers** (position-specific)
3. **TD odds adjustments**
4. **Distribution parameters** (mu, sigma, shift)

#### Game Script Multipliers (V2 - Rebalanced!)

**Key changes in V2:**
- ✅ ALL ceiling multipliers >= 1.0 (no more < baseline!)
- ✅ Wider floor ranges for more variance
- ✅ Mathematically consistent with consensus mean

**Example: QB Multipliers**

| Scenario | Floor | Ceiling | Effect |
|----------|-------|---------|--------|
| Shootout | 0.75x | 1.40x | Huge upside, wide range |
| Defensive | 0.75x | 1.15x | Limited upside |
| Blowout Fav | 0.80x | 1.15x | Limited (runs clock) |
| Blowout Dog | 0.80x | 1.35x | High upside (garbage time) |
| Competitive | 0.85x | 1.25x | Moderate range |

**Example: RB Multipliers**

| Scenario | Floor | Ceiling | Effect |
|----------|-------|---------|--------|
| Shootout | 0.90x | 1.05x | Limited (pass-heavy) |
| Defensive | 0.95x | 1.15x | Stable usage |
| Blowout Fav | 0.95x | 1.30x | **Clock management boost** |
| Blowout Dog | 0.75x | 1.05x | Abandoned run game |
| Competitive | 0.85x | 1.20x | Moderate range |

#### Distribution Fitting (V2 - Scenario-Specific Means!)

**Major change:** Distributions now fit to P10/P90 only, allowing scenario-specific means.

**Old approach (V1):**
```python
# Forced all scenarios to have mean = consensus
fit_shifted_lognormal(mean=consensus, p10=floor, p90=ceiling)
# Problem: Boom/bust scenarios had no effect on expected scores!
```

**New approach (V2):**
```python
# Let mean vary by scenario
fit_lognormal_to_percentiles(p10=floor, p90=ceiling)
# Result: Shootout scenarios actually score higher!
```

**Impact:**
- Josh Allen shootout: 23.07 pts (+5.3% vs consensus)
- Josh Allen defensive: 20.08 pts (-8.3% vs consensus)
- **BOOM-BUST GAP: 2.69 pts (12.3%)** ✅

### Output Columns (88 total!)

For each scenario (shootout, defensive, blowout_favorite, blowout_underdog, competitive):
- `floor_{scenario}`: P10 value
- `ceiling_{scenario}`: P90 value
- `mu_{scenario}`: Log-normal μ parameter
- `sigma_{scenario}`: Log-normal σ parameter
- `shift_{scenario}`: Shift parameter

Plus game context:
- `proj_team_pts`: Team's implied total
- `proj_opp_pts`: Opponent's implied total
- `is_favorite`: Boolean flag
- `td_floor_mult`, `td_ceiling_mult`: TD odds adjustments

---

## Phase 3: Optimization

**Script:** `3_run_optimizer.py`
**Runtime:** 70-90 minutes (quick-test: 2-3 minutes)
**Output:** `outputs/run_YYYYMMDD_HHMMSS/BEST_LINEUPS.csv`

### Three Sub-Phases

```
Phase 3.1: MILP Candidates (3 min)
    ↓
Phase 3.2: Monte Carlo Evaluation (25-30 min)
    ↓
Phase 3.3: Genetic Refinement (40-60 min/iteration)
```

---

### Phase 3.1: Candidate Generation (MILP)

**File:** `optimizer/generate_candidates.py`
**Runtime:** ~3 minutes for 1000 candidates (quick-test: 18 sec for 50)
**Algorithm:** Mixed Integer Linear Programming

#### Tiered Sampling Strategy

1. **Tier 1: Chalk lineups** (20% of candidates)
   - Pure deterministic MILP
   - Maximizes consensus projections
   - Temperature = 0 (no randomness)

2. **Tier 2: Temperature-based variation** (60% of candidates)
   - Add random noise: `projection + Gumbel(0, T)`
   - Temperature increases: 0.2 → 1.0
   - More variance as tier progresses

3. **Tier 3: Pure random** (20% of candidates)
   - Random salary-valid lineups
   - Maximum exploration
   - (Currently disabled in quick-test)

#### MILP Formulation

**Decision Variables:**
```python
x[i] ∈ {0, 1}  # Binary: player i selected or not
```

**Objective:**
```python
maximize: Σ(projection[i] × x[i])  # With temperature noise
```

**Constraints:**
```python
# Salary cap
Σ(salary[i] × x[i]) ≤ 60,000

# Position requirements
Σ(x[i] : position[i] == QB) == 1
Σ(x[i] : position[i] == RB) ∈ {2, 3}
Σ(x[i] : position[i] == WR) ∈ {3, 4}
Σ(x[i] : position[i] == TE) ∈ {1, 2}
Σ(x[i] : position[i] == DEF) == 1
Σ(x[i]) == 9  # Total roster size

# Stacking constraints (optional)
# - Force QB + at least 1 pass catcher from same team
# - Currently disabled for more diversity
```

**Output:** `candidates.csv` with position-based columns:
```
QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DEF,
lineup_id, player_ids, tier, temperature, total_projection, total_salary
```

---

### Phase 3.2: Monte Carlo Evaluation

**File:** `optimizer/evaluate_lineups.py`
**Runtime:** ~25-30 minutes for 1000 lineups × 10k sims (quick-test: 3 sec for 50 × 1k)
**Speedup:** 12x with parallel processing

#### Simulation Process

For each lineup:
1. **Sample game script** from probabilities
2. **For each player:**
   - Get scenario-specific distribution params (mu, sigma, shift)
   - Sample score: `X = exp(mu + sigma × Z) + shift` where Z ~ N(0,1)
3. **Sum scores** for lineup total
4. **Repeat 10,000 times**

#### Game Script Sampling

Probabilistic sampling based on continuous probabilities:

```python
# For each game, sample one scenario
scenario = random.choice(
    ['shootout', 'defensive', 'blowout_fav', 'blowout_dog', 'competitive'],
    p=[shootout_prob, defensive_prob, blowout_prob, ..., competitive_prob]
)

# Players in that game use the sampled scenario's distribution
```

**Key insight:** Different scenarios produce different means, creating realistic boom/bust variance.

#### Statistics Calculated

For each lineup (from 10,000 simulations):
- `mean`: Average score
- `median`: 50th percentile (fitness metric)
- `p10`: 10th percentile (floor)
- `p90`: 90th percentile (ceiling)
- `std`: Standard deviation (variance)
- `skewness`: (mean - median) / std (right-tail measure)

**Output:** `evaluations.csv` with position-based columns:
```
QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DEF,
lineup_id, player_ids, tier, temperature, milp_projection, total_salary,
mean, median, p10, p90, std, skewness
```

---

### Phase 3.3: Genetic Algorithm Refinement

**File:** `optimizer/optimize_genetic.py`
**Runtime:** ~40-60 minutes per iteration (quick-test: 45-60 sec)
**Typical:** 3-5 iterations until convergence

#### Algorithm Overview

```
1. Initialize population (from Phase 2 evaluations)
2. For each generation:
   a. Selection (tournament)
   b. Crossover (position-aware)
   c. Mutation (salary-preserving, adaptive)
   d. Evaluation (Monte Carlo, parallel)
   e. Replacement (elitism + diversity enforcement)
   f. Check convergence
3. Return best lineups
```

#### 1. Selection: Tournament Selection

```python
def tournament_select(population, tournament_size=5, n_parents=50):
    parents = []
    for _ in range(n_parents):
        # Random tournament of 5
        tournament = random.sample(population, 5)
        # Pick best by fitness
        winner = max(tournament, key=fitness_func)
        parents.append(winner)
    return parents
```

**Fitness functions:**
- `conservative`: median - 0.5 × std (high floor)
- `balanced`: median (default)
- `aggressive`: p90 - p10 (boom potential)
- `tournament`: p90 (pure upside)

#### 2. Crossover: Position-Aware

```python
def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Group players by position
    p1_positions = group_by_position(parent1)
    p2_positions = group_by_position(parent2)

    # For each position, randomly inherit from one parent
    child1, child2 = [], []
    for position in ['QB', 'RB', 'WR', 'TE', 'DEF']:
        if random.random() < 0.5:
            child1 += p1_positions[position]
            child2 += p2_positions[position]
        else:
            child1 += p2_positions[position]
            child2 += p1_positions[position]

    return child1, child2
```

**Key insight:** Preserves position-specific combos (e.g., QB-WR stacks)

#### 3. Mutation: Salary-Preserving with Adaptive Aggression (V2!)

```python
def mutate(lineup, mutation_rate=0.3, aggressive=False):
    if random.random() > mutation_rate:
        return lineup.copy()

    # Adaptive: more swaps when diversity is low
    if aggressive:
        n_swaps = random.randint(2, 4)  # More disruption
        salary_tolerance = 1000          # Wider search
    else:
        n_swaps = random.randint(1, 2)
        salary_tolerance = 500

    # For each swap
    for _ in range(n_swaps):
        # Pick random player to replace
        old_player = random.choice(lineup)

        # Find candidates: same position, similar salary, not in lineup
        candidates = players[
            (position == old_player.position) &
            (abs(salary - old_player.salary) <= salary_tolerance) &
            (not in lineup)
        ]

        # Replace with random candidate
        new_player = random.choice(candidates)
        lineup.replace(old_player, new_player)

    return lineup
```

**V2 enhancement:** Aggressive mode activates when population uniqueness < 50%!

#### 4. Diversity Enforcement (V2 - NEW!)

After creating offspring population:

```python
# Deduplicate: remove lineups within 2 players of each other
new_population = deduplicate_population(
    elite + offspring,
    min_distance=2  # Hamming distance threshold
)

# If too homogeneous, inject diversity
if len(new_population) < population_size * 0.7:
    n_to_inject = population_size - len(new_population)

    # Sample diverse lineups from Phase 2 pool
    injection = available_evals.sample(n_to_inject)
    new_population.extend(injection)
```

**Diversity metrics tracked:**
- **Uniqueness:** % of truly unique lineups (98-100% in V2!)
- **Avg Hamming distance:** Average # different players (6.6-7.5 in V2!)
- **Player diversity:** # unique players used

**Output each generation:**
```
Diversity: 99.0% unique, avg distance: 7.1 players
```

#### 5. Convergence Detection

```python
# Stop if no improvement for N generations
if len(best_fitness_history) > patience:
    recent = best_fitness_history[-patience:]
    improvements = [recent[i] > recent[i-1] for i in range(1, len(recent))]
    if not any(improvements):
        return  # Converged!
```

**Default:** patience = 5 generations

**Final output:** `BEST_LINEUPS.csv` with all lineups from final population (70-100 typically):
```
QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DEF,
lineup_id, player_ids, mean, median, p10, p90, std, skewness, fitness
```

---

## Recent Major Improvements

### 1. Game Script Multiplier Rebalancing (Nov 30, 2024)

**Problem:** Old multipliers had ceiling < 1.0, creating mathematically incompatible constraints.

**Solution:**
- Rebalanced all ceiling multipliers >= 1.0
- Widened floor ranges for more variance
- Example: QB shootout ceiling 1.15 → 1.40

**Files changed:**
- `2_data_integration.py`: Updated GAME_SCRIPT_FLOOR and GAME_SCRIPT_CEILING
- `optimizer/utils/distribution_fit.py`: Added `fit_lognormal_to_percentiles()`

**Impact:**
- Boom/bust gap increased from 0.08 pts (0.4%) to 2.69 pts (12.3%)
- Game scripts now meaningfully affect projections

### 2. Scenario-Specific Means (Nov 30, 2024)

**Problem:** Old approach forced all scenarios to have mean = consensus, eliminating boom/bust effect.

**Solution:**
- Fit distributions to P10/P90 only
- Let mean vary by scenario
- Preserves floor/ceiling constraints

**Files changed:**
- `optimizer/utils/distribution_fit.py`: New `fit_lognormal_to_percentiles()`
- `2_data_integration.py`: Use new fitting function

**Impact:**
- Shootout scenarios: +5% to +17% vs consensus
- Defensive scenarios: -8% to -14% vs consensus
- MILP vs MC alignment still good: +0.6%

### 3. Diversity Enforcement in GA (Nov 30, 2024)

**Problem:** Final populations had 10-30% unique lineups, mostly identical cores.

**Solution:**
- Added diversity metrics (uniqueness, Hamming distance)
- Deduplication after replacement (min 2 player difference)
- Diversity injection when population < 70%
- Adaptive aggressive mutation when diversity < 50%

**Files added:**
- `optimizer/utils/diversity_tools.py`: Diversity metrics and deduplication

**Files changed:**
- `optimizer/optimize_genetic.py`: Diversity enforcement loop
- `optimizer/utils/genetic_operators.py`: Aggressive mutation parameter

**Impact:**
- Uniqueness increased from 10-30% to 98-100%
- Average Hamming distance: 6.6-7.5 players (vs 1-2 before)
- 100+ unique players used (vs 30-50 before)

### 4. Position-Based CSV Formatting (Nov 30, 2024)

**Problem:** Lineup CSVs had single `player_ids` column, hard to read.

**Solution:**
- Split into position-based columns: QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DEF
- Applied to all output CSVs (candidates, evaluations, BEST_LINEUPS)
- Preserve `player_ids` for backwards compatibility

**Files changed:**
- `3_run_optimizer.py`: Added `format_lineups_by_position()`

**Impact:**
- Much easier to read and analyze lineups
- Can quickly scan for specific players or stacks

---

## Output Files

### Run Directory Structure

```
outputs/run_YYYYMMDD_HHMMSS/
├── candidates.csv              # Phase 1 output (1000 lineups)
├── evaluations.csv             # Phase 2 output (1000 lineups + stats)
├── lineups_after_phase2.csv    # Same as evaluations (formatted)
├── iteration_N/
│   ├── lineups_iteration_N.csv # GA population each iteration
│   └── optimal_lineups.csv     # Best lineups that iteration
├── BEST_LINEUPS.csv            # Final best lineups (70-100)
├── final_summary.json          # Run statistics
├── optimizer_state.json        # State for resumption
└── checkpoints/
    └── generation_N.json       # GA checkpoints
```

### CSV Column Order

All lineup CSVs now use position-based format:
```
QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DEF,  # Player names
lineup_id, player_ids,                        # Identifiers
tier, temperature, total_projection,          # Phase 1 info
milp_projection, total_salary,                # MILP details
mean, median, p10, p90, std, skewness,        # MC stats
fitness                                        # GA fitness
```

Example row:
```csv
drake maye,devon achane,christian mccaffrey,tetairoa mcmillan,wandale robinson,troy franklin,hunter henry,treveyon henderson,buccaneers,child_684308,"drake maye,buccaneers,...",138.44,136.96,112.99,165.20,20.98,0.071,136.96
```

---

## Performance Metrics

### Runtime Breakdown (8-core M1 Mac)

| Operation | Count | Time | Notes |
|-----------|-------|------|-------|
| **Data Fetching** | | | |
| FanDuel CSV read | 378 rows | <1 sec | Local file |
| FantasyPros API | 1 request | 2-3 sec | Rate limited |
| DK game lines scrape | 12 games | 1-2 sec | Cached |
| DK TD odds scrape | 378 players | 5-10 sec | Cached |
| ESPN projections scrape | 378 players | 30-60 sec | Rate limited |
| **Total Phase 1** | | **2-5 min** | |
| | | | |
| **Data Integration** | | | |
| Game script analysis | 12 games | <1 sec | |
| Floor/ceiling calc | 378 × 5 scenarios | 1 sec | |
| Distribution fitting | 378 × 5 scenarios | 1 sec | Vectorized |
| **Total Phase 2** | | **1-2 sec** | |
| | | | |
| **Optimization** | | | |
| MILP candidates | 1000 lineups | 3 min | Parallel |
| Monte Carlo eval | 1000 × 10k sims | 25-30 min | 12 cores |
| GA refinement | 16 generations | 40-60 min | MC bottleneck |
| **Total Phase 3** | | **70-90 min** | |
| | | | |
| **GRAND TOTAL** | | **75-95 min** | Full pipeline |
| **Quick-test** | 50 cands × 1k sims | **2-3 min** | For testing |

### Scalability

| Parameter | Value | Impact |
|-----------|-------|--------|
| Candidates | 1000 → 2000 | +3 min (Phase 1) + 2x MC time |
| MC sims | 10k → 20k | 2x MC time |
| Processes | 12 → 24 | 1.8x speedup (diminishing returns) |
| Population size | 100 → 200 | +40% GA time |
| Max generations | 30 → 50 | Linear in GA time |

### Memory Usage

| Phase | Peak RAM | Notes |
|-------|----------|-------|
| Data fetching | 100 MB | Small datasets |
| Integration | 200 MB | Distribution params |
| MILP | 300 MB | PuLP solver |
| Monte Carlo | 2-3 GB | 12 processes × 250 MB |
| Genetic | 500 MB | Population storage |

**Recommendation:** 4 GB RAM minimum, 8 GB recommended

---

## Troubleshooting

### "Game script multipliers incompatible"
**Status:** ✅ FIXED in V2
**Solution:** Rebalanced multipliers, all ceiling >= 1.0

### "Boom/bust scenarios have no effect"
**Status:** ✅ FIXED in V2
**Solution:** Allow scenario-specific means

### "Final lineups all look the same"
**Status:** ✅ FIXED in V2
**Solution:** Diversity enforcement in GA

### "MILP and MC projections don't match"
**Status:** ✅ VERIFIED
**Current:** Within 0.6% (excellent alignment)

### "Out of memory"
**Solution:** Reduce `--processes` or `--candidates`

### "KeyError: 'fdSalary'"
**Status:** ✅ HANDLED
**Solution:** Column normalization in data fetching

---

## Summary of Changes (V1 → V2)

| Component | V1 | V2 | Impact |
|-----------|----|----|--------|
| **Game script multipliers** | Some ceiling < 1.0 | All ceiling >= 1.0 | Mathematically consistent |
| **Distribution fitting** | Force mean = consensus | Fit P10/P90 only | Boom/bust gap 12.3% |
| **GA diversity** | 10-30% unique | 98-100% unique | Huge improvement |
| **CSV format** | Single player_ids column | Position-based columns | Much more readable |
| **Hamming distance** | 1-2 players | 6.6-7.5 players | Real diversity |
| **Player variety** | 30-50 unique players | 100+ unique players | Better exploration |

**Overall:** V2 is a major upgrade with meaningful boom/bust effects and real diversity!

---

## References

- Old pipeline review: `docs/PIPELINE_REVIEW_v1_DEPRECATED.md`
- Game script fix summary: `GAME_SCRIPT_FIX_SUMMARY.md`
- Diversity enhancements: `DIVERSITY_ENHANCEMENTS.md`
- Quick start guide: `docs/QUICK_START.md`
