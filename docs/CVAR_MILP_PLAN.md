# CVaR-MILP Implementation Plan

## Overview

Replace the 3-phase pipeline (MILP → Monte Carlo → Genetic Algorithm) with a single-phase CVaR-based MILP that directly optimizes for p80 (lineup ceiling).

### Current Pipeline (to be replaced)
```
Phase 1: MILP (maximize consensus projection)
    ↓
Phase 2: Monte Carlo (evaluate distributions)
    ↓
Phase 3: Genetic Algorithm (evolve for fitness)
```

### New Pipeline
```
Step 1: Pre-generate scenario matrix (vectorized Monte Carlo)
    ↓
Step 2: CVaR-MILP (directly optimize for p80)
    ↓
Step 3: Repeat with diversity strategies for 100 lineups
```

---

## Mathematical Formulation

### CVaR for Maximizing Upside

Standard CVaR minimizes tail risk. We flip it to **maximize tail gain** (top 20% = p80).

**Inputs:**
- `points[i][s]` = simulated score for player `i` in scenario `s`
- `N` = number of players (~200-300 after filtering)
- `S` = number of scenarios (1000, configurable)
- `α = 0.20` (top 20% = p80)

**Variables:**
- `x[i]` ∈ {0, 1} = binary, 1 if player `i` selected
- `t` = continuous, threshold approximating p80
- `y[s]` ≥ 0 = continuous, excess above threshold in scenario `s`

**Objective:**
```
Maximize: t + (1 / (S × α)) × Σ y[s]
```

**CVaR Constraints:**
```
For each scenario s:
    y[s] ≥ Σ(x[i] × points[i][s]) - t    ∀s ∈ {1..S}
    y[s] ≥ 0
```

**Standard DFS Constraints:**
```
Σ(x[i] × salary[i]) ≤ 60000              # Salary cap
Σ x[i] = 9                                # Total players
Σ(x[i] : pos[i] = QB) = 1                 # Exactly 1 QB
Σ(x[i] : pos[i] = RB) ∈ {2, 3}            # 2-3 RBs
Σ(x[i] : pos[i] = WR) ∈ {3, 4}            # 3-4 WRs
Σ(x[i] : pos[i] = TE) ∈ {1, 2}            # 1-2 TEs
Σ(x[i] : pos[i] = DEF) = 1                # Exactly 1 DEF
```

**Force Include/Exclude:**
```
x[i] = 1  for i in FORCE_INCLUDE
x[i] = 0  for i in EXCLUDE_PLAYERS
```

---

## Diversity Strategies

We use two simple, effective strategies for generating 100 diverse lineups:

### Strategy 1: Exact Lineup Exclusion (Anti-Overlap)

After finding lineup L, add constraint to exclude that exact combination:
```
Σ(x[i] : i ∈ L) ≤ 8    # Can't have all 9 of the same players
```

This is linear and ensures we never get the exact same lineup twice.
Close lineups (8 overlapping players) are still allowed - only exact duplicates are blocked.

### Strategy 2: Anchored Lineups

Force specific players and re-optimize to ensure diversity across key positions:

**QB-Anchored** (~25 lineups):
- Force each top-25 QB by projection, optimize the rest
- Ensures we explore lineups built around different QBs

**DEF-Anchored** (~16 lineups):
- Force each DEF (one per team), optimize the rest
- Ensures we explore lineups with different game stacks

**RB-Anchored** (~20 lineups):
- Force each top-20 RB by projection, optimize the rest
- Ensures we explore different RB-heavy builds

**WR-Anchored** (~20 lineups):
- Force each top-20 WR by projection, optimize the rest
- Ensures we explore different WR-heavy builds

**General** (~19 lineups):
- No anchor, just exclude previous lineups
- Fills remaining slots with best available

### Why No Scenario Biasing (For Now)

Game script probabilities are already calculated in `2_data_integration.py` and baked into the
scenario matrix. The CVaR objective naturally finds lineups that boom in correlated scenarios.
We can add scenario biasing later if diversity is insufficient.

---

## Implementation Plan

### File Structure

```
code/
├── config.py                    # Simplified config (remove GA stuff)
├── 3_run_optimizer.py           # Complete rewrite - CVaR orchestrator
└── optimizer/
    ├── __init__.py
    ├── scenario_generator.py    # NEW: Generate scenario matrix
    ├── cvar_optimizer.py        # NEW: CVaR-MILP solver
    └── utils/
        ├── distribution_fit.py  # Keep (for sampling)
        ├── monte_carlo.py       # Simplify (just for scenario generation)
        └── milp_solver.py       # REPLACE with CVaR version
```

### Files to Remove (or Archive)
- `optimizer/generate_candidates.py` - Replaced by CVaR
- `optimizer/evaluate_lineups.py` - Absorbed into scenario generation
- `optimizer/optimize_genetic.py` - No longer needed
- `optimizer/utils/genetic_operators.py` - No longer needed
- `optimizer/utils/diversity_tools.py` - No longer needed

---

## Phase 1: Scenario Generator (`scenario_generator.py`)

**Purpose**: Pre-generate the `points[i][s]` matrix efficiently.

### Core Function
```python
def generate_scenario_matrix(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_scenarios: int = 1000,
    bias: str = 'baseline'  # or 'shootout', 'blowout', etc.
) -> np.ndarray:
    """
    Generate scenario matrix: points[player_idx][scenario_idx]

    Returns:
        np.ndarray of shape (n_players, n_scenarios)
    """
```

### Implementation Details

1. **Vectorized Game Script Sampling**
   - Sample all S game scripts at once per game
   - Result: `game_script_samples[game_id][s]` = script for game in scenario s

2. **Vectorized Player Sampling**
   - For each player, look up their mu/sigma/shift based on sampled script
   - Sample all S scores at once: `np.exp(mu + sigma * randn(S)) + shift`

3. **Bias Modes**
   - `baseline`: Use game script probabilities as-is
   - `shootout`: Multiply shootout_prob by 2, renormalize
   - `blowout`: Multiply blowout_prob by 2, renormalize
   - `defensive`: Multiply defensive_prob by 2, renormalize
   - `competitive`: Multiply competitive_prob by 2, renormalize

### Performance Target
- 300 players × 1000 scenarios = 300,000 samples
- Should complete in < 1 second (vectorized numpy)

---

## Phase 2: CVaR Optimizer (`cvar_optimizer.py`)

**Purpose**: Solve the CVaR-MILP formulation.

### Core Function
```python
def optimize_lineup_cvar(
    players_df: pd.DataFrame,
    scenario_matrix: np.ndarray,
    salary_cap: float = 60000,
    alpha: float = 0.10,
    force_include: List[str] = None,
    exclude_players: List[str] = None,
    exclude_lineups: List[List[str]] = None,
    time_limit: int = 60
) -> Optional[Dict]:
    """
    Solve CVaR-MILP for optimal ceiling lineup.

    Args:
        players_df: Player data with id, position, salary
        scenario_matrix: Pre-computed points[player_idx][scenario_idx]
        alpha: CVaR tail probability (0.10 = optimize for p90)
        exclude_lineups: Previous lineups to exclude (exact match)

    Returns:
        Dict with lineup info or None if infeasible
    """
```

### Key Implementation Notes

1. **Variable Count**
   - Binary: ~300 (one per player)
   - Continuous: 1001 (t + 1000 y[s] variables)
   - Total: ~1300 variables

2. **Constraint Count**
   - CVaR: 1000 (one per scenario)
   - Salary: 1
   - Positions: ~10
   - Diversity: N (one per excluded lineup)
   - Total: ~1000 + N

3. **Solver Settings**
   - Use CBC (default in PuLP)
   - Time limit: 60 seconds (should solve in < 30s)
   - Gap tolerance: 0.5% (good enough for DFS)

---

## Phase 3: Orchestrator (`3_run_optimizer.py`)

**Purpose**: Generate 100 diverse lineups using anchored strategies.

### Main Function
```python
def run_cvar_optimizer(
    week_dir: str,
    n_lineups: int = 100,
    n_scenarios: int = 1000,
    alpha: float = 0.20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate N diverse lineups using CVaR-MILP.

    Strategies (in order):
    1. QB-anchored: Top 25 QBs (~25 lineups)
    2. DEF-anchored: All DEFs (~16 lineups)
    3. RB-anchored: Top 20 RBs (~20 lineups)
    4. WR-anchored: Top 20 WRs (~20 lineups)
    5. General: No anchor, fill remaining (~19 lineups)
    """
```

### Lineup Generation Flow

```python
all_lineups = []
excluded_lineups = []

# Generate scenario matrix ONCE (reused for all lineups)
scenario_matrix = generate_scenario_matrix(players_df, game_scripts_df, n_scenarios)

# Strategy 1: QB-anchored (top 25 QBs by projection)
top_qbs = players_df[players_df['position'] == 'QB'].nlargest(25, 'fpProjPts')
for _, qb in top_qbs.iterrows():
    lineup = optimize_lineup_cvar(
        players_df, scenario_matrix,
        force_include=[qb['id']],
        exclude_lineups=excluded_lineups
    )
    if lineup:
        lineup['strategy'] = 'qb_anchor'
        lineup['anchor_player'] = qb['name']
        all_lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])

# Strategy 2: DEF-anchored (all defenses)
all_defs = players_df[players_df['position'] == 'D']
for _, defense in all_defs.iterrows():
    lineup = optimize_lineup_cvar(
        players_df, scenario_matrix,
        force_include=[defense['id']],
        exclude_lineups=excluded_lineups
    )
    if lineup:
        lineup['strategy'] = 'def_anchor'
        lineup['anchor_player'] = defense['name']
        all_lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])

# Strategy 3: RB-anchored (top 20 RBs)
top_rbs = players_df[players_df['position'] == 'RB'].nlargest(20, 'fpProjPts')
for _, rb in top_rbs.iterrows():
    lineup = optimize_lineup_cvar(
        players_df, scenario_matrix,
        force_include=[rb['id']],
        exclude_lineups=excluded_lineups
    )
    if lineup:
        lineup['strategy'] = 'rb_anchor'
        lineup['anchor_player'] = rb['name']
        all_lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])

# Strategy 4: WR-anchored (top 20 WRs)
top_wrs = players_df[players_df['position'] == 'WR'].nlargest(20, 'fpProjPts')
for _, wr in top_wrs.iterrows():
    lineup = optimize_lineup_cvar(
        players_df, scenario_matrix,
        force_include=[wr['id']],
        exclude_lineups=excluded_lineups
    )
    if lineup:
        lineup['strategy'] = 'wr_anchor'
        lineup['anchor_player'] = wr['name']
        all_lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])

# Strategy 5: General (fill remaining slots)
remaining = n_lineups - len(all_lineups)
for i in range(remaining):
    lineup = optimize_lineup_cvar(
        players_df, scenario_matrix,
        exclude_lineups=excluded_lineups
    )
    if lineup:
        lineup['strategy'] = 'general'
        lineup['anchor_player'] = None
        all_lineups.append(lineup)
        excluded_lineups.append(lineup['player_ids'])
    else:
        break  # Can't generate more unique lineups
```

---

## Config Simplification

### Remove from `config.py`:
```python
# REMOVE - Genetic Algorithm (no longer used)
DEFAULT_MAX_GENERATIONS = 50
DEFAULT_CONVERGENCE_PATIENCE = 50
DEFAULT_CONVERGENCE_THRESHOLD = 0.003
GENETIC_TOURNAMENT_SIZE = 2
GENETIC_MUTATION_RATE = 0.30
GENETIC_NUM_PARENTS = 50
ELITE_ARCHIVE_SIZE = 100
MUTATION_SALARY_TOLERANCE = 500

# REMOVE - Tiered temperature sampling (replaced by CVaR)
TEMP_DETERMINISTIC = 0.0
TEMP_MODERATE_MIN = 0.3
TEMP_MODERATE_MAX = 1.1
TEMP_CONTRARIAN_MIN = 1.5
TEMP_CONTRARIAN_MAX = 3.0
MAX_OVERLAP_CHALK = 7
MAX_OVERLAP_MODERATE = 6
MAX_OVERLAP_CONTRARIAN = 4

# REMOVE - Fitness functions (CVaR replaces this)
CUSTOM_FITNESS_WEIGHTS = {...}
DEFAULT_FITNESS = 'tournament'
```

### Keep in `config.py`:
```python
# DATA PATHS
DATA_INPUT_DIR = 'data/input'
DATA_INTERMEDIATE_DIR = 'data/intermediate'
OUTPUTS_DIR = 'outputs'
PLAYERS_INTEGRATED = 'data/intermediate/players_integrated.csv'
GAME_SCRIPTS = 'data/intermediate/game_script.csv'

# CVaR OPTIMIZER SETTINGS
DEFAULT_N_LINEUPS = 100          # Number of lineups to generate
DEFAULT_N_SCENARIOS = 1000       # Scenarios for CVaR (configurable)
DEFAULT_CVAR_ALPHA = 0.20        # Optimize for p80 (top 20%)
DEFAULT_SOLVER_TIME_LIMIT = 60   # Seconds per MILP solve

# PLAYER CONSTRAINTS
FORCE_INCLUDE = []
EXCLUDE_PLAYERS = []

# LINEUP CONSTRAINTS (FanDuel)
SALARY_CAP = 60000
LINEUP_SIZE = 9
MIN_QB = 1, MAX_QB = 1
MIN_RB = 2, MAX_RB = 3
MIN_WR = 3, MAX_WR = 4
MIN_TE = 1, MAX_TE = 2
MIN_DEF = 1, MAX_DEF = 1

# LOGGING
VERBOSE = True
```

**Note**: Game script floor/ceiling multipliers stay in `2_data_integration.py` where they're used.
No need to duplicate in config.

---

## Output Format

### `3_lineups.csv` (Final Output)
```csv
QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DEF,lineup_id,player_ids,total_salary,strategy,anchor_player,mean,median,p10,p80,cvar_score
```

### `results.json`
```json
{
  "run_name": "run_20251203_143022",
  "timestamp": "2025-12-03T14:30:22",
  "config": {
    "n_lineups": 100,
    "n_scenarios": 1000,
    "alpha": 0.20,
    "solver_time_limit": 60
  },
  "summary": {
    "lineups_generated": 100,
    "total_time_seconds": 180.5,
    "avg_time_per_lineup": 1.8,
    "best_cvar_score": 158.3,
    "strategy_breakdown": {
      "qb_anchor": 25,
      "def_anchor": 16,
      "rb_anchor": 20,
      "wr_anchor": 20,
      "general": 19
    }
  }
}
```

---

## Implementation Order

### Step 1: Core CVaR Solver (2-3 hours)
1. Create `optimizer/cvar_optimizer.py`
2. Implement basic CVaR-MILP formulation
3. Test with dummy scenario matrix
4. Verify lineup validity (positions, salary)

### Step 2: Scenario Generator (1-2 hours)
1. Create `optimizer/scenario_generator.py`
2. Implement vectorized sampling
3. Add bias modes (shootout, blowout, etc.)
4. Test scenario statistics match expected distributions

### Step 3: Orchestrator (2-3 hours)
1. Rewrite `3_run_optimizer.py`
2. Implement all diversity strategies
3. Add progress reporting
4. Save outputs in expected format

### Step 4: Config Cleanup (30 min)
1. Remove deprecated settings
2. Add new CVaR settings
3. Update validation

### Step 5: Testing & Validation (1-2 hours)
1. Run on real data
2. Compare lineup quality to old approach
3. Verify stacks emerge naturally
4. Check diversity across lineups

---

## Expected Benefits

1. **Simpler Pipeline**: 1 phase instead of 3
2. **Faster**: ~2-3 min total vs 1-2 hours
3. **Better Lineups**: Directly optimizes for ceiling
4. **Natural Stacks**: Correlated players boom together in scenarios
5. **More Principled**: Mathematical optimization vs evolutionary heuristics
6. **Easier to Debug**: Single optimization problem, clear objective

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MILP too slow (>1000 scenarios) | Reduce to 500-1000, use solver time limits |
| Not enough diversity | Anchored strategies ensure different builds |
| PuLP/CBC can't handle size | Consider HiGHS solver (faster), or reduce variables |
| Lineups look too similar | Exact-lineup exclusion prevents duplicates |

---

## Configuration Summary

Based on discussion:
- **Scenarios**: 1000 (configurable via `--scenarios`)
- **Alpha**: 0.20 (p80 optimization)
- **Lineups**: 100 (configurable via `--n-lineups`)
- **Diversity**: Anchored strategies + exact-lineup exclusion (no scenario biasing)

---

## Ready to Implement

The plan is complete. Implementation order:

1. **`optimizer/scenario_generator.py`** - Vectorized scenario matrix generation
2. **`optimizer/cvar_optimizer.py`** - CVaR-MILP solver with PuLP
3. **`3_run_optimizer.py`** - Orchestrator with anchored strategies
4. **`config.py`** - Cleanup and add new settings
5. **Testing** - Run on real data, verify lineup quality
