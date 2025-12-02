# Optimizer Usage Guide

Advanced usage, parameter tuning, and optimization strategies.

## Table of Contents

1. [Command Reference](#command-reference)
2. [Parameter Tuning](#parameter-tuning)
3. [Fitness Functions](#fitness-functions)
4. [Contest Strategies](#contest-strategies)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Command Reference

### Basic Usage

```bash
# Full optimization with defaults
python code/3_run_optimizer.py --week-dir data/2025_12_08

# Quick test (fast, good for validation)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test
```

### All Parameters

```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \           # Week directory (required)
  --run-name run_20251208_143022 \       # Resume existing run
  --candidates 1000 \                     # Phase 1: lineups to generate
  --simulations 10000 \                   # Phase 2: MC sims per lineup
  --fitness balanced \                    # Fitness function
  --iterations 3 \                        # Optimization iterations
  --max-generations 30 \                  # GA max generations
  --convergence-patience 5 \              # Stop after N stale generations
  --convergence-threshold 1.0 \           # Min improvement % to continue
  --processes 12 \                        # Parallel processes (auto-detect)
  --force-restart                         # Ignore existing checkpoint
```

### Quick Test Mode

Optimized for fast validation (~2 minutes):

```bash
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test
```

**Quick Test Settings**:
- Candidates: 50 (vs 1000 default)
- Simulations: 1000 (vs 10000 default)
- Iterations: 1 (vs 1 default)
- Max generations: 30 (same)

Use this to:
- Verify data integrity
- Test parameter changes
- Debug lineup generation issues

## Parameter Tuning

### Phase 1: Candidate Generation

**`--candidates N`** (default: 1000)

Controls candidate pool size. More candidates = more diversity but slower.

**Recommendations**:
- **Quick test**: 50-100
- **Normal run**: 500-1000
- **Thorough search**: 1500-2000

**Trade-offs**:
- More candidates = better coverage of solution space
- Diminishing returns after ~1000 candidates
- Memory usage: ~1MB per 100 candidates

### Phase 2: Monte Carlo Simulations

**`--simulations N`** (default: 10000)

Number of simulations per lineup. More sims = better distribution estimates.

**Recommendations**:
- **Quick test**: 1000
- **Normal run**: 5000-10000
- **High precision**: 15000-20000

**Trade-offs**:
- More sims = more accurate percentile estimates
- Diminishing returns after ~10000 sims
- Time: ~0.5-1 sec per lineup per 10k sims (parallelized)

**Statistical Accuracy**:
```
1000 sims:  ±3% percentile accuracy
5000 sims:  ±1.5% percentile accuracy
10000 sims: ±1% percentile accuracy
20000 sims: ±0.7% percentile accuracy
```

### Phase 3: Genetic Algorithm

**`--max-generations N`** (default: 30)

Maximum GA generations before stopping.

**Recommendations**:
- **Quick convergence**: 20-30
- **Thorough search**: 40-50
- **Exhaustive**: 60-100

**`--convergence-patience N`** (default: 5)

Stop if no improvement for N generations.

**Recommendations**:
- **Fast**: 3-5 (stops quickly)
- **Thorough**: 7-10 (explores more)

**`--convergence-threshold PCT`** (default: 1.0)

Minimum improvement % to count as progress.

**Recommendations**:
- **Strict**: 0.5% (requires substantial improvement)
- **Normal**: 1.0% (balanced)
- **Lenient**: 2.0% (stops sooner)

### Parallel Processing

**`--processes N`** (default: auto)

Number of parallel processes for simulation.

**Auto-Detection**:
```python
# Uses all cores except 1
n_processes = max(1, cpu_count() - 1)
```

**Manual Override**:
```bash
# Use 8 cores explicitly
python code/3_run_optimizer.py --week-dir data/2025_12_08 --processes 8

# Use all cores (may slow down system)
python code/3_run_optimizer.py --week-dir data/2025_12_08 --processes -1
```

**Performance**:
- Linear speedup up to ~8-12 cores
- Diminishing returns beyond 16 cores (overhead)
- Memory: ~500MB per process

## Fitness Functions

The fitness function determines which lineups are "best". Choose based on contest type.

### Available Functions

#### `balanced` (default)
```python
fitness = median + 0.3×P90 - 0.1×P10
```

**Best for**: Mixed contests, general purpose

**Philosophy**: Maximize median (consistency) while valuing upside and penalizing downside.

#### `median`
```python
fitness = median
```

**Best for**: Cash games, 50/50s, head-to-head

**Philosophy**: Pure consistency, ignore variance.

#### `upside`
```python
fitness = median + P90
```

**Best for**: Large field GPPs, Milly Maker

**Philosophy**: Need huge upside to win big tournaments.

#### `safe`
```python
fitness = median - 0.5×std
```

**Best for**: Risk-averse, guaranteed cash

**Philosophy**: Minimize variance, avoid blowups.

### Custom Fitness Functions

Edit `optimizer/fitness.py` to create custom functions:

```python
def custom_fitness(row: pd.Series) -> float:
    """
    Your custom fitness calculation.

    Available metrics:
    - row['projected_median']
    - row['projected_mean']
    - row['p10'], row['p90']
    - row['std'], row['skewness']
    """
    # Example: Maximize P75 (75th percentile)
    return row['projected_median'] + 0.5 * (row['p90'] - row['projected_median'])
```

Then use:
```bash
python code/3_run_optimizer.py --week-dir data/2025_12_08 --fitness custom
```

## Contest Strategies

### Cash Games (50/50, Double-Up)

**Goal**: Finish in top 50%

**Strategy**:
- Use `--fitness median` or `--fitness safe`
- Prefer high-floor players (RBs with volume)
- Avoid extreme contrarian plays
- Target correlation (QB+pass catchers)

**Lineup Construction**:
```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness median \
  --candidates 500 \
  --simulations 10000
```

**What to look for in results**:
- High median (125+)
- Low std (<15)
- P10 > 105 (limited downside)

### Small GPPs (100-1000 entries)

**Goal**: Top 10-20%

**Strategy**:
- Use `--fitness balanced`
- Mix chalk with 1-2 contrarian plays
- Target game stacks (QB+WRs+Opp WR)
- Moderate variance acceptable

**Lineup Construction**:
```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness balanced \
  --candidates 1000 \
  --simulations 10000
```

**What to look for**:
- Median 120-125
- P90 > 160 (decent upside)
- 2-3 players <20% owned (contrarian edge)

### Large GPPs (10k+ entries)

**Goal**: Top 0.1% (need huge score)

**Strategy**:
- Use `--fitness upside`
- Heavy contrarian emphasis
- Stack game scripts (shootout games)
- Accept high variance

**Lineup Construction**:
```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness upside \
  --candidates 1500 \
  --simulations 15000 \
  --iterations 3
```

**What to look for**:
- P90 > 170 (ceiling matters more than floor)
- Multiple low-owned players (<10%)
- Game stacks from projected shootouts

### Multi-Entry Strategy

Generate diverse lineups across multiple runs:

```bash
# Run 1: Balanced core
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness balanced \
  --candidates 1000

# Run 2: Contrarian upside
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness upside \
  --candidates 1000

# Run 3: Safe cash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --fitness safe \
  --candidates 1000
```

Then pick best lineups from each run's `3_lineups.csv`.

## Performance Optimization

### Time Estimates

**Default Settings** (1000 candidates, 10k sims, 1 iteration):
- Phase 1: 1-2 minutes
- Phase 2: 20-40 minutes (parallelized)
- Phase 3: 30-60 minutes
- **Total**: ~1-2 hours

**Quick Test** (50 candidates, 1k sims):
- Phase 1: 10 seconds
- Phase 2: 1 minute
- Phase 3: 30 seconds
- **Total**: ~2 minutes

### Speedup Strategies

#### 1. Reduce Candidates

Most impact with least quality loss:

```bash
# 50% time reduction, minimal quality loss
python code/3_run_optimizer.py --week-dir data/2025_12_08 --candidates 500
```

#### 2. Reduce Simulations

Moderate impact on quality:

```bash
# 50% time reduction, acceptable quality loss
python code/3_run_optimizer.py --week-dir data/2025_12_08 --simulations 5000
```

#### 3. Use More Cores

Linear speedup up to ~12 cores:

```bash
# Explicit core count
python code/3_run_optimizer.py --week-dir data/2025_12_08 --processes 16
```

#### 4. Aggressive Convergence

Stop earlier if good enough:

```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --convergence-patience 3 \
  --convergence-threshold 0.5
```

### Memory Usage

**Typical Usage**:
- Phase 1: ~100MB
- Phase 2: ~500MB per process
- Phase 3: ~200MB

**High Memory Settings** (16GB+ recommended):
```bash
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --candidates 2000 \
  --simulations 20000 \
  --processes 16
```

## Resuming Runs

### Automatic Checkpointing

The optimizer automatically saves state after each generation:
- `optimizer_state.json` - Current GA state
- `generation_N.json` - Per-generation history

### Resume Interrupted Run

```bash
# Resume by run name
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --run-name run_20251208_143022

# Force restart (ignore checkpoint)
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --run-name run_20251208_143022 \
  --force-restart
```

### Resume Behavior

**What's Preserved**:
- All evaluated lineups
- Generation history
- Best fitness achieved
- Diversity pool

**What's Re-run**:
- Remaining generations
- Convergence check
- Final lineup selection

## Troubleshooting

### No Lineups Generated

**Symptom**:
```
Generated 0 chalk lineups
```

**Causes & Fixes**:

1. **Missing DEF players**
   ```bash
   # Check player data
   python -c "import pandas as pd; print(pd.read_csv('data/WEEK/intermediate/1_players.csv')['position'].value_counts())"

   # Should show 'D' or 'DEF' position
   # If not, re-run data integration
   python code/2_data_integration.py --week-dir data/WEEK
   ```

2. **Salary cap too tight**
   - Check if expensive players dominate
   - Optimizer can't find valid combinations
   - Solution: Check `1_players.csv` for salary distribution

3. **Position constraints impossible**
   - Not enough players at some position
   - Check IR list (players marked Injured Reserve)

### Slow Performance

**Symptom**: Takes >4 hours for default settings

**Checks**:

1. **CPU throttling**
   ```bash
   # Check CPU usage during run
   top -pid $(pgrep -f 3_run_optimizer)

   # Should see ~800-1200% CPU (8-12 cores)
   ```

2. **Memory pressure**
   ```bash
   # Check memory usage
   ps aux | grep python

   # If using swap, reduce --processes
   ```

3. **Disk I/O bottleneck**
   - Move data directory to SSD
   - Reduce checkpoint frequency

**Solutions**:
```bash
# Reduce workload
python code/3_run_optimizer.py \
  --week-dir data/2025_12_08 \
  --candidates 500 \
  --simulations 5000

# Or use quick test
python code/3_run_optimizer.py --week-dir data/2025_12_08 --quick-test
```

### Poor Quality Results

**Symptom**: Low fitness scores, dominated by obvious lineups

**Checks**:

1. **Not enough diversity**
   ```bash
   # Increase candidates
   python code/3_run_optimizer.py --week-dir data/2025_12_08 --candidates 1500
   ```

2. **Premature convergence**
   ```bash
   # More GA exploration
   python code/3_run_optimizer.py \
     --week-dir data/2025_12_08 \
     --max-generations 50 \
     --convergence-patience 10
   ```

3. **Wrong fitness function**
   - Using `median` for GPPs → no upside
   - Using `upside` for cash → too risky
   - Switch to appropriate function

### Out of Memory

**Symptom**: Process killed, `Killed` message

**Solutions**:

1. **Reduce parallel processes**
   ```bash
   python code/3_run_optimizer.py --week-dir data/2025_12_08 --processes 4
   ```

2. **Reduce candidates**
   ```bash
   python code/3_run_optimizer.py --week-dir data/2025_12_08 --candidates 500
   ```

3. **Reduce simulations**
   ```bash
   python code/3_run_optimizer.py --week-dir data/2025_12_08 --simulations 5000
   ```

### Correlation Warnings

**Symptom**:
```
Warning: High correlation between players
```

**Explanation**: Some player pairs have strong correlation (e.g., QB+WR on same team)

**Action**: Usually OK, optimizer accounts for this. Only concern if:
- Many lineups have same correlation stack
- Reduces diversity

**Fix**:
```bash
# Generate more candidates for diversity
python code/3_run_optimizer.py --week-dir data/2025_12_08 --candidates 1500
```

## Advanced Usage

### Custom Player Projections

Override consensus with your own projections:

```bash
# Edit intermediate/1_players.csv
# Modify 'fpProjPts' column with custom values

# Then run optimizer
python code/3_run_optimizer.py --week-dir data/2025_12_08
```

### Lock/Exclude Players

Edit `optimizer/constraints.py` to add player locks:

```python
# Lock specific players into all lineups
LOCKED_PLAYERS = ['Patrick Mahomes', 'Travis Kelce']

# Exclude players from consideration
EXCLUDED_PLAYERS = ['Injured Player', 'Inactive Player']
```

### Game Script Adjustments

Override game script probabilities in `intermediate/2_game_scripts.csv`:

```csv
game_id,shootout,defensive,blowout,competitive
LAR@SF,0.8,0.1,0.05,0.05  # Force shootout scenario
```

This affects player distributions in Monte Carlo simulation.

---

**Next Steps**:
- See WORKFLOW.md for data pipeline details
- See QUICK_START.md for basic setup
- See docs/ARCHITECTURE_REFACTOR.md for technical architecture
