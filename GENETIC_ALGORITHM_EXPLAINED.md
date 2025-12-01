# Genetic Algorithm - How It Works

## Overview

The genetic algorithm in Phase 3 is **100% custom-built** - no external GA library. It uses evolutionary principles to iteratively improve lineups.

## The Three Core Operators

### 1. **Selection** (Tournament Selection)

**Purpose**: Choose which lineups get to "reproduce"

**How it works**:
```python
def tournament_select(population, fitness_func, tournament_size=5, n_parents=50):
    """
    For each parent slot:
    1. Randomly pick 5 lineups from population
    2. Calculate fitness for each
    3. Pick the best one as parent
    4. Repeat until we have 50 parents
    """
```

**Why tournament selection?**
- Simple and fast
- Maintains diversity (not just picking top N)
- Gives weaker lineups a chance (prevents premature convergence)

**Example**:
```
Population: 100 lineups with fitness scores
Want 50 parents:

Tournament 1: Pick random 5 → [lineup_3, lineup_47, lineup_12, lineup_89, lineup_23]
             → Best is lineup_47 (fitness: 112.3) → Parent 1

Tournament 2: Pick random 5 → [lineup_1, lineup_55, lineup_67, lineup_31, lineup_99]
             → Best is lineup_99 (fitness: 111.8) → Parent 2

... repeat 48 more times
```

### 2. **Crossover** (Position-Aware)

**Purpose**: Combine two parent lineups to create children

**Special Challenge**: Can't just randomly swap players - must maintain valid DFS lineups!
- Must have 1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DEF
- Must stay under $60K salary cap

**How it works**:
```python
def crossover(parent1, parent2, players_df):
    """
    1. Group each parent's players by position
    2. For each position, randomly choose parent to inherit from
    3. Combine to create valid child lineup
    """
```

**Example**:
```
Parent 1: [Mahomes(QB), CMC(RB), Kamara(RB), Jefferson(WR), Lamb(WR), Hill(WR), Kelce(TE), Bills(DEF)]
Parent 2: [Allen(QB), Henry(RB), Barkley(RB), Chase(WR), Adams(WR), Kupp(WR), Kittle(TE), 49ers(DEF)]

Step 1: Group by position
Parent 1 by position:
  QB: [Mahomes]
  RB: [CMC, Kamara]
  WR: [Jefferson, Lamb, Hill]
  TE: [Kelce]
  DEF: [Bills]

Parent 2 by position:
  QB: [Allen]
  RB: [Henry, Barkley]
  WR: [Chase, Adams, Kupp]
  TE: [Kittle]
  DEF: [49ers]

Step 2: Randomly pick which parent to inherit each position from
  QB: Pick Parent 2 → Allen
  RB: Pick Parent 1 → CMC, Kamara
  WR: Pick Parent 2 → Chase, Adams, Kupp
  TE: Pick Parent 1 → Kelce
  DEF: Pick Parent 2 → 49ers

Child 1: [Allen(QB), CMC(RB), Kamara(RB), Chase(WR), Adams(WR), Kupp(WR), Kelce(TE), 49ers(DEF)]
```

This ensures:
- Child has valid position distribution
- Child inherits good "building blocks" from parents
- Child is different from both parents (exploration)

### 3. **Mutation** (Salary-Preserving)

**Purpose**: Introduce random changes to prevent getting stuck

**How it works**:
```python
def mutate(lineup, players_df, mutation_rate=0.3, salary_tolerance=500):
    """
    30% chance of mutation:
    1. Pick 1-2 random players from lineup
    2. For each, find candidates with:
       - Same position
       - Similar salary (±$500)
       - Not already in lineup
    3. Replace with random candidate
    """
```

**Example**:
```
Original: [Mahomes($9000,QB), CMC($9500,RB), ...]

Mutation triggers (30% chance):
1. Pick random player to mutate: CMC
2. Find RBs with salary $9000-$10000 not in lineup:
   - Barkley ($9700)
   - Kamara ($9200)
   - Henry ($9400)
3. Randomly pick one: Kamara
4. Replace: [Mahomes($9000,QB), Kamara($9200,RB), ...]

New lineup stays under salary cap and maintains validity!
```

## The Evolution Loop

```python
for generation in range(max_generations):
    # 1. Selection
    parents = tournament_select(population, fitness_func, n_parents=50)

    # 2. Crossover
    offspring = []
    for i in range(0, len(parents), 2):
        child1, child2 = crossover(parents[i], parents[i+1], players_df)
        offspring.append(child1)
        offspring.append(child2)

    # 3. Mutation
    for child in offspring:
        mutated_child = mutate(child, players_df, mutation_rate=0.3)
        # Replace original with mutated version

    # 4. Evaluation
    # Run Monte Carlo simulation (10,000 sims) on all offspring
    evaluate_lineups_parallel(offspring, game_scripts_df, n_sims=10000)

    # 5. Survival of the fittest (Elitism)
    # Keep top 20% of old population
    elite = top_20_percent(population)

    # Combine elite + offspring, keep best 100
    population = select_best_100(elite + offspring)

    # 6. Check convergence
    if no_improvement_in_last_3_generations():
        break
```

## Example Evolution

```
Generation 0 (initial):
  Best: 107.26 (lineup from Phase 2)
  Population: 100 lineups

Generation 1:
  Selected 50 parents (tournament selection)
  Created 50 offspring (crossover)
  Mutated 14 offspring (30% mutation rate)
  Evaluated all 50 with Monte Carlo
  → Best: 108.32 ✨ (NEW BEST!)
  → Improvement: +1.06

Generation 2:
  Selected 50 parents
  Created 50 offspring
  Mutated 18 offspring
  Evaluated
  → Best: 108.98 ✨ (NEW BEST!)
  → Improvement: +0.66

Generation 3:
  → Best: 109.28 ✨
  → Improvement: +0.30

Generation 4:
  → Best: 109.82 ✨
  → Improvement: +0.54

Generation 5:
  → Best: 110.09 ✨
  → Improvement: +0.27

Generations 6-8:
  → Best: 110.09 (no improvement)

Generation 9:
  → Best: 110.09 (no improvement)
  → CONVERGED! No improvement in last 3 generations

Final best: 110.09 (improved 2.83 points from start!)
```

## Why This Works

1. **Tournament Selection** - Balances exploitation (picking good lineups) with exploration (giving weaker ones a chance)

2. **Position-Aware Crossover** - Preserves valid lineup structure while mixing good player combinations

3. **Salary-Preserving Mutation** - Introduces variety without breaking constraints

4. **Elitism** - Keeps best lineups from being lost (top 20% always survive)

5. **Parallel Evaluation** - Uses 12 CPU cores to run Monte Carlo fast

6. **Convergence Detection** - Stops when no improvement for N generations (saves time)

## Key Insights

### Why Not Just Run More Phase 1 Candidates?

Phase 1 (MILP) is **greedy** - it optimizes for median projection alone.

Phase 3 (GA) finds lineups that:
- Have better P90 upside
- Have better correlation structure
- Balance median + variance + upside better

**Example**:
```
Phase 1 best lineup:
  Median: 107.26
  P90: 120.15
  Variance: medium
  Fitness (balanced): 107.26

Phase 3 best lineup:
  Median: 106.89 (slightly lower)
  P90: 124.32 (much higher!)
  Variance: higher (more boom potential)
  Fitness (balanced): 110.09 (better overall!)
```

### Why Custom vs Package?

Popular GA libraries (DEAP, PyGAD, etc.) are general-purpose. Our problem has:
- **Custom constraints** (salary cap, positions, DFS rules)
- **Expensive fitness function** (10,000 Monte Carlo sims)
- **Domain-specific operators** (position-aware crossover)

Building custom lets us:
- Optimize for exactly our constraints
- Parallelize Monte Carlo efficiently
- Use domain knowledge (player positions, salaries)
- Checkpoint and resume

## No External Dependencies

The GA uses only:
- `numpy` for random number generation
- `pandas` for player data
- Built-in `random` module

Everything else is custom logic!

## Performance

Quick test mode (50 candidates → 2 iterations):
- Phase 1: 10 seconds
- Phase 2: 6 seconds
- Phase 3: ~30 seconds (9 generations × 3 seconds each)
- **Total: ~46 seconds**

Production mode (1000 candidates → 10 iterations):
- Phase 1: ~60 seconds
- Phase 2: ~30 seconds
- Phase 3: ~5-10 minutes (depends on convergence)
- **Total: ~7-12 minutes**

Each generation in Phase 3:
- 50 offspring × 10,000 simulations = 500,000 simulations
- With 12 cores: ~3 seconds per generation
- Typically converges in 5-15 generations
