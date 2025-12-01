# Genetic Algorithm Diversity Enhancements

## Problem
The GA was experiencing diversity collapse - final populations had mostly identical lineups with only small variations.

## Root Causes
1. **Tournament selection with replacement** - same high-fitness lineups selected repeatedly
2. **Strong elitism** - top 20 always survive, dominating gene pool
3. **Conservative mutation** - only 1-2 player swaps with tight salary constraints
4. **No diversity metrics** - no visibility into or enforcement of uniqueness
5. **No recovery mechanism** - once diversity lost, hard to recover

## Solutions Implemented

### 1. Diversity Metrics (`diversity_tools.py`)
- **calculate_uniqueness()**: Percentage of truly unique lineups in population
- **calculate_player_diversity()**: How many unique players used, usage distribution
- **hamming_distance()**: Number of different players between two lineups
- **calculate_avg_distance()**: Average Hamming distance across population
- **deduplicate_population()**: Remove near-duplicates (configurable threshold)

### 2. Diversity Enforcement in GA Loop
**Deduplication** (after replacement):
```python
if enforce_diversity:
    # Remove lineups within min_hamming_distance of each other
    new_population = deduplicate_population(new_population, min_distance=2)
```

**Diversity Injection** (when population too homogeneous):
```python
if len(new_population) < population_size * 0.7:
    # Inject diverse lineups from Phase 2 pool
    n_to_inject = population_size - len(new_population)
    injection = available_evals.sample(n_to_inject)
    new_population.extend(injection)
```

### 3. Adaptive Mutation
**Aggressive mode** activates when uniqueness < 50%:
- Swaps 2-4 players instead of 1-2
- Wider salary tolerance (2x normal)
- More disruption to break out of local optima

```python
current_uniqueness = calculate_uniqueness(population)
use_aggressive = current_uniqueness < 0.5

mutate(child, players_df, aggressive=use_aggressive)
```

### 4. Diversity Reporting
Each generation now shows:
```
Diversity: 85.0% unique, avg distance: 4.2 players
```

This gives immediate feedback on population health.

## Parameters

**New GA parameters:**
- `enforce_diversity=True`: Enable all diversity mechanisms
- `min_hamming_distance=2`: Minimum player difference for unique lineups
- `diversity_injection_rate=0.1`: Fraction to inject when needed

**Mutation enhancement:**
- `aggressive=False/True`: Controlled by adaptive logic based on diversity

## Expected Improvements

1. **Higher final diversity**: 70-90% unique lineups vs 10-30% before
2. **More player variety**: 100+ unique players used vs 30-50 before
3. **Better exploration**: Less premature convergence
4. **Maintained quality**: Diversity shouldn't hurt best fitness
5. **Observable metrics**: Clear visibility into population health

## Usage

```python
optimize_genetic(
    ...
    enforce_diversity=True,      # Enable diversity features
    min_hamming_distance=2,      # At least 2 different players
    diversity_injection_rate=0.1 # 10% injection when needed
)
```

## Trade-offs

**Pros:**
- Much more diverse final output
- Better exploration of solution space
- Reduced risk of missing good lineups
- Self-regulating (adaptive)

**Cons:**
- Slightly slower (deduplication overhead)
- May take more generations to converge
- Best fitness might improve slower initially

The trade-off is worth it for getting a diverse set of competitive lineups!
