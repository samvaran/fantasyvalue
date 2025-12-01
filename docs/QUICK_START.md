# Quick Start Guide

The optimizer is ready to use! Here's how to get started:

## 1. Test Run (2-3 minutes)

```bash
python3 3_run_optimizer.py --quick-test
```

This runs a quick test with:
- 50 candidate lineups
- 1,000 simulations per lineup
- 2 refinement iterations

## 2. Monitor Progress

While the optimizer is running, open a new terminal and run:

```bash
python3 view_progress.py
```

This shows real-time progress including:
- Phase completion status
- Iteration history
- Best lineups found so far
- Fitness progression

## 3. Production Run (~90 minutes)

Once the test run completes successfully:

```bash
python3 3_run_optimizer.py --candidates 1000 --sims 10000
```

This runs the full optimizer:
- 1,000 candidate lineups (~3 min)
- 10,000 simulations per lineup (~25-30 min)
- Genetic refinement iterations (~40-60 min each)
- Automatically stops when converged (typically 3-5 iterations)

## 4. View Results

After completion, your best lineups are in:

```bash
outputs/run_YYYYMMDD_HHMMSS/BEST_LINEUPS.csv
```

This file contains the top 10 lineups found across all iterations, with:
- Player IDs and names
- Salary usage
- Projected scores (mean, median, P10, P90)
- Risk metrics (std, skewness)
- Fitness score

## 5. Resume Interrupted Run

If the optimizer is interrupted, resume with:

```bash
python3 run_optimizer.py --run-name YYYYMMDD_HHMMSS
```

The optimizer automatically:
- Detects completed phases
- Continues from last iteration
- Preserves all previous results

## Expected Performance (8-core machine)

- **Phase 1** (Candidate Generation): ~3 minutes
- **Phase 2** (Monte Carlo Evaluation): ~25-30 minutes
- **Phase 3** (Genetic Refinement): ~40-60 minutes per iteration
- **Total**: ~70-90 minutes for first 2 iterations, typically converges in 3-5 iterations

## Files Created

Each run creates a timestamped directory with:

- **BEST_LINEUPS.csv** - Top 10 lineups (use these!)
- **final_summary.json** - Complete run statistics
- **optimizer_state.json** - State for resumption
- **candidates.csv** - All generated candidates
- **evaluations.csv** - Monte Carlo results
- **iteration_N/** - Results from each genetic refinement

## Fitness Functions

Choose different strategies with `--fitness`:

```bash
# Conservative (high floor, low risk)
python3 3_run_optimizer.py --fitness conservative

# Balanced (solid expected value) - DEFAULT
python3 3_run_optimizer.py --fitness balanced

# Aggressive (boom/bust potential)
python3 3_run_optimizer.py --fitness aggressive

# Tournament (pure upside, swing for fences)
python3 3_run_optimizer.py --fitness tournament
```

## Troubleshooting

**"No DEFs available"** - Your data is missing defense projections. The optimizer automatically fills these with defaults.

**"KeyError: 'id'"** - Column name mismatch. The optimizer auto-normalizes columns, but if you see this, check your input CSV structure.

**Optimizer is slow** - Reduce `--candidates` (try 500) or `--sims` (try 5000) for faster testing.

**Out of memory** - Reduce `--processes` or `--candidates`.

## More Information

- **OPTIMIZER_USAGE.md** - Complete usage guide with all options
- **OPTIMIZER_DESIGN.md** - Technical design and architecture
- **WORKFLOW.md** - Data pipeline overview

---

**You're all set!** Run the quick test first to verify everything works, then let it run the full pipeline overnight for best results.
