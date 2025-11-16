# Simulation Variance Fix

## Problem Identified

The Monte Carlo simulations were producing **unrealistically narrow distributions** where all percentiles (p10, p50, p75, p90) were bunched together within 1-5 points of the consensus mean.

### Symptoms:
- `sim_p90` was barely higher than `sim_p50` (only 1-3 points difference)
- `sim_mean` ≈ `sim_p50` ≈ `consensus` (all within 0.1 points)
- P90 total from optimization was HIGHER than simulated P90 (backwards!)
- Example: Lineup with consensus=121.91, p90=126.01, sim_p90=123.35, sim_mean=121.9

### Root Cause:

The `uncertainty` column in `knapsack.csv` contained values that were **far too small**:
- Values ranged from 0.3 to 0.45 (absolute values)
- For a player with consensus=20 pts, uncertainty=0.35 is only **1.75%** of the mean
- This created almost zero variance in the log-normal distributions

**Proper fantasy football variance should be 25-40% of the mean**, not 1.5-2%!

## Solution

Modified `league_optimizer.py` in two places to use **position-specific uncertainty multipliers**:

### 1. Ceiling Value Calculation (Line 62-77)
```python
# OLD (uses tiny uncertainty values from CSV):
uncertainty = row['uncertainty'] if pd.notna(row['uncertainty']) else consensus * 0.3

# NEW (position-specific, realistic variance):
if row['position'] == 'QB':
    uncertainty = consensus * 0.30  # 30% - QBs have moderate variance
elif row['position'] == 'RB':
    uncertainty = consensus * 0.35  # 35% - RBs: injury/game script risk
elif row['position'] == 'WR':
    uncertainty = consensus * 0.40  # 40% - WRs: highest variance (targets)
elif row['position'] == 'TE':
    uncertainty = consensus * 0.35  # 35% - TEs: high variance (low volume)
elif row['position'] == 'D':
    uncertainty = consensus * 0.45  # 45% - DEF: extreme game script variance
else:
    uncertainty = consensus * 0.35
```

### 2. Monte Carlo Simulation (Line 278-290)
Applied the same position-specific uncertainty in the simulation loop.

## Expected Results

After re-running `python3 league_optimizer.py`, you should see:

### Realistic Variance:
- **P90 - P50 spread**: 15-25 points (was: 1-3 points)
- **P90 - Mean spread**: 8-15 points (was: 0-2 points)
- **P50 - P10 spread**: 15-25 points (was: 1-3 points)

### Example Lineup:
```
OLD: consensus=121.9, sim_p50=121.9, sim_p90=123.3  (2.4 pt spread)
NEW: consensus=121.9, sim_p50=121.0, sim_p90=138.5  (17.5 pt spread) ✓
```

### Why This Matters:
- **Tournament optimization**: You need high ceiling lineups, not average ones
- **Risk assessment**: Wide spreads = high variance = tournament-winning potential
- **Proper ranking**: Lineups should be ranked by SIMULATED P90, not theoretical P90

## Position-Specific Variance Rationale

| Position | Variance | Reasoning |
|----------|----------|-----------|
| QB | 30% | Rushing floors, passing ceilings, relatively stable |
| RB | 35% | Game script, injury risk, TD variance |
| WR | 40% | Target share volatility, boom/bust, DB matchups |
| TE | 35% | Low volume, high TD dependency |
| DEF | 45% | Most game script dependent, sack/turnover luck |

## Files Modified
- `league_optimizer.py` (lines 62-77, 278-290)

## Next Steps
1. Run: `python3 league_optimizer.py`
2. Check `LEAGUE_LINEUPS.csv` for proper variance
3. Verify P90 - P50 spreads are 15-25 points
4. Confirm lineups are ranked by `sim_p90` (highest upside)
