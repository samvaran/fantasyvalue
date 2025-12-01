# Game Script Multiplier Fix - Summary

## Problem
Game script scenarios (shootout, defensive, etc.) were not meaningfully affecting player scores because:
1. Old ceiling multipliers < 1.0 created mathematically incompatible constraints
2. Distribution fitting forced all scenarios to have mean = consensus
3. Result: boom/bust scenarios only affected variance, not actual expected scores

## Solution

### Part 1: Rebalanced Multipliers
Updated `GAME_SCRIPT_FLOOR` and `GAME_SCRIPT_CEILING` in 2_data_integration.py:

**Key Principles:**
- ALL ceiling multipliers >= 1.0 (ceiling must be above baseline)
- Floor multipliers lowered to create wider ranges
- Variance expressed via floor reduction + ceiling boost
- Preserves game script intent (shootout = QB/WR boost, defensive = RB/DEF boost, etc.)

**Example changes:**
- QB shootout: ceiling 1.15 → 1.40 (massive upside)
- QB defensive: floor 0.85 → 0.75, ceiling 0.80 → 1.15 (wider range, compatible)
- RB blowout_favorite: ceiling 1.10 → 1.30 (good upside)

### Part 2: Flexible Distribution Fitting
Created new `fit_lognormal_to_percentiles()` function that:
- Fits to P10 and P90 constraints ONLY
- Lets the mean vary by scenario
- Allows boom scenarios to have higher means, bust scenarios to have lower means

**Technical details:**
```python
# Old approach: fit_shifted_lognormal(mean=consensus, p10=floor, p90=ceiling)
# Forced all scenarios to have same mean

# New approach: fit_lognormal_to_percentiles(p10=floor, p90=ceiling)
# Mean = exp(mu + sigma^2/2) + shift (varies by scenario)
```

## Results

### Boom/Bust Gaps (now significant!)
- **Josh Allen (QB)**: BOOM 23.07 vs BUST 20.38 = **2.69 pts gap (12.3%)**
- **Jonathan Taylor (RB)**: BOOM 19.50 vs BUST 16.24 = **3.26 pts gap (17.8%)**
- **Christian McCaffrey (RB)**: BOOM 21.87 vs BUST 18.17 = **3.70 pts gap (20.3%)**

### MILP vs MC Alignment
- Weighted average of scenario means ≈ consensus
- Difference: **+0.6%** (well within acceptable range)

### Validation
✓ BOOM scenarios score significantly higher than BUST (>5% gap)
✓ BUST scenarios score lower than consensus  
✓ Competitive scenario close to consensus
✓ MILP and MC remain aligned
✓ Game script intent preserved (shootout helps QB/WR, defensive helps RB/DEF, etc.)

## Files Modified

1. **2_data_integration.py**
   - Updated GAME_SCRIPT_FLOOR (lines 47-83)
   - Updated GAME_SCRIPT_CEILING (lines 86-122)
   - Changed to use fit_lognormal_to_percentiles (line 388)

2. **optimizer/utils/distribution_fit.py**
   - Added fit_lognormal_to_percentiles() function (lines 36-70)

## Impact

Players in favorable game scripts now have meaningfully higher projections:
- QB in shootout: +5% to +17% vs consensus
- RB in blowout_favorite: +6% to +28% vs consensus
- WR/TE in shootout: similar boosts

Players in unfavorable game scripts have lower projections:
- QB in defensive: -8% to -14% vs consensus
- RB in blowout_underdog: -11% to -14% vs consensus

This creates **real strategic value** in targeting game scripts when building lineups!
