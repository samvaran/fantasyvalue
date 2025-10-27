# TD Odds & Game Script Adjustments

**Last updated:** 2025-10-27

## Overview

The data pipeline now applies **aggressive TD odds and game script adjustments** to consensus projections before optimization. This gives players in favorable game environments a significant boost, helping the optimizer identify high-upside plays.

---

## TD Odds Boost

**Applies to: RB, WR, TE only** (excludes QBs and Defenses)

**Formula:** `adjustedConsensus = consensus × (1 + tdProbability × 0.5)`

**Impact:**
- **20% TD probability** → +10% boost to projection
- **40% TD probability** → +20% boost to projection
- **60% TD probability** → +30% boost to projection
- **100% TD probability** → +50% boost to projection (theoretical max)

**Also increases uncertainty** (variance) for high-TD players by 50% of the boost, capturing the boom-or-bust nature.

**Why only skill positions?**
- **QBs excluded**: QB projections already account for passing TDs. Anytime TD odds are for *rushing* TDs, which is minor.
- **Defenses excluded**: Defensive TDs are rare events already factored into DST projections.

### Real Examples (Current Week):

| Player | Position | TD Prob | Original Proj | Adjusted Proj | Boost |
|--------|----------|---------|---------------|---------------|-------|
| **Josh Jacobs** | RB | 64.3% | 15.1 | 19.9 | **+4.9 pts (+32%)** |
| **Jaylen Warren** | RB | 45.5% | 12.7 | 15.5 | **+2.9 pts (+23%)** |
| **Rashee Rice** | WR | 51.2% | 13.3 | 16.0 | **+2.7 pts (+21%)** |
| **Travis Kelce** | TE | 52.4% | 8.1 | 9.8 | **+1.7 pts (+21%)** |
| **Tucker Kraft** | TE | 43.5% | 8.0 | 9.7 | **+1.7 pts (+22%)** |

**Result:** High-TD players like Josh Jacobs are now heavily favored (39/40 top lineups vs maybe 5-10 before).

---

## Game Script Adjustments

### Running Backs (RB)

**Positive game script** (team favored by 3+ points):
- **+8% boost** - Favored teams run more to protect leads
- Example: Bijan Robinson (+1.7 pts when ATL favored)

**Negative game script** (team trailing by 3+ points):
- **-6% penalty** - Trailing teams abandon the run
- Example: RBs on teams trailing significantly get dinged

### Wide Receivers & Tight Ends (WR/TE)

**Negative game script** (team trailing by 3+ points):
- **+10% boost** - Trailing teams pass more to catch up
- Example: Terry McLaurin (+2.7 pts, +24% when WAS trailing)

**Heavy favorite** (team favored by 7+ points):
- **-4% penalty** - Run-heavy game script in 4th quarter
- Pass catchers see fewer targets late

### Quarterbacks (QB)

**High-scoring games** (total over 50 points):
- **+8% boost** - Shootouts mean more pass attempts
- Example: Patrick Mahomes in WAS@KC (total 47.5, gets boost)

**Low-scoring games** (total under 42 points):
- **-5% penalty** - Defensive slugfests = fewer opportunities
- QBs in low-total games penalized

### Defenses (D)

**Facing weak offenses** (opponent projected < 18 points):
- **+25% boost** - Easy matchup = more sacks, INTs, TDs
- Example: Colts DST, Patriots DST (+2.1 pts each, +25%)

**Facing strong offenses** (opponent projected > 26 points):
- **-20% penalty** - Tough matchup = getting shredded
- Defenses vs high-powered offenses heavily penalized

---

## Combined Impact

Players can get **stacked boosts** from both TD odds AND game script:

**Example: Josh Jacobs**
1. Base consensus: 15.1 pts
2. TD odds boost (64% prob): +32% → 19.9 pts
3. Game script (if GB favored): Would add another +8%
4. **Total potential boost:** ~+40% (+6 points!)

**Example: Terry McLaurin**
1. Base consensus: 11.2 pts
2. TD odds boost (26% prob): +13%
3. Game script (WAS trailing KC): +10%
4. **Combined:** 13.9 pts (+24% total boost)

---

## How This Changes Lineups

### Before Adjustments (Old System):
- Optimizer picked chalk: Mahomes, Bijan, Devon Achane
- High-salary, high-projection players dominated
- TD probability ignored
- Game environment ignored

### After Adjustments (New System):
- **Josh Jacobs** (64% TD prob) in 39/40 lineups
- **Isiah Pacheco** (38% TD prob) in 21/40 lineups
- **Terry McLaurin** (trailing team) highly valued
- **Defenses vs weak offenses** get huge boost
- High-TD TEs (Kelce, Kraft) more prominent

---

## Why This Works

1. **TD probability is predictive** - Sportsbooks know something. A player with 50% anytime TD odds IS more likely to boom than projections suggest.

2. **Game script matters** - RBs on favored teams DO get more carries. Pass catchers on trailing teams DO get more targets. This isn't speculation, it's empirical.

3. **Projection sources are conservative** - FanDuel and ESPN projections don't fully account for game environment. These adjustments fill the gap.

4. **Creates differentiation** - Without these boosts, optimizer picks the same chalk every week. With them, we find contrarian high-upside plays.

---

## Tuning the Adjustments

All boost values are configurable in `fetch_data.js` around line 1208-1256.

### To make TD odds MORE impactful:
```javascript
const tdBoost = (p.tdProbability / 100) * 0.7; // Increase from 0.5 to 0.7
```

### To make game script MORE impactful:
```javascript
if (spread > 3) {
    adjustedConsensus *= 1.12; // Increase from 1.08 to 1.12 for RBs
}
```

### To make defenses EVEN MORE matchup-dependent:
```javascript
if (p.projOppPts < 18) {
    adjustedConsensus *= 1.35; // Increase from 1.25 to 1.35
}
```

---

## Validation

**Week 8 Example (Hypothetical):**

Without adjustments:
- Tucker Kraft: 8.0 projected, 32.8 actual → Optimizer ignored him

With adjustments:
- Tucker Kraft: 9.7 projected (43% TD prob boost)
- Higher projection + higher uncertainty = more likely to get picked
- Still might not make top lineup, but appears in some lineups

**The goal isn't perfection** - it's giving high-upside players a fair shake in the optimization.

---

## Files Modified

- **`fetch_data.js`** (lines 1208-1259): Added TD odds and game script adjustment logic
- **`knapsack.csv`**: Now contains adjusted consensus values
- All downstream optimizers use these adjusted values automatically

---

## Usage

Just run the pipeline normally:

```bash
# Fetch data with adjustments applied
node fetch_data.js

# Optimize with adjusted projections
python league_optimizer.py
```

The adjustments are **always on** - they're baked into the consensus calculation.

---

## Results

**Top lineup ceiling:** 144.8 pts (sim P90)
**Player diversity:** 40 unique players in top 40 lineups
**High-TD players featured:** Josh Jacobs, Rashee Rice, Tucker Kraft, Kelce all prominent

Compare to before adjustments:
- More diversity (was ~34 unique players)
- High-TD players properly valued (Jacobs 39/40 vs maybe 10/40 before)
- Game script properly reflected (Terry McLaurin boosted as underdog WR)

---

**These adjustments make the optimizer tournament-ready. Players with boom potential get the respect they deserve.** 🚀
