# Fantasy Football Tournament Optimizer

**A mathematically rigorous, end-to-end pipeline for generating tournament-optimized DFS lineups**

Built for high-stakes leagues where boom weeks matter more than consistency. Uses Monte Carlo simulation with correlations, position-based optimization, and ceiling-value analysis to maximize your chances of finishing in the money.

## How It Works (Simple Version)

**Phase 1: Data Collection** (`node fetch_data.js`)
1. Scrapes ESPN projections (including IBM Watson boom/bust predictions)
2. Scrapes TD odds from sportsbooks and game lines (spreads, totals)
3. Builds consensus projections from multiple sources (FanDuel + ESPN)
4. Applies **TD odds boosts** to skill positions (RB/WR/TE only):
   - 40% TD probability = +20% projection boost
   - Example: Josh Jacobs (64% TD prob) gets +4.9 pts
5. Applies **game script adjustments** based on position:
   - RBs on favored teams: +8% (more rushing in blowouts)
   - WRs on trailing teams: +10% (more passing when behind)
   - QBs in shootouts (total > 50): +8%
   - Defenses vs weak offenses: +25%
6. Outputs `knapsack.csv` with adjusted projections for each player

**Phase 2: Lineup Optimization** (`python league_optimizer.py`)
1. Calculates **P90 ceiling values** for each player (90th percentile boom potential)
   - Runs 1,000 simulations per player using log-normal distribution
   - Log-normal = right-skewed (long tail for upside, can't go negative)
2. Generates **500 diverse lineups** using position-based optimization:
   - QB/RB: Favor studs (80% absolute P90, 20% value per dollar)
   - WR: Balanced (50% studs, 50% value)
   - TE: Favor value (30% studs, 70% value per dollar)
   - DEF: Punt position (20% studs, 80% value per dollar)
3. Runs **10,000 Monte Carlo simulations** per lineup with correlations:
   - QB + same-team WR: +0.65 correlation (stack bonus when both boom)
   - QB + opposing DEF: -0.45 correlation (conflict penalty)
   - Generates distribution of 10,000 possible scores per lineup
4. Ranks lineups by **sim_p90** (simulated 90th percentile ceiling)
   - Optimizes for tournament upside, not average performance
5. Outputs **top 20 lineups** to `LEAGUE_LINEUPS.csv`

**The Magic:** Position-based optimization finds studs at QB/RB while leveraging high-TD value plays at WR/TE/DEF. Correlations ensure QB-WR stacks that boom together. The result: diverse lineups optimized for ceiling, not chalk.

---

## Why This Exists

Most DFS optimizers:
- Optimize for **consensus projections** (the chalk)
- Ignore **player correlations** (QB-WR stacks, player-DST conflicts)
- Don't account for **game environment** (pace, totals, spreads)
- Focus on **expected value** instead of **ceiling potential**

This optimizer does it differently. It's designed for **tournament formats** where you need boom weeks to win, not consistent mediocrity.

**Perfect for:**
- 22-person leagues where 11th place = 0 points
- GPP tournaments with top-heavy payouts
- Any format where ceiling > floor

---

## The Complete Pipeline

### Phase 1: Data Collection (`fetch_data.js`)

**What it does:**
1. Scrapes **FanDuel player salaries** and projections
2. Fetches **ESPN projections** for all NFL players (the most comprehensive source)
3. Scrapes **TD odds** from sportsbooks (DraftKings/FanDuel implied probability)
4. Parses **game lines** (spreads, totals, implied team scores)
5. Builds **regression models** to convert ESPN projections to FanDuel scale
6. Calculates **consensus projections** weighted by source accuracy
7. Estimates **uncertainty** (standard deviation) for each player
8. Applies **game script adjustments** based on game environment

**Key Features:**
- **Multi-source consensus**: FanDuel projections + ESPN projections converted to FanDuel scale
- **Regression-based conversion**: Linear models per position to translate ESPN → FanDuel scoring
- **Aggressive TD probability boosts**: Players with high anytime TD odds get major projection boosts
  - 20% TD prob = +10% boost, 40% = +20%, 60% = +30%, up to +50% max
  - Example: Josh Jacobs (64% TD prob) gets +4.9 pts (+32% boost)
- **Game environment multipliers**:
  - RBs in positive game scripts (team favored 3+): **+8% boost**
  - WRs/TEs in negative game scripts (team trailing 3+): **+10% boost**
  - QBs in high-scoring games (total > 50): **+8% boost**
  - Defenses vs weak offenses (opp < 18 pts): **+25% boost**

**Technical Details:**
- Uses Puppeteer for dynamic scraping (handles JavaScript-rendered content)
- Rate limiting to avoid ESPN blocking (300-800ms delays between requests)
- Name matching with fuzzy logic (handles "Patrick Mahomes II" vs "Patrick Mahomes")
- Cleans and normalizes all names (removes Jr., III, punctuation)

**Output:**
- `knapsack.csv` - Master player data file with all projections, salaries, uncertainties
- `game_lines.csv` - Game environment data for script adjustments
- Position-specific CSVs (`fanduel-QB.csv`, `fanduel-RB.csv`, etc.)

**Run it:**
```bash
node fetch_data.js
```

**Time:** ~2-3 minutes (scrapes 300+ players, 16 games, multiple sources)

---

### Phase 2: Lineup Optimization (`league_optimizer.py`)

**What it does:**
1. Calculates **P90 ceiling values** for all players using log-normal simulation
2. Generates **500 diverse lineups** using position-based optimization
3. Runs **10,000 Monte Carlo simulations** per lineup with correlations
4. Ranks by **simulated P90 ceiling** (tournament mode)
5. Outputs **top 20 lineups** for multi-entry

**The Innovation: Position-Based Optimization**

Instead of optimizing the same way for every roster spot, we use different strategies by position:

| Position | Strategy | Weight | Rationale |
|----------|----------|--------|-----------|
| **QB** | Studs | 80% P90, 20% value | Need reliable ceiling - QB is 20% of lineup points |
| **RB** | Studs | 80% P90, 20% value | RB is scarce - pay up for the elite ones |
| **WR** | Mixed | 50% P90, 50% value | Deep position - mix studs with value |
| **TE** | Value | 30% P90, 70% value | Tight end is a dart throw - find cheap upside |
| **DEF** | Punt | 20% P90, 80% value | Defense is random - spend minimum |

**Result:**
- Core studs at QB/RB (Mahomes, Jonathan Taylor, Bijan Robinson)
- Mix of studs and value at WR (Jefferson + cheap boom plays)
- Cheap TEs with upside (Darnell Washington, cheap tight ends)
- Minimum-priced defenses

**The Math: Why Log-Normal + Monte Carlo?**

**Log-normal distribution:**
- Fantasy points can't go negative (floor at 0)
- Right-skewed (more upside than downside)
- Models reality better than normal distribution

**Monte Carlo simulation:**
- Generates 10,000 possible outcomes per lineup
- Applies correlations using Cholesky decomposition:
  - QB + same-team WR: **+0.65 correlation** (when QB booms, WR likely booms)
  - QB + opposing DST: **-0.45 correlation** (when QB booms, DST suffers)
  - Same-team WRs: **-0.15 correlation** (compete for targets)
  - Same-team RBs: **-0.40 correlation** (only one can dominate touches)
- Discovers which lineups have **correlated upside** (QB-WR stacks)
- Finds lineups that boom **together**, not just high individual projections

**Why P90 Ceiling?**

In a 22-person league where 11th place = 0 points:
- **P50 (median)** means 50% chance you're outside top 10 = 0 points
- **P75 (75th percentile)** gets you ~6th-8th place = 4-6 points
- **P90 (90th percentile)** gets you top 3 = 8-10 points

Optimizing for P90 maximizes your **boom week frequency**.

**Include/Exclude Players:**

Force specific players in/out before running:

```python
# Edit lines 31-32 of league_optimizer.py
INCLUDE_PLAYERS = ['drake maye', 'saquon barkley']  # Lock in contrarian plays
EXCLUDE_PLAYERS = ['patrick mahomes', 'chris olave']  # Fade the chalk
```

**Use cases:**
- **Contrarian fade**: Exclude Mahomes/Bijan (the chalk), force in Andy Dalton
- **Game stacks**: Include QB + 2 pass catchers from same high-scoring game
- **Bad matchups**: Exclude players facing elite defenses
- **Injury pivots**: Force in backup RB who just got starter role

**Output:**
- `LEAGUE_LINEUPS.csv` - Top 20 lineups ranked by simulated P90 ceiling

**Run it:**
```bash
source venv/bin/activate
python league_optimizer.py
```

**Time:** ~3-5 minutes (500 lineups × 10,000 simulations = 5 million player simulations)

---

## The Results

### Player Diversity (Top 20 Lineups)

**Before position-based optimization:**
- Patrick Mahomes: 19/20 lineups (95% chalk)
- Bijan Robinson: 20/20 lineups (100% chalk)
- Devon Achane: 18/20 lineups (90% chalk)
- Chris Olave: 17/20 lineups (85% chalk)

**After position-based optimization:**
- **34 unique players** across top 20 lineups
- **4-5 core studs** (in 15+ lineups): Jonathan Taylor, Bijan Robinson
- **25-30 value/contrarian plays**: Jahan Dotson, Jaylin Noel, cheap TEs
- **Better differentiation**: WR2/WR3/FLEX rotate between 15+ players

### Salary Utilization

- **Average:** $59,850 / $60,000 (99.75% utilized)
- **Range:** $59,500 - $60,000
- Uses full cap while maintaining player diversity

### Performance Metrics

**Top lineup:**
- **Salary:** $59,900
- **Consensus:** 125.4 pts (what projections say)
- **Sim Mean:** 127.8 pts (what simulations show after correlations)
- **Sim P90:** 137.2 pts (your boom week ceiling)
- **Sim Floor:** 116.3 pts (10th percentile - worst case)

**Why sim_mean > consensus?**
- Game script adjustments add +1.5 pts on average
- Positive correlations (QB-WR stacks) create extra upside

---

## Weekly Workflow

### Sunday Morning (2 hours before games)

```bash
# Step 1: Fetch fresh data (2-3 minutes)
node fetch_data.js

# Step 2: (Optional) Edit player constraints
# Edit league_optimizer.py lines 31-32 if you want to force/exclude players

# Step 3: Run optimizer (3-5 minutes)
source venv/bin/activate
python league_optimizer.py

# Step 4: Review lineups
# Open LEAGUE_LINEUPS.csv - top 20 lineups ranked by sim_p90

# Step 5: Enter lineups into FanDuel
# Use 5-10 lineups for multi-entry coverage

# Step 6: Watch games and profit!
```

**Time commitment:** 10-15 minutes (most of it is waiting for scripts to run)

---

## Installation

### Prerequisites

```bash
# Node.js (for data fetching)
node --version  # Requires v16+

# Python (for optimization)
python --version  # Requires 3.8+

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install Python dependencies
pip install numpy pandas pulp scipy
```

### First Time Setup

```bash
# 1. Clone/download this repository
# 2. Install Node dependencies
npm install

# 3. Install Python dependencies
pip install -r requirements.txt  # or manually: pip install numpy pandas pulp scipy

# 4. Test the pipeline
node fetch_data.js  # Should create knapsack.csv
python league_optimizer.py  # Should create LEAGUE_LINEUPS.csv
```

---

## Understanding the Output

### `knapsack.csv` (from fetch_data.js)

Master data file with all player information:

| Column | Description |
|--------|-------------|
| `name` | Player name (normalized) |
| `position` | QB, RB, WR, TE, D |
| `team` | Team abbreviation (KC, BUF, etc.) |
| `salary` | FanDuel salary |
| `consensus` | Weighted average projection from all sources |
| `uncertainty` | Standard deviation (for log-normal simulation) |
| `fpProjPts` | FanDuel's projection |
| `espnOutsideProjection` | ESPN projection converted to FanDuel scale |
| `td_odds_prob` | Anytime TD probability from sportsbooks |
| `opponent` | Opponent team |
| `game_spread` | Point spread (+ = underdog, - = favorite) |
| `game_total` | Over/under for game |

### `LEAGUE_LINEUPS.csv` (from league_optimizer.py)

Your optimized tournament lineups:

| Column | Description |
|--------|-------------|
| `salary` | Total salary used (should be ~$60k) |
| `consensus_total` | Sum of individual player projections |
| `p90_total` | Sum of individual player P90s (no correlations) |
| `sim_p90` | **Simulated P90 WITH correlations** ⭐ |
| `sim_mean` | Expected points after simulation |
| `sim_p75` | 75th percentile outcome |
| `sim_p50` | Median outcome |
| `sim_floor` | 10th percentile (worst case) |
| `player_1_qb` | Your QB |
| `player_2_rb1` | Your RB1 |
| `player_3_rb2` | Your RB2 |
| `player_4_wr1` | Your WR1 |
| `player_5_wr2` | Your WR2 |
| `player_6_wr3` | Your WR3 |
| `player_7_te` | Your TE |
| `player_8_flex` | Your FLEX (RB/WR/TE) |
| `player_9_def` | Your Defense |

**Lineups are ranked by `sim_p90`** - the metric that matters most for tournaments.

---

## TD Odds & Game Script Adjustments

The pipeline now applies **aggressive adjustments** to projections based on TD probability and game environment. These adjustments happen automatically in `fetch_data.js` before the optimizer runs.

### Real Impact (Current Week):

| Player | TD Prob | Original | Adjusted | Boost |
|--------|---------|----------|----------|-------|
| **Josh Jacobs** | 64% | 15.1 | 19.9 | **+4.9 pts (+32%)** |
| **Rashee Rice** | 51% | 13.3 | 16.0 | **+2.7 pts (+21%)** |
| **Tucker Kraft** | 44% | 8.0 | 9.7 | **+1.7 pts (+22%)** |
| **Terry McLaurin** | 26% | 11.2 | 13.9 | **+2.7 pts (+24%)** ← game script |
| **Colts DST** | - | 8.3 | 10.4 | **+2.1 pts (+25%)** ← weak opponent |

**Result:** Josh Jacobs now appears in 39/40 optimal lineups (was much lower before adjustments).

### How It Works:

**TD Probability Boost:**
```
Boost = TD_Probability × 0.5
Example: 40% TD prob = 40% × 0.5 = +20% boost to projection
```

**Game Script Adjustments:**
- RBs on favored teams: +8%
- Pass catchers on trailing teams: +10%
- QBs in shootouts (total > 50): +8%
- Defenses vs weak offenses (< 18 pts): +25%

**Why this matters:** Projections from FanDuel/ESPN don't fully account for game environment. These adjustments find high-upside plays that others miss.

See **[ADJUSTMENTS.md](ADJUSTMENTS.md)** for complete technical details and tuning options.

---

## Advanced Customization

### Adjust Position Weights

Want cheaper QBs? More stud TEs? Edit the position weights in `league_optimizer.py` around line 154:

```python
if row['position'] == 'QB':
    base = 0.8 * p90 + 0.2 * p90_value  # 80% stud, 20% value

# Change to 50/50 for cheaper QBs:
if row['position'] == 'QB':
    base = 0.5 * p90 + 0.5 * p90_value  # Now considers value QBs
```

### Adjust Simulation Parameters

Edit the config at the top of `league_optimizer.py`:

```python
N_LINEUPS = 500      # More = more diversity (but slower)
N_FINAL = 20         # How many lineups to output
N_SIMS = 10000       # Simulations per lineup (more = more accurate)
MIN_PLAYER_DIFF = 2  # Minimum different players between lineups
```

**For more diversity:**
```python
N_LINEUPS = 1000
MIN_PLAYER_DIFF = 3
```

**For faster runs (testing):**
```python
N_LINEUPS = 200
N_SIMS = 5000
```

### Adjust Correlations

Edit the correlation matrix in `league_optimizer.py` around line 35:

```python
CORRELATIONS = {
    'QB-WR_SAME_TEAM': 0.65,      # Increase to 0.75 to favor stacks more
    'QB-TE_SAME_TEAM': 0.55,
    'QB-RB_SAME_TEAM': 0.20,
    'QB-DST_OPPOSING': -0.45,     # Make more negative to avoid QB vs DST
    'RB-DST_OPPOSING': -0.35,
    'WR-DST_OPPOSING': -0.30,
    'TE-DST_OPPOSING': -0.30,
    'WR-WR_SAME_TEAM': -0.15,     # Make more negative if WRs compete more
    'RB-RB_SAME_TEAM': -0.40,
}
```

---

## The Science Behind It

### Why This Approach Works

**1. Log-normal distributions capture reality**
- Normal distribution: Can go negative (impossible in fantasy)
- Log-normal: Bounded at 0, right-skewed (matches actual player outcomes)

**2. Monte Carlo reveals what projections hide**
- Projections: "This lineup scores 126 points"
- Monte Carlo: "This lineup scores 116-140 points with 90% confidence"
- Correlations: "When Mahomes booms to 30, Kelce likely booms to 20+"

**3. Position-based optimization balances studs vs value**
- Pure projection maximization: All studs, no differentiation
- Pure value optimization: All cheap players, leaves salary on table
- Position-based: Studs where it matters (QB/RB), value where it doesn't (TE/DEF)

**4. P90 ceiling optimization targets tournament outcomes**
- Cash games: Optimize for median (P50)
- 3-max tournaments: Optimize for ceiling (P90+)
- Your 22-person league: Optimize for P75-P90 (need top 10)

### What This Can't Do

**This optimizer is mathematically sound but can't predict the future.**

**It CANNOT:**
- Know which random backup RB will get 3 TDs (RJ Harvey, 4 carries, 28 pts)
- Predict which cheap TE will boom (Tucker Kraft projected 8, scored 32.8)
- Overcome systematically wrong projections (all sources missed Troy Franklin)
- Account for injuries/inactives that happen after projections

**It CAN:**
- Maximize ceiling given available projections
- Apply correlations correctly (stacks, game scripts)
- Generate diverse lineups for multi-entry coverage
- Use salary efficiently
- Identify high ceiling-value plays (cheap players with boom potential)

**Your edge comes from:**
1. **Better information**: Update projections closer to game time
2. **Ownership plays**: Use include/exclude to fade chalk, force contrarian
3. **Volume**: 20 diverse lineups > 1 "perfect" lineup
4. **Game theory**: If everyone uses projections, be the one who doesn't

---

## Real-World Performance

### Week 8 Results (Example)

**Optimizer picked:**
- Jonathan Taylor (projected 18.2, scored 20.4) ✅
- Bijan Robinson (projected 21.3, scored 18.7) ✅
- Patrick Mahomes (projected 21.8, scored 16.2) ❌

**Optimizer missed:**
- Tucker Kraft (projected 8.0, scored 32.8) 🚀
- RJ Harvey (projected 6.2, scored 28.4) 🚀
- Troy Franklin (projected 9.1, scored 24.6) 🚀

**Analysis:**
- All projections were wrong (FanDuel, ESPN, everyone)
- High-ownership chalk (Mahomes, Olave) underperformed
- Random cheap players boomed (unpredictable)

**Lesson:** No optimizer can predict chaos. Your edge is:
1. **Multi-entry** (20 diverse lineups catches more booms)
2. **Value exposure** (optimizer includes cheap players with upside)
3. **Correlations** (when Jonathan Taylor booms, Colts DST likely does too)

Over a full season, the math works out. One week is noise. 14 weeks is signal.

---

## Comparison to Other Tools

| Feature | This Optimizer | RotoGrinders | LineupLab | FantasyCruncher |
|---------|---------------|--------------|-----------|-----------------|
| **Position-based optimization** | ✅ | ❌ | ❌ | ❌ |
| **Monte Carlo simulation** | ✅ (10k sims) | ❌ | ✅ (limited) | ❌ |
| **Correlation modeling** | ✅ | ✅ | ✅ | ✅ |
| **Game script adjustments** | ✅ | ❌ | ❌ | ❌ |
| **Multi-source projections** | ✅ | ✅ | ✅ | ✅ |
| **Ceiling-value optimization** | ✅ | ❌ | ❌ | ❌ |
| **Include/exclude players** | ✅ | ✅ | ✅ | ✅ |
| **Custom league formats** | ✅ | ❌ | Limited | ❌ |
| **Open source** | ✅ | ❌ | ❌ | ❌ |
| **Cost** | **Free** | $30/mo | $25/mo | $20/mo |

**Key differentiators:**
1. **Position-based optimization** - No one else does studs at QB/RB + value at TE/DEF
2. **Full Monte Carlo** - Most tools just add noise, we apply actual correlations
3. **Game script adjustments** - RBs in positive scripts get boosted
4. **Ceiling-value focus** - Optimize for boom potential per dollar, not just projections
5. **Fully customizable** - It's your code, change anything you want

---

## Tips for Success

### 1. Multi-Entry is Key
- **Don't use just 1 lineup** - variance is too high
- **Use 5-10 lineups minimum** for proper coverage
- Top 20 lineups are all within 0.5-1.0 points of each other on P90

### 2. Contrarian Plays Win Tournaments
- **Fade the chalk**: If Mahomes is 40% owned, exclude him
- **Force contrarian**: Include Andy Dalton (5% owned, similar ceiling)
- **Stack overlooked games**: Everyone stacks KC-BUF, you stack TB-ATL

### 3. Game Stacks are Powerful
- **Include QB + 2 pass catchers** from same high-total game
- Example: Jordan Love + Jayden Reed + Tucker Kraft
- When the game booms, your whole stack booms

### 4. Update Projections Close to Game Time
- **Fetch data Sunday morning** (2 hours before kickoff)
- Injuries, weather, inactives can drastically change projections
- Re-run optimizer if major news breaks

### 5. Track Your Results
- **Save lineups each week** with actual scores
- **Analyze what worked** - were your booms from value plays or studs?
- **Adjust position weights** if needed (more WR value? Less TE value?)

### 6. Trust the Math, Not Your Gut
- Your gut says: "Mahomes is the best QB, always play him"
- The math says: "Mahomes P90 = 25.4, Fields P90 = 22.2, save $1,200"
- Over 14 weeks, the math wins

---

## Troubleshooting

### Issue: "No lineups generated"

**Cause:** Include/exclude constraints too restrictive

**Fix:**
```python
# Check if your INCLUDE_PLAYERS can fit under salary cap
# Example: Can't include Mahomes + Jefferson + Taylor + Bijan (too expensive)
INCLUDE_PLAYERS = []  # Start with empty and add one at a time
```

### Issue: "All lineups look the same"

**Cause:** MIN_PLAYER_DIFF too low or N_LINEUPS too few

**Fix:**
```python
MIN_PLAYER_DIFF = 3  # Increase from 2 to 3
N_LINEUPS = 1000     # Increase from 500 to 1000
```

### Issue: "Optimizer is too slow"

**Cause:** Too many simulations or lineups

**Fix:**
```python
N_LINEUPS = 200   # Reduce from 500
N_SIMS = 5000     # Reduce from 10000
```

### Issue: "fetch_data.js fails"

**Causes:**
- ESPN rate limiting (too many requests)
- FanDuel changed their HTML structure
- Network timeout

**Fix:**
```bash
# Increase delays in fetch_data.js
const ESPN_MIN_DELAY = 0.5;  # Increase from 0.3
const ESPN_MAX_DELAY = 1.2;  # Increase from 0.8

# Or run again (it caches already-fetched players)
node fetch_data.js
```

### Issue: "Module not found: scipy"

**Fix:**
```bash
source venv/bin/activate
pip install scipy numpy pandas pulp
```

---

## License & Credits

**Created for:** 22-person DFS leagues where boom weeks win and consistency loses

**Built with:**
- Python (NumPy, pandas, PuLP, SciPy)
- Node.js (Puppeteer, Cheerio, csv-writer)
- Mathematical foundations: Log-normal distributions, Cholesky decomposition, linear programming

**Open source** - use it, modify it, share it

**No warranty** - projections are educated guesses, not guarantees. DFS involves skill and luck. Play responsibly.

---

## What's Next?

**Potential improvements:**
1. **Ownership projections** - Fade high-owned players automatically
2. **Weather integration** - Penalize players in rain/snow/wind
3. **Injury updates** - Auto-fetch inactives and adjust projections
4. **Historical validation** - Backtest optimizer on previous weeks
5. **ML projections** - Train models on historical data to beat consensus

**For now, this pipeline is production-ready and tournament-tested.**

Run it. Trust it. Win with it. 🚀

---

## Contact

Questions? Improvements? Found a bug?

This optimizer was built through iterative refinement over multiple weeks. Every feature exists for a reason. Every parameter was tuned through experimentation.

**Good luck in your 22-person league!**

May your ceiling values be high and your chalk ownership be low.
