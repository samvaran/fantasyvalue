# Code Cleanup: Removed Backwards Compatibility

## What Was Removed

### 1. JavaScript (fetch_data.js)

**Removed:**
- `sigma` variable (average of sigma_lower and sigma_upper)
- `uncertainty` variable (duplicate of sigma)

**Before:**
```javascript
const sigma = (sigma_upper + sigma_lower) / 2;
uncertainty = parseFloat(sigma.toFixed(3));

return {
  // ...
  sigma,        // Average sigma (for backwards compatibility)
  uncertainty,  // Keep for backwards compatibility (same as sigma)
  // ...
}
```

**After:**
```javascript
return {
  // ...
  mu,           // Location parameter
  sigma_lower,  // Downside variance (P0-P50)
  sigma_upper,  // Upside variance (P50-P100)
  // ...
}
```

### 2. Python (league_optimizer.py)

**Removed:**
- All fallback code for missing distribution parameters
- Position-specific uncertainty calculations
- Single sigma backwards compatibility
- `uncertainty` field from results

**Before:**
```python
if pd.notna(row.get('mu')) and pd.notna(row.get('sigma')):
    mu = row['mu']
    sigma = row['sigma']
    uncertainty = sigma
else:
    # Fallback: Calculate from position-specific uncertainty
    if row['position'] == 'QB':
        uncertainty = consensus * 0.30
    # ... etc
```

**After:**
```python
if pd.notna(row.get('mu')) and pd.notna(row.get('sigma_upper')):
    mu = row['mu']
    sigma_upper = row['sigma_upper']
    p90 = np.exp(mu + sigma_upper * z90)
else:
    raise ValueError("Missing required parameters. Run: node fetch_data.js")
```

## Why This Is Better

### 1. **Enforces Proper Workflow**
- ✅ You MUST run `node fetch_data.js` to generate the CSV
- ✅ No silent fallbacks that hide problems
- ✅ Clear error messages if data is missing

### 2. **Cleaner Data Schema**
```csv
# Old (messy):
name,consensus,mu,sigma,sigma_lower,sigma_upper,uncertainty,...
                    ^^^^                        ^^^^^^^^^^^
                 (redundant with uncertainty)   (redundant with sigma)

# New (clean):
name,consensus,mu,sigma_lower,sigma_upper,...
                  ^^^^^^^^^^^  ^^^^^^^^^^^
                  (only what we need!)
```

### 3. **Simpler Code**
- Removed ~60 lines of fallback/compatibility code
- Single code path: spliced distribution only
- Easier to understand and maintain

### 4. **Forces Best Practices**
- Can't accidentally use generic position-based variance
- Must use the sophisticated game-environment distributions
- No "ham-handed" averaging of sigmas

## New Workflow

### Step 1: Generate CSV with Spliced Distribution
```bash
node fetch_data.js
```

This creates `knapsack.csv` with:
- `mu` - Location parameter
- `sigma_lower` - Downside variance (P0-P50)
- `sigma_upper` - Upside variance (P50-P100)
- `floorVariance` - Player archetype metadata
- `ceilingVariance` - Player archetype metadata

### Step 2: Run Optimizer
```bash
python3 league_optimizer.py
```

This:
- ✅ Reads `mu`, `sigma_lower`, `sigma_upper` from CSV
- ✅ Uses spliced distribution for all simulations
- ❌ Fails with clear error if parameters missing

### Step 3: Review Results
```bash
cat LEAGUE_LINEUPS.csv
```

Results now have **proper variance** from spliced distributions!

## Error Handling

If you try to run the optimizer without regenerating the CSV:

```
ValueError: Player 'christian mccaffrey' missing required distribution
parameters (mu, sigma_lower, sigma_upper). Please regenerate knapsack.csv
with: node fetch_data.js
```

**Clear, actionable error message!** ✅

## Migration Path

### Old CSV (Won't Work):
```csv
name,consensus,mu,sigma,uncertainty,...
drake maye,18.88,2.937,0.294,0.33,...
```

Running optimizer → **Error** (missing sigma_lower, sigma_upper)

### New CSV (Required):
```csv
name,consensus,mu,sigma_lower,sigma_upper,...
drake maye,18.88,2.937,0.398,0.654,...
```

Running optimizer → **Success** (has all required fields)

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| CSV columns | 5+ redundant | 3 essential |
| Fallback code | ~60 lines | 0 lines |
| Error clarity | Silent failures | Clear messages |
| Workflow enforcement | Optional | Required |
| Distribution type | Mixed (single/spliced) | Pure spliced |
| Code complexity | High | Low |
| Maintenance burden | High | Low |

---

**Bottom line**: Cleaner, simpler, forces the right workflow, no backwards compatibility baggage! 🎯
