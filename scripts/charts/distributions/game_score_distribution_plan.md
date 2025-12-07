# Game Score Distribution Analysis Plan

## Objective

Create distribution models for game scores (total points and margin) to enable realistic game script sampling in CVaR simulations.

## Why This Matters

We found:
- Betting lines → player residuals: weak signal (r ≈ 0.03)
- Actual game outcomes → player residuals: strong signal (r ≈ 0.15-0.49)

By sampling realistic game outcomes from betting lines, then using the strong actual→residual equations, we recover the full signal strength with uncertainty naturally baked in.

## Analysis Steps

### 1. Game Total Distribution
- Residual: `total_residual = actual_total - total_line`
- Fit distribution (normal or skew-normal)
- Check if σ scales with total_line
- Check for any systematic bias

### 2. Game Margin Distribution
- Note: spread predicts margin from favorite's perspective
- For home team: `margin_residual = (home_score - away_score) - (-spread_line)`
- Fit distribution
- Check σ scaling with spread magnitude

### 3. Joint Distribution / Correlation
- Check correlation between total_residual and margin_residual
- Determine if we can sample independently or need joint sampling

### 4. Output
- Distribution parameters for total and margin
- Charts showing fits
- Markdown documentation of results

## Pipeline Integration

```
CVaR Simulation Step:
1. For each game: sample (total, margin) from fitted distributions
2. Calculate team_score = (total + margin) / 2 for home team
3. Use team_score → player_residual equations to adjust μ
4. Sample player outcomes from adjusted distributions
5. MILP optimizes for P90 across simulations
```
