# Game Score Distribution Analysis Results

Distribution parameters for sampling game outcomes in CVaR simulations.

## Sign Convention

```
spread_line = predicted home margin (from nflreadpy)
  positive = home team favored
  negative = away team favored
```

## 1. Total Points Residual

```
total_residual = actual_total - total_line
```

| Statistic | Value |
|-----------|-------|
| n | 2,476 |
| Mean | 0.31 |
| Std | 13.22 |
| Skewness | 0.325 |
| P10 | -15.5 |
| P50 | -0.5 |
| P90 | 17.5 |

**Normal fit**: μ = 0.31, σ = 13.22

## 2. Margin Residual (Home Team Perspective)

```
predicted_margin = spread_line  (positive = home favored)
actual_margin = home_score - away_score
margin_residual = actual_margin - predicted_margin
```

| Statistic | Value |
|-----------|-------|
| n | 2,476 |
| Mean | 0.04 |
| Std | 12.75 |
| Skewness | 0.107 |
| P10 | -15.5 |
| P50 | 0.0 |
| P90 | 16.5 |

**Normal fit**: μ = 0.04, σ = 12.75

## 3. Correlation Between Residuals

- **r = 0.026** (can sample independently)

## 4. Sampling Formula for CVaR

```python
# For each game simulation:
total_residual = np.random.normal(0.31, 13.22)
margin_residual = np.random.normal(0.04, 12.75)

sampled_total = total_line + total_residual
sampled_margin = spread_line + margin_residual  # home perspective

# Calculate team scores:
home_score = (sampled_total + sampled_margin) / 2
away_score = (sampled_total - sampled_margin) / 2
```
