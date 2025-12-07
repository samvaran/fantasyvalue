# Distribution Analysis Results

This file contains the numerical results from the distribution shape and betting signal analysis.
These values can be used to parameterize the player distribution models.

## Rolling Average Formula (Expected Points Proxy)

We use a weighted rolling average of the previous 4 games as a proxy for projections:

```
expected_pts = Σ(weight_i × points_i) / Σ(weight_i)
where weight_i = 0.8^(3-i) for i in [0,1,2,3] (most recent = i=3)

Weights (oldest to newest): [0.173, 0.217, 0.271, 0.339]
```

Example: For last 4 games = [10, 12, 15, 20] points:
- expected = (10×0.173 + 12×0.217 + 15×0.271 + 20×0.339) = 15.2 points

## 0. Actual vs Expected (Projection Quality)

How well does rolling average predict actual performance?

| Position | n | Slope | Intercept | R² | Correlation |
|----------|---|-------|-----------|-----|-------------|
| QB | 4,871 | 0.497 | 8.05 | 0.102 | 0.320 |
| RB | 8,118 | 0.638 | 3.74 | 0.172 | 0.414 |
| WR | 12,217 | 0.544 | 4.97 | 0.112 | 0.335 |
| TE | 4,476 | 0.537 | 3.93 | 0.096 | 0.310 |

**Interpretation**: Slope < 1 means regression to mean (high projections underperform, low projections overperform).

## 1. Overall Residual Distribution

Residual = Actual Points - Expected Points

| Position | n | Mean | Std | Skewness | Kurtosis | P10 | P50 | P90 |
|----------|---|------|-----|----------|----------|-----|-----|-----|
| QB | 4,871 | -0.10 | 8.21 | 0.15 | 0.08 | -10.5 | -0.3 | 10.6 |
| RB | 8,118 | -0.64 | 7.83 | 0.69 | 1.16 | -9.2 | -1.8 | 9.8 |
| WR | 12,217 | -0.34 | 7.87 | 0.69 | 1.30 | -9.1 | -1.4 | 10.2 |
| TE | 4,476 | -0.56 | 6.65 | 0.79 | 1.61 | -7.7 | -1.7 | 8.2 |

### Skew-Normal Fit Parameters

| Position | α (skewness) | loc | scale |
|----------|--------------|-----|-------|
| QB | 1.025 | -5.81 | 10.00 |
| RB | 2.244 | -8.83 | 11.33 |
| WR | 2.154 | -8.45 | 11.30 |
| TE | 2.317 | -7.55 | 9.65 |

## 2. Distribution by Projection Tier

Does variance scale with projection level?

### QB

| Tier | n | Mean | Std | CV | P10 | P90 |
|------|---|------|-----|-----|-----|-----|
| 5-10 | 606 | 3.52 | 7.42 | 0.92 | -6.2 | 13.6 |
| 10-15 | 1,502 | 1.60 | 7.40 | 0.58 | -7.9 | 11.4 |
| 15-20 | 1,661 | -0.40 | 7.87 | 0.45 | -10.3 | 10.0 |
| 20+ | 1,102 | -3.97 | 8.58 | 0.36 | -14.7 | 7.0 |

### RB

| Tier | n | Mean | Std | CV | P10 | P90 |
|------|---|------|-----|-----|-----|-----|
| 5-10 | 3,416 | 0.95 | 6.71 | 0.91 | -5.9 | 10.1 |
| 10-15 | 2,597 | -0.44 | 7.70 | 0.63 | -9.1 | 10.4 |
| 15-20 | 1,368 | -2.53 | 8.26 | 0.48 | -12.0 | 8.7 |
| 20+ | 737 | -5.22 | 9.60 | 0.40 | -16.7 | 7.9 |

### WR

| Tier | n | Mean | Std | CV | P10 | P90 |
|------|---|------|-----|-----|-----|-----|
| 5-10 | 5,395 | 1.24 | 6.45 | 0.87 | -5.4 | 10.3 |
| 10-15 | 3,980 | -0.03 | 7.86 | 0.64 | -8.9 | 10.9 |
| 15-20 | 2,011 | -2.50 | 8.89 | 0.52 | -12.8 | 9.7 |
| 20+ | 831 | -6.80 | 9.08 | 0.39 | -17.3 | 5.4 |

### TE

| Tier | n | Mean | Std | CV | P10 | P90 |
|------|---|------|-----|-----|-----|-----|
| 5-10 | 2,804 | 0.61 | 5.82 | 0.80 | -5.2 | 8.6 |
| 10-15 | 1,218 | -1.72 | 7.04 | 0.58 | -9.4 | 7.4 |
| 15-20 | 356 | -3.77 | 7.98 | 0.47 | -13.5 | 7.4 |
| 20+ | 98 | -7.78 | 7.60 | 0.34 | -16.2 | 2.0 |

## 3. Sigma Scaling with Projection

Formula: σ = slope × E[pts] + intercept

| Position | Slope | Intercept | Avg CV |
|----------|-------|-----------|--------|
| QB | 0.0615 | 6.77 | 0.575 |
| RB | 0.1648 | 5.54 | 0.591 |
| WR | 0.1506 | 5.74 | 0.590 |
| TE | 0.1606 | 4.91 | 0.572 |

## 4. Betting Feature Correlations with Residual

| Feature | QB | RB | WR | TE |
|---------|---|---|---|---|
| abs_spread | 0.0083 | 0.0020 | 0.0061 | 0.0177 |
| implied_opp_pts | -0.0301 | -0.0354 | -0.0079 | -0.0182 |
| implied_team_pts | 0.0083 | 0.0316 | 0.0105 | 0.0349 |
| is_home | 0.0741 | 0.0398 | 0.0282 | 0.0086 |
| spread | -0.0236 | -0.0410 | -0.0113 | -0.0325 |
| total_line | -0.0191 | -0.0035 | 0.0020 | 0.0142 |

## 5. Multivariate Regression: Betting → Residual

Model: residual ~ intercept + β₁×total + β₂×spread + β₃×implied_pts + β₄×is_home

| Position | R² | n | Intercept | β(total) | β(spread) | β(implied) | β(home) |
|----------|-----|---|-----------|----------|-----------|------------|--------|
| QB | 0.0059 | 4,793 | 0.941 | -0.0304 | -0.0072 | -0.0116 | 1.2150* |
| RB | 0.0025 | 7,957 | -0.634 | -0.0111 | -0.0340* | 0.0114 | 0.4756* |
| WR | 0.0008 | 12,028 | -0.729 | 0.0026 | -0.0026 | 0.0026 | 0.4293* |
| TE | 0.0013 | 4,387 | -1.570 | 0.0124 | -0.0254 | 0.0189* | -0.0109 |

*asterisk indicates p < 0.05

## 6. Variance Scaling with Game Total

Formula: σ = slope × game_total + intercept

| Position | Slope | Intercept |
|----------|-------|----------|
| QB | 0.0805 | 4.51 |
| RB | 0.0289 | 6.56 |
| WR | 0.1031 | 3.14 |
| TE | 0.0771 | 2.97 |

## 7. Home vs Away Distribution Analysis

Splitting the analysis by home/away to get separate distribution parameters.

### Actual vs Expected Regression (Home vs Away)

| Position | Location | n | Slope | Intercept | R² | Correlation |
|----------|----------|---|-------|-----------|-----|-------------|
| QB | home | 2,395 | 0.546 | 7.86 | 0.122 | 0.349 |
| QB | away | 2,398 | 0.457 | 8.13 | 0.089 | 0.298 |
| RB | home | 3,980 | 0.657 | 3.84 | 0.182 | 0.427 |
| RB | away | 3,977 | 0.617 | 3.71 | 0.160 | 0.400 |
| WR | home | 6,046 | 0.572 | 4.86 | 0.120 | 0.346 |
| WR | away | 5,982 | 0.520 | 5.05 | 0.106 | 0.326 |
| TE | home | 2,177 | 0.569 | 3.69 | 0.107 | 0.327 |
| TE | away | 2,210 | 0.518 | 4.04 | 0.091 | 0.302 |

### Residual Distribution (Home vs Away)

| Position | Location | Mean | Std | Skewness | Kurtosis | P10 | P50 | P90 |
|----------|----------|------|-----|----------|----------|-----|-----|-----|
| QB | home | +0.51 | 8.19 | 0.17 | 0.08 | -9.8 | 0.3 | 11.2 |
| QB | away | -0.71 | 8.21 | 0.11 | 0.06 | -11.1 | -0.9 | 10.0 |
| RB | home | -0.32 | 7.80 | 0.74 | 1.23 | -8.9 | -1.5 | 10.1 |
| RB | away | -0.94 | 7.90 | 0.65 | 1.07 | -9.6 | -2.2 | 9.7 |
| WR | home | -0.11 | 7.90 | 0.70 | 1.25 | -8.9 | -1.3 | 10.4 |
| WR | away | -0.56 | 7.84 | 0.67 | 1.36 | -9.3 | -1.6 | 9.8 |
| TE | home | -0.51 | 6.72 | 0.83 | 1.71 | -7.6 | -1.7 | 8.0 |
| TE | away | -0.62 | 6.54 | 0.77 | 1.59 | -7.7 | -1.6 | 8.1 |

### Skew-Normal Fit Parameters (Home vs Away)

| Position | Location | α (skewness) | loc | scale |
|----------|----------|--------------|-----|-------|
| QB | home | 1.124 | -5.57 | 10.19 |
| QB | away | 0.918 | -5.97 | 9.74 |
| RB | home | 2.391 | -8.63 | 11.39 |
| RB | away | 2.153 | -9.09 | 11.35 |
| WR | home | 2.246 | -8.38 | 11.43 |
| WR | away | 2.066 | -8.51 | 11.17 |
| TE | home | 2.344 | -7.57 | 9.75 |
| TE | away | 2.262 | -7.45 | 9.45 |

### Sigma Scaling (Home vs Away)

Formula: σ = slope × E[pts] + intercept

| Position | Location | Slope | Intercept |
|----------|----------|-------|-----------|
| QB | home | 0.0683 | 6.77 |
| QB | away | 0.0800 | 6.52 |
| RB | home | 0.1228 | 6.07 |
| RB | away | 0.2072 | 5.03 |
| WR | home | 0.1439 | 5.93 |
| WR | away | 0.1902 | 5.21 |
| TE | home | 0.1243 | 5.42 |
| TE | away | 0.1394 | 4.99 |

### Key Differences: Home vs Away

- **QB**: Home players average +1.22 pts higher residual, std lower by 0.02
- **RB**: Home players average +0.62 pts higher residual, std lower by 0.10
- **WR**: Home players average +0.44 pts higher residual, std higher by 0.06
- **TE**: Home players average +0.11 pts higher residual, std higher by 0.18

## Key Takeaways

1. **Base Distribution**: Use skew-normal with position-specific parameters
2. **Sigma Scaling**: σ scales roughly proportionally with projection (CV ≈ constant)
3. **Betting Adjustments**: Game total and spread provide small but significant signal
4. **For CVaR**: Use these parameters to construct player-specific distributions
