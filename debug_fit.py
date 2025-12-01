"""Debug specific distribution fits."""
import pandas as pd
import numpy as np

# Load integrated data
df = pd.read_csv('data/intermediate/players_integrated.csv')

# Check Josh Allen defensive scenario
josh = df[df['name'] == 'josh allen'].iloc[0]

print("=" * 80)
print("JOSH ALLEN DEFENSIVE SCENARIO")
print("=" * 80)
print(f"\nConsensus: {josh['fpProjPts']:.2f}")
print(f"\nOriginal floor/ceiling (ESPN):")
print(f"  Floor:   {josh['espnLowScore']:.2f}")
print(f"  Ceiling: {josh['espnHighScore']:.2f}")
print(f"\nDefensive scenario distribution params:")
print(f"  mu:    {josh['mu_defensive']:.4f}")
print(f"  sigma: {josh['sigma_defensive']:.4f}")
print(f"  shift: {josh['shift_defensive']:.4f}")

# Calculate theoretical mean from these parameters
fitted_mean = np.exp(josh['mu_defensive'] + josh['sigma_defensive']**2 / 2) + josh['shift_defensive']
Z10 = -1.2816
Z90 = 1.2816
fitted_p10 = np.exp(josh['mu_defensive'] + Z10 * josh['sigma_defensive']) + josh['shift_defensive']
fitted_p90 = np.exp(josh['mu_defensive'] + Z90 * josh['sigma_defensive']) + josh['shift_defensive']

print(f"\nFitted distribution properties:")
print(f"  Mean:  {fitted_mean:.2f}")
print(f"  P10:   {fitted_p10:.2f}")
print(f"  P90:   {fitted_p90:.2f}")

# Check what floor/ceiling were used
# Defensive: floor 0.75, ceiling 1.15
floor_mult = 0.75
ceiling_mult = 1.15
floor_val = josh['espnLowScore'] * floor_mult
ceiling_val = josh['espnHighScore'] * ceiling_mult

print(f"\nTarget floor/ceiling for fitting:")
print(f"  Floor (P10):   {floor_val:.2f}  (ESPN {josh['espnLowScore']:.2f} * {floor_mult:.2f})")
print(f"  Ceiling (P90): {ceiling_val:.2f}  (ESPN {josh['espnHighScore']:.2f} * {ceiling_mult:.2f})")

print(f"\nConstraint compatibility check:")
print(f"  Floor < Consensus < Ceiling? {floor_val:.2f} < {josh['fpProjPts']:.2f} < {ceiling_val:.2f}")
if floor_val < josh['fpProjPts'] < ceiling_val:
    print("  ✓ Constraints are compatible!")
else:
    print("  ✗ Constraints are INCOMPATIBLE!")
    
# Check what got stored in floor_defensive, ceiling_defensive
print(f"\nStored floor/ceiling values:")
print(f"  floor_defensive:   {josh['floor_defensive']:.2f}")
print(f"  ceiling_defensive: {josh['ceiling_defensive']:.2f}")
    
# Check competitive scenario too
print("\n" + "=" * 80)
print("COMPETITIVE SCENARIO")
print("=" * 80)
floor_mult_comp = 0.85
ceiling_mult_comp = 1.25
floor_val_comp = josh['espnLowScore'] * floor_mult_comp
ceiling_val_comp = josh['espnHighScore'] * ceiling_mult_comp

print(f"Target floor/ceiling:")
print(f"  Floor (P10):   {floor_val_comp:.2f}")
print(f"  Ceiling (P90): {ceiling_val_comp:.2f}")
print(f"  Consensus:     {josh['fpProjPts']:.2f}")

if floor_val_comp < josh['fpProjPts'] < ceiling_val_comp:
    print("  ✓ Constraints are compatible!")
else:
    print("  ✗ Constraints are INCOMPATIBLE!")

fitted_mean_comp = np.exp(josh['mu_competitive'] + josh['sigma_competitive']**2 / 2) + josh['shift_competitive']
fitted_p10_comp = np.exp(josh['mu_competitive'] + Z10 * josh['sigma_competitive']) + josh['shift_competitive']
fitted_p90_comp = np.exp(josh['mu_competitive'] + Z90 * josh['sigma_competitive']) + josh['shift_competitive']

print(f"\nFitted distribution properties:")
print(f"  Mean:  {fitted_mean_comp:.2f}")
print(f"  P10:   {fitted_p10_comp:.2f}")
print(f"  P90:   {fitted_p90_comp:.2f}")

print(f"\nStored floor/ceiling values:")
print(f"  floor_competitive:   {josh['floor_competitive']:.2f}")
print(f"  ceiling_competitive: {josh['ceiling_competitive']:.2f}")
