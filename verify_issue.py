"""Verify the core issue with current approach."""
import pandas as pd

df = pd.read_csv('data/intermediate/players_integrated.csv')

# Josh Allen example
josh = df[df['name'] == 'josh allen'].iloc[0]

print("=" * 80)
print("JOSH ALLEN - DEFENSIVE vs SHOOTOUT COMPARISON")
print("=" * 80)
print(f"\nConsensus: {josh['fpProjPts']:.2f}")

scenarios = ['defensive', 'shootout', 'competitive']
for scenario in scenarios:
    floor = josh[f'floor_{scenario}']
    ceiling = josh[f'ceiling_{scenario}']
    midpoint = (floor + ceiling) / 2
    range_val = ceiling - floor
    
    print(f"\n{scenario.upper()}")
    print(f"  Floor:    {floor:.2f}")
    print(f"  Ceiling:  {ceiling:.2f}")
    print(f"  Midpoint: {midpoint:.2f}")
    print(f"  Range:    {range_val:.2f}")

print("\n" + "=" * 80)
print("THE PROBLEM")
print("=" * 80)
print("\nCurrent approach: Game scripts only change VARIANCE (range), not LOCATION")
print("  - Defensive has narrower range than shootout ✓")
print("  - But BOTH have same midpoint (≈22-23) ✓")
print("  - So BOTH have same mean when fitted! ✗")
print("\nWhat we ACTUALLY want:")
print("  - Shootout (BOOM for QB): HIGHER mean, WIDER range")
print("  - Defensive (BUST for QB): LOWER mean, NARROWER range")
print("  - This requires shifting the entire distribution, not just changing variance")
