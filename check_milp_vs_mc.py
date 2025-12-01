"""Check MILP projection vs MC mean."""
import pandas as pd

df = pd.read_csv('outputs/run_20251130_191609/evaluations.csv')

print("=" * 80)
print("MILP PROJECTION vs MC MEAN COMPARISON")
print("=" * 80)
print()

# Calculate difference
df['diff'] = df['mean'] - df['milp_projection']
df['diff_pct'] = (df['diff'] / df['milp_projection'] * 100)

print(f"Sample of top 10 lineups:")
print(df[['lineup_id', 'milp_projection', 'mean', 'diff', 'diff_pct']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"  Mean MILP projection: {df['milp_projection'].mean():.2f}")
print(f"  Mean MC mean:         {df['mean'].mean():.2f}")
print(f"  Average difference:   {df['diff'].mean():.2f} ({df['diff_pct'].mean():.1f}%)")
print(f"  Std dev of diff:      {df['diff'].std():.2f}")
print(f"  Min diff:             {df['diff'].min():.2f} ({df['diff_pct'].min():.1f}%)")
print(f"  Max diff:             {df['diff'].max():.2f} ({df['diff_pct'].max():.1f}%)")

if abs(df['diff_pct'].mean()) < 2.0:
    print("\n  ✓ SUCCESS: MILP and MC are well-aligned (within 2%)!")
else:
    print(f"\n  ✗ ISSUE: MILP and MC differ by {df['diff_pct'].mean():.1f}% on average")
