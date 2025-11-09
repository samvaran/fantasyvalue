const fs = require('fs');
const csv = require('csv-parse/sync');

const data = fs.readFileSync('knapsack.csv', 'utf8');
const players = csv.parse(data, { columns: true });

// Filter to players with complete data
const complete = players.filter(p =>
  p.consensus && p.p90 && p.fpProjPts &&
  p.espnLowScore && p.espnHighScore &&
  p.tdProbability && parseFloat(p.tdProbability) > 10
);

console.log('\n=== DETERMINISTIC MODEL VALIDATION ===\n');
console.log('Players with TD Probability > 10% and complete data:\n');

complete.slice(0, 10).forEach(p => {
  const consensus = parseFloat(p.consensus);
  const fpProj = parseFloat(p.fpProjPts);
  const p90 = parseFloat(p.p90);
  const low = parseFloat(p.espnLowScore);
  const high = parseFloat(p.espnHighScore);
  const tdProb = parseFloat(p.tdProbability);

  // Calculate how much consensus moved from base
  const consensusDiff = consensus - fpProj;
  const consensusPct = ((consensusDiff / fpProj) * 100).toFixed(1);

  // Calculate floor and ceiling variance
  const floorVar = ((consensus - low) / consensus).toFixed(2);
  const ceilVar = ((high - consensus) / consensus).toFixed(2);

  // Calculate P90 boost over consensus
  const p90Boost = ((p90 / consensus - 1) * 100).toFixed(1);

  console.log(`${p.name.padEnd(20)} ${p.position} | TD: ${tdProb.toFixed(0)}%`);
  console.log(`  Consensus: ${consensus.toFixed(1)} (${consensusPct > 0 ? '+' : ''}${consensusPct}% vs FP ${fpProj})`);
  console.log(`  Floor Var: ${floorVar} | Ceil Var: ${ceilVar}`);
  console.log(`  P90: ${p90.toFixed(1)} (+${p90Boost}% over consensus)`);
  console.log('');
});

// Summary stats
const avgConsensusDiff = complete.reduce((sum, p) => {
  return sum + Math.abs(parseFloat(p.consensus) - parseFloat(p.fpProjPts));
}, 0) / complete.length;

console.log(`\nAverage absolute consensus difference from FP: ${avgConsensusDiff.toFixed(2)} pts`);
console.log('(Lower is better - means we\'re locking consensus as intended)\n');
