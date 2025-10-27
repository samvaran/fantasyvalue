#!/bin/bash

# Create _OLD directory if it doesn't exist
mkdir -p _OLD

echo "Moving old/unused files to _OLD folder..."
echo ""

# OLD DOCUMENTATION (replaced by README.md)
mv -v CEILING_VALUE_GUIDE.md _OLD/ 2>/dev/null
mv -v LEAGUE_OPTIMIZER_GUIDE.md _OLD/ 2>/dev/null
mv -v LEAGUE_OPTIMIZER_README.md _OLD/ 2>/dev/null
mv -v MONTE_CARLO_SUMMARY.md _OLD/ 2>/dev/null
mv -v MONTE_CARLO_VALUE.md _OLD/ 2>/dev/null
mv -v SIMULATION_TODO.md _OLD/ 2>/dev/null
mv -v STRATEGY.md _OLD/ 2>/dev/null

# OLD OPTIMIZER VERSIONS (replaced by league_optimizer.py)
mv -v knapsack_solver.js _OLD/ 2>/dev/null
mv -v knapsack_solver.py _OLD/ 2>/dev/null
mv -v knapsack_solver_diverse.py _OLD/ 2>/dev/null
mv -v simulation_optimizer.py _OLD/ 2>/dev/null
mv -v tournament_optimizer.py _OLD/ 2>/dev/null
mv -v ceiling_optimizer.py _OLD/ 2>/dev/null
mv -v calculate_ceiling_values.py _OLD/ 2>/dev/null
mv -v monte_carlo_optimizer.py _OLD/ 2>/dev/null

# OLD OUTPUT FILES (from previous versions)
mv -v LINEUPS.csv _OLD/ 2>/dev/null
mv -v LINEUPS_DIVERSE.csv _OLD/ 2>/dev/null
mv -v LINEUPS_TOURNAMENT.csv _OLD/ 2>/dev/null
mv -v lineups_cash.csv _OLD/ 2>/dev/null
mv -v lineups_ceiling.csv _OLD/ 2>/dev/null
mv -v lineups_initial.csv _OLD/ 2>/dev/null
mv -v lineups_league.csv _OLD/ 2>/dev/null
mv -v lineups_simulated_all.csv _OLD/ 2>/dev/null
mv -v simulation_comparison.csv _OLD/ 2>/dev/null
mv -v monte_carlo_run.log _OLD/ 2>/dev/null

# OLD FETCHER SCRIPTS (replaced by fetch_data.js)
mv -v fetch_espn_projections.js _OLD/ 2>/dev/null
mv -v fetch_espn_projections.py _OLD/ 2>/dev/null
mv -v fetch_game_lines.js _OLD/ 2>/dev/null
mv -v parse_game_lines.js _OLD/ 2>/dev/null
mv -v test-game-lines.js _OLD/ 2>/dev/null
mv -v fantasyvalue.js _OLD/ 2>/dev/null
mv -v test.js _OLD/ 2>/dev/null
mv -v scratch.js _OLD/ 2>/dev/null

# OLD/UNUSED DATA FILES
mv -v knapsack_with_espn.csv _OLD/ 2>/dev/null
mv -v player_ceiling_values.csv _OLD/ 2>/dev/null
mv -v actual_results.csv _OLD/ 2>/dev/null
mv -v projection_stats.csv _OLD/ 2>/dev/null
mv -v regression_models.csv _OLD/ 2>/dev/null
mv -v fd_week_results.json _OLD/ 2>/dev/null
mv -v sample_copy_paste_text.txt _OLD/ 2>/dev/null
mv -v game_lines_manual.csv _OLD/ 2>/dev/null

# DRAFTKINGS DATA (not used, only using FanDuel)
mv -v draftkings-data.json _OLD/ 2>/dev/null
mv -v draftkings-sample.html _OLD/ 2>/dev/null
mv -v draftkings_extracted.txt _OLD/ 2>/dev/null
mv -v draftkings_initial_state.json _OLD/ 2>/dev/null
mv -v draftkings_output.txt _OLD/ 2>/dev/null
mv -v draftkings_output2.txt _OLD/ 2>/dev/null
mv -v draftkings_raw.html _OLD/ 2>/dev/null

# ESPN SAMPLE DATA (not needed)
mv -v jamarr-chase-sample-espn-projections.json _OLD/ 2>/dev/null

# OLD PACKAGE FILE
mv -v package_OLD.json _OLD/ 2>/dev/null

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "CURRENT PRODUCTION FILES:"
echo "========================="
echo ""
echo "📊 Data Pipeline:"
echo "  - fetch_data.js (fetches projections, odds, game lines)"
echo ""
echo "🎯 Optimizer:"
echo "  - league_optimizer.py (one script to rule them all)"
echo ""
echo "📄 Documentation:"
echo "  - README.md (complete pipeline guide)"
echo ""
echo "📁 Current Data Files:"
echo "  - knapsack.csv (player data)"
echo "  - game_lines.csv (game environment)"
echo "  - td_odds.json (anytime TD odds)"
echo "  - espn_players.csv/.json (ESPN projections)"
echo "  - espn_projections.json (converted projections)"
echo "  - fanduel-*.csv (position-specific data)"
echo ""
echo "📈 Current Outputs:"
echo "  - LEAGUE_LINEUPS.csv (your optimized lineups)"
echo ""
echo "🗂️  Old files moved to: _OLD/"
