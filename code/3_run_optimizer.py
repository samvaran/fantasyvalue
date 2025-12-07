"""
CVaR-MILP DFS Lineup Optimizer

Generates optimal DFS lineups by directly optimizing for ceiling (p80) using
Conditional Value at Risk (CVaR) formulation.

Pipeline:
1. Load player data and game scripts
2. Generate scenario matrix (vectorized Monte Carlo)
3. Generate diverse lineups using CVaR-MILP with anchored strategies

Diversity Strategies:
- QB-anchored: Force each top QB, optimize rest
- DEF-anchored: Force each DEF, optimize rest
- RB-anchored: Force each top RB, optimize rest
- WR-anchored: Force each top WR, optimize rest
- General: No anchor, just exclude previous lineups

Usage:
    python 3_run_optimizer.py --week-dir data/2025_12_01
    python 3_run_optimizer.py --week-dir data/2025_12_01 --n-lineups 50 --scenarios 500
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import sys
from tqdm import tqdm

# Import configuration
from config import (
    SALARY_CAP, FORCE_INCLUDE, EXCLUDE_PLAYERS,
    DEFAULT_N_LINEUPS, DEFAULT_N_SCENARIOS, DEFAULT_CVAR_ALPHA,
    DEFAULT_SOLVER_TIME_LIMIT, PLAYERS_INTEGRATED, GAME_SCRIPTS, OUTPUTS_DIR,
    GENERAL_ONLY, PARALLEL_QB_ANCHOR, N_TOP_QBS, N_LINEUPS_PER_ANCHOR
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add optimizer to path
sys.path.insert(0, str(Path(__file__).parent / 'optimizer'))

from optimizer.scenario_generator import generate_scenario_matrix
from optimizer.scenario_generator_v2 import generate_scenario_matrix_v2
from optimizer.cvar_optimizer import optimize_lineup_cvar


def _generate_lineups_for_qb(args):
    """
    Worker function for parallel QB-anchored lineup generation.
    Must be at module level for multiprocessing.

    Args:
        args: Tuple of (qb_id, qb_name, n_lineups, players_df, scenario_matrix,
              player_index, alpha, time_limit)

    Returns:
        List of lineup dicts for this QB
    """
    (qb_id, qb_name, n_lineups, players_df, scenario_matrix,
     player_index, alpha, time_limit) = args

    lineups = []
    excluded_lineups = []  # Only track exclusions within this QB's lineups

    for i in range(n_lineups):
        lineup = optimize_lineup_cvar(
            players_df=players_df,
            scenario_matrix=scenario_matrix,
            player_index=player_index,
            alpha=alpha,
            salary_cap=SALARY_CAP,
            force_include=[qb_id] + list(FORCE_INCLUDE),
            exclude_players=list(EXCLUDE_PLAYERS),
            exclude_lineups=excluded_lineups,
            time_limit=time_limit,
            verbose=False
        )

        if lineup:
            lineup['strategy'] = 'qb_anchor'
            lineup['anchor_player'] = qb_name
            lineups.append(lineup)
            excluded_lineups.append(lineup['player_ids'])
        else:
            break  # Can't generate more unique lineups for this QB

    return lineups


class CVaROptimizer:
    """Orchestrates CVaR-MILP lineup optimization with diversity strategies."""

    def __init__(
        self,
        week_dir: str = None,
        run_name: str = None
    ):
        # Week directory setup
        if week_dir:
            week_path = Path(week_dir)
            self.players_path = week_path / 'intermediate' / '1_players.csv'
            self.game_scripts_path = week_path / 'intermediate' / '2_game_scripts.csv'
            self.output_dir = week_path / 'outputs'
        else:
            # Fallback to old paths
            self.players_path = Path(PLAYERS_INTEGRATED)
            self.game_scripts_path = Path(GAME_SCRIPTS)
            self.output_dir = Path(OUTPUTS_DIR)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create run-specific directory
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'run_{run_name}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Run directory: {self.run_dir}")

    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80 + "\n")

    def load_data(self):
        """Load player and game script data."""
        print("Loading data...")

        if not self.players_path.exists():
            raise FileNotFoundError(f"Players file not found: {self.players_path}")
        if not self.game_scripts_path.exists():
            raise FileNotFoundError(f"Game scripts file not found: {self.game_scripts_path}")

        self.players_df = pd.read_csv(self.players_path)
        self.game_scripts_df = pd.read_csv(self.game_scripts_path)

        print(f"  Loaded {len(self.players_df)} players")
        print(f"  Loaded {len(self.game_scripts_df)} games")

        # Show position breakdown
        pos_counts = self.players_df['position'].value_counts()
        print(f"  Positions: {dict(pos_counts)}")

    def generate_scenarios(self, n_scenarios: int, seed: int = None, use_v2: bool = True):
        """Generate scenario matrix for CVaR optimization.

        Args:
            n_scenarios: Number of scenarios to generate
            seed: Random seed for reproducibility
            use_v2: If True, use v2 generator with game-score-based sampling.
                   If False, use v1 generator with categorical game scripts.
        """
        print(f"\nGenerating {n_scenarios} scenarios (v2={use_v2})...")

        start = time.time()
        if use_v2:
            # V2: Game-score-based sampling with player adjustments
            self.scenario_matrix, self.player_index = generate_scenario_matrix_v2(
                self.players_df,
                self.game_scripts_df,
                n_scenarios=n_scenarios,
                seed=seed,
                use_game_adjustments=True
            )
        else:
            # V1: Categorical game script sampling
            self.scenario_matrix, self.player_index = generate_scenario_matrix(
                self.players_df,
                self.game_scripts_df,
                n_scenarios=n_scenarios,
                seed=seed
            )
        elapsed = time.time() - start

        print(f"  Scenario matrix shape: {self.scenario_matrix.shape}")
        print(f"  Generation time: {elapsed:.2f}s")

        # Show some stats
        mean_scores = self.scenario_matrix.mean(axis=1)
        print(f"  Mean player score: {mean_scores.mean():.2f} (std: {mean_scores.std():.2f})")

    def generate_anchored_lineups(
        self,
        position: str,
        n_top: int,
        strategy_name: str,
        alpha: float,
        time_limit: int,
        existing_lineups: list,
        verbose: bool = True
    ) -> list:
        """Generate lineups anchored to top players at a position."""
        lineups = []

        # Get top players at position by projection
        pos_players = self.players_df[self.players_df['position'] == position].copy()
        pos_players = pos_players.nlargest(n_top, 'fpProjPts')

        if verbose:
            print(f"\n--- {strategy_name.upper()} ({len(pos_players)} players) ---")

        iterator = tqdm(pos_players.iterrows(), total=len(pos_players),
                       desc=strategy_name, disable=not verbose)

        for _, player in iterator:
            player_id = str(player['id'])

            lineup = optimize_lineup_cvar(
                players_df=self.players_df,
                scenario_matrix=self.scenario_matrix,
                player_index=self.player_index,
                alpha=alpha,
                salary_cap=SALARY_CAP,
                force_include=[player_id] + list(FORCE_INCLUDE),
                exclude_players=list(EXCLUDE_PLAYERS),
                exclude_lineups=[l['player_ids'] for l in existing_lineups + lineups],
                time_limit=time_limit,
                verbose=False
            )

            if lineup:
                lineup['strategy'] = strategy_name
                lineup['anchor_player'] = player['name']
                lineup['lineup_id'] = len(existing_lineups) + len(lineups)
                lineups.append(lineup)

        if verbose:
            print(f"  Generated {len(lineups)} {strategy_name} lineups")

        return lineups

    def generate_qb_anchored_parallel(
        self,
        n_top_qbs: int,
        n_lineups_per_qb: int,
        alpha: float,
        time_limit: int,
        verbose: bool = True,
        max_workers: int = None
    ) -> list:
        """
        Generate lineups anchored on top QBs in parallel.

        Each QB gets n_lineups_per_qb lineups with uniqueness constraints
        only within that QB's group (not across all lineups).

        Args:
            n_top_qbs: Number of top QBs to anchor on
            n_lineups_per_qb: Number of lineups per QB
            alpha: CVaR tail probability
            time_limit: Solver time limit per lineup
            verbose: Print progress
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            List of all generated lineups
        """
        # Get top QBs by projection
        qb_players = self.players_df[self.players_df['position'] == 'QB'].copy()
        top_qbs = qb_players.nlargest(n_top_qbs, 'fpProjPts')

        if verbose:
            print(f"\n--- PARALLEL QB-ANCHORED ({len(top_qbs)} QBs x {n_lineups_per_qb} lineups) ---")
            print(f"  Top QBs: {', '.join(top_qbs['name'].tolist())}")

        # Prepare args for each QB
        qb_args = []
        for _, qb in top_qbs.iterrows():
            qb_args.append((
                str(qb['id']),
                qb['name'],
                n_lineups_per_qb,
                self.players_df,
                self.scenario_matrix,
                self.player_index,
                alpha,
                time_limit
            ))

        # Run in parallel
        if max_workers is None:
            max_workers = min(len(qb_args), multiprocessing.cpu_count())

        all_lineups = []

        if verbose:
            print(f"  Using {max_workers} parallel workers...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_generate_lineups_for_qb, args): args[1] for args in qb_args}

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="QBs", disable=not verbose) as pbar:
                for future in as_completed(futures):
                    qb_name = futures[future]
                    try:
                        lineups = future.result()
                        all_lineups.extend(lineups)
                        if verbose:
                            pbar.set_postfix({'last': qb_name, 'lineups': len(lineups)})
                    except Exception as e:
                        if verbose:
                            tqdm.write(f"  Error for {qb_name}: {e}")
                    pbar.update(1)

        if verbose:
            print(f"  Generated {len(all_lineups)} total lineups from {len(top_qbs)} QBs")

        return all_lineups

    def generate_general_lineups(
        self,
        n_lineups: int,
        alpha: float,
        time_limit: int,
        existing_lineups: list,
        verbose: bool = True,
        skip_exclusions: bool = False
    ) -> list:
        """Generate general lineups without anchoring.

        Args:
            skip_exclusions: If True, don't add exclusion constraints (faster but may get duplicates)
        """
        lineups = []

        if verbose:
            mode = " (no exclusions)" if skip_exclusions else ""
            print(f"\n--- GENERAL LINEUPS ({n_lineups} target){mode} ---")

        iterator = tqdm(range(n_lineups), desc="General", disable=not verbose)

        for _ in iterator:
            # Skip exclusion constraints if requested (faster, dedupe later)
            exclude_list = [] if skip_exclusions else [l['player_ids'] for l in existing_lineups + lineups]

            lineup = optimize_lineup_cvar(
                players_df=self.players_df,
                scenario_matrix=self.scenario_matrix,
                player_index=self.player_index,
                alpha=alpha,
                salary_cap=SALARY_CAP,
                force_include=list(FORCE_INCLUDE),
                exclude_players=list(EXCLUDE_PLAYERS),
                exclude_lineups=exclude_list,
                time_limit=time_limit,
                verbose=False
            )

            if lineup:
                lineup['strategy'] = 'general'
                lineup['anchor_player'] = None
                lineup['lineup_id'] = len(existing_lineups) + len(lineups)
                lineups.append(lineup)
            else:
                if verbose:
                    tqdm.write(f"  Could not generate more general lineups. Got {len(lineups)}.")
                break

        if verbose:
            print(f"  Generated {len(lineups)} general lineups")

        return lineups

    def format_lineups_by_position(self, lineups: list) -> pd.DataFrame:
        """Convert lineup list to DataFrame with position-based columns."""
        rows = []

        for lineup in lineups:
            # Group players by position
            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'D': []}
            for player in lineup['players']:
                pos = player['position']
                if pos in positions:
                    positions[pos].append(player['name'])

            # Build row
            row = {
                'QB': positions['QB'][0] if positions['QB'] else '',
                'RB1': positions['RB'][0] if len(positions['RB']) > 0 else '',
                'RB2': positions['RB'][1] if len(positions['RB']) > 1 else '',
                'WR1': positions['WR'][0] if len(positions['WR']) > 0 else '',
                'WR2': positions['WR'][1] if len(positions['WR']) > 1 else '',
                'WR3': positions['WR'][2] if len(positions['WR']) > 2 else '',
                'TE': positions['TE'][0] if positions['TE'] else '',
                'DEF': positions['D'][0] if positions['D'] else '',
            }

            # Determine FLEX
            flex = ''
            if len(positions['RB']) > 2:
                flex = positions['RB'][2]
            elif len(positions['WR']) > 3:
                flex = positions['WR'][3]
            elif len(positions['TE']) > 1:
                flex = positions['TE'][1]
            row['FLEX'] = flex

            # Add stats
            row['lineup_id'] = lineup.get('lineup_id', 0)
            row['player_ids'] = ','.join(lineup['player_ids'])
            row['total_salary'] = lineup['total_salary']
            row['strategy'] = lineup.get('strategy', '')
            row['anchor_player'] = lineup.get('anchor_player', '')
            row['mean'] = lineup['mean']
            row['median'] = lineup['median']
            row['std'] = lineup['std']
            row['p10'] = lineup['p10']
            row['p80'] = lineup['p80']
            row['p90'] = lineup['p90']
            row['cvar_score'] = lineup['cvar_score']

            rows.append(row)

        # Create DataFrame with ordered columns
        columns = [
            'QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF',
            'lineup_id', 'player_ids', 'total_salary', 'strategy', 'anchor_player',
            'mean', 'median', 'std', 'p10', 'p80', 'p90', 'cvar_score'
        ]

        return pd.DataFrame(rows)[columns]

    def run(
        self,
        n_lineups: int = DEFAULT_N_LINEUPS,
        n_scenarios: int = DEFAULT_N_SCENARIOS,
        alpha: float = DEFAULT_CVAR_ALPHA,
        time_limit: int = DEFAULT_SOLVER_TIME_LIMIT,
        seed: int = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run the full CVaR optimization pipeline.

        Args:
            n_lineups: Total number of lineups to generate
            n_scenarios: Number of scenarios for CVaR
            alpha: CVaR tail probability (0.20 = p80)
            time_limit: Solver time limit per lineup
            seed: Random seed for reproducibility
            verbose: Print progress

        Returns:
            DataFrame with all generated lineups
        """
        self.print_header("CVaR-MILP DFS LINEUP OPTIMIZER")

        pipeline_start = time.time()

        # Save configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'n_lineups': n_lineups,
            'n_scenarios': n_scenarios,
            'alpha': alpha,
            'time_limit': time_limit,
            'seed': seed,
            'salary_cap': SALARY_CAP,
            'force_include': list(FORCE_INCLUDE),
            'exclude_players': list(EXCLUDE_PLAYERS),
            'data_sources': {
                'players': str(self.players_path),
                'game_scripts': str(self.game_scripts_path)
            }
        }

        config_file = self.run_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration:")
        print(f"  Lineups to generate: {n_lineups}")
        print(f"  Scenarios: {n_scenarios}")
        print(f"  Alpha (CVaR): {alpha} (optimizing for p{int((1-alpha)*100)})")
        print(f"  Solver time limit: {time_limit}s per lineup")
        print(f"  Salary cap: ${SALARY_CAP:,}")

        # Step 1: Load data
        self.print_header("STEP 1: LOAD DATA")
        self.load_data()

        # Step 2: Generate scenarios
        self.print_header("STEP 2: GENERATE SCENARIOS")
        self.generate_scenarios(n_scenarios, seed)

        # Step 3: Generate lineups with diversity strategies
        self.print_header("STEP 3: GENERATE LINEUPS")

        all_lineups = []

        if PARALLEL_QB_ANCHOR:
            # Use parallel QB-anchored approach
            print(f"\nStrategy: PARALLEL QB-ANCHORED mode")
            print(f"  Top {N_TOP_QBS} QBs x {N_LINEUPS_PER_ANCHOR} lineups = {N_TOP_QBS * N_LINEUPS_PER_ANCHOR} target lineups")

            all_lineups = self.generate_qb_anchored_parallel(
                n_top_qbs=N_TOP_QBS,
                n_lineups_per_qb=N_LINEUPS_PER_ANCHOR,
                alpha=alpha,
                time_limit=time_limit,
                verbose=verbose
            )

        elif GENERAL_ONLY:
            # Skip anchored strategies, generate all as general (no exclusions for speed)
            print(f"\nStrategy: GENERAL_ONLY mode (no anchored lineups, no exclusion constraints)")
            print(f"  Target: {n_lineups} lineups")

            general_lineups = self.generate_general_lineups(
                n_lineups=n_lineups, alpha=alpha, time_limit=time_limit,
                existing_lineups=all_lineups, verbose=verbose,
                skip_exclusions=True  # Skip exclusions for speed, dedupe later
            )

            # Dedupe by player_ids (keep first occurrence, which has best CVaR due to same optimization)
            seen_lineups = set()
            unique_lineups = []
            for lineup in general_lineups:
                lineup_key = tuple(sorted(lineup['player_ids']))
                if lineup_key not in seen_lineups:
                    seen_lineups.add(lineup_key)
                    unique_lineups.append(lineup)

            if verbose and len(unique_lineups) < len(general_lineups):
                print(f"  Deduped: {len(general_lineups)} -> {len(unique_lineups)} unique lineups")

            all_lineups.extend(unique_lineups)
        else:
            # Calculate strategy allocations (proportional to total)
            # Target: ~25% QB, ~20% DEF, ~20% RB, ~20% WR, ~15% General
            n_qb = max(1, int(n_lineups * 0.25))
            n_def = max(1, int(n_lineups * 0.20))
            n_rb = max(1, int(n_lineups * 0.20))
            n_wr = max(1, int(n_lineups * 0.20))
            n_general = max(0, n_lineups - n_qb - n_def - n_rb - n_wr)

            print(f"\nStrategy allocation:")
            print(f"  QB-anchored: up to {n_qb}")
            print(f"  DEF-anchored: up to {n_def}")
            print(f"  RB-anchored: up to {n_rb}")
            print(f"  WR-anchored: up to {n_wr}")
            print(f"  General: up to {n_general}")

            # Generate QB-anchored lineups
            qb_lineups = self.generate_anchored_lineups(
                position='QB', n_top=n_qb, strategy_name='qb_anchor',
                alpha=alpha, time_limit=time_limit,
                existing_lineups=all_lineups, verbose=verbose
            )
            all_lineups.extend(qb_lineups)

            # Early stop check
            if len(all_lineups) >= n_lineups:
                all_lineups = all_lineups[:n_lineups]
            else:
                # Generate DEF-anchored lineups
                def_lineups = self.generate_anchored_lineups(
                    position='D', n_top=n_def, strategy_name='def_anchor',
                    alpha=alpha, time_limit=time_limit,
                    existing_lineups=all_lineups, verbose=verbose
                )
                all_lineups.extend(def_lineups)

            # Early stop check
            if len(all_lineups) >= n_lineups:
                all_lineups = all_lineups[:n_lineups]
            else:
                # Generate RB-anchored lineups
                rb_lineups = self.generate_anchored_lineups(
                    position='RB', n_top=n_rb, strategy_name='rb_anchor',
                    alpha=alpha, time_limit=time_limit,
                    existing_lineups=all_lineups, verbose=verbose
                )
                all_lineups.extend(rb_lineups)

            # Early stop check
            if len(all_lineups) >= n_lineups:
                all_lineups = all_lineups[:n_lineups]
            else:
                # Generate WR-anchored lineups
                wr_lineups = self.generate_anchored_lineups(
                    position='WR', n_top=n_wr, strategy_name='wr_anchor',
                    alpha=alpha, time_limit=time_limit,
                    existing_lineups=all_lineups, verbose=verbose
                )
                all_lineups.extend(wr_lineups)

            # Generate general lineups to fill remaining
            remaining = n_lineups - len(all_lineups)
            if remaining > 0:
                general_lineups = self.generate_general_lineups(
                    n_lineups=remaining, alpha=alpha, time_limit=time_limit,
                    existing_lineups=all_lineups, verbose=verbose
                )
                all_lineups.extend(general_lineups)

        # Trim to exact count (only for non-parallel modes)
        if not PARALLEL_QB_ANCHOR and len(all_lineups) > n_lineups:
            all_lineups = all_lineups[:n_lineups]

        # Step 4: Format and save results
        self.print_header("STEP 4: SAVE RESULTS")

        # Reassign lineup IDs in order
        for i, lineup in enumerate(all_lineups):
            lineup['lineup_id'] = i

        # Sort by CVaR score (best first)
        all_lineups.sort(key=lambda x: x['cvar_score'], reverse=True)

        # Format as DataFrame
        lineups_df = self.format_lineups_by_position(all_lineups)

        # Round numerical columns to 2 decimal places
        numeric_cols = lineups_df.select_dtypes(include=['float64', 'float32']).columns
        lineups_df[numeric_cols] = lineups_df[numeric_cols].round(2)

        # Save lineups
        lineups_file = self.run_dir / '3_lineups.csv'
        lineups_df.to_csv(lineups_file, index=False)
        print(f"Lineups saved to: {lineups_file}")

        # Save results summary
        pipeline_elapsed = time.time() - pipeline_start

        strategy_counts = lineups_df['strategy'].value_counts().to_dict()

        results = {
            'run_name': self.run_dir.name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'summary': {
                'lineups_generated': len(all_lineups),
                'total_time_seconds': pipeline_elapsed,
                'avg_time_per_lineup': pipeline_elapsed / len(all_lineups) if all_lineups else 0,
                'best_cvar_score': float(lineups_df['cvar_score'].max()) if len(lineups_df) > 0 else 0,
                'avg_cvar_score': float(lineups_df['cvar_score'].mean()) if len(lineups_df) > 0 else 0,
                'strategy_breakdown': strategy_counts
            }
        }

        results_file = self.run_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

        # Print summary
        self.print_header("OPTIMIZATION COMPLETE")

        print(f"Total lineups generated: {len(all_lineups)}")
        print(f"Total time: {pipeline_elapsed / 60:.1f} minutes")
        print(f"Average time per lineup: {pipeline_elapsed / len(all_lineups):.2f}s" if all_lineups else "")

        print(f"\nStrategy breakdown:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}")

        if len(lineups_df) > 0:
            print(f"\nCVaR Score range: {lineups_df['cvar_score'].min():.2f} - {lineups_df['cvar_score'].max():.2f}")
            print(f"P80 range: {lineups_df['p80'].min():.2f} - {lineups_df['p80'].max():.2f}")

            print(f"\nTop 5 lineups by CVaR:")
            for i, row in lineups_df.head(5).iterrows():
                print(f"  {i+1}. CVaR: {row['cvar_score']:.2f}, P80: {row['p80']:.2f}, "
                      f"Strategy: {row['strategy']}, Salary: ${row['total_salary']:,}")

        print(f"\nAll results saved to: {self.run_dir}")

        return lineups_df


def main():
    parser = argparse.ArgumentParser(
        description="CVaR-MILP DFS Lineup Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python 3_run_optimizer.py --week-dir data/2025_12_01

  # Generate fewer lineups with more scenarios (higher precision)
  python 3_run_optimizer.py --week-dir data/2025_12_01 --n-lineups 50 --scenarios 2000

  # More aggressive optimization (p90 instead of p80)
  python 3_run_optimizer.py --week-dir data/2025_12_01 --alpha 0.10
        """
    )

    # Data location
    parser.add_argument('--week-dir', type=str, default=None,
                        help='Week directory (e.g., data/2025_12_01)')

    # CVaR settings
    parser.add_argument('--n-lineups', type=int, default=DEFAULT_N_LINEUPS,
                        help=f'Number of lineups to generate (default: {DEFAULT_N_LINEUPS})')
    parser.add_argument('--scenarios', type=int, default=DEFAULT_N_SCENARIOS,
                        help=f'Number of scenarios for CVaR (default: {DEFAULT_N_SCENARIOS})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_CVAR_ALPHA,
                        help=f'CVaR alpha (default: {DEFAULT_CVAR_ALPHA}, meaning p{int((1-DEFAULT_CVAR_ALPHA)*100)})')
    parser.add_argument('--time-limit', type=int, default=DEFAULT_SOLVER_TIME_LIMIT,
                        help=f'Solver time limit per lineup (default: {DEFAULT_SOLVER_TIME_LIMIT}s)')

    # Other settings
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Create optimizer
    optimizer = CVaROptimizer(
        week_dir=args.week_dir,
        run_name=args.run_name
    )

    # Run optimization
    optimizer.run(
        n_lineups=args.n_lineups,
        n_scenarios=args.scenarios,
        alpha=args.alpha,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
