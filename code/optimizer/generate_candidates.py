"""
Phase 1: Candidate Generation via MILP + Diversity Strategies

Generates diverse lineup candidates using multiple strategies:
1. QB-anchored: One lineup per top 32 QBs (by projection)
2. DEF-anchored: One lineup per allowable DEF
3. RB-anchored: One lineup per top 32 RBs (by projection)
4. WR-anchored: One lineup per top 32 WRs (by projection)
5. Blowout-only: Lineups using only players from blowout games
6. Shootout-only: Lineups using only players from shootout games
7. Non-shootout-or-blowout: Lineups avoiding extreme game scripts
8. Top-blowout stacked: Max players from the top blowout game
9. Top-shootout stacked: Max players from the top shootout game
10. Tiered general diversity:
    - Tier 1 (chalk): Deterministic optimal lineups
    - Tier 2 (temperature): Temperature-based variation
    - Tier 3 (contrarian): Stricter diversity constraints

Then selects top N by fitness for the initial GA population.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import argparse
import sys
from tqdm import tqdm

from utils.milp_solver import create_lineup_milp
from utils.genetic_operators import FITNESS_FUNCTIONS

# Import config values (add parent dir to path temporarily)
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from config import (
    DEFAULT_CANDIDATES, SALARY_CAP, DEFAULT_FITNESS,
    FORCE_INCLUDE, EXCLUDE_PLAYERS,
    MAX_OVERLAP_CHALK, MAX_OVERLAP_MODERATE, MAX_OVERLAP_CONTRARIAN,
    TEMP_DETERMINISTIC, TEMP_MODERATE_MIN, TEMP_MODERATE_MAX,
    TEMP_CONTRARIAN_MIN, TEMP_CONTRARIAN_MAX
)


def get_lineup_key(player_ids: List[str]) -> str:
    """Get a unique key for a lineup based on sorted player IDs."""
    return ','.join(sorted(player_ids))


def lineup_exists(player_ids: List[str], existing_keys: Set[str]) -> bool:
    """Check if lineup already exists in the set."""
    return get_lineup_key(player_ids) in existing_keys


def preprocess_players(players_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess player data: fill missing floor/ceiling with defaults.
    """
    scripts = ['shootout', 'defensive', 'blowout_favorite', 'blowout_underdog', 'competitive']
    missing_count = 0

    for idx in players_df.index:
        if pd.isna(players_df.loc[idx, 'floor_competitive']):
            missing_count += 1
            consensus = players_df.loc[idx, 'fpProjPts']
            for script in scripts:
                players_df.loc[idx, f'floor_{script}'] = consensus * 0.5
                players_df.loc[idx, f'ceiling_{script}'] = consensus * 1.5

    if verbose and missing_count > 0:
        print(f"  Filled {missing_count} players with default floor/ceiling values")

    return players_df


def get_allowable_players(
    players_df: pd.DataFrame,
    position: str,
    force_include: List[str] = None,
    exclude_players: List[str] = None
) -> pd.DataFrame:
    """Get allowable players for a position, respecting include/exclude lists."""
    force_include = force_include or []
    exclude_players = exclude_players or []

    # Filter by position
    pos_players = players_df[players_df['position'] == position].copy()

    # Apply exclusions (case-insensitive)
    exclude_lower = [p.lower() for p in exclude_players]
    pos_players = pos_players[~pos_players['name'].str.lower().isin(exclude_lower)]

    return pos_players


def generate_qb_anchored_lineups(
    players_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    force_include: List[str],
    exclude_players: List[str],
    n_top: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate one lineup per top N QBs (by projection).

    Returns:
        Tuple of (new_lineups, updated_existing_lineups, updated_existing_keys)
    """
    lineups = []

    # Get allowable QBs and sort by projection
    qbs = get_allowable_players(players_df, 'QB', force_include, exclude_players)
    qbs = qbs.sort_values('fpProjPts', ascending=False).head(n_top)

    if verbose:
        print(f"\n=== QB-ANCHORED LINEUPS (top {len(qbs)} QBs) ===")

    qb_iter = tqdm(qbs.iterrows(), total=len(qbs), desc="QB-anchored", disable=not verbose)
    for _, qb in qb_iter:
        qb_id = qb['id'] if 'id' in qb.index else qb['name']

        # Create player pool excluding other QBs
        other_qbs = set(qbs['id'] if 'id' in qbs.columns else qbs['name']) - {qb_id}
        pool = players_df[~players_df['id'].isin(other_qbs) if 'id' in players_df.columns
                         else ~players_df['name'].isin(other_qbs)].copy()
        pool['projection'] = pool['fpProjPts']

        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_CHALK
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 1,
                'strategy': 'qb_anchor',
                'anchor_player': qb_id,
                'temperature': TEMP_DETERMINISTIC,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} QB-anchored lineups")

    return lineups, existing_lineups, existing_keys


def generate_def_anchored_lineups(
    players_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    force_include: List[str],
    exclude_players: List[str],
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate one lineup per allowable defense.
    """
    lineups = []

    # Get allowable defenses
    defs = get_allowable_players(players_df, 'D', force_include, exclude_players)

    if verbose:
        print(f"\n=== DEF-ANCHORED LINEUPS ({len(defs)} DEFs) ===")

    def_iter = tqdm(defs.iterrows(), total=len(defs), desc="DEF-anchored", disable=not verbose)
    for _, defense in def_iter:
        def_id = defense['id'] if 'id' in defense.index else defense['name']

        # Create player pool excluding other DEFs
        other_defs = set(defs['id'] if 'id' in defs.columns else defs['name']) - {def_id}
        pool = players_df[~players_df['id'].isin(other_defs) if 'id' in players_df.columns
                         else ~players_df['name'].isin(other_defs)].copy()
        pool['projection'] = pool['fpProjPts']

        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_CHALK
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 1,
                'strategy': 'def_anchor',
                'anchor_player': def_id,
                'temperature': TEMP_DETERMINISTIC,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} DEF-anchored lineups")

    return lineups, existing_lineups, existing_keys


def generate_rb_anchored_lineups(
    players_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    force_include: List[str],
    exclude_players: List[str],
    n_top: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate one lineup per top N RBs (by projection).
    """
    lineups = []

    # Get allowable RBs and sort by projection
    rbs = get_allowable_players(players_df, 'RB', force_include, exclude_players)
    rbs = rbs.sort_values('fpProjPts', ascending=False).head(n_top)

    if verbose:
        print(f"\n=== RB-ANCHORED LINEUPS (top {len(rbs)} RBs) ===")

    rb_iter = tqdm(rbs.iterrows(), total=len(rbs), desc="RB-anchored", disable=not verbose)
    for _, rb in rb_iter:
        rb_id = rb['id'] if 'id' in rb.index else rb['name']

        # Create player pool - boost this RB's projection to force inclusion
        pool = players_df.copy()
        pool['projection'] = pool['fpProjPts']

        # Boost the anchor RB significantly
        id_col = 'id' if 'id' in pool.columns else 'name'
        anchor_mask = pool[id_col] == rb_id
        pool.loc[anchor_mask, 'projection'] = pool.loc[anchor_mask, 'fpProjPts'] * 2.0

        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_CHALK
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            # Verify the anchor RB is actually in the lineup
            if rb_id in lineup['player_ids']:
                lineups.append({
                    'lineup_id': len(existing_lineups) + len(lineups),
                    'tier': 1,
                    'strategy': 'rb_anchor',
                    'anchor_player': rb_id,
                    'temperature': TEMP_DETERMINISTIC,
                    'player_ids': ','.join(lineup['player_ids']),
                    'total_salary': lineup['total_salary'],
                    'total_projection': lineup['total_projection']
                })
                existing_lineups.append(lineup['player_ids'])
                existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} RB-anchored lineups")

    return lineups, existing_lineups, existing_keys


def generate_wr_anchored_lineups(
    players_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    force_include: List[str],
    exclude_players: List[str],
    n_top: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate one lineup per top N WRs (by projection).
    """
    lineups = []

    # Get allowable WRs and sort by projection
    wrs = get_allowable_players(players_df, 'WR', force_include, exclude_players)
    wrs = wrs.sort_values('fpProjPts', ascending=False).head(n_top)

    if verbose:
        print(f"\n=== WR-ANCHORED LINEUPS (top {len(wrs)} WRs) ===")

    wr_iter = tqdm(wrs.iterrows(), total=len(wrs), desc="WR-anchored", disable=not verbose)
    for _, wr in wr_iter:
        wr_id = wr['id'] if 'id' in wr.index else wr['name']

        # Create player pool - boost this WR's projection to force inclusion
        pool = players_df.copy()
        pool['projection'] = pool['fpProjPts']

        # Boost the anchor WR significantly
        id_col = 'id' if 'id' in pool.columns else 'name'
        anchor_mask = pool[id_col] == wr_id
        pool.loc[anchor_mask, 'projection'] = pool.loc[anchor_mask, 'fpProjPts'] * 2.0

        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_CHALK
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            # Verify the anchor WR is actually in the lineup
            if wr_id in lineup['player_ids']:
                lineups.append({
                    'lineup_id': len(existing_lineups) + len(lineups),
                    'tier': 1,
                    'strategy': 'wr_anchor',
                    'anchor_player': wr_id,
                    'temperature': TEMP_DETERMINISTIC,
                    'player_ids': ','.join(lineup['player_ids']),
                    'total_salary': lineup['total_salary'],
                    'total_projection': lineup['total_projection']
                })
                existing_lineups.append(lineup['player_ids'])
                existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} WR-anchored lineups")

    return lineups, existing_lineups, existing_keys


def generate_blowout_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups using only players from blowout games.
    """
    lineups = []

    # Find blowout games (where blowout is the primary script OR has high probability)
    blowout_games = game_scripts_df[
        (game_scripts_df['primary_script'] == 'blowout') |
        (game_scripts_df['blowout_prob'] >= 0.3)
    ]['game_id'].tolist()

    if not blowout_games:
        if verbose:
            print("\n=== BLOWOUT-ONLY LINEUPS (0 blowout games found) ===")
        return lineups, existing_lineups, existing_keys

    # Filter players to only those in blowout games
    blowout_players = players_df[players_df['game_id'].isin(blowout_games)].copy()
    blowout_players['projection'] = blowout_players['fpProjPts']

    if verbose:
        print(f"\n=== BLOWOUT-ONLY LINEUPS ({len(blowout_games)} games, {len(blowout_players)} players) ===")

    blowout_iter = tqdm(range(n_lineups), desc="Blowout-only", disable=not verbose)
    for i in blowout_iter:
        lineup = create_lineup_milp(
            blowout_players,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_MODERATE
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 2,
                'strategy': 'blowout_only',
                'anchor_player': None,
                'temperature': TEMP_MODERATE_MIN + (TEMP_MODERATE_MAX - TEMP_MODERATE_MIN) * (i / n_lineups),
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
        else:
            # Can't generate more unique lineups from this pool
            break

    if verbose:
        print(f"  Generated {len(lineups)} blowout-only lineups")

    return lineups, existing_lineups, existing_keys


def generate_shootout_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups using only players from shootout games.
    """
    lineups = []

    # Find shootout games (where shootout is the primary script OR has high probability)
    shootout_games = game_scripts_df[
        (game_scripts_df['primary_script'] == 'shootout') |
        (game_scripts_df['shootout_prob'] >= 0.3)
    ]['game_id'].tolist()

    if not shootout_games:
        if verbose:
            print("\n=== SHOOTOUT-ONLY LINEUPS (0 shootout games found) ===")
        return lineups, existing_lineups, existing_keys

    # Filter players to only those in shootout games
    shootout_players = players_df[players_df['game_id'].isin(shootout_games)].copy()
    shootout_players['projection'] = shootout_players['fpProjPts']

    if verbose:
        print(f"\n=== SHOOTOUT-ONLY LINEUPS ({len(shootout_games)} games, {len(shootout_players)} players) ===")

    shootout_iter = tqdm(range(n_lineups), desc="Shootout-only", disable=not verbose)
    for i in shootout_iter:
        lineup = create_lineup_milp(
            shootout_players,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_MODERATE
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 2,
                'strategy': 'shootout_only',
                'anchor_player': None,
                'temperature': TEMP_MODERATE_MIN + (TEMP_MODERATE_MAX - TEMP_MODERATE_MIN) * (i / n_lineups),
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
        else:
            # Can't generate more unique lineups from this pool
            break

    if verbose:
        print(f"  Generated {len(lineups)} shootout-only lineups")

    return lineups, existing_lineups, existing_keys


def generate_non_extreme_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups using only players from non-shootout and non-blowout games.
    (competitive/defensive games)
    """
    lineups = []

    # Find games that are neither shootout nor blowout
    non_extreme_games = game_scripts_df[
        (game_scripts_df['shootout_prob'] < 0.3) &
        (game_scripts_df['blowout_prob'] < 0.3) &
        (game_scripts_df['primary_script'].isin(['competitive', 'defensive']))
    ]['game_id'].tolist()

    if not non_extreme_games:
        if verbose:
            print("\n=== NON-EXTREME LINEUPS (0 competitive/defensive games found) ===")
        return lineups, existing_lineups, existing_keys

    # Filter players to only those in non-extreme games
    non_extreme_players = players_df[players_df['game_id'].isin(non_extreme_games)].copy()
    non_extreme_players['projection'] = non_extreme_players['fpProjPts']

    if verbose:
        print(f"\n=== NON-SHOOTOUT/BLOWOUT LINEUPS ({len(non_extreme_games)} games, {len(non_extreme_players)} players) ===")

    non_extreme_iter = tqdm(range(n_lineups), desc="Non-extreme", disable=not verbose)
    for i in non_extreme_iter:
        lineup = create_lineup_milp(
            non_extreme_players,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_MODERATE
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 2,
                'strategy': 'non_extreme',
                'anchor_player': None,
                'temperature': TEMP_MODERATE_MIN + (TEMP_MODERATE_MAX - TEMP_MODERATE_MIN) * (i / n_lineups),
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
        else:
            break

    if verbose:
        print(f"  Generated {len(lineups)} non-extreme lineups")

    return lineups, existing_lineups, existing_keys


def generate_non_blowout_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 10,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups using only players from non-blowout games.
    """
    lineups = []

    # Find non-blowout games
    non_blowout_games = game_scripts_df[
        (game_scripts_df['primary_script'] != 'blowout') &
        (game_scripts_df['blowout_prob'] < 0.3)
    ]['game_id'].tolist()

    if not non_blowout_games:
        if verbose:
            print("\n=== NON-BLOWOUT LINEUPS (0 non-blowout games found) ===")
        return lineups, existing_lineups, existing_keys

    # Filter players to only those in non-blowout games
    non_blowout_players = players_df[players_df['game_id'].isin(non_blowout_games)].copy()
    non_blowout_players['projection'] = non_blowout_players['fpProjPts']

    if verbose:
        print(f"\n=== NON-BLOWOUT LINEUPS ({len(non_blowout_games)} games, {len(non_blowout_players)} players) ===")

    non_blowout_iter = tqdm(range(n_lineups), desc="Non-blowout", disable=not verbose)
    for i in non_blowout_iter:
        lineup = create_lineup_milp(
            non_blowout_players,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_MODERATE
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 2,
                'strategy': 'non_blowout',
                'anchor_player': None,
                'temperature': TEMP_MODERATE_MIN + (TEMP_MODERATE_MAX - TEMP_MODERATE_MIN) * (i / n_lineups),
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
        else:
            break

    if verbose:
        print(f"  Generated {len(lineups)} non-blowout lineups")

    return lineups, existing_lineups, existing_keys


def generate_top_blowout_stacked_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 5,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups maximizing players from the top blowout game.
    Uses both favorite and underdog stacks.
    """
    lineups = []

    # Find the top blowout game (highest blowout probability)
    blowout_games = game_scripts_df.sort_values('blowout_prob', ascending=False)

    if len(blowout_games) == 0 or blowout_games.iloc[0]['blowout_prob'] < 0.2:
        if verbose:
            print("\n=== TOP BLOWOUT STACKED (no significant blowout games) ===")
        return lineups, existing_lineups, existing_keys

    top_game = blowout_games.iloc[0]
    game_id = top_game['game_id']
    favorite = top_game.get('favorite', game_id.split('@')[1] if '@' in game_id else None)
    underdog = top_game.get('underdog', game_id.split('@')[0] if '@' in game_id else None)

    if verbose:
        print(f"\n=== TOP BLOWOUT STACKED ({game_id}, {top_game['blowout_prob']:.1%} blowout) ===")
        print(f"    Favorite: {favorite}, Underdog: {underdog}")

    # Get players from this game
    game_players = players_df[players_df['game_id'] == game_id]

    if len(game_players) < 5:
        if verbose:
            print(f"  Not enough players in game ({len(game_players)})")
        return lineups, existing_lineups, existing_keys

    # Strategy 1: Stack the favorite (likely to dominate)
    if favorite:
        favorite_players = game_players[game_players['team'] == favorite]
        if len(favorite_players) >= 3:
            # Boost projections for favorite players
            pool = players_df.copy()
            pool['projection'] = pool['fpProjPts']
            # Give significant boost to favorite team players
            favorite_mask = (pool['game_id'] == game_id) & (pool['team'] == favorite)
            pool.loc[favorite_mask, 'projection'] = pool.loc[favorite_mask, 'fpProjPts'] * 1.5

            for i in range(min(n_lineups // 2 + 1, 3)):
                lineup = create_lineup_milp(
                    pool,
                    salary_cap=salary_cap,
                    max_overlap_with=existing_lineups if existing_lineups else None,
                    max_overlap=MAX_OVERLAP_MODERATE
                )

                if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
                    lineups.append({
                        'lineup_id': len(existing_lineups) + len(lineups),
                        'tier': 2,
                        'strategy': 'blowout_favorite_stack',
                        'anchor_player': game_id,
                        'temperature': TEMP_MODERATE_MIN,
                        'player_ids': ','.join(lineup['player_ids']),
                        'total_salary': lineup['total_salary'],
                        'total_projection': lineup['total_projection']
                    })
                    existing_lineups.append(lineup['player_ids'])
                    existing_keys.add(get_lineup_key(lineup['player_ids']))

    # Strategy 2: Stack the underdog (garbage time potential)
    if underdog:
        underdog_players = game_players[game_players['team'] == underdog]
        if len(underdog_players) >= 3:
            pool = players_df.copy()
            pool['projection'] = pool['fpProjPts']
            # Boost underdog team players (garbage time upside)
            underdog_mask = (pool['game_id'] == game_id) & (pool['team'] == underdog)
            pool.loc[underdog_mask, 'projection'] = pool.loc[underdog_mask, 'fpProjPts'] * 1.3

            for i in range(min(n_lineups // 2, 2)):
                lineup = create_lineup_milp(
                    pool,
                    salary_cap=salary_cap,
                    max_overlap_with=existing_lineups if existing_lineups else None,
                    max_overlap=MAX_OVERLAP_MODERATE
                )

                if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
                    lineups.append({
                        'lineup_id': len(existing_lineups) + len(lineups),
                        'tier': 2,
                        'strategy': 'blowout_underdog_stack',
                        'anchor_player': game_id,
                        'temperature': TEMP_MODERATE_MAX,
                        'player_ids': ','.join(lineup['player_ids']),
                        'total_salary': lineup['total_salary'],
                        'total_projection': lineup['total_projection']
                    })
                    existing_lineups.append(lineup['player_ids'])
                    existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} top-blowout stacked lineups")

    return lineups, existing_lineups, existing_keys


def generate_top_shootout_stacked_lineups(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    n_lineups: int = 5,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups maximizing players from the top shootout game.
    Stacks both teams since both are expected to score.
    """
    lineups = []

    # Find the top shootout game (highest shootout probability)
    shootout_games = game_scripts_df.sort_values('shootout_prob', ascending=False)

    if len(shootout_games) == 0 or shootout_games.iloc[0]['shootout_prob'] < 0.2:
        if verbose:
            print("\n=== TOP SHOOTOUT STACKED (no significant shootout games) ===")
        return lineups, existing_lineups, existing_keys

    top_game = shootout_games.iloc[0]
    game_id = top_game['game_id']

    # Parse teams from game_id (format: "AWAY@HOME")
    if '@' in game_id:
        away_team = game_id.split('@')[0]
        home_team = game_id.split('@')[1]
    else:
        away_team = None
        home_team = None

    if verbose:
        print(f"\n=== TOP SHOOTOUT STACKED ({game_id}, {top_game['shootout_prob']:.1%} shootout) ===")
        print(f"    Away: {away_team}, Home: {home_team}")

    # Get players from this game
    game_players = players_df[players_df['game_id'] == game_id]

    if len(game_players) < 5:
        if verbose:
            print(f"  Not enough players in game ({len(game_players)})")
        return lineups, existing_lineups, existing_keys

    # In a shootout, stack BOTH teams - they're both expected to score a lot
    # Strategy 1: Home team stack
    if home_team:
        home_players = game_players[game_players['team'] == home_team]
        if len(home_players) >= 3:
            pool = players_df.copy()
            pool['projection'] = pool['fpProjPts']
            # Boost home team players
            home_mask = (pool['game_id'] == game_id) & (pool['team'] == home_team)
            pool.loc[home_mask, 'projection'] = pool.loc[home_mask, 'fpProjPts'] * 1.4

            for i in range(min(n_lineups // 2 + 1, 3)):
                lineup = create_lineup_milp(
                    pool,
                    salary_cap=salary_cap,
                    max_overlap_with=existing_lineups if existing_lineups else None,
                    max_overlap=MAX_OVERLAP_MODERATE
                )

                if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
                    lineups.append({
                        'lineup_id': len(existing_lineups) + len(lineups),
                        'tier': 2,
                        'strategy': 'shootout_home_stack',
                        'anchor_player': game_id,
                        'temperature': TEMP_MODERATE_MIN,
                        'player_ids': ','.join(lineup['player_ids']),
                        'total_salary': lineup['total_salary'],
                        'total_projection': lineup['total_projection']
                    })
                    existing_lineups.append(lineup['player_ids'])
                    existing_keys.add(get_lineup_key(lineup['player_ids']))

    # Strategy 2: Away team stack
    if away_team:
        away_players = game_players[game_players['team'] == away_team]
        if len(away_players) >= 3:
            pool = players_df.copy()
            pool['projection'] = pool['fpProjPts']
            # Boost away team players
            away_mask = (pool['game_id'] == game_id) & (pool['team'] == away_team)
            pool.loc[away_mask, 'projection'] = pool.loc[away_mask, 'fpProjPts'] * 1.4

            for i in range(min(n_lineups // 2, 2)):
                lineup = create_lineup_milp(
                    pool,
                    salary_cap=salary_cap,
                    max_overlap_with=existing_lineups if existing_lineups else None,
                    max_overlap=MAX_OVERLAP_MODERATE
                )

                if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
                    lineups.append({
                        'lineup_id': len(existing_lineups) + len(lineups),
                        'tier': 2,
                        'strategy': 'shootout_away_stack',
                        'anchor_player': game_id,
                        'temperature': TEMP_MODERATE_MAX,
                        'player_ids': ','.join(lineup['player_ids']),
                        'total_salary': lineup['total_salary'],
                        'total_projection': lineup['total_projection']
                    })
                    existing_lineups.append(lineup['player_ids'])
                    existing_keys.add(get_lineup_key(lineup['player_ids']))

    if verbose:
        print(f"  Generated {len(lineups)} top-shootout stacked lineups")

    return lineups, existing_lineups, existing_keys


def generate_tiered_lineups(
    players_df: pd.DataFrame,
    existing_lineups: List[List[str]],
    existing_keys: Set[str],
    salary_cap: float,
    target_total: int,
    verbose: bool = True
) -> Tuple[List[Dict], List[List[str]], Set[str]]:
    """
    Generate lineups using the original 3-tier approach:
    - Tier 1 (chalk): Deterministic optimal lineups with max_overlap=7
    - Tier 2 (temperature): Temperature-based variation with max_overlap=6-7
    - Tier 3 (contrarian): Higher diversity with max_overlap=5-6

    This fills the remaining slots after the diversity strategies.
    """
    lineups = []
    current_count = len(existing_lineups)
    needed = target_total - current_count

    if needed <= 0:
        return lineups, existing_lineups, existing_keys

    pool = players_df.copy()
    pool['projection'] = pool['fpProjPts']

    # Calculate tier sizes based on remaining slots
    # Original distribution: 20 chalk, 80 temp, rest contrarian
    # Scale proportionally
    tier1_count = max(10, int(needed * 0.1))  # ~10% chalk
    tier2_count = max(30, int(needed * 0.3))  # ~30% temperature
    tier3_count = needed - tier1_count - tier2_count  # Rest contrarian

    # ===== TIER 1: Deterministic chalk =====
    if verbose:
        print(f"\n=== TIER 1: DETERMINISTIC CHALK ({tier1_count} target) ===")

    tier1_iter = tqdm(range(tier1_count), desc="Tier 1 (chalk)", disable=not verbose)
    consecutive_failures = 0
    max_failures = 5

    for i in tier1_iter:
        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_CHALK
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 1,
                'strategy': 'chalk',
                'anchor_player': None,
                'temperature': TEMP_DETERMINISTIC,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break

    tier1_generated = len([lu for lu in lineups if lu['tier'] == 1 and lu['strategy'] == 'chalk'])
    if verbose:
        print(f"  Generated {tier1_generated} chalk lineups")

    # ===== TIER 2: Temperature-based variation =====
    if verbose:
        print(f"\n=== TIER 2: TEMPERATURE VARIATION ({tier2_count} target) ===")

    tier2_iter = tqdm(range(tier2_count), desc="Tier 2 (temp)", disable=not verbose)
    consecutive_failures = 0

    for i in tier2_iter:
        # Temperature ramps from TEMP_MODERATE_MIN to TEMP_MODERATE_MAX
        temperature = TEMP_MODERATE_MIN + (TEMP_MODERATE_MAX - TEMP_MODERATE_MIN) * (i / tier2_count)

        lineup = create_lineup_milp(
            pool,
            salary_cap=salary_cap,
            max_overlap_with=existing_lineups if existing_lineups else None,
            max_overlap=MAX_OVERLAP_MODERATE
        )

        if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
            lineups.append({
                'lineup_id': len(existing_lineups) + len(lineups),
                'tier': 2,
                'strategy': 'temperature',
                'anchor_player': None,
                'temperature': temperature,
                'player_ids': ','.join(lineup['player_ids']),
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection']
            })
            existing_lineups.append(lineup['player_ids'])
            existing_keys.add(get_lineup_key(lineup['player_ids']))
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break

    tier2_generated = len([lu for lu in lineups if lu['tier'] == 2 and lu['strategy'] == 'temperature'])
    if verbose:
        print(f"  Generated {tier2_generated} temperature-based lineups")

    # ===== TIER 3: Contrarian/random =====
    if tier3_count > 0:
        if verbose:
            print(f"\n=== TIER 3: CONTRARIAN ({tier3_count} target) ===")

        tier3_iter = tqdm(range(tier3_count), desc="Tier 3 (contrarian)", disable=not verbose)
        consecutive_failures = 0
        max_failures = 10

        for i in tier3_iter:
            # High temperature for contrarian plays
            temperature = np.random.uniform(TEMP_CONTRARIAN_MIN, TEMP_CONTRARIAN_MAX)

            lineup = create_lineup_milp(
                pool,
                salary_cap=salary_cap,
                max_overlap_with=existing_lineups if existing_lineups else None,
                max_overlap=MAX_OVERLAP_CONTRARIAN
            )

            if lineup and not lineup_exists(lineup['player_ids'], existing_keys):
                lineups.append({
                    'lineup_id': len(existing_lineups) + len(lineups),
                    'tier': 3,
                    'strategy': 'contrarian',
                    'anchor_player': None,
                    'temperature': temperature,
                    'player_ids': ','.join(lineup['player_ids']),
                    'total_salary': lineup['total_salary'],
                    'total_projection': lineup['total_projection']
                })
                existing_lineups.append(lineup['player_ids'])
                existing_keys.add(get_lineup_key(lineup['player_ids']))
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    if verbose:
                        tqdm.write(f"  Stopping early: {consecutive_failures} consecutive failures")
                    break

        tier3_generated = len([lu for lu in lineups if lu['tier'] == 3])
        if verbose:
            print(f"  Generated {tier3_generated} contrarian lineups")

    total_tiered = len(lineups)
    if verbose:
        print(f"\n  Total tiered lineups: {total_tiered}")

    return lineups, existing_lineups, existing_keys


def select_top_by_fitness(
    all_lineups: List[Dict],
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_select: int,
    fitness_name: str = DEFAULT_FITNESS,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate all lineups and select top N by fitness proxy.

    For speed, we use a quick proxy: just evaluate using consensus projections
    rather than full Monte Carlo (which happens in Phase 2).
    """
    if verbose:
        print(f"\n=== SELECTING TOP {n_select} BY FITNESS ({fitness_name}) ===")

    # Get fitness function
    fitness_func = FITNESS_FUNCTIONS.get(fitness_name, FITNESS_FUNCTIONS['tournament'])

    # Quick evaluation: use sum of projections as proxy for fitness
    # (Full MC happens in Phase 2)
    for lineup in all_lineups:
        player_ids = lineup['player_ids'].split(',')
        id_col = 'id' if 'id' in players_df.columns else 'name'

        total_proj = 0
        for pid in player_ids:
            player = players_df[players_df[id_col] == pid]
            if len(player) > 0:
                total_proj += player.iloc[0]['fpProjPts']

        # Use projection as proxy for fitness (higher is better)
        lineup['proxy_fitness'] = total_proj

    # Sort by proxy fitness and take top N
    all_lineups.sort(key=lambda x: x['proxy_fitness'], reverse=True)
    selected = all_lineups[:n_select]

    if verbose:
        print(f"  Total lineups generated: {len(all_lineups)}")
        print(f"  Selected top {len(selected)} by projection")

        # Show strategy breakdown in selected
        strategy_counts = {}
        tier_counts = {1: 0, 2: 0, 3: 0}
        for lu in selected:
            strat = lu.get('strategy', 'unknown')
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            tier = lu.get('tier', 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        print(f"\n  Tier breakdown in top {len(selected)}:")
        print(f"    Tier 1 (chalk): {tier_counts.get(1, 0)}")
        print(f"    Tier 2 (moderate): {tier_counts.get(2, 0)}")
        print(f"    Tier 3 (contrarian): {tier_counts.get(3, 0)}")

        print(f"\n  Strategy breakdown in top {len(selected)}:")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"    {strat}: {count}")

    # Convert to DataFrame with expected columns
    result = []
    for i, lu in enumerate(selected):
        result.append({
            'lineup_id': i,
            'tier': lu.get('tier', 3),
            'strategy': lu.get('strategy', 'general'),
            'temperature': lu.get('temperature', 0.0),
            'player_ids': lu['player_ids'],
            'total_salary': lu['total_salary'],
            'total_projection': lu['total_projection']
        })

    return pd.DataFrame(result)


def generate_candidates(
    players_df: pd.DataFrame,
    game_scripts_df: pd.DataFrame,
    n_lineups: int = DEFAULT_CANDIDATES,
    salary_cap: float = SALARY_CAP,
    output_path: str = 'outputs/lineups_candidates.csv',
    fitness_name: str = DEFAULT_FITNESS,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate diverse lineup candidates using multiple strategies.

    Strategies:
    1. QB-anchored: One lineup per allowable QB
    2. DEF-anchored: One lineup per allowable DEF
    3. RB-anchored: One lineup per top 32 RBs
    4. WR-anchored: One lineup per top 32 WRs
    5. Blowout-only: Lineups using only players from blowout games
    6. Shootout-only: Lineups using only players from shootout games
    7. Non-extreme: Lineups avoiding shootout/blowout games
    8. Top-blowout stacked: Max players from the top blowout game
    9. Top-shootout stacked: Max players from the top shootout game
    10. Tiered general:
        - Tier 1 (chalk): Deterministic optimal
        - Tier 2 (temperature): Moderate variation
        - Tier 3 (contrarian): High diversity

    Args:
        players_df: DataFrame with integrated player data
        game_scripts_df: DataFrame with game script probabilities
        n_lineups: Number of lineups to select for initial population (default: 200)
        salary_cap: Salary cap (default: 60000)
        output_path: Where to save candidates CSV
        fitness_name: Fitness function to use for selection
        verbose: Print progress

    Returns:
        DataFrame with selected candidate lineups
    """
    if verbose:
        print("=" * 80)
        print("PHASE 1: CANDIDATE GENERATION (DIVERSITY STRATEGIES + TIERS)")
        print("=" * 80)
        print(f"\nTarget: Generate diverse pool, select top {n_lineups}")

    # Preprocess
    players_df = preprocess_players(players_df, verbose)

    # Get config lists
    force_include = list(FORCE_INCLUDE) if FORCE_INCLUDE else []
    exclude_players = list(EXCLUDE_PLAYERS) if EXCLUDE_PLAYERS else []

    if verbose and force_include:
        print(f"\nForce include: {force_include}")
    if verbose and exclude_players:
        print(f"Exclude players: {exclude_players}")

    # Track all lineups and existing lineup keys
    all_lineups = []
    existing_lineups = []  # List of player_id lists
    existing_keys = set()  # Set of sorted player_id strings

    # Generate ~3x more lineups than needed, then select top N
    target_pool_size = n_lineups * 3

    # ===== Strategy 1: QB-anchored lineups (top 32) =====
    qb_lineups, existing_lineups, existing_keys = generate_qb_anchored_lineups(
        players_df, existing_lineups, existing_keys, salary_cap,
        force_include, exclude_players, n_top=32, verbose=verbose
    )
    all_lineups.extend(qb_lineups)

    # ===== Strategy 2: DEF-anchored lineups =====
    def_lineups, existing_lineups, existing_keys = generate_def_anchored_lineups(
        players_df, existing_lineups, existing_keys, salary_cap,
        force_include, exclude_players, verbose
    )
    all_lineups.extend(def_lineups)

    # ===== Strategy 3: RB-anchored lineups (top 32) =====
    rb_lineups, existing_lineups, existing_keys = generate_rb_anchored_lineups(
        players_df, existing_lineups, existing_keys, salary_cap,
        force_include, exclude_players, n_top=32, verbose=verbose
    )
    all_lineups.extend(rb_lineups)

    # ===== Strategy 4: WR-anchored lineups (top 32) =====
    wr_lineups, existing_lineups, existing_keys = generate_wr_anchored_lineups(
        players_df, existing_lineups, existing_keys, salary_cap,
        force_include, exclude_players, n_top=32, verbose=verbose
    )
    all_lineups.extend(wr_lineups)

    # ===== Strategy 5: Blowout-only lineups =====
    blowout_lineups, existing_lineups, existing_keys = generate_blowout_lineups(
        players_df, game_scripts_df, existing_lineups, existing_keys,
        salary_cap, n_lineups=15, verbose=verbose
    )
    all_lineups.extend(blowout_lineups)

    # ===== Strategy 6: Shootout-only lineups =====
    shootout_lineups, existing_lineups, existing_keys = generate_shootout_lineups(
        players_df, game_scripts_df, existing_lineups, existing_keys,
        salary_cap, n_lineups=15, verbose=verbose
    )
    all_lineups.extend(shootout_lineups)

    # ===== Strategy 7: Non-extreme (no shootout/blowout) lineups =====
    non_extreme_lineups, existing_lineups, existing_keys = generate_non_extreme_lineups(
        players_df, game_scripts_df, existing_lineups, existing_keys,
        salary_cap, n_lineups=15, verbose=verbose
    )
    all_lineups.extend(non_extreme_lineups)

    # ===== Strategy 8: Top blowout stacked lineups =====
    blowout_stack_lineups, existing_lineups, existing_keys = generate_top_blowout_stacked_lineups(
        players_df, game_scripts_df, existing_lineups, existing_keys,
        salary_cap, n_lineups=10, verbose=verbose
    )
    all_lineups.extend(blowout_stack_lineups)

    # ===== Strategy 9: Top shootout stacked lineups =====
    shootout_stack_lineups, existing_lineups, existing_keys = generate_top_shootout_stacked_lineups(
        players_df, game_scripts_df, existing_lineups, existing_keys,
        salary_cap, n_lineups=10, verbose=verbose
    )
    all_lineups.extend(shootout_stack_lineups)

    # ===== Strategy 10: Tiered general lineups (chalk, temp, contrarian) =====
    # Only generate tiered lineups if we need more to reach target
    current_count = len(all_lineups)
    if current_count < target_pool_size:
        tiered_lineups, existing_lineups, existing_keys = generate_tiered_lineups(
            players_df, existing_lineups, existing_keys, salary_cap,
            target_total=target_pool_size, verbose=verbose
        )
        all_lineups.extend(tiered_lineups)

    # Convert all lineups to DataFrame (selection happens AFTER Phase 2 MC evaluation)
    if verbose:
        print(f"\n=== CANDIDATE POOL COMPLETE ===")
        print(f"  Total unique lineups generated: {len(all_lineups)}")
        print(f"  All will be sent to Phase 2 for Monte Carlo evaluation")
        print(f"  Top {n_lineups} will be selected after MC evaluation based on fitness")

        # Show strategy breakdown
        strategy_counts = {}
        tier_counts = {1: 0, 2: 0, 3: 0}
        for lu in all_lineups:
            strat = lu.get('strategy', 'unknown')
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            tier = lu.get('tier', 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        print(f"\n  Tier breakdown:")
        print(f"    Tier 1 (chalk): {tier_counts.get(1, 0)}")
        print(f"    Tier 2 (moderate): {tier_counts.get(2, 0)}")
        print(f"    Tier 3 (contrarian): {tier_counts.get(3, 0)}")

        print(f"\n  Strategy breakdown:")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"    {strat}: {count}")

    # Convert to DataFrame
    result = []
    for i, lu in enumerate(all_lineups):
        result.append({
            'lineup_id': i,
            'tier': lu.get('tier', 3),
            'strategy': lu.get('strategy', 'general'),
            'temperature': lu.get('temperature', 0.0),
            'player_ids': lu['player_ids'],
            'total_salary': lu['total_salary'],
            'total_projection': lu['total_projection']
        })

    lineups_df = pd.DataFrame(result)

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lineups_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"  Total lineups in pool: {len(all_lineups)}")
        print(f"  Selected for initial population: {len(lineups_df)}")
        print(f"  Saved to: {output_file}")
        print("=" * 80)

    return lineups_df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate lineup candidates")
    parser.add_argument('--input', default='players_integrated.csv', help='Input players CSV')
    parser.add_argument('--game-scripts', default='game_script.csv', help='Game scripts CSV')
    parser.add_argument('--output', default='outputs/lineups_candidates.csv', help='Output lineups CSV')
    parser.add_argument('--n-lineups', type=int, default=DEFAULT_CANDIDATES,
                       help=f'Number of lineups to select (default: {DEFAULT_CANDIDATES})')
    parser.add_argument('--salary-cap', type=float, default=SALARY_CAP,
                       help=f'Salary cap (default: {SALARY_CAP})')
    parser.add_argument('--fitness', default=DEFAULT_FITNESS,
                       choices=['conservative', 'balanced', 'aggressive', 'tournament'],
                       help=f'Fitness function for selection (default: {DEFAULT_FITNESS})')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Load data
    players_df = pd.read_csv(args.input)
    game_scripts_df = pd.read_csv(args.game_scripts)

    # Generate candidates
    generate_candidates(
        players_df=players_df,
        game_scripts_df=game_scripts_df,
        n_lineups=args.n_lineups,
        salary_cap=args.salary_cap,
        output_path=args.output,
        fitness_name=args.fitness,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
