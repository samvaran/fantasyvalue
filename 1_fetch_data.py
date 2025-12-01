"""
Main script for fetching and merging fantasy football data.

Usage:
    python fetch_data.py --all              # Fetch all data sources
    python fetch_data.py --fp               # Fetch FantasyPros only
    python fetch_data.py --dk               # Fetch DraftKings TD odds only
    python fetch_data.py --lines            # Fetch DraftKings game lines only
    python fetch_data.py --espn             # Fetch ESPN projections only
    python fetch_data.py                    # Use all cached data
"""
import argparse
from pathlib import Path

import pandas as pd

from scrapers import (
    FantasyProsScraper,
    FanDuelLoader,
    DraftKingsGameLinesScraper,
    DraftKingsTdOddsScraper,
    EspnPlayerListScraper,
    EspnProjectionsScraper,
)
from utils import normalize_name


def main():
    parser = argparse.ArgumentParser(description='Fetch fantasy football data')
    parser.add_argument('--all', action='store_true', help='Fetch all data sources')
    parser.add_argument('--fp', action='store_true', help='Fetch FantasyPros projections')
    parser.add_argument('--lines', action='store_true', help='Fetch DraftKings game lines')
    parser.add_argument('--dk', action='store_true', help='Fetch DraftKings TD odds')
    parser.add_argument('--espn', action='store_true', help='Fetch ESPN projections')
    parser.add_argument('--skip-fp', action='store_true', help='Skip FantasyPros (use cache)')
    args = parser.parse_args()

    # Determine what to fetch
    fetch_fp = (args.fp or args.all) and not args.skip_fp
    fetch_lines = args.lines or args.all
    fetch_dk = args.dk or args.all
    fetch_espn = args.espn or args.all

    print("\n" + "=" * 80)
    print("FANTASY DATA COLLECTION PIPELINE")
    print("=" * 80)
    print("\nData Sources:")
    print(f"  FanDuel Salaries: LOAD (local CSV)")
    print(f"  FantasyPros:     {'FETCH' if fetch_fp else 'CACHE'}")
    print(f"  DK Game Lines:   {'FETCH' if fetch_lines else 'CACHE'}")
    print(f"  DK TD Odds:      {'FETCH' if fetch_dk else 'CACHE'}")
    print(f"  ESPN Players:    {'FETCH' if fetch_espn else 'CACHE'}")
    print(f"  ESPN Projections: {'FETCH' if fetch_espn else 'CACHE'}")

    # ========================================================================
    # STEP 1: FANDUEL SALARIES (CHEAPEST - FAIL FAST)
    # ========================================================================
    print("\n=== STEP 1: FanDuel Salaries ===\n")
    fd_loader = FanDuelLoader()
    fd_data = fd_loader.get_data(use_cache=True)  # Always load from CSV

    # ========================================================================
    # STEP 2: FANTASYPROS PROJECTIONS
    # ========================================================================
    print("\n=== STEP 2: FantasyPros Projections ===\n")
    fp_scraper = FantasyProsScraper()
    fp_data = fp_scraper.get_data(use_cache=not fetch_fp)

    # ========================================================================
    # STEP 3: DRAFTKINGS GAME LINES
    # ========================================================================
    print("\n=== STEP 3: DraftKings Game Lines ===\n")
    lines_scraper = DraftKingsGameLinesScraper()
    game_lines = lines_scraper.get_data(use_cache=not fetch_lines)

    # Build lookup by team abbreviation
    game_lines_by_team = {}
    for line in game_lines:
        team_abbr = line['team_abbr']
        game_lines_by_team[team_abbr] = {
            'proj_team_pts': line['projected_pts'],
            'proj_opp_pts': None,  # Will be filled below
        }

    # Set opponent projected points
    for line in game_lines:
        team_abbr = line['team_abbr']
        opp_abbr = line['opponent_abbr']
        if team_abbr in game_lines_by_team and opp_abbr in game_lines_by_team:
            game_lines_by_team[team_abbr]['proj_opp_pts'] = game_lines_by_team[opp_abbr]['proj_team_pts']

    # ========================================================================
    # STEP 4: DRAFTKINGS TD ODDS
    # ========================================================================
    print("\n=== STEP 4: DraftKings TD Odds ===\n")
    td_scraper = DraftKingsTdOddsScraper()
    td_odds = td_scraper.get_data(use_cache=not fetch_dk)

    # ========================================================================
    # STEP 5: ESPN PLAYER IDS
    # ========================================================================
    print("\n=== STEP 5: ESPN Player IDs ===\n")
    espn_players_scraper = EspnPlayerListScraper()
    espn_ids = espn_players_scraper.get_data(use_cache=not fetch_espn)

    # ========================================================================
    # STEP 6: MERGE ALL DATA
    # ========================================================================
    print("\n=== STEP 6: Merging Data ===\n")

    all_players = []

    for position, players in fp_data.items():
        print(f"  Processing {position}...")

        for player in players:
            name = player['name']

            # Join with FanDuel data
            fd = fd_data.get(name, {})

            # Skip if no salary or on IR
            if not fd.get('salary') or fd.get('injury_status') == 'IR':
                continue

            # Join with TD odds
            td = td_odds.get(name, {})

            # Join with ESPN ID
            espn_id = espn_ids.get(name, '')

            # Join with game lines (by team)
            team = fd.get('team', '')
            game_line = game_lines_by_team.get(team, {})

            # Create merged player dict
            merged = {
                'name': name,
                'position': position,
                'team': team,
                'game': fd.get('game', ''),
                'opponent': fd.get('opponent', ''),
                'salary': fd.get('salary', 0),
                'fppg': fd.get('fppg', 0),
                'injury_status': fd.get('injury_status', ''),
                'injury_detail': fd.get('injury_detail', ''),
                'fpProjPts': player['fpProjPts'],
                'tdOdds': td.get('tdOdds', ''),
                'tdProbability': td.get('tdProbability', 0),
                'espnId': espn_id,
                'proj_team_pts': game_line.get('proj_team_pts', ''),
                'proj_opp_pts': game_line.get('proj_opp_pts', ''),
            }

            all_players.append(merged)

        print(f"    Merged {len([p for p in all_players if p['position'] == position])} {position} players")

    # ========================================================================
    # STEP 7: ESPN PROJECTIONS (if requested)
    # ========================================================================
    if fetch_espn:
        print("\n=== STEP 7: ESPN Projections ===\n")
        espn_proj_scraper = EspnProjectionsScraper()

        # Only fetch for players with ESPN ID and FP projection >= 2.5
        espn_projections = espn_proj_scraper.fetch_all(all_players, min_projection=2.5)

        # Save to cache
        espn_proj_scraper._save_cache(espn_projections)
    else:
        print("\n=== STEP 7: ESPN Projections (cached) ===\n")
        espn_proj_scraper = EspnProjectionsScraper()
        espn_projections = espn_proj_scraper._load_cache()

    # Merge ESPN projections
    print("\n  Merging ESPN projections...")
    for player in all_players:
        espn_proj = espn_projections.get(player['name'], {})
        player.update({
            'espnScoreProjection': espn_proj.get('espnScoreProjection', ''),
            'espnLowScore': espn_proj.get('espnLowScore', ''),
            'espnHighScore': espn_proj.get('espnHighScore', ''),
            'espnOutsideProjection': espn_proj.get('espnOutsideProjection', ''),
            'espnSimulationProjection': espn_proj.get('espnSimulationProjection', ''),
        })

    # ========================================================================
    # STEP 8: WRITE OUTPUT CSV
    # ========================================================================
    print("\n=== STEP 8: Writing Output ===\n")

    df = pd.DataFrame(all_players)

    # Write combined CSV
    output_dir = Path('data/intermediate')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'players_raw.csv'
    df.to_csv(output_file, index=False)
    print(f"  Wrote {len(df)} players to {output_file}")

    # Write position-specific CSVs
    for position in ['QB', 'RB', 'WR', 'TE', 'D']:
        pos_df = df[df['position'] == position].copy()
        pos_file = output_dir / f'players_{position.lower()}.csv'
        pos_df.to_csv(pos_file, index=False)
        print(f"  Wrote {len(pos_df)} {position}s to {pos_file.name}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - data/intermediate/players_raw.csv (all players with raw data)")
    print("  - data/intermediate/players_{position}.csv (position-specific files)")
    print("  - cache/ directory (cached scraped data)")
    print("\nNext steps:")
    print("  - python game_script_continuous.py")
    print("  - python data_integration.py")
    print("  - python run_optimizer.py")


if __name__ == '__main__':
    main()
