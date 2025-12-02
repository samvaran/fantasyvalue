"""
Data models for fantasy football projections.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Position(Enum):
    """Player positions."""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    DST = "D"


@dataclass
class Player:
    """
    Player with all projection data.

    Data flows through pipeline stages:
    1. Raw (after scraping/merging)
    2. Processed (after consensus/variance)
    3. Final (after distribution fitting)
    """

    # ==================== IDENTITY ====================
    name: str
    position: str
    team: str = ""
    opponent: str = ""
    game: str = ""  # e.g., "KC@BUF"

    # ==================== SALARY ====================
    salary: int = 0
    fppg: float = 0.0  # FanDuel's own historical average

    # ==================== RAW PROJECTIONS ====================
    # FantasyPros
    fp_projection: Optional[float] = None

    # ESPN (converted to FP scale via regression)
    espn_score: Optional[float] = None
    espn_low: Optional[float] = None
    espn_high: Optional[float] = None
    espn_outside: Optional[float] = None
    espn_simulation: Optional[float] = None
    espn_id: Optional[str] = None

    # ==================== GAME ENVIRONMENT ====================
    proj_team_pts: Optional[float] = None
    proj_opp_pts: Optional[float] = None

    # ==================== TD PROBABILITY ====================
    td_odds: Optional[str] = None       # e.g., "+150"
    td_probability: Optional[float] = None  # 0-100 scale

    # ==================== INJURY ====================
    injury_status: Optional[str] = None     # "Q", "D", "O", "IR"
    injury_detail: Optional[str] = None

    # ==================== PROCESSED VALUES (filled by pipeline) ====================
    # Consensus (weighted projection)
    consensus: Optional[float] = None

    # Variance metrics (player archetype)
    floor_variance: Optional[float] = None      # 0.1-1.0
    ceiling_variance: Optional[float] = None    # 0.3-2.0
    variance_source: Optional[str] = None       # "ESPN" or "FALLBACK"

    # Adjusted floor/ceiling (after game script, pace, TD)
    floor: Optional[float] = None   # P10 target
    ceiling: Optional[float] = None # P90 target

    # ==================== DISTRIBUTION PARAMETERS ====================
    mu: Optional[float] = None      # log-space location parameter
    sigma: Optional[float] = None   # log-space scale parameter
    shift: Optional[float] = None   # shift parameter (3-param log-normal)

    # Key percentiles
    p10: Optional[float] = None
    p50: Optional[float] = None  # Median
    p90: Optional[float] = None
    mean: Optional[float] = None  # Expected value (should equal consensus)

    # ==================== COMPUTED PROPERTIES ====================
    @property
    def value(self) -> float:
        """Points per $1K salary (consensus-based)"""
        if self.salary == 0 or self.consensus is None:
            return 0.0
        return self.consensus / (self.salary / 1000)

    @property
    def ceiling_value(self) -> float:
        """P90 points per $1K salary"""
        if self.salary == 0 or self.p90 is None:
            return 0.0
        return self.p90 / (self.salary / 1000)

    @property
    def spread(self) -> Optional[float]:
        """Team spread (positive = favorite)"""
        if self.proj_team_pts is None or self.proj_opp_pts is None:
            return None
        return self.proj_team_pts - self.proj_opp_pts

    @property
    def total(self) -> Optional[float]:
        """Game total (over/under)"""
        if self.proj_team_pts is None or self.proj_opp_pts is None:
            return None
        return self.proj_team_pts + self.proj_opp_pts

    @property
    def is_injured(self) -> bool:
        """Check if player has injury designation"""
        return self.injury_status in ["Q", "D", "O", "IR"]

    @property
    def is_favorite(self) -> bool:
        """Check if team is favored"""
        if self.spread is None:
            return False
        return self.spread > 0

    @property
    def is_underdog(self) -> bool:
        """Check if team is underdog"""
        if self.spread is None:
            return False
        return self.spread < 0

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export"""
        return {
            'name': self.name,
            'position': self.position,
            'team': self.team,
            'game': self.game,
            'salary': self.salary,
            'value': self.value,
            'ceiling_value': self.ceiling_value,
            'consensus': self.consensus,
            'p90': self.p90,
            'p10': self.p10,
            'mu': self.mu,
            'sigma': self.sigma,
            'shift': self.shift,
            'mean': self.mean,
            'p50': self.p50,
            'floor_variance': self.floor_variance,
            'ceiling_variance': self.ceiling_variance,
            'variance_source': self.variance_source,
            'fp_projection': self.fp_projection,
            'espn_score': self.espn_score,
            'espn_low': self.espn_low,
            'espn_high': self.espn_high,
            'espn_outside': self.espn_outside,
            'espn_simulation': self.espn_simulation,
            'proj_team_pts': self.proj_team_pts,
            'proj_opp_pts': self.proj_opp_pts,
            'td_odds': self.td_odds,
            'td_probability': self.td_probability,
            'espn_id': self.espn_id,
            'injury': f"{self.injury_status or ''}: {self.injury_detail or ''}".strip(": "),
        }
