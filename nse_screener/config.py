"""Central configuration for the NSE screener.

All tunable constants live here. Nothing downstream should hard-code a
threshold; it should read it from a ``ScreenerConfig`` instance instead.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

#: Minimum holding horizon, in trading sessions, for which the momentum
#: edge has documented positive net-of-cost expectancy. This is a hard
#: constraint (see build spec section 0.2) and is not configurable from
#: the CLI.
HOLDING_HORIZON_SESSIONS: int = 63


class ScreenerConfig(BaseModel):
    """All tunable constants for a single screener run.

    Instances are immutable after construction so that a run's parameters
    cannot drift mid-execution.
    """

    model_config = ConfigDict(frozen=True)

    # -- Data acquisition -------------------------------------------------
    price_history_period: str = "3y"
    index_symbol: str = "^NSEI"

    # -- Data integrity (data.py) ------------------------------------------
    min_sessions: int = 300
    calendar_alignment_lookback: int = 252
    max_missing_calendar_fraction: float = 0.02
    max_recency_gap_days: int = 5
    impossible_move_abs_return: float = 0.50

    # -- Market gate (gate.py) ---------------------------------------------
    gate_ma_short: int = 50
    gate_ma_long: int = 200
    gate_slope_lookback: int = 21
    gate_breadth_high_proximity: float = 0.15

    # -- Momentum (factors/momentum.py) ------------------------------------
    # ``momentum_min_sessions`` is a floor only. The true requirement is
    # derived inside ``weighted_relative_strength`` as
    # ``max(momentum_min_sessions, 4 * momentum_quarter_sessions + 1)``
    # (253 with the defaults), because the oldest term looks back four
    # full quarters.
    momentum_min_sessions: int = 252
    momentum_quarter_sessions: int = 63
    momentum_weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)

    # -- Forensic screen (factors/forensic.py) ------------------------------
    forensic_cfo_ni_min: float = 0.8
    forensic_accrual_ratio_max: float = 0.10
    forensic_dso_vs_revenue_growth_pp: float = 15.0
    forensic_inventory_vs_cogs_growth_pp: float = 20.0
    forensic_leverage_jump_fraction: float = 0.50
    forensic_disqualify_flag_count: int = 2

    # -- Eligibility / liquidity (ranking.py) --------------------------------
    min_avg_traded_value_crore: float = 25.0
    liquidity_window_sessions: int = 50
    eligibility_min_sessions: int = 300
    fundamentals_min_years: int = 2

    # -- Composite ranking weights (ranking.py) ------------------------------
    # WEIGHTS ARE UNVALIDATED. They reflect evidence strength of each factor,
    # NOT an optimised blend. Do not tune these against historical returns
    # without a proper out-of-sample split -- in-sample optimisation of these
    # weights will produce a curve-fitted result that fails live. Changing
    # them requires a fresh backtest with a matched random-entry benchmark.
    #
    # weight_value was added after the original four and rebalances all
    # five to sum to 1.0. Value (earnings yield / book-to-price) has
    # first-tier academic evidence on its own, on par with momentum and
    # profitability -- but it is weighted below rs and f_score here because
    # (a) it has no NSE-specific validation in this tool, unlike momentum's
    # backing, and (b) value and momentum are documented to be negatively
    # correlated factors, so overweighting value directly fights the
    # tool's primary signal rather than diversifying it. gross_profitability
    # and roe were trimmed to make room, roe more so since it is the most
    # redundant of the four with gross_profitability (both are "quality"
    # proxies) rather than an independent signal.
    weight_rs: float = 0.35
    weight_f_score: float = 0.20
    weight_value: float = 0.20
    weight_gross_profitability: float = 0.15
    weight_roe: float = 0.10

    top_n: int = 5

    #: See module-level constant. Kept on the config object for convenience
    #: when building reports, but never overridable to a smaller value.
    holding_horizon_sessions: int = HOLDING_HORIZON_SESSIONS

    @field_validator("holding_horizon_sessions")
    @classmethod
    def _horizon_is_fixed(cls, value: int) -> int:
        if value != HOLDING_HORIZON_SESSIONS:
            raise ValueError(
                "holding_horizon_sessions is a hard constraint and cannot be "
                f"changed from {HOLDING_HORIZON_SESSIONS} sessions"
            )
        return value

    @field_validator(
        "weight_rs",
        "weight_f_score",
        "weight_value",
        "weight_gross_profitability",
        "weight_roe",
    )
    @classmethod
    def _weight_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("composite weights must be within [0, 1]")
        return value

    @field_validator("momentum_weights")
    @classmethod
    def _momentum_weights_sum_to_one(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if abs(sum(value) - 1.0) > 1e-9:
            raise ValueError("momentum_weights must sum to 1.0")
        if value[0] < max(value[1:]):
            # Mirrors the check in weighted_relative_strength. Enforced here
            # so a misconfiguration fails at construction rather than being
            # caught by ranking.py's per-symbol ValueError handling and
            # reported as every symbol being momentum-uncomputable.
            raise ValueError(
                "momentum_weights[0] (most recent quarter) must be at least as "
                "large as every other weight"
            )
        return value

    @model_validator(mode="after")
    def _composite_weights_sum_to_one(self) -> "ScreenerConfig":
        total = (
            self.weight_rs
            + self.weight_f_score
            + self.weight_value
            + self.weight_gross_profitability
            + self.weight_roe
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("composite ranking weights must sum to 1.0")
        return self

    @model_validator(mode="after")
    def _eligibility_not_looser_than_integrity(self) -> "ScreenerConfig":
        if self.eligibility_min_sessions < self.min_sessions:
            raise ValueError(
                "eligibility_min_sessions cannot be lower than min_sessions "
                f"({self.eligibility_min_sessions} < {self.min_sessions}): every "
                "symbol accepted by data-integrity validation already has at "
                "least min_sessions of history, so a lower eligibility floor "
                "would be silently unenforceable and just misleading config"
            )
        return self
