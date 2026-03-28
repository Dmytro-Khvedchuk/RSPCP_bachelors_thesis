"""Position sizing implementations — fixed-fractional and regime-conditional."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import PortfolioSnapshot


# ---------------------------------------------------------------------------
# FixedFractionalSizer
# ---------------------------------------------------------------------------


class FixedFractionalSizer(BaseModel, frozen=True):
    """Fixed-fraction position sizer.

    Allocates a constant fraction of portfolio equity per trade, scaled
    by signal strength.  Returns a **notional USD amount** that the
    execution engine converts to base-asset units by dividing by the
    fill price.

    Attributes:
        fraction: Fraction of equity to risk per trade (default 2 %).
    """

    fraction: Annotated[
        float,
        PydanticField(
            default=0.02,
            gt=0.0,
            le=1.0,
            description="Fraction of equity per trade (e.g. 0.02 = 2 %)",
        ),
    ]

    def size(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        volatility: float,  # noqa: ARG002
    ) -> float:
        """Compute notional position size for a given signal.

        Args:
            signal: Trading signal with direction and conviction strength.
            portfolio: Current portfolio state (equity, cash, positions).
            volatility: Unused by this sizer; accepted for protocol
                compatibility.

        Returns:
            Notional USD amount.  The execution engine divides by the
            fill price to obtain base-asset units.  Returns zero when
            equity is non-positive.
        """
        if portfolio.equity <= 0.0:
            return 0.0
        notional: float = portfolio.equity * self.fraction * signal.strength
        return notional


# ---------------------------------------------------------------------------
# RegimeConditionalSizer
# ---------------------------------------------------------------------------


class RegimeConditionalSizer(BaseModel, frozen=True):
    """Regime-aware position sizer informed by RC2 analysis.

    Scales position size using the gap between estimated directional
    accuracy and the break-even threshold.  Applies additional
    reductions when permutation entropy is high (random-walk regime) or
    when the asset belongs to a lower-confidence tier.

    Attributes:
        base_fraction: Base equity fraction (same semantics as
            :class:`FixedFractionalSizer`).
        break_even_da: Directional-accuracy break-even threshold.
        high_entropy_threshold: Permutation-entropy level above which
            the market is treated as near-random.
        high_entropy_reduction: Multiplicative reduction applied in
            high-entropy regimes.
        da_estimate: Current directional-accuracy estimate.
        pe_estimate: Current permutation-entropy estimate.
        tier_b_kelly_multiplier: Kelly-fraction multiplier for Tier-B
            assets (e.g. SOLUSDT).
        is_tier_b: Whether the asset is classified as Tier B.
    """

    base_fraction: Annotated[
        float,
        PydanticField(
            default=0.02,
            gt=0.0,
            le=1.0,
            description="Base equity fraction per trade",
        ),
    ]

    break_even_da: Annotated[
        float,
        PydanticField(
            default=0.55,
            gt=0.0,
            le=1.0,
            description="DA break-even threshold from RC2/RC7",
        ),
    ]

    high_entropy_threshold: Annotated[
        float,
        PydanticField(
            default=0.98,
            gt=0.0,
            le=1.0,
            description="PE level above which market is near-random",
        ),
    ]

    high_entropy_reduction: Annotated[
        float,
        PydanticField(
            default=0.5,
            gt=0.0,
            le=1.0,
            description="Multiplicative reduction in high-entropy regimes",
        ),
    ]

    da_estimate: Annotated[
        float,
        PydanticField(
            default=0.55,
            ge=0.0,
            le=1.0,
            description="Current directional-accuracy estimate",
        ),
    ]

    pe_estimate: Annotated[
        float,
        PydanticField(
            default=0.95,
            ge=0.0,
            le=1.0,
            description="Current permutation-entropy estimate",
        ),
    ]

    tier_b_kelly_multiplier: Annotated[
        float,
        PydanticField(
            default=0.5,
            gt=0.0,
            le=1.0,
            description="Kelly multiplier for Tier-B assets",
        ),
    ]

    is_tier_b: Annotated[
        bool,
        PydanticField(
            default=False,
            description="Whether the asset is Tier B (e.g. SOLUSDT)",
        ),
    ]

    def size(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        volatility: float,  # noqa: ARG002
    ) -> float:
        """Compute regime-adjusted notional position size.

        The sizing formula scales the base fraction by the gap between
        the estimated directional accuracy and the break-even DA.  When
        DA is at or below break-even, the sizer returns zero (NO-TRADE).

        Args:
            signal: Trading signal with direction and conviction strength.
            portfolio: Current portfolio state (equity, cash, positions).
            volatility: Unused by this sizer; accepted for protocol
                compatibility.

        Returns:
            Notional USD amount (zero when DA <= break-even or equity
            is non-positive).
        """
        if portfolio.equity <= 0.0:
            return 0.0

        # DA-based edge scaling — zero when at or below break-even
        da_scale: float = max(0.0, (self.da_estimate - self.break_even_da) / self.break_even_da)
        if da_scale == 0.0:
            return 0.0

        notional: float = portfolio.equity * self.base_fraction * signal.strength * da_scale

        # High-entropy reduction
        if self.pe_estimate > self.high_entropy_threshold:
            notional *= self.high_entropy_reduction

        # Tier-B reduction (e.g. SOLUSDT)
        if self.is_tier_b:
            notional *= self.tier_b_kelly_multiplier

        return notional
