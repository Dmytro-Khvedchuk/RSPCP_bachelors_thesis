"""Baseline strategies — buy-and-hold floor and random null hypothesis."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import numpy as np
import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import PortfolioSnapshot, Side
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# BuyAndHoldStrategy
# ---------------------------------------------------------------------------


class BuyAndHoldStrategy(BaseModel, frozen=True):
    """Enter long on the first bar and hold forever.

    Emits a single LONG signal with full conviction on the first bar
    (detected by an empty positions dict).  All subsequent bars produce
    no signals.  Buy-and-hold Sharpe (0.576 from RC2) is the minimum
    hurdle: any strategy below this has negative alpha.

    Attributes:
        asset: Trading asset to hold.
    """

    asset: Asset

    def on_bar(
        self,
        timestamp: datetime,
        features: pl.DataFrame,  # noqa: ARG002
        portfolio: PortfolioSnapshot,
    ) -> list[Signal]:
        """Emit a LONG signal on the first bar only.

        Args:
            timestamp: Current bar timestamp (UTC).
            features: Feature DataFrame for the current bar (unused).
            portfolio: Current portfolio snapshot.

        Returns:
            Single-element list on the first bar, empty list thereafter.
        """
        has_position: bool = len(portfolio.positions) > 0
        if has_position:
            return []
        signal: Signal = Signal(
            asset=self.asset,
            side=Side.LONG,
            strength=1.0,
            timestamp=timestamp,
        )
        return [signal]


# ---------------------------------------------------------------------------
# RandomStrategy
# ---------------------------------------------------------------------------


_EQUAL_PROBABILITY: float = 0.5


class RandomStrategy(BaseModel, frozen=True):
    """Random signal generator preserving frequency — the null hypothesis.

    On each bar, emits a signal with probability ``signal_frequency``.
    Direction is uniformly random (50/50 LONG/SHORT).  This baseline
    implements the White (2000) null: same signal count, random timing.
    Phase 15 Monte Carlo validation runs many seeds to build the null
    distribution.

    Attributes:
        asset: Trading asset.
        signal_frequency: Probability of emitting a signal per bar.
        seed: RNG seed for reproducibility.
    """

    asset: Asset

    signal_frequency: Annotated[
        float,
        PydanticField(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Probability of emitting a signal on any given bar",
        ),
    ]

    seed: Annotated[
        int,
        PydanticField(default=42, ge=0, description="RNG seed for reproducibility"),
    ]

    # Private RNG — excluded from serialisation and hashing via model_config
    _rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        """Lazily initialise the NumPy random generator.

        Returns:
            Seeded ``numpy.random.Generator`` instance.
        """
        cached: np.random.Generator | None = self._rng
        if cached is not None:
            return cached
        rng: np.random.Generator = np.random.default_rng(self.seed)
        object.__setattr__(self, "_rng", rng)
        return rng

    def on_bar(
        self,
        timestamp: datetime,
        features: pl.DataFrame,  # noqa: ARG002
        portfolio: PortfolioSnapshot,  # noqa: ARG002
    ) -> list[Signal]:
        """Possibly emit a random signal based on ``signal_frequency``.

        Args:
            timestamp: Current bar timestamp (UTC).
            features: Feature DataFrame for the current bar (unused).
            portfolio: Current portfolio snapshot (unused).

        Returns:
            Single-element list with probability ``signal_frequency``,
            empty list otherwise.
        """
        rng: np.random.Generator = self._get_rng()
        emit: bool = bool(rng.random() < self.signal_frequency)
        if not emit:
            return []
        side: Side = Side.LONG if bool(rng.random() < _EQUAL_PROBABILITY) else Side.SHORT
        signal: Signal = Signal(
            asset=self.asset,
            side=side,
            strength=1.0,
            timestamp=timestamp,
        )
        return [signal]
