"""Feature engineering domain value objects — indicator configuration."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator


class IndicatorConfig(BaseModel, frozen=True):
    """Configuration for all technical indicator parameters.

    Every parameter controlling indicator computation is centralised here
    so that no magic numbers appear in the indicator functions themselves.
    The config is immutable (``frozen=True``) and validated at construction
    time.

    Invariants:
        * ``ema_fast_span`` must be strictly less than ``ema_slow_span``.
        * ``clip_lower`` must be strictly less than ``clip_upper``.
        * All window / period parameters have minimum-value constraints
          that ensure statistical validity of the rolling computations.
    """

    # ----- Returns -----
    return_horizons: tuple[int, ...] = (1, 4, 12, 24)

    # ----- Volatility -----
    realized_vol_windows: tuple[int, ...] = (12, 24, 48)

    garman_klass_window: Annotated[
        int,
        PydanticField(default=24, ge=2, description="Rolling window for Garman-Klass volatility"),
    ]

    parkinson_window: Annotated[
        int,
        PydanticField(default=24, ge=2, description="Rolling window for Parkinson volatility"),
    ]

    atr_period: Annotated[
        int,
        PydanticField(default=14, ge=2, description="ATR lookback period"),
    ]

    atr_wilder: bool = True

    # ----- Momentum -----
    ema_fast_span: Annotated[
        int,
        PydanticField(default=8, ge=2, description="Fast EMA span for crossover signal"),
    ]

    ema_slow_span: Annotated[
        int,
        PydanticField(default=21, ge=5, description="Slow EMA span for crossover signal"),
    ]

    rsi_period: Annotated[
        int,
        PydanticField(default=14, ge=2, description="RSI lookback period (Wilder smoothing)"),
    ]

    roc_periods: tuple[int, ...] = (1, 4, 12)

    slope_window: Annotated[
        int,
        PydanticField(default=14, ge=3, description="Rolling linear regression slope window"),
    ]

    # ----- Volume -----
    volume_zscore_window: Annotated[
        int,
        PydanticField(default=24, ge=5, description="Rolling window for volume z-score"),
    ]

    obv_slope_window: Annotated[
        int,
        PydanticField(default=14, ge=3, description="Rolling window for OBV slope"),
    ]

    amihud_window: Annotated[
        int,
        PydanticField(default=24, ge=5, description="Rolling window for Amihud illiquidity ratio"),
    ]

    # ----- Statistical -----
    hurst_window: Annotated[
        int,
        PydanticField(default=100, ge=20, description="Rolling window for Hurst exponent (R/S analysis)"),
    ]

    return_zscore_window: Annotated[
        int,
        PydanticField(default=24, ge=5, description="Rolling window for return z-score"),
    ]

    bollinger_window: Annotated[
        int,
        PydanticField(default=20, ge=5, description="Bollinger Bands lookback window"),
    ]

    bollinger_num_std: Annotated[
        float,
        PydanticField(default=2.0, gt=0, description="Number of standard deviations for Bollinger Bands"),
    ]

    # ----- Clipping -----
    clip_lower: float = -5.0
    clip_upper: float = 5.0

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> Self:
        """Ensure fast EMA span is strictly less than slow EMA span.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``ema_fast_span`` is not less than ``ema_slow_span``.
        """
        if self.ema_fast_span >= self.ema_slow_span:
            msg: str = (
                f"ema_fast_span ({self.ema_fast_span}) must be strictly less than ema_slow_span ({self.ema_slow_span})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _clip_bounds_ordered(self) -> Self:
        """Ensure clip_lower is strictly less than clip_upper.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``clip_lower`` is not less than ``clip_upper``.
        """
        if self.clip_lower >= self.clip_upper:
            msg: str = f"clip_lower ({self.clip_lower}) must be strictly less than clip_upper ({self.clip_upper})"
            raise ValueError(msg)
        return self
