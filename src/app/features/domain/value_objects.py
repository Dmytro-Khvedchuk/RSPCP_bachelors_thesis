"""Feature engineering domain value objects — indicator, target, and matrix configuration."""

from __future__ import annotations

from typing import Annotated, Self

import polars as pl
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator


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


_MIN_VOL_HORIZON: int = 2
"""Minimum forward volatility horizon — rolling std needs >= 2 observations."""


class TargetConfig(BaseModel, frozen=True):
    """Configuration for forward-looking regression targets.

    Targets are fundamentally different from indicators: they use future
    data (negative shifts) and must **never** appear in live inference.
    A separate config class enforces this semantic boundary.

    Default horizon rationale (for dollar bars producing ~2-3 bars/day):
        * 1-bar  ≈ 8-12 hours
        * 4-bar  ≈ 1-2 days
        * 24-bar ≈ 8-12 days

    Invariants:
        * ``forward_return_horizons`` must be non-empty with all values >= 1.
        * ``forward_vol_horizons`` must have all values >= 2
          (rolling std requires at least 2 observations).
        * No duplicate values in either horizon tuple.
    """

    forward_return_horizons: tuple[int, ...] = (1, 4, 24)
    """Bar-count horizons for forward log returns."""

    forward_vol_horizons: tuple[int, ...] = (4, 24)
    """Bar-count horizons for forward realized volatility."""

    close_col: str = "close"
    """Configurable close column name."""

    @field_validator("forward_return_horizons")
    @classmethod
    def _validate_return_horizons(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Validate forward return horizons are non-empty, positive, and unique.

        Args:
            v: Tuple of horizon values.

        Returns:
            Validated tuple.

        Raises:
            ValueError: If the tuple is empty, contains values < 1,
                or has duplicates.
        """
        if len(v) == 0:
            msg: str = "forward_return_horizons must be non-empty"
            raise ValueError(msg)
        if any(h < 1 for h in v):
            msg = f"All forward_return_horizons must be >= 1, got {v}"
            raise ValueError(msg)
        if len(v) != len(set(v)):
            msg = f"forward_return_horizons must not contain duplicates, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("forward_vol_horizons")
    @classmethod
    def _validate_vol_horizons(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Validate forward volatility horizons have values >= 2 and no duplicates.

        Args:
            v: Tuple of horizon values.

        Returns:
            Validated tuple.

        Raises:
            ValueError: If any value < 2 or there are duplicates.
        """
        if any(h < _MIN_VOL_HORIZON for h in v):
            msg: str = f"All forward_vol_horizons must be >= {_MIN_VOL_HORIZON} (std needs >= 2 observations), got {v}"
            raise ValueError(msg)
        if len(v) != len(set(v)):
            msg = f"forward_vol_horizons must not contain duplicates, got {v}"
            raise ValueError(msg)
        return v


class FeatureConfig(BaseModel, frozen=True):
    """Composite configuration for the feature matrix build pipeline.

    Bundles indicator and target settings so that downstream callers
    (profiling, backtest, model training) pass a single config object.

    Attributes:
        indicator_config: Controls all backward-looking indicator parameters.
        target_config: Controls all forward-looking target horizons.
        drop_na: Whether to drop rows with any NaN in computed columns.
        compute_targets: Set ``False`` for live inference where future data
            does not exist.
    """

    indicator_config: IndicatorConfig = PydanticField(
        default_factory=IndicatorConfig,  # type: ignore[reportArgumentType]  # pyright vs pydantic Annotated defaults
    )
    target_config: TargetConfig = PydanticField(default_factory=TargetConfig)
    drop_na: bool = True
    compute_targets: bool = True


class FeatureSet(BaseModel, frozen=True):
    """Structured output from :class:`FeatureMatrixBuilder`.

    Contains the ready-to-use DataFrame together with metadata that
    downstream consumers need (column partitioning, row-count diagnostics).

    Attributes:
        df: Full DataFrame with OHLCV + indicators + optional targets.
        feature_columns: Sorted tuple of backward-looking indicator column names.
        target_columns: Sorted tuple of forward-looking target column names
            (empty when ``compute_targets=False``).
        n_rows_raw: Row count before NaN dropping (for diagnostics).
        n_rows_clean: Row count after NaN dropping.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pl.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    n_rows_raw: int
    n_rows_clean: int

    @model_validator(mode="after")
    def _clean_leq_raw(self) -> Self:
        """Ensure clean row count does not exceed raw row count.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``n_rows_clean`` exceeds ``n_rows_raw``.
        """
        if self.n_rows_clean > self.n_rows_raw:
            msg: str = f"n_rows_clean ({self.n_rows_clean}) must be <= n_rows_raw ({self.n_rows_raw})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _clean_matches_df(self) -> Self:
        """Ensure clean row count matches the DataFrame length.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``n_rows_clean`` does not equal ``len(df)``.
        """
        df_len: int = len(self.df)
        if self.n_rows_clean != df_len:
            msg: str = f"n_rows_clean ({self.n_rows_clean}) must equal len(df) ({df_len})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _columns_exist(self) -> Self:
        """Ensure all declared feature and target columns exist in the DataFrame.

        Returns:
            Validated instance.

        Raises:
            ValueError: If any declared column is missing from the DataFrame.
        """
        df_cols: set[str] = set(self.df.columns)
        missing_features: set[str] = set(self.feature_columns) - df_cols
        if missing_features:
            msg: str = f"Feature columns missing from df: {sorted(missing_features)}"
            raise ValueError(msg)
        missing_targets: set[str] = set(self.target_columns) - df_cols
        if missing_targets:
            msg = f"Target columns missing from df: {sorted(missing_targets)}"
            raise ValueError(msg)
        return self
