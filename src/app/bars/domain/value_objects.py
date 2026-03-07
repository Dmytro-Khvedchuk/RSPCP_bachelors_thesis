"""Bar domain value objects — bar types and configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator


class BarType(StrEnum):
    """Supported bar aggregation types.

    Standard bars sample at fixed thresholds (time, tick count, volume,
    dollar volume).  Information-driven bars (imbalance & run) use adaptive
    thresholds based on the sequential trade-flow structure described in
    López de Prado, *Advances in Financial Machine Learning* (2018).
    """

    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK_IMBALANCE = "tick_imbalance"
    VOLUME_IMBALANCE = "volume_imbalance"
    DOLLAR_IMBALANCE = "dollar_imbalance"
    TICK_RUN = "tick_run"
    VOLUME_RUN = "volume_run"
    DOLLAR_RUN = "dollar_run"


# ---------------------------------------------------------------------------
# BarConfig
# ---------------------------------------------------------------------------


class BarConfig(BaseModel, frozen=True):
    """Configuration for alternative bar construction.

    Combines the bar type with its sampling parameters.  For standard bars
    only ``threshold`` is required.  Information-driven bars additionally
    use ``ewm_span`` and ``warmup_period`` for the adaptive EMA-based
    expected-imbalance / expected-run estimator.

    Invariants:
        * ``threshold`` must be positive.
        * ``ewm_span`` must be >= 10.
        * ``warmup_period`` must be >= 1.
        * ``warmup_period`` must not exceed ``ewm_span``.
    """

    bar_type: BarType

    threshold: Annotated[
        float,
        PydanticField(gt=0, description="Sampling threshold (ticks / volume / dollars)"),
    ]

    ewm_span: Annotated[
        int,
        PydanticField(
            default=100,
            ge=10,
            description="EWMA span for adaptive threshold estimation (information-driven bars)",
        ),
    ]

    warmup_period: Annotated[
        int,
        PydanticField(
            default=100,
            ge=1,
            description="Number of initial bars used as warmup before adaptive thresholds kick in",
        ),
    ]

    @model_validator(mode="after")
    def _warmup_lte_ewm_span(self) -> Self:
        """Ensure warmup period does not exceed the EWM span.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``warmup_period`` exceeds ``ewm_span``.
        """
        if self.warmup_period > self.ewm_span:
            msg: str = f"warmup_period ({self.warmup_period}) must not exceed ewm_span ({self.ewm_span})"
            raise ValueError(msg)
        return self

    @property
    def is_information_driven(self) -> bool:
        """Return ``True`` if this bar type uses adaptive thresholds.

        Returns:
            Whether the bar type is an imbalance or run variant.
        """
        info_types: set[BarType] = {
            BarType.TICK_IMBALANCE,
            BarType.VOLUME_IMBALANCE,
            BarType.DOLLAR_IMBALANCE,
            BarType.TICK_RUN,
            BarType.VOLUME_RUN,
            BarType.DOLLAR_RUN,
        }
        return self.bar_type in info_types

    @property
    def config_hash(self) -> str:
        """Return a deterministic hash string for this configuration.

        Used as a storage key so that bars produced with different parameters
        are stored separately.

        Returns:
            Hex digest uniquely identifying this configuration.
        """
        import hashlib

        payload: str = self.model_dump_json()
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
