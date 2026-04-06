"""Strategy domain protocols — batch signal generation interface."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from src.app.features.domain.value_objects import FeatureSet


class IStrategy(Protocol):
    """Structural interface for batch signal generation strategies.

    Unlike the per-bar :class:`backtest.domain.protocols.IStrategy`, which
    processes one bar at a time inside the event loop, this protocol
    operates on an entire :class:`FeatureSet` at once and returns a
    signal DataFrame.  A downstream adapter bridges this batch interface
    to the backtest engine's event-driven contract.

    The returned DataFrame **must** contain at least the following
    columns (additional strategy-specific columns are permitted):

    ============  ==========  ==========================================
    Column        Dtype       Description
    ============  ==========  ==========================================
    ``timestamp`` Datetime    Bar timestamp (UTC, from the FeatureSet).
    ``side``      Utf8        Direction: ``"long"``, ``"short"``, or
                              ``"flat"`` (no position).
    ``strength``  Float64     Conviction score in ``[0.0, 1.0]``.
                              ``0.0`` means no conviction; ``1.0``
                              means maximum conviction.
    ============  ==========  ==========================================
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name.

        Returns:
            Strategy identifier (e.g. ``"ema_crossover"``).
        """
        ...

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce directional signals for every bar in *feature_set*.

        Implementations must respect temporal ordering: the signal for
        bar *t* may only use features available at or before *t*.
        Using future feature values constitutes look-ahead bias.

        Args:
            feature_set: Structured output from the feature matrix
                builder, containing backward-looking indicators and
                (optionally) forward-looking targets.  Strategies
                must use only ``feature_set.feature_columns`` — never
                ``feature_set.target_columns``.

        Returns:
            Polars DataFrame with at least ``timestamp``, ``side``, and
            ``strength`` columns.  One row per bar in the input.
        """
        ...
