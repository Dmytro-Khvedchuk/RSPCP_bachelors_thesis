"""Bar comparison service — duration, count variability, and volume profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.research.domain.value_objects import BarDurationStats


class BarComparator:
    """Stateless utility for comparing alternative bar types.

    Provides methods to analyse bar count regularity, duration statistics,
    count variability across bar types, and volume profiles.  All inputs and
    outputs are Pandas DataFrames/Series so that results integrate naturally
    with the research notebook ecosystem.
    """

    # -- Bar count per period ------------------------------------------------

    def bar_count_per_period(  # noqa: PLR6301
        self,
        bars_df: pd.DataFrame,
        freq: str = "W",
    ) -> pd.Series:  # type: ignore[type-arg]
        """Count the number of bars in each calendar period.

        Groups bars by a calendar frequency (e.g. weekly, monthly) using the
        ``start_ts`` column, then counts rows in each group.

        Args:
            bars_df: Bar DataFrame with a ``start_ts`` datetime column.
            freq: Pandas offset alias for the grouping period
                (e.g. ``"W"`` for weekly, ``"ME"`` for month-end).

        Returns:
            Series indexed by period start with bar counts as values.
        """
        grouped: pd.core.groupby.DataFrameGroupBy = bars_df.groupby(  # type: ignore[type-arg]
            pd.Grouper(key="start_ts", freq=freq),
        )
        counts: pd.Series = grouped.size()  # type: ignore[assignment]
        return counts

    # -- Duration statistics -------------------------------------------------

    def compute_duration_stats(  # noqa: PLR6301
        self,
        bars_df: pd.DataFrame,
        asset: str,
        bar_type: str,
    ) -> BarDurationStats:
        """Compute descriptive statistics for bar durations.

        Duration is defined as ``end_ts - start_ts`` for each bar, converted
        to minutes.

        Args:
            bars_df: Bar DataFrame with ``start_ts`` and ``end_ts`` datetime
                columns.
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``).

        Returns:
            Frozen :class:`BarDurationStats` value object with mean, median,
            std, min, max, and coefficient of variation.
        """
        durations: pd.Series = (  # type: ignore[type-arg]
            (bars_df["end_ts"] - bars_df["start_ts"]).dt.total_seconds() / 60.0
        )
        duration_values: np.ndarray = durations.to_numpy(dtype=np.float64)  # type: ignore[assignment]

        mean_dur: float = float(np.mean(duration_values))
        median_dur: float = float(np.median(duration_values))
        std_dur: float = float(np.std(duration_values, ddof=1)) if len(duration_values) > 1 else 0.0
        min_dur: float = float(np.min(duration_values))
        max_dur: float = float(np.max(duration_values))
        cv: float = std_dur / mean_dur if mean_dur > 0.0 else 0.0

        return BarDurationStats(
            asset=asset,
            bar_type=bar_type,
            mean_duration_minutes=mean_dur,
            median_duration_minutes=median_dur,
            std_duration_minutes=std_dur,
            min_duration_minutes=min_dur,
            max_duration_minutes=max_dur,
            cv=cv,
        )

    # -- Cross-bar-type count variability ------------------------------------

    def compare_bar_count_variability(
        self,
        bar_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compare weekly bar-count variability across bar types.

        For each bar type, computes weekly bar counts using
        :meth:`bar_count_per_period`, then derives the mean count, standard
        deviation, and coefficient of variation (CV = std / mean).

        Args:
            bar_data: Mapping from bar-type name to the corresponding bar
                DataFrame (must contain a ``start_ts`` column).

        Returns:
            DataFrame with columns: ``bar_type``, ``mean_count``,
            ``std_count``, ``cv``.  One row per bar type.
        """
        records: list[dict[str, object]] = []
        for bt, df in bar_data.items():
            weekly_counts: pd.Series = self.bar_count_per_period(df, freq="W")  # type: ignore[type-arg]
            count_values: np.ndarray = weekly_counts.to_numpy(dtype=np.float64)  # type: ignore[assignment]

            mean_c: float = float(np.mean(count_values))
            std_c: float = float(np.std(count_values, ddof=1)) if len(count_values) > 1 else 0.0
            cv_c: float = std_c / mean_c if mean_c > 0.0 else 0.0

            records.append(
                {
                    "bar_type": bt,
                    "mean_count": mean_c,
                    "std_count": std_c,
                    "cv": cv_c,
                }
            )

        result: pd.DataFrame = pd.DataFrame(records)
        return result

    # -- Volume profile ------------------------------------------------------

    def compute_volume_profile(  # noqa: PLR6301
        self,
        ohlcv_df: pd.DataFrame,
        freq: str = "W",
    ) -> pd.DataFrame:
        """Aggregate volume by calendar period.

        Groups the OHLCV DataFrame by the specified frequency using the
        ``timestamp`` column, then sums volume within each period.

        Args:
            ohlcv_df: OHLCV DataFrame with ``timestamp`` and ``volume``
                columns.
            freq: Pandas offset alias for the grouping period
                (e.g. ``"W"`` for weekly, ``"ME"`` for month-end).

        Returns:
            DataFrame with columns: ``period``, ``total_volume``.
        """
        grouped: pd.core.groupby.DataFrameGroupBy = ohlcv_df.groupby(  # type: ignore[type-arg]
            pd.Grouper(key="timestamp", freq=freq),
        )
        volume_sums: pd.Series = grouped["volume"].sum()  # type: ignore[assignment]

        result: pd.DataFrame = pd.DataFrame(
            {
                "period": volume_sums.index,
                "total_volume": volume_sums.values,
            }
        )
        return result
