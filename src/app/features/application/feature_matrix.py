"""Feature matrix builder — orchestrates indicators + targets into a ready-to-use FeatureSet.

The :class:`FeatureMatrixBuilder` is the single entry point for downstream
code (profiling, backtest, model training) to obtain a clean feature matrix
from raw OHLCV data.  It chains indicator computation, optional target
construction, NaN removal, and metadata assembly.
"""

from __future__ import annotations

import polars as pl

from src.app.features.application.indicators import compute_all_indicators
from src.app.features.application.targets import compute_all_targets
from src.app.features.domain.value_objects import FeatureConfig, FeatureSet


class FeatureMatrixBuilder:
    """Stateless builder that assembles a complete feature matrix.

    No constructor dependencies — pure computation, no I/O.  Call
    :meth:`build` with an OHLCV DataFrame and a :class:`FeatureConfig`
    to get a :class:`FeatureSet` back.
    """

    def build(self, df: pl.DataFrame, config: FeatureConfig) -> FeatureSet:
        """Build a complete feature matrix from raw OHLCV data.

        Pipeline:
            1. Compute backward-looking indicators.
            2. Compute forward-looking targets (if ``config.compute_targets``).
            3. Identify new feature and target columns by diffing column sets.
            4. Drop rows with NaN in computed columns (if ``config.drop_na``).
            5. Return a :class:`FeatureSet` with metadata.

        Args:
            df: Polars DataFrame with OHLCV columns
                (``open``, ``high``, ``low``, ``close``, ``volume``).
            config: Composite configuration controlling the build pipeline.

        Returns:
            A :class:`FeatureSet` containing the clean DataFrame and metadata.
        """
        base_columns: list[str] = df.columns

        # Step 1: backward-looking indicators
        result: pl.DataFrame = compute_all_indicators(df, config.indicator_config)
        feature_columns: tuple[str, ...] = self._identify_new_columns(base_columns, result.columns)

        # Step 2: forward-looking targets (optional)
        target_columns: tuple[str, ...] = ()
        if config.compute_targets:
            cols_before_targets: list[str] = result.columns
            result = compute_all_targets(result, config.target_config)
            target_columns = self._identify_new_columns(cols_before_targets, result.columns)

        # Step 3: record raw row count, then drop NaNs
        n_rows_raw: int = len(result)
        if config.drop_na:
            subset: list[str] = list(feature_columns) + list(target_columns)
            result = result.drop_nulls(subset=subset)

        n_rows_clean: int = len(result)

        return FeatureSet(
            df=result,
            feature_columns=feature_columns,
            target_columns=target_columns,
            n_rows_raw=n_rows_raw,
            n_rows_clean=n_rows_clean,
        )

    @staticmethod
    def _identify_new_columns(before: list[str], after: list[str]) -> tuple[str, ...]:
        """Return sorted tuple of columns present in *after* but not in *before*.

        Args:
            before: Column names before computation.
            after: Column names after computation.

        Returns:
            Sorted tuple of newly added column names.
        """
        before_set: set[str] = set(before)
        return tuple(sorted(col for col in after if col not in before_set))
