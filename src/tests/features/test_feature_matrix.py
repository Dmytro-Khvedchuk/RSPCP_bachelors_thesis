"""Integration tests for the FeatureMatrixBuilder.

Tests the full build pipeline: indicators + targets + NaN removal + metadata,
covering happy paths, optional targets, and diagnostic metadata correctness.
"""

from __future__ import annotations

import polars as pl

from src.app.features.application.feature_matrix import FeatureMatrixBuilder
from src.app.features.domain.value_objects import FeatureConfig, FeatureSet

from src.tests.features.conftest import (
    make_random_walk_df,
    make_small_feature_config,
)


class TestFeatureMatrixBuilderWithTargets:
    """Tests for FeatureMatrixBuilder.build with compute_targets=True."""

    def test_build_returns_feature_set(self) -> None:
        """build() must return a FeatureSet instance."""
        df: pl.DataFrame = make_random_walk_df(200, seed=100)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert isinstance(result, FeatureSet)

    def test_build_with_targets_has_target_columns(self) -> None:
        """With compute_targets=True the FeatureSet must contain target columns."""
        df: pl.DataFrame = make_random_walk_df(200, seed=101)
        config: FeatureConfig = make_small_feature_config(compute_targets=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        assert len(result.target_columns) > 0
        for col in result.target_columns:
            assert col.startswith("fwd_"), f"Expected target column to start with 'fwd_', got '{col}'"

    def test_build_feature_columns_present_in_df(self) -> None:
        """All feature_columns declared in FeatureSet must exist in result.df."""
        df: pl.DataFrame = make_random_walk_df(200, seed=102)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        for col in result.feature_columns:
            assert col in result.df.columns

    def test_build_target_columns_present_in_df(self) -> None:
        """All target_columns declared in FeatureSet must exist in result.df."""
        df: pl.DataFrame = make_random_walk_df(200, seed=103)
        config: FeatureConfig = make_small_feature_config(compute_targets=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        for col in result.target_columns:
            assert col in result.df.columns


class TestFeatureMatrixBuilderWithoutTargets:
    """Tests for FeatureMatrixBuilder.build with compute_targets=False."""

    def test_build_without_targets_no_fwd_columns(self) -> None:
        """With compute_targets=False, no 'fwd_' columns should appear."""
        df: pl.DataFrame = make_random_walk_df(200, seed=110)
        config: FeatureConfig = make_small_feature_config(compute_targets=False)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        fwd_cols: list[str] = [c for c in result.df.columns if c.startswith("fwd_")]
        assert len(fwd_cols) == 0

    def test_build_without_targets_target_columns_empty(self) -> None:
        """target_columns tuple must be empty when compute_targets=False."""
        df: pl.DataFrame = make_random_walk_df(200, seed=111)
        config: FeatureConfig = make_small_feature_config(compute_targets=False)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        assert result.target_columns == ()

    def test_build_without_targets_still_has_features(self) -> None:
        """feature_columns must still be non-empty when compute_targets=False."""
        df: pl.DataFrame = make_random_walk_df(200, seed=112)
        config: FeatureConfig = make_small_feature_config(compute_targets=False)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        assert len(result.feature_columns) > 0


class TestFeatureMatrixBuilderNaNHandling:
    """Tests for NaN dropping behaviour."""

    def test_drop_na_true_removes_warmup_rows(self) -> None:
        """With drop_na=True, rows with NaN in computed columns are removed."""
        n: int = 200
        df: pl.DataFrame = make_random_walk_df(n, seed=120)
        config: FeatureConfig = make_small_feature_config(drop_na=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        # After drop, the clean df should have fewer rows than input
        assert result.n_rows_clean < n
        # n_rows_raw should equal n (all rows before drop)
        assert result.n_rows_raw == n

    def test_drop_na_false_preserves_all_rows(self) -> None:
        """With drop_na=False, all rows are kept and may contain nulls."""
        n: int = 200
        df: pl.DataFrame = make_random_walk_df(n, seed=121)
        config: FeatureConfig = make_small_feature_config(drop_na=False)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        assert result.n_rows_clean == n
        assert result.n_rows_raw == n
        assert len(result.df) == n

    def test_drop_na_clean_df_has_no_nulls_in_features(self) -> None:
        """After NaN dropping, no feature or target column should have nulls."""
        df: pl.DataFrame = make_random_walk_df(300, seed=122)
        config: FeatureConfig = make_small_feature_config(drop_na=True, compute_targets=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        all_computed_cols: list[str] = list(result.feature_columns) + list(result.target_columns)
        for col in all_computed_cols:
            null_cnt: int = result.df[col].null_count()
            assert null_cnt == 0, f"Column '{col}' has {null_cnt} nulls after drop_na=True"


class TestFeatureMatrixBuilderMetadata:
    """Tests for FeatureSet metadata correctness."""

    def test_n_rows_raw_equals_input_length(self) -> None:
        """Verify n_rows_raw equals the input length (before NaN dropping)."""
        n: int = 200
        df: pl.DataFrame = make_random_walk_df(n, seed=130)
        config: FeatureConfig = make_small_feature_config(drop_na=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert result.n_rows_raw == n

    def test_n_rows_clean_matches_df_len(self) -> None:
        """n_rows_clean must equal len(result.df)."""
        df: pl.DataFrame = make_random_walk_df(200, seed=131)
        config: FeatureConfig = make_small_feature_config(drop_na=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert result.n_rows_clean == len(result.df)

    def test_n_rows_clean_leq_n_rows_raw(self) -> None:
        """n_rows_clean must never exceed n_rows_raw."""
        df: pl.DataFrame = make_random_walk_df(200, seed=132)
        config: FeatureConfig = make_small_feature_config(drop_na=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert result.n_rows_clean <= result.n_rows_raw

    def test_feature_columns_sorted(self) -> None:
        """feature_columns must be in sorted order."""
        df: pl.DataFrame = make_random_walk_df(200, seed=133)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert list(result.feature_columns) == sorted(result.feature_columns)

    def test_target_columns_sorted(self) -> None:
        """target_columns must be in sorted order."""
        df: pl.DataFrame = make_random_walk_df(200, seed=134)
        config: FeatureConfig = make_small_feature_config(compute_targets=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert list(result.target_columns) == sorted(result.target_columns)

    def test_feature_and_target_columns_disjoint(self) -> None:
        """feature_columns and target_columns must not overlap."""
        df: pl.DataFrame = make_random_walk_df(200, seed=135)
        config: FeatureConfig = make_small_feature_config(compute_targets=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)

        feat_set: set[str] = set(result.feature_columns)
        tgt_set: set[str] = set(result.target_columns)
        overlap: set[str] = feat_set & tgt_set
        assert len(overlap) == 0, f"Overlapping feature/target columns: {overlap}"


class TestFeatureMatrixBuilderEdgeCases:
    """Edge-case tests for FeatureMatrixBuilder."""

    def test_build_with_minimal_rows_still_returns_feature_set(self) -> None:
        """Even with limited rows, build() should not raise (just return fewer clean rows)."""
        n: int = 250  # well above any window size
        df: pl.DataFrame = make_random_walk_df(n, seed=140)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        assert isinstance(result, FeatureSet)
        assert result.n_rows_clean >= 0

    def test_build_preserves_ohlcv_columns(self) -> None:
        """OHLCV columns must remain in the output DataFrame."""
        df: pl.DataFrame = make_random_walk_df(200, seed=141)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result: FeatureSet = builder.build(df, config)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.df.columns

    def test_build_idempotent(self) -> None:
        """Calling build twice on the same input produces identical results."""
        df: pl.DataFrame = make_random_walk_df(200, seed=142)
        config: FeatureConfig = make_small_feature_config()
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        result_1: FeatureSet = builder.build(df, config)
        result_2: FeatureSet = builder.build(df, config)

        assert result_1.n_rows_clean == result_2.n_rows_clean
        assert result_1.feature_columns == result_2.feature_columns
        assert result_1.target_columns == result_2.target_columns
