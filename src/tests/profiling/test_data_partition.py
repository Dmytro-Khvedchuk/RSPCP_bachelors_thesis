"""Tests for DataPartition construction, validation, and filtering."""

from __future__ import annotations

from datetime import datetime, UTC

import polars as pl
import pytest

from src.app.profiling.domain.value_objects import DataPartition


class TestDataPartitionConstruction:
    """Tests for DataPartition construction and default()."""

    def test_default_partition_creates_valid_instance(self) -> None:
        """DataPartition.default() must return a valid instance."""
        partition = DataPartition.default()
        assert partition.feature_selection_start == datetime(2020, 1, 1, tzinfo=UTC)
        assert partition.feature_selection_end == datetime(2023, 1, 1, tzinfo=UTC)
        assert partition.model_dev_start == datetime(2020, 1, 1, tzinfo=UTC)
        assert partition.model_dev_end == datetime(2024, 1, 1, tzinfo=UTC)
        assert partition.holdout_start == datetime(2024, 1, 1, tzinfo=UTC)

    def test_custom_partition_accepts_valid_boundaries(self) -> None:
        """Custom valid boundaries should be accepted."""
        partition = DataPartition(
            feature_selection_start=datetime(2021, 1, 1, tzinfo=UTC),
            feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
            model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
            model_dev_end=datetime(2023, 1, 1, tzinfo=UTC),
            holdout_start=datetime(2023, 1, 1, tzinfo=UTC),
        )
        assert partition.feature_selection_start == datetime(2021, 1, 1, tzinfo=UTC)

    def test_partition_is_frozen(self) -> None:
        """DataPartition instances must be immutable."""
        from pydantic import ValidationError

        partition = DataPartition.default()
        with pytest.raises(ValidationError):
            partition.holdout_start = datetime(2025, 1, 1, tzinfo=UTC)  # type: ignore[misc]


class TestDataPartitionValidation:
    """Tests for DataPartition validation rules."""

    def test_feature_selection_start_ge_end_raises(self) -> None:
        """feature_selection_start >= feature_selection_end must raise ValueError."""
        with pytest.raises(ValueError, match="feature_selection_start"):
            DataPartition(
                feature_selection_start=datetime(2023, 1, 1, tzinfo=UTC),
                feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
                model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
                model_dev_end=datetime(2024, 1, 1, tzinfo=UTC),
                holdout_start=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_model_dev_start_ge_end_raises(self) -> None:
        """model_dev_start >= model_dev_end must raise ValueError."""
        with pytest.raises(ValueError, match="model_dev_start"):
            DataPartition(
                feature_selection_start=datetime(2020, 1, 1, tzinfo=UTC),
                feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
                model_dev_start=datetime(2024, 1, 1, tzinfo=UTC),
                model_dev_end=datetime(2023, 1, 1, tzinfo=UTC),
                holdout_start=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_holdout_before_model_dev_end_raises(self) -> None:
        """holdout_start < model_dev_end must raise ValueError."""
        with pytest.raises(ValueError, match="holdout_start"):
            DataPartition(
                feature_selection_start=datetime(2020, 1, 1, tzinfo=UTC),
                feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
                model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
                model_dev_end=datetime(2024, 1, 1, tzinfo=UTC),
                holdout_start=datetime(2023, 6, 1, tzinfo=UTC),
            )

    def test_feature_selection_before_model_dev_start_raises(self) -> None:
        """feature_selection_start < model_dev_start must raise ValueError."""
        with pytest.raises(ValueError, match="feature_selection_start"):
            DataPartition(
                feature_selection_start=datetime(2019, 1, 1, tzinfo=UTC),
                feature_selection_end=datetime(2022, 1, 1, tzinfo=UTC),
                model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
                model_dev_end=datetime(2024, 1, 1, tzinfo=UTC),
                holdout_start=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_feature_selection_end_after_model_dev_end_raises(self) -> None:
        """feature_selection_end > model_dev_end must raise ValueError."""
        with pytest.raises(ValueError, match="feature_selection_end"):
            DataPartition(
                feature_selection_start=datetime(2020, 1, 1, tzinfo=UTC),
                feature_selection_end=datetime(2025, 1, 1, tzinfo=UTC),
                model_dev_start=datetime(2020, 1, 1, tzinfo=UTC),
                model_dev_end=datetime(2024, 1, 1, tzinfo=UTC),
                holdout_start=datetime(2024, 1, 1, tzinfo=UTC),
            )


class TestDataPartitionFiltering:
    """Tests for DataPartition filter_* methods."""

    def test_filter_feature_selection(self) -> None:
        """filter_feature_selection keeps only rows in [start, end)."""
        partition = DataPartition.default()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2019, 12, 31, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2022, 6, 15, tzinfo=UTC),
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )
        result = partition.filter_feature_selection(df, "timestamp")
        assert result.height == 2
        assert result["value"].to_list() == [2, 3]

    def test_filter_model_dev(self) -> None:
        """filter_model_dev keeps only rows in [start, end)."""
        partition = DataPartition.default()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2019, 12, 31, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2023, 6, 15, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                ],
                "value": [1, 2, 3, 4],
            }
        )
        result = partition.filter_model_dev(df, "timestamp")
        assert result.height == 2
        assert result["value"].to_list() == [2, 3]

    def test_filter_holdout(self) -> None:
        """filter_holdout keeps only rows where timestamp >= holdout_start."""
        partition = DataPartition.default()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 12, 31, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 6, 15, tzinfo=UTC),
                ],
                "value": [1, 2, 3],
            }
        )
        result = partition.filter_holdout(df, "timestamp")
        assert result.height == 2
        assert result["value"].to_list() == [2, 3]

    def test_filter_on_empty_dataframe(self) -> None:
        """Filtering an empty DataFrame returns an empty DataFrame."""
        partition = DataPartition.default()
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )
        assert partition.filter_feature_selection(df, "timestamp").height == 0
        assert partition.filter_model_dev(df, "timestamp").height == 0
        assert partition.filter_holdout(df, "timestamp").height == 0

    def test_filter_no_matching_rows(self) -> None:
        """Filtering when no rows match returns an empty DataFrame."""
        partition = DataPartition.default()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2019, 1, 1, tzinfo=UTC)],
                "value": [42],
            }
        )
        assert partition.filter_feature_selection(df, "timestamp").height == 0
        assert partition.filter_model_dev(df, "timestamp").height == 0
        assert partition.filter_holdout(df, "timestamp").height == 0
