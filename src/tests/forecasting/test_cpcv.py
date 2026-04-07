"""Unit tests for the CPCV (Combinatorial Purged Cross-Validation) splitter.

Tests cover combinatorial generation, temporal purging, embargo windows,
cross-asset purging, single-asset operation, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.infrastructure.cpcv import CPCVConfig, CPCVSplitter


# ---------------------------------------------------------------------------
# CPCVConfig tests
# ---------------------------------------------------------------------------


class TestCPCVConfig:
    """Tests for CPCVConfig validation and defaults."""

    def test_default_config(self) -> None:
        config: CPCVConfig = CPCVConfig()
        assert config.n_blocks == 6
        assert config.purge_window == 1
        assert config.embargo_window == 1

    def test_custom_config(self) -> None:
        config: CPCVConfig = CPCVConfig(n_blocks=10, purge_window=3, embargo_window=2)
        assert config.n_blocks == 10
        assert config.purge_window == 3
        assert config.embargo_window == 2

    def test_frozen(self) -> None:
        config: CPCVConfig = CPCVConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            config.n_blocks = 10  # type: ignore[misc]

    def test_n_blocks_minimum(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 3"):
            CPCVConfig(n_blocks=2)

    def test_purge_window_non_negative(self) -> None:
        config: CPCVConfig = CPCVConfig(purge_window=0)
        assert config.purge_window == 0

    def test_embargo_window_non_negative(self) -> None:
        config: CPCVConfig = CPCVConfig(embargo_window=0)
        assert config.embargo_window == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_contiguous_segments(sorted_values: list[int]) -> list[tuple[int, int]]:
    """Detect contiguous segments in a sorted list of integers.

    Two values are contiguous when they differ by exactly 1.

    Args:
        sorted_values: Sorted unique integer values.

    Returns:
        List of (start, end) tuples for each contiguous segment.
    """
    if not sorted_values:
        return []
    segments: list[tuple[int, int]] = []
    seg_start: int = sorted_values[0]
    seg_end: int = sorted_values[0]
    for i in range(1, len(sorted_values)):
        if sorted_values[i] == seg_end + 1:
            seg_end = sorted_values[i]
        else:
            segments.append((seg_start, seg_end))
            seg_start = sorted_values[i]
            seg_end = sorted_values[i]
    segments.append((seg_start, seg_end))
    return segments


def _make_timestamps(n: int, start: int = 1000, step: int = 1) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Generate monotonically increasing integer timestamps.

    Args:
        n: Number of timestamps.
        start: First timestamp value.
        step: Increment between timestamps.

    Returns:
        1-D int64 array of timestamps.
    """
    return np.arange(start, start + n * step, step, dtype=np.int64)


def _make_multi_asset_data(
    n_per_asset: int,
    assets: list[str],
    start: int = 1000,
    step: int = 1,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.str_]]]:
    """Generate interleaved multi-asset timestamps and labels.

    Assets are interleaved so that at each time step all assets have a sample,
    mimicking a pooled dataset sorted by time.

    Args:
        n_per_asset: Number of bars per asset.
        assets: List of asset identifiers.
        start: Starting timestamp.
        step: Increment between successive timestamps.

    Returns:
        Tuple of (timestamps, group_labels) arrays.
    """
    n_assets: int = len(assets)
    total: int = n_per_asset * n_assets
    timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = np.empty(total, dtype=np.int64)
    labels: list[str] = []

    idx: int = 0
    for t in range(n_per_asset):
        ts_val: int = start + t * step
        for asset in assets:
            timestamps[idx] = ts_val
            labels.append(asset)
            idx += 1

    group_labels: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(labels, dtype=np.str_)
    return timestamps, group_labels


# ---------------------------------------------------------------------------
# Combinatorial generation
# ---------------------------------------------------------------------------


class TestCombinations:
    """Verify that C(n_blocks, 2) combinations are generated."""

    def test_default_6_blocks_yields_15(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=6, purge_window=0, embargo_window=0))
        assert splitter.n_combinations == 15

        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(60)
        splits: list[
            tuple[
                np.ndarray[tuple[int], np.dtype[np.intp]],
                np.ndarray[tuple[int], np.dtype[np.intp]],
            ]
        ] = splitter.split(timestamps)
        assert len(splits) == 15

    def test_3_blocks_yields_3(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=3, purge_window=0, embargo_window=0))
        assert splitter.n_combinations == 3

        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(30)
        splits: list[
            tuple[
                np.ndarray[tuple[int], np.dtype[np.intp]],
                np.ndarray[tuple[int], np.dtype[np.intp]],
            ]
        ] = splitter.split(timestamps)
        assert len(splits) == 3

    def test_5_blocks_yields_10(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=5, purge_window=0, embargo_window=0))
        assert splitter.n_combinations == 10

    def test_test_indices_cover_2_blocks(self) -> None:
        """Each split should have test indices from exactly 2 blocks."""
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=6, purge_window=0, embargo_window=0))
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(60)
        splits = splitter.split(timestamps)

        # With 60 samples and 6 blocks, each block has 10 samples.
        # Test set should have ~20 samples (2 blocks).
        for train_idx, test_idx in splits:
            assert len(test_idx) == 20
            assert len(train_idx) == 40  # noqa: PLR2004


# ---------------------------------------------------------------------------
# No temporal overlap
# ---------------------------------------------------------------------------


class TestNoTemporalOverlap:
    """Verify that train and test indices never overlap after purging."""

    def test_no_index_overlap_without_purge(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=6, purge_window=0, embargo_window=0))
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(120)

        for train_idx, test_idx in splitter.split(timestamps):
            overlap: set[int] = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0, f"Found overlap: {overlap}"

    def test_no_index_overlap_with_purge(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=6, purge_window=2, embargo_window=1))
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(120)

        for train_idx, test_idx in splitter.split(timestamps):
            overlap: set[int] = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0, f"Found overlap: {overlap}"


# ---------------------------------------------------------------------------
# Purge + embargo gaps
# ---------------------------------------------------------------------------


class TestPurgeEmbargo:
    """Verify that purge + embargo windows create correct gaps."""

    def test_purge_removes_boundary_samples(self) -> None:
        """With purge=2, embargo=0, training samples within 2 positions of
        a test block boundary should be removed."""
        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=2, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(30)

        for train_idx, test_idx in splitter.split(timestamps):
            train_set: set[int] = set(train_idx.tolist())
            test_sorted: list[int] = sorted(test_idx.tolist())

            # Find contiguous test segments.
            segments: list[tuple[int, int]] = []
            seg_start: int = test_sorted[0]
            seg_end: int = test_sorted[0]
            for i in range(1, len(test_sorted)):
                if test_sorted[i] == seg_end + 1:
                    seg_end = test_sorted[i]
                else:
                    segments.append((seg_start, seg_end))
                    seg_start = test_sorted[i]
                    seg_end = test_sorted[i]
            segments.append((seg_start, seg_end))

            # Verify no training sample is within purge_window of any test segment.
            for seg_lo, seg_hi in segments:
                for offset in range(1, config.purge_window + config.embargo_window + 1):
                    before: int = seg_lo - offset
                    after: int = seg_hi + offset
                    assert before not in train_set, (
                        f"Sample {before} should be purged (within {offset} of test block [{seg_lo}, {seg_hi}])"
                    )
                    if after < 30:  # noqa: PLR2004
                        assert after not in train_set, (
                            f"Sample {after} should be purged (within {offset} of test block [{seg_lo}, {seg_hi}])"
                        )

    def test_embargo_extends_purge(self) -> None:
        """With purge=1, embargo=2, total danger zone = 3 positions."""
        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=1, embargo_window=2)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(30)
        total_window: int = config.purge_window + config.embargo_window  # 3

        for train_idx, test_idx in splitter.split(timestamps):
            train_set: set[int] = set(train_idx.tolist())

            test_min: int = int(test_idx.min())

            # No training sample within 3 positions before test_min.
            for offset in range(1, total_window + 1):
                pos: int = test_min - offset
                if pos >= 0:
                    assert pos not in train_set

    def test_zero_purge_and_embargo_preserves_all_train(self) -> None:
        """With purge=0, embargo=0, all non-test indices are training."""
        config: CPCVConfig = CPCVConfig(n_blocks=4, purge_window=0, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(40)

        for train_idx, test_idx in splitter.split(timestamps):
            assert len(train_idx) + len(test_idx) == 40  # noqa: PLR2004

    def test_large_purge_window_small_dataset(self) -> None:
        """Large purge windows on small datasets should still produce valid
        (possibly empty) training sets without errors."""
        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=5, embargo_window=5)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(9)

        splits = splitter.split(timestamps)
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            overlap: set[int] = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Cross-asset purging
# ---------------------------------------------------------------------------


class TestCrossAssetPurging:
    """Verify that cross-asset purging prevents contemporaneous leakage."""

    def test_eth_purged_when_btc_in_test(self) -> None:
        """When BTC samples are in the test window, ETH training data at the
        same timestamps (and within the purge+embargo buffer) must be removed.

        Verifies per-segment danger zones: for each contiguous test segment,
        no training sample (from ANY asset) has a timestamp within
        ``purge + embargo`` of that segment's boundaries.
        """
        n_per_asset: int = 120
        assets: list[str] = ["BTC", "ETH"]
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]]
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]]
        timestamps, group_labels = _make_multi_asset_data(n_per_asset, assets, start=1000, step=1)

        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=1, embargo_window=1)
        splitter: CPCVSplitter = CPCVSplitter(config)
        total_window: int = config.purge_window + config.embargo_window

        for train_idx, test_idx in splitter.split(timestamps, group_labels=group_labels):
            # Build per-segment danger zones from test timestamps.
            test_ts_sorted: list[int] = sorted(set(timestamps[test_idx].tolist()))
            segments: list[tuple[int, int]] = _find_contiguous_segments(test_ts_sorted)

            # Every training sample must NOT fall within any segment's danger zone.
            for idx in train_idx:
                ts: int = int(timestamps[idx])
                for seg_lo, seg_hi in segments:
                    danger_lo: int = seg_lo - total_window
                    danger_hi: int = seg_hi + total_window
                    assert not (danger_lo <= ts <= danger_hi), (
                        f"Training sample at index {idx} (asset={group_labels[idx]}, ts={ts}) "
                        f"is within danger zone [{danger_lo}, {danger_hi}] "
                        f"for test segment [{seg_lo}, {seg_hi}]"
                    )

    def test_contemporaneous_eth_removed_from_btc_test(self) -> None:
        """Specifically verify ETH samples at the SAME timestamp as BTC test
        samples are purged from training."""
        n_per_asset: int = 60
        assets: list[str] = ["BTC", "ETH"]
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]]
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]]
        timestamps, group_labels = _make_multi_asset_data(n_per_asset, assets, start=1000, step=1)

        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=0, embargo_window=1)
        splitter: CPCVSplitter = CPCVSplitter(config)

        for train_idx, test_idx in splitter.split(timestamps, group_labels=group_labels):
            test_ts_set: set[int] = set(timestamps[test_idx].tolist())
            train_ts_set: set[int] = set(timestamps[train_idx].tolist())

            # No training timestamp should be in the test timestamp set
            # (since embargo=1 covers at least the exact overlap).
            overlap: set[int] = train_ts_set & test_ts_set
            assert len(overlap) == 0, f"Found timestamps in both train and test: {overlap}"

    def test_all_assets_purged_symmetrically(self) -> None:
        """Cross-asset purging should apply to ALL assets equally.

        Verifies per-segment danger zones for a 3-asset pooled dataset.
        """
        n_per_asset: int = 120
        assets: list[str] = ["BTC", "ETH", "SOL"]
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]]
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]]
        timestamps, group_labels = _make_multi_asset_data(n_per_asset, assets, start=1000, step=1)

        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=1, embargo_window=1)
        splitter: CPCVSplitter = CPCVSplitter(config)
        total_window: int = config.purge_window + config.embargo_window

        for train_idx, test_idx in splitter.split(timestamps, group_labels=group_labels):
            test_ts_sorted: list[int] = sorted(set(timestamps[test_idx].tolist()))
            segments: list[tuple[int, int]] = _find_contiguous_segments(test_ts_sorted)

            for idx in train_idx:
                ts: int = int(timestamps[idx])
                asset: str = str(group_labels[idx])
                for seg_lo, seg_hi in segments:
                    danger_lo: int = seg_lo - total_window
                    danger_hi: int = seg_hi + total_window
                    assert not (danger_lo <= ts <= danger_hi), (
                        f"Training sample at index {idx} (asset={asset}, ts={ts}) "
                        f"is within danger zone [{danger_lo}, {danger_hi}] "
                        f"for test segment [{seg_lo}, {seg_hi}]"
                    )


# ---------------------------------------------------------------------------
# Single-asset (no group_labels) case
# ---------------------------------------------------------------------------


class TestSingleAsset:
    """Verify correct operation without group_labels (position-based purging)."""

    def test_single_asset_basic(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=4, purge_window=1, embargo_window=1))
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(40)

        splits = splitter.split(timestamps)
        assert len(splits) == 6  # C(4,2) = 6

        for train_idx, test_idx in splits:
            overlap: set[int] = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0

    def test_single_asset_no_group_labels(self) -> None:
        """Calling without group_labels should work identically to None."""
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=4, purge_window=1, embargo_window=0))
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(40)

        splits_no_labels = splitter.split(timestamps)
        splits_none = splitter.split(timestamps, group_labels=None)

        assert len(splits_no_labels) == len(splits_none)

        for (train_a, test_a), (train_b, test_b) in zip(splits_no_labels, splits_none, strict=True):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test boundary conditions and degenerate inputs."""

    def test_minimum_samples_equal_n_blocks(self) -> None:
        """Should work when n_samples == n_blocks (1 sample per block)."""
        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=0, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(3)

        splits = splitter.split(timestamps)
        assert len(splits) == 3  # C(3,2) = 3

        for train_idx, test_idx in splits:
            assert len(test_idx) == 2  # noqa: PLR2004
            assert len(train_idx) == 1

    def test_too_few_samples_raises(self) -> None:
        config: CPCVConfig = CPCVConfig(n_blocks=6, purge_window=0, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(3)

        with pytest.raises(ValueError, match="at least n_blocks=6"):
            splitter.split(timestamps)

    def test_group_labels_length_mismatch_raises(self) -> None:
        config: CPCVConfig = CPCVConfig(n_blocks=3, purge_window=0, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(30)
        bad_labels: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(["BTC"] * 10, dtype=np.str_)

        with pytest.raises(ValueError, match="group_labels length"):
            splitter.split(timestamps, group_labels=bad_labels)

    def test_non_divisible_block_sizes(self) -> None:
        """When n_samples is not divisible by n_blocks, blocks differ by at most 1."""
        config: CPCVConfig = CPCVConfig(n_blocks=4, purge_window=0, embargo_window=0)
        splitter: CPCVSplitter = CPCVSplitter(config)
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]] = _make_timestamps(13)

        splits = splitter.split(timestamps)
        assert len(splits) == 6  # C(4,2)

        # All 13 indices should appear across all test sets (but distribution
        # across splits can vary).
        for train_idx, test_idx in splits:
            combined: set[int] = set(train_idx.tolist()) | set(test_idx.tolist())
            assert combined == set(range(13))

    def test_n_combinations_property(self) -> None:
        splitter: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=6))
        assert splitter.n_combinations == 15

        splitter_8: CPCVSplitter = CPCVSplitter(CPCVConfig(n_blocks=8))
        assert splitter_8.n_combinations == 28  # C(8,2)

    def test_config_property(self) -> None:
        config: CPCVConfig = CPCVConfig(n_blocks=5, purge_window=3, embargo_window=2)
        splitter: CPCVSplitter = CPCVSplitter(config)
        assert splitter.config is config
        assert splitter.config.n_blocks == 5
