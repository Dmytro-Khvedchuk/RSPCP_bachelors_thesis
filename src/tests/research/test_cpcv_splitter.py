"""Tests for Combinatorial Purged Cross-Validation (CPCV) splitter.

Covers fold generation correctness, purge/embargo logic, edge cases
(adjacent test groups, uneven group sizes), and config validation.
"""

from __future__ import annotations

from math import comb

import pytest

from src.app.research.application.cpcv_splitter import (
    CPCVConfig,
    CPCVSplitter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> CPCVConfig:
    return CPCVConfig(n_groups=6, k_test=2, purge_bars=1, embargo_bars=5)


@pytest.fixture
def default_splitter(default_config: CPCVConfig) -> CPCVSplitter:
    return CPCVSplitter(default_config)


@pytest.fixture
def small_n_samples() -> int:
    return 120


@pytest.fixture
def realistic_n_samples() -> int:
    return 5286


# ---------------------------------------------------------------------------
# TestCPCVConfig
# ---------------------------------------------------------------------------


class TestCPCVConfig:
    """Tests for CPCVConfig validation."""

    def test_default_config(self) -> None:
        config = CPCVConfig()
        assert config.n_groups == 6
        assert config.k_test == 2
        assert config.purge_bars == 1
        assert config.embargo_bars == 5

    def test_n_groups_must_be_ge_2(self) -> None:
        with pytest.raises(ValueError, match="n_groups must be >= 2"):
            CPCVConfig(n_groups=1)

    def test_k_test_must_be_ge_1(self) -> None:
        with pytest.raises(ValueError, match="k_test must be >= 1"):
            CPCVConfig(k_test=0)

    def test_purge_non_negative(self) -> None:
        with pytest.raises(ValueError, match="purge_bars must be >= 0"):
            CPCVConfig(purge_bars=-1)

    def test_embargo_non_negative(self) -> None:
        with pytest.raises(ValueError, match="embargo_bars must be >= 0"):
            CPCVConfig(embargo_bars=-1)

    def test_frozen(self) -> None:
        config = CPCVConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011
            config.n_groups = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCPCVSplitter — Init and properties
# ---------------------------------------------------------------------------


class TestCPCVSplitterInit:
    """Tests for CPCVSplitter initialization and properties."""

    def test_k_test_must_be_less_than_n_groups(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=4)
        with pytest.raises(ValueError, match="k_test .* must be < n_groups"):
            CPCVSplitter(config)

    def test_n_folds_c62(self, default_splitter: CPCVSplitter) -> None:
        assert default_splitter.n_folds == comb(6, 2)
        assert default_splitter.n_folds == 15

    def test_n_folds_c53(self) -> None:
        config = CPCVConfig(n_groups=5, k_test=3, purge_bars=0, embargo_bars=0)
        splitter = CPCVSplitter(config)
        assert splitter.n_folds == comb(5, 3)
        assert splitter.n_folds == 10


# ---------------------------------------------------------------------------
# TestGetGroupBoundaries
# ---------------------------------------------------------------------------


class TestGetGroupBoundaries:
    """Tests for group boundary computation."""

    def test_even_division(self, default_splitter: CPCVSplitter) -> None:
        boundaries = default_splitter.get_group_boundaries(120)
        assert len(boundaries) == 6
        for start, end in boundaries:
            assert end - start == 20

    def test_uneven_division(self, default_splitter: CPCVSplitter) -> None:
        boundaries = default_splitter.get_group_boundaries(125)
        assert len(boundaries) == 6
        sizes = [end - start for start, end in boundaries]
        # First 5 groups get 21 bars, last 1 gets 20 bars
        assert sum(sizes) == 125
        assert max(sizes) - min(sizes) <= 1

    def test_covers_all_indices(self, default_splitter: CPCVSplitter, realistic_n_samples: int) -> None:
        boundaries = default_splitter.get_group_boundaries(realistic_n_samples)
        assert boundaries[0][0] == 0
        assert boundaries[-1][1] == realistic_n_samples
        # No gaps between groups
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] == boundaries[i + 1][0]

    def test_too_few_samples(self, default_splitter: CPCVSplitter) -> None:
        with pytest.raises(ValueError, match="n_samples .* must be >= n_groups"):
            default_splitter.get_group_boundaries(3)


# ---------------------------------------------------------------------------
# TestSplit — Core logic
# ---------------------------------------------------------------------------


class TestSplitCore:
    """Tests for the main split() method — fold count, indices, no overlap."""

    def test_correct_number_of_folds(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        assert len(folds) == 15

    def test_no_train_test_overlap(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        for fold in folds:
            train_set = set(fold.train_indices)
            test_set = set(fold.test_indices)
            assert train_set.isdisjoint(test_set), f"Fold {fold.fold_index}: train/test overlap"

    def test_test_indices_are_sorted(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        for fold in folds:
            assert fold.test_indices == tuple(sorted(fold.test_indices))
            assert fold.train_indices == tuple(sorted(fold.train_indices))

    def test_each_bar_in_test_set_correct_times(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        """Each bar appears in the test set C(n_groups-1, k_test-1) = C(5,1) = 5 times."""
        folds = default_splitter.split(small_n_samples)
        test_counts: dict[int, int] = {}
        for fold in folds:
            for idx in fold.test_indices:
                test_counts[idx] = test_counts.get(idx, 0) + 1

        expected_count = comb(6 - 1, 2 - 1)  # C(5,1) = 5
        assert all(v == expected_count for v in test_counts.values()), (
            f"Expected each bar in test {expected_count} times, got distribution: {set(test_counts.values())}"
        )

    def test_all_bars_appear_in_test(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        all_test_indices: set[int] = set()
        for fold in folds:
            all_test_indices.update(fold.test_indices)
        assert all_test_indices == set(range(small_n_samples))

    def test_n_train_n_test_consistent(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        for fold in folds:
            assert fold.n_train == len(fold.train_indices)
            assert fold.n_test == len(fold.test_indices)

    def test_fold_index_sequential(self, default_splitter: CPCVSplitter, small_n_samples: int) -> None:
        folds = default_splitter.split(small_n_samples)
        for i, fold in enumerate(folds):
            assert fold.fold_index == i


# ---------------------------------------------------------------------------
# TestSplit — Purge logic
# ---------------------------------------------------------------------------


class TestSplitPurge:
    """Tests for purge behavior at train-test boundaries."""

    def test_purge_removes_bars_before_test_group(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=2, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        # Test group 1 (indices 10-19), training group 0 (0-9)
        fold_with_group1 = [f for f in folds if f.test_groups == (1,)][0]
        train_set = set(fold_with_group1.train_indices)

        # Bars 8, 9 (just before group 1 boundary at index 10) should be purged
        assert 8 not in train_set
        assert 9 not in train_set
        # Bar 7 should still be in training
        assert 7 in train_set

    def test_purge_removes_bars_after_test_group(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=2, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        # Test group 1 (indices 10-19), training group 2 (20-29)
        fold_with_group1 = [f for f in folds if f.test_groups == (1,)][0]
        train_set = set(fold_with_group1.train_indices)

        # Bars 20, 21 (just after group 1 ends at index 20) should be purged
        assert 20 not in train_set
        assert 21 not in train_set
        # Bar 22 should still be in training
        assert 22 in train_set

    def test_purge_at_dataset_start(self) -> None:
        """Purge should not go below index 0."""
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=5, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        fold_with_group0 = [f for f in folds if f.test_groups == (0,)][0]
        # Should not crash; purge before group 0 has nothing to remove
        assert fold_with_group0.n_test == 10

    def test_zero_purge(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=0, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        for fold in folds:
            assert fold.n_purged == 0
            assert fold.n_train + fold.n_test == 40

    def test_n_purged_reported_correctly(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=2, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        # Group 1 test: purge 2 before (from group 0) + 2 after (from group 2) = 4
        fold_g1 = [f for f in folds if f.test_groups == (1,)][0]
        assert fold_g1.n_purged == 4


# ---------------------------------------------------------------------------
# TestSplit — Embargo logic
# ---------------------------------------------------------------------------


class TestSplitEmbargo:
    """Tests for embargo behavior after test groups."""

    def test_embargo_after_test_group(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=0, embargo_bars=3)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        # Test group 1 (indices 10-19), next training = group 2 (20-29)
        fold_g1 = [f for f in folds if f.test_groups == (1,)][0]
        train_set = set(fold_g1.train_indices)

        # Bars 20, 21, 22 should be embargoed
        assert 20 not in train_set
        assert 21 not in train_set
        assert 22 not in train_set
        # Bar 23 should be in training
        assert 23 in train_set

    def test_embargo_at_dataset_end(self) -> None:
        """Embargo after last test group should clip to dataset end."""
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=0, embargo_bars=20)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        fold_g3 = [f for f in folds if f.test_groups == (3,)][0]
        # Embargo after group 3 (end of dataset) — nothing to embargo
        # Should not crash
        assert fold_g3.n_embargoed == 0

    def test_embargo_follows_purge(self) -> None:
        """Embargo starts after purge window, not at test group end."""
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=2, embargo_bars=3)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        # Test group 1 (10-19), purge 2 after (20,21), embargo 3 after purge (22,23,24)
        fold_g1 = [f for f in folds if f.test_groups == (1,)][0]
        train_set = set(fold_g1.train_indices)

        assert 20 not in train_set  # purged
        assert 21 not in train_set  # purged
        assert 22 not in train_set  # embargoed
        assert 23 not in train_set  # embargoed
        assert 24 not in train_set  # embargoed
        assert 25 in train_set  # should be in training

    def test_zero_embargo(self) -> None:
        config = CPCVConfig(n_groups=4, k_test=1, purge_bars=0, embargo_bars=0)
        splitter = CPCVSplitter(config)
        folds = splitter.split(40)

        for fold in folds:
            assert fold.n_embargoed == 0


# ---------------------------------------------------------------------------
# TestSplit — Adjacent test groups
# ---------------------------------------------------------------------------


class TestSplitAdjacentGroups:
    """Tests for adjacent test groups (no purge/embargo between them)."""

    def test_adjacent_test_groups_no_internal_purge(self) -> None:
        """When groups 1 and 2 are both test, the boundary between them is inside
        the test set.  Purge/embargo only applies at train-test boundaries."""
        config = CPCVConfig(n_groups=6, k_test=2, purge_bars=2, embargo_bars=3)
        splitter = CPCVSplitter(config)
        folds = splitter.split(120)

        # Find fold where groups 1 and 2 are both test
        fold_12 = [f for f in folds if f.test_groups == (1, 2)][0]

        # Test indices should include all of groups 1 and 2 (indices 20-59)
        test_set = set(fold_12.test_indices)
        for idx in range(20, 60):
            assert idx in test_set

        # Purge/embargo should only be at train-test boundaries:
        # Before group 1 (from group 0) and after group 2 (from group 3)
        train_set = set(fold_12.train_indices)

        # Before group 1: purge bars 18, 19 from training (group 0 is 0-19)
        assert 18 not in train_set
        assert 19 not in train_set
        assert 17 in train_set

        # After group 2: purge bars 60, 61 then embargo 62, 63, 64
        assert 60 not in train_set  # purged
        assert 61 not in train_set  # purged
        assert 62 not in train_set  # embargoed
        assert 63 not in train_set  # embargoed
        assert 64 not in train_set  # embargoed
        assert 65 in train_set


# ---------------------------------------------------------------------------
# TestSplit — Realistic scenario (BTCUSDT dollar bars)
# ---------------------------------------------------------------------------


class TestSplitRealistic:
    """Tests with realistic N=5286 BTCUSDT dollar bar count."""

    def test_realistic_fold_sizes(self, default_splitter: CPCVSplitter) -> None:
        folds = default_splitter.split(5286)
        assert len(folds) == 15

        for fold in folds:
            # Each fold has 2 test groups (~881*2 = ~1762 test bars)
            assert fold.n_test > 1700
            assert fold.n_test < 1800
            # Training should be most of the remaining bars
            assert fold.n_train > 3400
            # Purge + embargo should be small relative to training
            assert fold.n_purged + fold.n_embargoed < 30

    def test_realistic_no_train_test_overlap(self, default_splitter: CPCVSplitter) -> None:
        folds = default_splitter.split(5286)
        for fold in folds:
            assert set(fold.train_indices).isdisjoint(set(fold.test_indices))
