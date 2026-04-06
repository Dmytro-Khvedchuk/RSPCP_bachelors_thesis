"""Combinatorial Purged Cross-Validation (CPCV) splitter.

Implements the CPCV algorithm from Lopez de Prado (2018), Chapter 12.
Generates C(n_groups, k_test) fold combinations with purging and embargo
to prevent information leakage in time-series cross-validation.

Purging removes training samples whose labels overlap with the test fold
boundary.  Embargo removes additional training samples after each test
fold to handle residual autocorrelation.

Uses the ML-research path (NumPy arrays) per CLAUDE.md.
"""

from __future__ import annotations

from itertools import combinations
from typing import Final

import numpy as np
from loguru import logger
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

_MAX_PURGE_WARNING_FRACTION: Final[float] = 0.20
"""Warn if purge + embargo removes more than 20% of training data."""


class CPCVConfig(BaseModel, frozen=True):
    """Configuration for Combinatorial Purged Cross-Validation.

    Attributes:
        n_groups: Number of contiguous groups to partition data into.
        k_test: Number of groups held out for testing per fold.
        purge_bars: Training bars to remove at each train-test boundary.
            Should match the forward label horizon (h=1 for fwd_logret_1).
        embargo_bars: Training bars to skip after each test group ends.
            Should match the ACF decay length of the target series.
    """

    n_groups: int = 6
    k_test: int = 2
    purge_bars: int = 1
    embargo_bars: int = 5

    @field_validator("n_groups")
    @classmethod
    def _n_groups_ge_2(cls, v: int) -> int:
        if v < 2:
            msg: str = f"n_groups must be >= 2, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("k_test")
    @classmethod
    def _k_test_ge_1(cls, v: int) -> int:
        if v < 1:
            msg: str = f"k_test must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("purge_bars")
    @classmethod
    def _purge_non_negative(cls, v: int) -> int:
        if v < 0:
            msg: str = f"purge_bars must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("embargo_bars")
    @classmethod
    def _embargo_non_negative(cls, v: int) -> int:
        if v < 0:
            msg: str = f"embargo_bars must be >= 0, got {v}"
            raise ValueError(msg)
        return v


class CPCVFold(BaseModel, frozen=True):
    """One fold of CPCV with train and test index arrays.

    Attributes:
        fold_index: Zero-based fold number within the C(n, k) sequence.
        test_groups: Which group IDs (0-indexed) form the test set.
        train_indices: Integer indices into the original array for training.
        test_indices: Integer indices into the original array for testing.
        n_train: Number of training samples (after purge + embargo).
        n_test: Number of test samples.
        n_purged: Number of samples removed by purging.
        n_embargoed: Number of samples removed by embargo.
    """

    fold_index: int
    test_groups: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    n_train: int
    n_test: int
    n_purged: int
    n_embargoed: int

    model_config = {"arbitrary_types_allowed": True}
    """Allow numpy arrays for convenient construction, stored as tuples."""


# ---------------------------------------------------------------------------
# CPCVSplitter
# ---------------------------------------------------------------------------


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation splitter.

    Generates C(n_groups, k_test) folds with purging and embargo to prevent
    information leakage in financial time-series cross-validation.

    Reference:
        Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
        Chapter 12: "Backtesting through Cross-Validation."

    Example::

        config = CPCVConfig(n_groups=6, k_test=2, purge_bars=1, embargo_bars=5)
        splitter = CPCVSplitter(config)
        folds = splitter.split(n_samples=5286)
        for fold in folds:
            X_train, y_train = X[list(fold.train_indices)], y[list(fold.train_indices)]
            X_test, y_test = X[list(fold.test_indices)], y[list(fold.test_indices)]
    """

    def __init__(self, config: CPCVConfig) -> None:
        """Initialize the splitter.

        Args:
            config: CPCV configuration with group count, test count, purge, embargo.

        Raises:
            ValueError: If k_test >= n_groups.
        """
        if config.k_test >= config.n_groups:
            msg: str = f"k_test ({config.k_test}) must be < n_groups ({config.n_groups})"
            raise ValueError(msg)
        self._config: CPCVConfig = config

    @property
    def config(self) -> CPCVConfig:
        """Return the splitter configuration."""
        return self._config

    @property
    def n_folds(self) -> int:
        """Return the total number of folds: C(n_groups, k_test)."""
        from math import comb

        return comb(self._config.n_groups, self._config.k_test)

    def get_group_boundaries(self, n_samples: int) -> list[tuple[int, int]]:
        """Compute (start, end) index pairs for each contiguous group.

        Groups are approximately equal in size; the last group absorbs
        any remainder so groups differ by at most 1 bar.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of (start_idx, end_idx) tuples (end exclusive).

        Raises:
            ValueError: If n_samples < n_groups.
        """
        n_groups: int = self._config.n_groups
        if n_samples < n_groups:
            msg: str = f"n_samples ({n_samples}) must be >= n_groups ({n_groups})"
            raise ValueError(msg)

        base_size: int = n_samples // n_groups
        remainder: int = n_samples % n_groups

        boundaries: list[tuple[int, int]] = []
        start: int = 0
        for i in range(n_groups):
            # Distribute remainder across the first `remainder` groups
            size: int = base_size + (1 if i < remainder else 0)
            boundaries.append((start, start + size))
            start += size

        return boundaries

    def split(self, n_samples: int) -> list[CPCVFold]:
        """Generate all C(n_groups, k_test) folds with purging and embargo.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of CPCVFold objects, one per fold combination.
        """
        boundaries: list[tuple[int, int]] = self.get_group_boundaries(n_samples)
        all_indices: set[int] = set(range(n_samples))
        n_groups: int = self._config.n_groups
        k_test: int = self._config.k_test
        purge_bars: int = self._config.purge_bars
        embargo_bars: int = self._config.embargo_bars

        folds: list[CPCVFold] = []

        for fold_idx, test_group_combo in enumerate(combinations(range(n_groups), k_test)):
            # --- Identify test indices ---
            test_indices: set[int] = set()
            for g in test_group_combo:
                g_start: int = boundaries[g][0]
                g_end: int = boundaries[g][1]
                test_indices.update(range(g_start, g_end))

            # --- Identify raw train indices (everything not in test) ---
            raw_train_indices: set[int] = all_indices - test_indices

            # --- Apply purge and embargo ---
            purge_set: set[int] = set()
            embargo_set: set[int] = set()

            for g in test_group_combo:
                g_start = boundaries[g][0]
                g_end = boundaries[g][1]

                # Purge BEFORE the test group: remove training bars
                # in [g_start - purge_bars, g_start) from training
                for idx in range(max(0, g_start - purge_bars), g_start):
                    if idx in raw_train_indices:
                        purge_set.add(idx)

                # Purge AFTER the test group: remove training bars
                # in [g_end, g_end + purge_bars) from training
                for idx in range(g_end, min(n_samples, g_end + purge_bars)):
                    if idx in raw_train_indices:
                        purge_set.add(idx)

                # Embargo AFTER the test group: remove training bars
                # in [g_end + purge_bars, g_end + purge_bars + embargo_bars)
                embargo_start: int = g_end + purge_bars
                for idx in range(
                    embargo_start,
                    min(n_samples, embargo_start + embargo_bars),
                ):
                    if idx in raw_train_indices and idx not in purge_set:
                        embargo_set.add(idx)

            # --- Final train indices ---
            final_train_indices: set[int] = raw_train_indices - purge_set - embargo_set
            n_purged: int = len(purge_set)
            n_embargoed: int = len(embargo_set)

            # --- Warn if too many bars removed ---
            total_removed: int = n_purged + n_embargoed
            removal_fraction: float = total_removed / len(raw_train_indices) if raw_train_indices else 0.0
            if removal_fraction > _MAX_PURGE_WARNING_FRACTION:
                logger.warning(
                    "Fold {}: purge+embargo removed {:.1%} of training data "
                    "({} purged + {} embargoed out of {} raw train bars)",
                    fold_idx,
                    removal_fraction,
                    n_purged,
                    n_embargoed,
                    len(raw_train_indices),
                )

            # --- Sort indices for deterministic ordering ---
            sorted_train: tuple[int, ...] = tuple(sorted(final_train_indices))
            sorted_test: tuple[int, ...] = tuple(sorted(test_indices))

            fold: CPCVFold = CPCVFold(
                fold_index=fold_idx,
                test_groups=test_group_combo,
                train_indices=sorted_train,
                test_indices=sorted_test,
                n_train=len(sorted_train),
                n_test=len(sorted_test),
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )
            folds.append(fold)

        logger.info(
            "CPCV generated {} folds (C({},{})): ~{} train / ~{} test bars per fold, purge={}, embargo={}",
            len(folds),
            n_groups,
            k_test,
            int(np.mean([f.n_train for f in folds])),
            int(np.mean([f.n_test for f in folds])),
            purge_bars,
            embargo_bars,
        )

        return folds
