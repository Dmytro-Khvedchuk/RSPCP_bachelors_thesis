"""Combinatorial Purged Cross-Validation (CPCV) splitter for financial time series.

Implements Lopez de Prado's CPCV (Advances in Financial Machine Learning,
Ch. 7 & 12) with cross-asset temporal purging to prevent information leakage
in pooled multi-asset datasets.
"""

from __future__ import annotations

import itertools
from typing import Annotated

import numpy as np
from pydantic import BaseModel
from pydantic import Field as PydanticField


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CPCVConfig(BaseModel, frozen=True):
    """Configuration for the CPCV splitter.

    Attributes:
        n_blocks: Number of contiguous time blocks to partition the index into.
            C(n_blocks, 2) combinations are generated as test sets.
        purge_window: Number of bars to purge from training at each boundary
            between a train block and a test block.  Prevents label overlap.
        embargo_window: Additional bars to embargo after the purge window to
            account for serial correlation in features.
    """

    n_blocks: Annotated[
        int,
        PydanticField(default=6, ge=3, description="Number of contiguous time blocks"),
    ]

    purge_window: Annotated[
        int,
        PydanticField(default=1, ge=0, description="Bars to purge at block boundaries"),
    ]

    embargo_window: Annotated[
        int,
        PydanticField(default=1, ge=0, description="Additional embargo bars after purge"),
    ]


# ---------------------------------------------------------------------------
# CPCV Splitter
# ---------------------------------------------------------------------------


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation splitter.

    Partitions a temporally sorted dataset into ``n_blocks`` contiguous blocks
    and generates all C(n_blocks, 2) test-set combinations.  For each
    combination the remaining blocks form the training set, after purging and
    embargoing samples at block boundaries to prevent look-ahead bias.

    When ``group_labels`` are supplied (one asset identifier per sample),
    cross-asset purging removes ALL asset samples that fall within the
    temporal danger zone around the test set, preventing leakage through
    cross-asset correlation (e.g. BTC <-> ETH ~0.85).

    Args:
        config: CPCV hyper-parameters (block count, purge, embargo).

    Example::

        splitter = CPCVSplitter(CPCVConfig(n_blocks=6, purge_window=2, embargo_window=1))
        for train_idx, test_idx in splitter.split(timestamps):
            model.fit(X[train_idx], y[train_idx])
            model.predict(X[test_idx])
    """

    def __init__(self, config: CPCVConfig) -> None:
        """Initialise the CPCV splitter with the given configuration.

        Args:
            config: CPCV hyper-parameters (block count, purge, embargo).
        """
        self._config: CPCVConfig = config

    @property
    def config(self) -> CPCVConfig:
        """Return the splitter configuration.

        Returns:
            The frozen CPCVConfig instance.
        """
        return self._config

    @property
    def n_combinations(self) -> int:
        """Return the total number of train/test splits.

        Returns:
            C(n_blocks, 2) — the number of 2-block test-set combinations.
        """
        n: int = self._config.n_blocks
        # C(n, 2) = n * (n - 1) / 2
        return n * (n - 1) // 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]],
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]] | None = None,
    ) -> list[tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]]:
        """Generate all CPCV train/test splits with purging and embargo.

        Args:
            timestamps: 1-D array of integer timestamps (e.g. Unix epoch).
                Must be sorted in ascending order within each asset group.
                When multiple assets are pooled, samples should be interleaved
                by time so that the block partition respects temporal ordering.
            group_labels: Optional 1-D string array identifying the asset for
                each sample (e.g. ``["BTC", "BTC", "ETH", ...]``).  When
                provided, cross-asset purging is applied: training samples from
                ALL assets within the temporal danger zone of the test set are
                removed.

        Returns:
            List of ``(train_indices, test_indices)`` tuples, one per
            C(n_blocks, 2) combination.  Indices reference positions in the
            original ``timestamps`` array.

        Raises:
            ValueError: If timestamps is empty or has fewer samples than
                n_blocks, or if group_labels length mismatches.
        """
        n_samples: int = len(timestamps)
        n_blocks: int = self._config.n_blocks

        if n_samples < n_blocks:
            msg: str = f"Need at least n_blocks={n_blocks} samples, got {n_samples}"
            raise ValueError(msg)

        if group_labels is not None and len(group_labels) != n_samples:
            msg = f"group_labels length {len(group_labels)} != timestamps length {n_samples}"
            raise ValueError(msg)

        # Step 1: Partition the sorted index into n_blocks contiguous blocks.
        block_indices: list[np.ndarray[tuple[int], np.dtype[np.intp]]] = self._partition_into_blocks(
            n_samples, n_blocks
        )

        # Step 2: For each C(n_blocks, 2) combination, build train/test with
        # purging + embargo.
        splits: list[tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]] = []
        combo: tuple[int, ...]
        for combo in itertools.combinations(range(n_blocks), 2):
            train_idx: np.ndarray[tuple[int], np.dtype[np.intp]]
            test_idx: np.ndarray[tuple[int], np.dtype[np.intp]]
            train_idx, test_idx = self._build_split(
                block_indices=block_indices,
                test_block_ids=combo,
                timestamps=timestamps,
                group_labels=group_labels,
            )
            splits.append((train_idx, test_idx))

        return splits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_into_blocks(
        n_samples: int,
        n_blocks: int,
    ) -> list[np.ndarray[tuple[int], np.dtype[np.intp]]]:
        """Split ``range(n_samples)`` into ``n_blocks`` contiguous blocks.

        Uses ``np.array_split`` to handle non-divisible sizes — earlier
        blocks get one extra sample when ``n_samples % n_blocks != 0``.

        Args:
            n_samples: Total number of samples.
            n_blocks: Number of contiguous blocks.

        Returns:
            List of index arrays, one per block.
        """
        all_indices: np.ndarray[tuple[int], np.dtype[np.intp]] = np.arange(n_samples, dtype=np.intp)
        blocks: list[np.ndarray[tuple[int], np.dtype[np.intp]]] = [
            block.astype(np.intp) for block in np.array_split(all_indices, n_blocks)
        ]
        return blocks

    def _build_split(
        self,
        block_indices: list[np.ndarray[tuple[int], np.dtype[np.intp]]],
        test_block_ids: tuple[int, ...],
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]],
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]] | None,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]:
        """Build a single train/test split for a given test-block combination.

        Args:
            block_indices: Pre-computed block partition.
            test_block_ids: Which blocks form the test set.
            timestamps: Full timestamp array.
            group_labels: Optional asset labels per sample.

        Returns:
            Tuple of (train_indices, test_indices) after purging and embargo.
        """
        # Collect test indices.
        test_idx: np.ndarray[tuple[int], np.dtype[np.intp]] = np.concatenate(
            [block_indices[b] for b in test_block_ids]
        ).astype(np.intp)

        # Candidate train indices = everything NOT in the test blocks.
        test_set: set[int] = set(test_idx.tolist())
        n_samples: int = len(timestamps)
        candidate_train: np.ndarray[tuple[int], np.dtype[np.intp]] = np.array(
            [i for i in range(n_samples) if i not in test_set],
            dtype=np.intp,
        )

        # Apply purging + embargo to remove leaked samples.
        train_idx: np.ndarray[tuple[int], np.dtype[np.intp]] = self._apply_purge_embargo(
            candidate_train=candidate_train,
            test_idx=test_idx,
            timestamps=timestamps,
            group_labels=group_labels,
        )

        return train_idx, test_idx

    def _apply_purge_embargo(
        self,
        candidate_train: np.ndarray[tuple[int], np.dtype[np.intp]],
        test_idx: np.ndarray[tuple[int], np.dtype[np.intp]],
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]],
        group_labels: np.ndarray[tuple[int], np.dtype[np.str_]] | None,
    ) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
        """Remove training samples that fall within the purge+embargo danger zone.

        The danger zone is defined as any timestamp in
        ``[test_min_ts - total_window, test_max_ts + total_window]`` where
        ``total_window = purge_window + embargo_window``.

        When ``group_labels`` is ``None`` (single-asset), we purge based on
        positional indices.  When ``group_labels`` is provided (multi-asset),
        we purge based on timestamps across ALL assets, preventing cross-asset
        information leakage.

        Args:
            candidate_train: Indices that are candidates for training.
            test_idx: Test set indices.
            timestamps: Full timestamp array.
            group_labels: Optional asset labels per sample.

        Returns:
            Filtered training indices with purge+embargo applied.
        """
        total_window: int = self._config.purge_window + self._config.embargo_window

        if total_window == 0:
            return candidate_train

        if group_labels is None:
            # Single-asset case: purge by positional proximity to test indices.
            return self._purge_by_position(candidate_train, test_idx, total_window)

        # Multi-asset case: purge by temporal proximity across all assets.
        return self._purge_by_timestamp(candidate_train, test_idx, timestamps, total_window)

    @staticmethod
    def _purge_by_position(
        candidate_train: np.ndarray[tuple[int], np.dtype[np.intp]],
        test_idx: np.ndarray[tuple[int], np.dtype[np.intp]],
        total_window: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
        """Purge training indices that are positionally near test indices.

        For each contiguous block of test indices, removes candidate training
        samples within ``total_window`` positions before or after.

        Args:
            candidate_train: Candidate training indices.
            test_idx: Test set indices.
            total_window: Total purge + embargo window (in positions).

        Returns:
            Filtered training indices.
        """
        # Build the danger zone: all indices within total_window of any test
        # block boundary.  Since test blocks are contiguous within each block
        # but may be non-contiguous across blocks, we compute per-block.
        sorted_test: np.ndarray[tuple[int], np.dtype[np.intp]] = np.sort(test_idx)

        # Detect contiguous segments in test_idx.
        segments: list[tuple[int, int]] = []
        seg_start: int = int(sorted_test[0])
        seg_end: int = int(sorted_test[0])
        for i in range(1, len(sorted_test)):
            current: int = int(sorted_test[i])
            if current == seg_end + 1:
                seg_end = current
            else:
                segments.append((seg_start, seg_end))
                seg_start = current
                seg_end = current
        segments.append((seg_start, seg_end))

        # Build set of indices to purge.
        purge_set: set[int] = set()
        for seg_start_val, seg_end_val in segments:
            purge_lo: int = max(0, seg_start_val - total_window)
            purge_hi: int = seg_end_val + total_window
            for idx in range(purge_lo, purge_hi + 1):
                purge_set.add(idx)

        # Filter: keep candidates NOT in purge zone and NOT already in test.
        test_set: set[int] = set(test_idx.tolist())
        mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.array(
            [int(c) not in purge_set and int(c) not in test_set for c in candidate_train],
            dtype=np.bool_,
        )
        result: np.ndarray[tuple[int], np.dtype[np.intp]] = candidate_train[mask]
        return result

    @staticmethod
    def _purge_by_timestamp(
        candidate_train: np.ndarray[tuple[int], np.dtype[np.intp]],
        test_idx: np.ndarray[tuple[int], np.dtype[np.intp]],
        timestamps: np.ndarray[tuple[int], np.dtype[np.int64]],
        total_window: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
        """Purge training indices by timestamp proximity to the test set.

        Computes the min/max timestamp in the test set for each contiguous
        test segment, then removes ALL candidate training samples (across
        all assets) whose timestamp falls within ``total_window`` bars of
        any test segment boundary.

        For cross-asset purging, the timestamp-based approach ensures that
        correlated assets (e.g. BTC/ETH with rho ~0.85) do not leak
        information through contemporaneous training samples.

        Args:
            candidate_train: Candidate training indices.
            test_idx: Test set indices.
            timestamps: Full timestamp array for all samples.
            total_window: Total purge + embargo window (in timestamp units).

        Returns:
            Filtered training indices.
        """
        # Detect contiguous timestamp segments in the test set.
        unique_test_ts: np.ndarray[tuple[int], np.dtype[np.int64]] = np.unique(timestamps[test_idx])

        # Compute timestamp differences to find gaps.
        if len(unique_test_ts) <= 1:
            # Single timestamp or empty — one segment.
            segments: list[tuple[int, int]] = [(int(unique_test_ts[0]), int(unique_test_ts[-1]))]
        else:
            ts_diffs: np.ndarray[tuple[int], np.dtype[np.int64]] = np.diff(unique_test_ts)
            median_diff: float = float(np.median(ts_diffs))
            # A gap > 2x the median spacing indicates non-contiguous segments.
            gap_threshold: float = max(median_diff * 2.0, 1.0)

            segments = []
            seg_start_ts: int = int(unique_test_ts[0])
            prev_ts: int = int(unique_test_ts[0])
            for i in range(1, len(unique_test_ts)):
                current_ts: int = int(unique_test_ts[i])
                if float(current_ts - prev_ts) > gap_threshold:
                    segments.append((seg_start_ts, prev_ts))
                    seg_start_ts = current_ts
                prev_ts = current_ts
            segments.append((seg_start_ts, prev_ts))

        # Build danger zone based on timestamp windows around each segment.
        candidate_ts: np.ndarray[tuple[int], np.dtype[np.int64]] = timestamps[candidate_train]
        mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.ones(len(candidate_train), dtype=np.bool_)

        for seg_lo, seg_hi in segments:
            purge_lo: int = seg_lo - total_window
            purge_hi: int = seg_hi + total_window
            in_danger: np.ndarray[tuple[int], np.dtype[np.bool_]] = (candidate_ts >= purge_lo) & (
                candidate_ts <= purge_hi
            )
            mask &= ~in_danger

        result: np.ndarray[tuple[int], np.dtype[np.intp]] = candidate_train[mask]
        return result
