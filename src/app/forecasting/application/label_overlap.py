"""Label overlap handling for overlapping forward-return labels (Lopez de Prado Ch. 4).

Provides sequential bootstrapping weights, effective sample size (N_eff),
and non-overlapping subsampling for ``fwd_logret_24`` where 24-bar labels
create ~24x overlap in adjacent observations.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator
from typing import Self


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LabelOverlapConfig(BaseModel, frozen=True):
    """Configuration for label overlap handling.

    Attributes:
        horizon: Number of forward bars the label spans (e.g. 24 for ``fwd_logret_24``).
        method: Strategy for handling overlap: ``"sequential_bootstrap"`` for
            uniqueness-weighted sampling, ``"subsample"`` for deterministic
            non-overlapping subsampling (every ``horizon``-th bar).
    """

    horizon: Annotated[
        int,
        PydanticField(ge=1, description="Number of forward bars the label spans"),
    ]

    method: Annotated[
        Literal["sequential_bootstrap", "subsample"],
        PydanticField(
            default="sequential_bootstrap",
            description="Overlap handling strategy",
        ),
    ]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class LabelUniquenessResult(BaseModel, frozen=True):
    """Result of label uniqueness computation.

    Attributes:
        weights: Per-sample average uniqueness scores, shape ``(n_samples,)``.
            Each weight is in ``(0, 1]`` — 1.0 means the label has no overlap
            with any other label.
        n_eff: Effective sample size computed as
            ``sum(weights)^2 / sum(weights^2)``.
        n_raw: Raw (unadjusted) number of samples.
    """

    weights: tuple[float, ...]
    """Per-sample average uniqueness scores."""

    n_eff: Annotated[
        float,
        PydanticField(ge=0.0, description="Effective sample size"),
    ]

    n_raw: Annotated[
        int,
        PydanticField(ge=0, description="Raw number of samples"),
    ]

    @model_validator(mode="after")
    def _weights_length_matches_n_raw(self) -> Self:
        """Ensure weights length equals n_raw.

        Returns:
            Validated instance.

        Raises:
            ValueError: If weights length != n_raw.
        """
        if len(self.weights) != self.n_raw:
            msg: str = f"weights length {len(self.weights)} != n_raw {self.n_raw}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _n_eff_bounded(self) -> Self:
        """Ensure n_eff does not exceed n_raw.

        Returns:
            Validated instance.

        Raises:
            ValueError: If n_eff > n_raw.
        """
        if self.n_eff > self.n_raw + 1e-9:
            msg: str = f"n_eff ({self.n_eff}) must not exceed n_raw ({self.n_raw})"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Core algorithm: indicator matrix + average uniqueness
# ---------------------------------------------------------------------------


def compute_indicator_matrix(
    n_samples: int,
    horizon: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Build the label indicator matrix (samples x time).

    Entry ``[i, t]`` is 1.0 if sample ``i``'s label spans time ``t``,
    i.e. ``t`` is in ``[i, i + horizon - 1]``.  The time axis has
    ``n_samples + horizon - 1`` columns to accommodate the last sample's
    label reaching beyond the sample index range.

    This is the ``indM`` matrix from Lopez de Prado (2018), Ch. 4.3.

    Args:
        n_samples: Number of observations.
        horizon: Forward-looking label span in bars.

    Returns:
        Binary indicator matrix of shape ``(n_samples, n_samples + horizon - 1)``.

    Raises:
        ValueError: If n_samples < 1 or horizon < 1.
    """
    if n_samples < 1:
        msg: str = f"n_samples must be >= 1, got {n_samples}"
        raise ValueError(msg)
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)

    n_cols: int = n_samples + horizon - 1
    ind_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros((n_samples, n_cols), dtype=np.float64)

    for i in range(n_samples):
        ind_matrix[i, i : i + horizon] = 1.0

    return ind_matrix


def compute_average_uniqueness(
    indicator_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute average uniqueness per sample from an indicator matrix.

    For each sample ``i``, the uniqueness at time ``t`` is ``1 / c_t``
    where ``c_t`` is the number of concurrent labels at time ``t``.  The
    average uniqueness of sample ``i`` is the mean of ``1 / c_t`` over
    all times where sample ``i``'s label is active.

    This implements Eq. 4.1 from Lopez de Prado (2018).

    Args:
        indicator_matrix: Binary indicator matrix of shape
            ``(n_samples, n_time_steps)`` from :func:`compute_indicator_matrix`.

    Returns:
        1-D array of average uniqueness scores, shape ``(n_samples,)``.
        Each value is in ``(0, 1]``.

    Raises:
        ValueError: If indicator_matrix is empty.
    """
    n_samples: int = indicator_matrix.shape[0]
    if n_samples == 0:
        msg: str = "indicator_matrix must have at least one row"
        raise ValueError(msg)

    # Concurrency: number of labels active at each time step
    concurrency: np.ndarray[tuple[int], np.dtype[np.float64]] = indicator_matrix.sum(axis=0)

    # Uniqueness: 1/c_t for each active time step, averaged per sample
    # Avoid division by zero for time steps with no active labels
    safe_concurrency: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(concurrency > 0.0, concurrency, 1.0)
    uniqueness_per_time: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        indicator_matrix / safe_concurrency[np.newaxis, :]
    )

    # Average over active time steps for each sample
    active_count: np.ndarray[tuple[int], np.dtype[np.float64]] = indicator_matrix.sum(axis=1)
    safe_active_count: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(active_count > 0.0, active_count, 1.0)
    avg_uniqueness: np.ndarray[tuple[int], np.dtype[np.float64]] = uniqueness_per_time.sum(axis=1) / safe_active_count

    return avg_uniqueness


def compute_n_eff(
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """Compute effective sample size from uniqueness weights.

    Uses the Kish (1965) effective sample size formula:
    ``N_eff = (sum(w))^2 / sum(w^2)``.

    When all weights are equal, ``N_eff = N``.  When labels heavily
    overlap, ``N_eff << N``.

    Args:
        weights: 1-D array of positive sample weights, shape ``(n_samples,)``.

    Returns:
        Effective sample size.

    Raises:
        ValueError: If weights is empty or contains non-positive values.
    """
    if weights.shape[0] == 0:
        msg: str = "weights must contain at least one element"
        raise ValueError(msg)
    if np.any(weights <= 0.0):
        msg = "all weights must be positive"
        raise ValueError(msg)

    sum_w: float = float(np.sum(weights))
    sum_w_sq: float = float(np.sum(weights**2))

    n_eff: float = (sum_w**2) / sum_w_sq
    return n_eff


# ---------------------------------------------------------------------------
# Sequential bootstrap draw
# ---------------------------------------------------------------------------


def sequential_bootstrap_draw(
    indicator_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Draw samples via Lopez de Prado's sequential bootstrapping algorithm.

    At each step, computes the average uniqueness of every candidate
    sample conditional on the currently selected set, then draws the
    next sample with probability proportional to its uniqueness.  This
    procedure de-biases bootstrap sampling against highly overlapping
    labels.

    Reference: Lopez de Prado (2018), Advances in Financial ML, Ch. 4.5.

    Args:
        indicator_matrix: Binary indicator matrix of shape
            ``(n_samples, n_time_steps)``.
        n_draws: Number of samples to draw (with replacement conceptually,
            but the probabilities adapt at each step).
        rng: NumPy random generator for reproducibility.

    Returns:
        1-D array of selected sample indices, shape ``(n_draws,)``.

    Raises:
        ValueError: If n_draws < 1, or indicator_matrix is empty.
    """
    n_samples: int = indicator_matrix.shape[0]
    if n_samples == 0:
        msg: str = "indicator_matrix must have at least one row"
        raise ValueError(msg)
    if n_draws < 1:
        msg = f"n_draws must be >= 1, got {n_draws}"
        raise ValueError(msg)

    n_time: int = indicator_matrix.shape[1]
    selected_indices: list[int] = []

    # Track the running concurrency from already-selected samples
    running_concurrency: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_time, dtype=np.float64)

    for _ in range(n_draws):
        # Compute conditional uniqueness for each candidate
        avg_u: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            # Concurrency if sample i were added
            row_i: np.ndarray[tuple[int], np.dtype[np.float64]] = indicator_matrix[i]
            candidate_concurrency: np.ndarray[tuple[int], np.dtype[np.float64]] = running_concurrency + row_i

            # Active time steps for sample i
            active_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = row_i > 0.0
            n_active: int = int(np.sum(active_mask))
            if n_active == 0:
                avg_u[i] = 0.0
                continue

            # Average uniqueness = mean(1/c_t) over active time steps
            avg_u[i] = float(np.sum(1.0 / candidate_concurrency[active_mask]) / n_active)

        # Convert uniqueness to probability distribution
        total_u: float = float(np.sum(avg_u))
        if total_u <= 0.0:
            # Degenerate: uniform draw
            probs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(n_samples, dtype=np.float64) / n_samples
        else:
            probs = avg_u / total_u

        # Draw next sample
        chosen: int = int(rng.choice(n_samples, p=probs))
        selected_indices.append(chosen)

        # Update running concurrency
        running_concurrency += indicator_matrix[chosen]

    result: np.ndarray[tuple[int], np.dtype[np.intp]] = np.array(selected_indices, dtype=np.intp)
    return result


# ---------------------------------------------------------------------------
# Non-overlapping subsampler
# ---------------------------------------------------------------------------


def subsample_non_overlapping(
    n_samples: int,
    horizon: int,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Return deterministic non-overlapping indices (every ``horizon``-th bar).

    This is the simplest approach to eliminate label overlap: retain only
    every ``h``-th observation so that consecutive retained labels do not
    share any forward-looking bars.

    Args:
        n_samples: Total number of observations.
        horizon: Forward-looking label span in bars.

    Returns:
        1-D array of selected indices, shape ``(n_selected,)`` where
        ``n_selected = ceil(n_samples / horizon)``.

    Raises:
        ValueError: If n_samples < 0 or horizon < 1.
    """
    if n_samples < 0:
        msg: str = f"n_samples must be >= 0, got {n_samples}"
        raise ValueError(msg)
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)

    indices: np.ndarray[tuple[int], np.dtype[np.intp]] = np.arange(0, n_samples, horizon, dtype=np.intp)
    return indices


# ---------------------------------------------------------------------------
# High-level facade
# ---------------------------------------------------------------------------


def compute_label_uniqueness(
    n_samples: int,
    config: LabelOverlapConfig,
) -> LabelUniquenessResult:
    """Compute label uniqueness weights for overlapping forward-return labels.

    Facade function that dispatches to sequential bootstrapping or
    non-overlapping subsampling based on the config.

    For ``"sequential_bootstrap"``, computes the average uniqueness of each
    sample's label and returns these as sample weights.  Models that accept
    ``sample_weight`` (sklearn, LightGBM) can use these directly.

    For ``"subsample"``, returns uniform weights (1.0) for the selected
    non-overlapping indices and zero for the rest.

    Args:
        n_samples: Total number of observations.
        config: Label overlap configuration specifying horizon and method.

    Returns:
        LabelUniquenessResult with per-sample weights, N_eff, and N_raw.

    Raises:
        ValueError: If n_samples < 1.
    """
    if n_samples < 1:
        msg: str = f"n_samples must be >= 1, got {n_samples}"
        raise ValueError(msg)

    horizon: int = config.horizon

    if config.method == "sequential_bootstrap":
        # Build indicator matrix and compute average uniqueness
        ind_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = compute_indicator_matrix(n_samples, horizon)
        weights_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = compute_average_uniqueness(ind_matrix)
        n_eff: float = compute_n_eff(weights_arr)
        weights_tuple: tuple[float, ...] = tuple(float(w) for w in weights_arr)

        logger.info(
            "Sequential bootstrap uniqueness: n_raw={}, n_eff={:.1f}, ratio={:.3f}, horizon={}",
            n_samples,
            n_eff,
            n_eff / n_samples,
            horizon,
        )

    else:
        # Non-overlapping subsampling: weight = 1.0 for selected, 0.0 otherwise
        selected: np.ndarray[tuple[int], np.dtype[np.intp]] = subsample_non_overlapping(n_samples, horizon)
        weights_list: list[float] = [0.0] * n_samples
        for idx in selected:
            weights_list[int(idx)] = 1.0
        weights_tuple = tuple(weights_list)
        weights_for_neff: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [w for w in weights_list if w > 0.0], dtype=np.float64
        )
        n_eff = compute_n_eff(weights_for_neff) if len(weights_for_neff) > 0 else 0.0

        logger.info(
            "Non-overlapping subsample: n_raw={}, n_selected={}, n_eff={:.1f}, horizon={}",
            n_samples,
            len(selected),
            n_eff,
            horizon,
        )

    return LabelUniquenessResult(
        weights=weights_tuple,
        n_eff=n_eff,
        n_raw=n_samples,
    )
