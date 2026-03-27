"""Phase 7 utility functions extracted from RC7 research notebooks.

Pure functions for stationarity transformations, discrete entropy,
feature degeneracy detection, and conditional break-even DA computation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.app.research.application.rc2_thresholds import (
    BreakevenDAResult,
    compute_breakeven_da,
)


# ---------------------------------------------------------------------------
# Stationarity transformations (Phase 7.3 — RC7_stationarity_policy.ipynb)
# ---------------------------------------------------------------------------


def apply_rolling_zscore(series: pd.Series, window: int = 24) -> pd.Series:  # type: ignore[type-arg]
    """Normalize a series via rolling z-score.

    Computes ``(x - rolling_mean) / rolling_std`` over the given window.
    The first ``window - 1`` values are NaN due to insufficient history.

    Args:
        series: Input Pandas Series (numeric).
        window: Rolling window size.

    Returns:
        Pandas Series of the same length with z-scored values.
    """
    rolling_mean: pd.Series = series.rolling(window=window).mean()  # type: ignore[type-arg]
    rolling_std: pd.Series = series.rolling(window=window).std()  # type: ignore[type-arg]
    return (series - rolling_mean) / rolling_std  # type: ignore[return-value]


def apply_first_difference(series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    """Compute the first difference of a series.

    Returns ``x_t - x_{t-1}`` with a single NaN at position 0.

    Args:
        series: Input Pandas Series (numeric).

    Returns:
        Pandas Series of the same length with differenced values.
    """
    return series.diff()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MI normalization (Phase 7.4 — RC7_mi_normalization.ipynb)
# ---------------------------------------------------------------------------


def compute_discrete_entropy(
    arr: np.ndarray,  # type: ignore[type-arg]
    n_bins: int | None = None,
) -> float:
    """Compute discrete Shannon entropy using histogram binning.

    Uses Sturges' rule for bin count when ``n_bins`` is not provided:
    ``k = 1 + floor(log2(N))``.

    Unlike Gaussian differential entropy, this is always non-negative,
    making it suitable for normalizing MI on small-variance targets.

    Args:
        arr: 1-D array of continuous values.
        n_bins: Number of histogram bins.  Defaults to Sturges' rule.

    Returns:
        Shannon entropy in nats (>= 0).
    """
    n: int = len(arr)
    if n <= 1:
        return 0.0

    if n_bins is None:
        n_bins = 1 + int(math.floor(math.log2(n)))

    counts: np.ndarray
    counts, _ = np.histogram(arr, bins=n_bins)

    probabilities: np.ndarray = counts / n
    # Filter out zero-probability bins to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    entropy: float = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy


_MI_STRONG_THRESHOLD: float = 0.05
"""MI above this value (nats) is classified as Strong."""

_MI_MODERATE_THRESHOLD: float = 0.01
"""MI above this value (nats) is classified as Moderate."""

_MI_WEAK_THRESHOLD: float = 0.001
"""MI above this value (nats) is classified as Weak."""


def classify_mi_effect_size(mi_nats: float) -> str:
    """Classify mutual information effect size on the RC7 qualitative scale.

    Args:
        mi_nats: Mutual information in nats (>= 0).

    Returns:
        One of ``"Strong"``, ``"Moderate"``, ``"Weak"``, ``"Negligible"``.
    """
    if mi_nats > _MI_STRONG_THRESHOLD:
        return "Strong"
    if mi_nats > _MI_MODERATE_THRESHOLD:
        return "Moderate"
    if mi_nats > _MI_WEAK_THRESHOLD:
        return "Weak"
    return "Negligible"


# ---------------------------------------------------------------------------
# Feature degeneracy (Phase 7.2 — RC7_profiling_closure.ipynb)
# ---------------------------------------------------------------------------

_DEGENERACY_THRESHOLD: float = 1e-10
"""Variance threshold below which a feature is considered degenerate."""


def is_feature_degenerate(
    series: np.ndarray,  # type: ignore[type-arg]
    threshold: float = _DEGENERACY_THRESHOLD,
) -> bool:
    """Check whether a feature series is degenerate (near-constant).

    A feature is degenerate when its sample variance falls below the
    given threshold, indicating it carries no useful information.

    Args:
        series: 1-D array of feature values.
        threshold: Variance threshold (default ``1e-10``).

    Returns:
        ``True`` if the feature is degenerate.
    """
    variance: float = float(np.var(series, ddof=1)) if len(series) > 1 else 0.0
    return variance < threshold


def compute_feature_variance(series: np.ndarray) -> float:  # type: ignore[type-arg]
    """Compute the sample variance (ddof=1) of a feature series.

    Args:
        series: 1-D array of feature values.

    Returns:
        Sample variance.
    """
    if len(series) <= 1:
        return 0.0
    return float(np.var(series, ddof=1))


# ---------------------------------------------------------------------------
# Conditional break-even DA (Phase 7.6 — RC7_conditional_breakeven.ipynb)
# ---------------------------------------------------------------------------


def compute_conditional_breakeven_da(
    returns: np.ndarray,  # type: ignore[type-arg]
    regime_labels: np.ndarray,  # type: ignore[type-arg]
    regime: str,
    round_trip_cost: float,
) -> BreakevenDAResult:
    """Compute break-even DA conditioned on a volatility regime subset.

    Filters returns to the specified regime and computes the break-even
    directional accuracy on that subset.  Since ``E[|r| | HIGH] > E[|r|]``,
    the conditional break-even DA is lower than the unconditional one.

    Args:
        returns: 1-D array of log returns.
        regime_labels: 1-D array of regime labels (same length as returns).
        regime: Regime label to filter on (e.g. ``"HIGH"``).
        round_trip_cost: Round-trip transaction cost (e.g. ``0.002``).

    Returns:
        ``BreakevenDAResult`` computed on the regime-filtered subset.

    Raises:
        ValueError: If no returns match the specified regime.
    """
    mask: np.ndarray = regime_labels == regime  # type: ignore[type-arg]
    subset: np.ndarray = returns[mask]  # type: ignore[type-arg]

    if len(subset) == 0:
        msg: str = f"No returns found for regime '{regime}'"
        raise ValueError(msg)

    mean_abs_return: float = float(np.mean(np.abs(subset)))
    return compute_breakeven_da(mean_abs_return, round_trip_cost)


def compute_amplification_ratio(
    returns: np.ndarray,  # type: ignore[type-arg]
    regime_labels: np.ndarray,  # type: ignore[type-arg]
    regime: str,
) -> float:
    """Compute the return amplification ratio for a volatility regime.

    The ratio is ``E[|r| | regime] / E[|r|]``.  For the HIGH regime
    this is expected to be > 1.0 (larger absolute returns during
    high-volatility periods).

    Args:
        returns: 1-D array of log returns.
        regime_labels: 1-D array of regime labels (same length as returns).
        regime: Regime label to compute the ratio for (e.g. ``"HIGH"``).

    Returns:
        Amplification ratio (float).

    Raises:
        ValueError: If no returns match the specified regime or overall mean is zero.
    """
    mask: np.ndarray = regime_labels == regime  # type: ignore[type-arg]
    subset: np.ndarray = returns[mask]  # type: ignore[type-arg]

    if len(subset) == 0:
        msg: str = f"No returns found for regime '{regime}'"
        raise ValueError(msg)

    overall_mean_abs: float = float(np.mean(np.abs(returns)))
    if overall_mean_abs == 0.0:
        msg = "Overall mean absolute return is zero"
        raise ValueError(msg)

    regime_mean_abs: float = float(np.mean(np.abs(subset)))
    return regime_mean_abs / overall_mean_abs
