"""Static Matplotlib chart functions for research analysis visualisation.

Provides standalone chart-building functions that return ``matplotlib.figure.Figure``
objects.  Charts cover Q-Q plots, ACF stem plots, return distributions, and
bar-duration boxplots.  Backend is determined by the environment (inline in
Jupyter, Agg in headless / CI).
"""

from __future__ import annotations

from math import ceil
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WIDTH: Final[float] = 10.0
_DEFAULT_HEIGHT: Final[float] = 6.0
_CONFIDENCE_Z: Final[float] = 1.96
_SUBPLOT_COLS: Final[int] = 2
_HIST_BINS: Final[int] = 50
_HIST_ALPHA: Final[float] = 0.4
_KDE_POINTS: Final[int] = 300

# Colour cycle for overlaid series (colourblind-friendly selection).
_SERIES_COLOURS: Final[list[str]] = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#9b59b6",
    "#f39c12",
    "#1abc9c",
    "#e67e22",
    "#34495e",
]


# ---------------------------------------------------------------------------
# Q-Q plots
# ---------------------------------------------------------------------------


def create_qq_plot(
    observed: np.ndarray,  # type: ignore[type-arg]
    theoretical: np.ndarray,  # type: ignore[type-arg]
    title: str = "Q-Q Plot",
    *,
    width: float = _DEFAULT_WIDTH,
    height: float = _DEFAULT_HEIGHT,
) -> MplFigure:
    """Create a single Q-Q plot of observed vs theoretical quantiles.

    Plots a scatter of ``(theoretical, observed)`` pairs plus a 45-degree
    diagonal reference line.

    Args:
        observed: Ordered sample values (y-axis).
        theoretical: Theoretical quantiles (x-axis).
        title: Plot title. Defaults to ``"Q-Q Plot"``.
        width: Figure width in inches. Defaults to 10.
        height: Figure height in inches. Defaults to 6.

    Returns:
        Matplotlib ``Figure`` containing the Q-Q scatter and reference line.
    """
    fig: MplFigure
    ax: Axes
    fig, ax = plt.subplots(figsize=(width, height))

    ax.scatter(theoretical, observed, s=12, alpha=0.6, color="#3498db", edgecolors="none")

    # Diagonal reference line spanning the data range.
    min_val: float = float(min(theoretical.min(), observed.min()))
    max_val: float = float(max(theoretical.max(), observed.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    ax.set_title(title)
    fig.tight_layout()

    return fig


def create_qq_grid(
    qq_data: dict[str, tuple[np.ndarray, np.ndarray]],  # type: ignore[type-arg]
    *,
    width: float = _DEFAULT_WIDTH,
    height_per_row: float = 4.0,
) -> MplFigure:
    """Create a subplot grid of Q-Q plots for multiple bar types.

    Arranges Q-Q plots in a 2-column grid.  Each subplot shows observed vs
    theoretical quantiles with a diagonal reference line.

    Args:
        qq_data: Mapping from bar-type name to
            ``(theoretical_quantiles, observed_values)``.
        width: Total figure width in inches. Defaults to 10.
        height_per_row: Height per subplot row in inches. Defaults to 4.

    Returns:
        Matplotlib ``Figure`` containing the subplot grid.
    """
    n_items: int = len(qq_data)
    n_cols: int = min(_SUBPLOT_COLS, n_items)
    n_rows: int = ceil(n_items / n_cols) if n_cols > 0 else 1
    total_height: float = height_per_row * n_rows

    fig: MplFigure
    axes: np.ndarray  # type: ignore[type-arg]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, total_height), squeeze=False)

    flat_axes: list[Axes] = [axes[r, c] for r in range(n_rows) for c in range(n_cols)]

    for idx, (bar_type, (theoretical, observed)) in enumerate(qq_data.items()):
        ax: Axes = flat_axes[idx]

        if theoretical.size == 0 or observed.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"Q-Q: {bar_type}")
            continue

        ax.scatter(theoretical, observed, s=10, alpha=0.6, color="#3498db", edgecolors="none")

        min_val: float = float(min(theoretical.min(), observed.min()))
        max_val: float = float(max(theoretical.max(), observed.max()))
        ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linewidth=1.5, linestyle="--")

        ax.set_title(f"Q-Q: {bar_type}")
        ax.set_xlabel("Theoretical")
        ax.set_ylabel("Observed")

    # Hide unused subplots.
    for idx in range(n_items, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ACF plots
# ---------------------------------------------------------------------------


def create_acf_stem_plot(
    acf_values: np.ndarray,  # type: ignore[type-arg]
    title: str = "ACF",
    n_obs: int = 500,
    *,
    width: float = _DEFAULT_WIDTH,
    height: float = _DEFAULT_HEIGHT,
) -> MplFigure:
    r"""Create a stem plot of ACF coefficients with 95 % confidence bands.

    Draws vertical stems for each lag's ACF coefficient and horizontal
    dashed lines at :math:`\pm 1.96 / \sqrt{n}`.

    Args:
        acf_values: Array of ACF coefficients (lag 0 to max_lag).
        title: Plot title. Defaults to ``"ACF"``.
        n_obs: Number of observations used to compute the confidence band.
            Defaults to 500.
        width: Figure width in inches. Defaults to 10.
        height: Figure height in inches. Defaults to 6.

    Returns:
        Matplotlib ``Figure`` with the ACF stem plot and confidence bands.
    """
    fig: MplFigure
    ax: Axes
    fig, ax = plt.subplots(figsize=(width, height))

    lags: np.ndarray = np.arange(len(acf_values))  # type: ignore[type-arg]
    ax.stem(lags, acf_values, linefmt="-", markerfmt="o", basefmt="k-")

    # 95 % confidence bands.
    conf_band: float = float(_CONFIDENCE_Z / np.sqrt(n_obs))
    ax.axhline(y=conf_band, color="#e74c3c", linestyle="--", linewidth=1, label="95% CI")
    ax.axhline(y=-conf_band, color="#e74c3c", linestyle="--", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


def create_acf_comparison_grid(
    acf_data: dict[str, np.ndarray],  # type: ignore[type-arg]
    *,
    n_obs: int = 500,
    width: float = _DEFAULT_WIDTH,
    height_per_row: float = 4.0,
) -> MplFigure:
    r"""Create a subplot grid comparing ACF across bar types.

    Each subplot shows one bar type's ACF stem plot with
    :math:`\pm 1.96 / \sqrt{n}` confidence bands.

    Args:
        acf_data: Mapping from bar-type name to ACF coefficient array.
        n_obs: Number of observations for the confidence band. Defaults to 500.
        width: Total figure width in inches. Defaults to 10.
        height_per_row: Height per subplot row in inches. Defaults to 4.

    Returns:
        Matplotlib ``Figure`` containing the comparison grid.
    """
    n_items: int = len(acf_data)
    n_cols: int = min(_SUBPLOT_COLS, n_items)
    n_rows: int = ceil(n_items / n_cols) if n_cols > 0 else 1
    total_height: float = height_per_row * n_rows

    fig: MplFigure
    axes: np.ndarray  # type: ignore[type-arg]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, total_height), squeeze=False)

    flat_axes: list[Axes] = [axes[r, c] for r in range(n_rows) for c in range(n_cols)]
    conf_band: float = float(_CONFIDENCE_Z / np.sqrt(n_obs))

    for idx, (bar_type, acf_vals) in enumerate(acf_data.items()):
        ax: Axes = flat_axes[idx]

        if acf_vals.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"ACF: {bar_type}")
            continue

        lags: np.ndarray = np.arange(len(acf_vals))  # type: ignore[type-arg]
        ax.stem(lags, acf_vals, linefmt="-", markerfmt="o", basefmt="k-")
        ax.axhline(y=conf_band, color="#e74c3c", linestyle="--", linewidth=1)
        ax.axhline(y=-conf_band, color="#e74c3c", linestyle="--", linewidth=1)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"ACF: {bar_type}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

    for idx in range(n_items, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Return distribution
# ---------------------------------------------------------------------------


def create_return_distribution(
    returns: dict[str, pd.Series],  # type: ignore[type-arg]
    *,
    width: float = _DEFAULT_WIDTH,
    height: float = _DEFAULT_HEIGHT,
    bins: int = _HIST_BINS,
) -> MplFigure:
    """Create overlaid histograms with KDE curves for each bar type's returns.

    Histograms use semi-transparent fills so overlapping regions are visible.
    A KDE curve (from ``pd.Series.plot.kde``) is overlaid for each series.

    Args:
        returns: Mapping from bar-type name to a return ``Series``.
        width: Figure width in inches. Defaults to 10.
        height: Figure height in inches. Defaults to 6.
        bins: Number of histogram bins. Defaults to 50.

    Returns:
        Matplotlib ``Figure`` with overlaid histograms and KDE curves.
    """
    fig: MplFigure
    ax: Axes
    fig, ax = plt.subplots(figsize=(width, height))

    _min_hist_samples: int = 3

    for idx, (bar_type, series) in enumerate(returns.items()):
        color: str = _SERIES_COLOURS[idx % len(_SERIES_COLOURS)]
        values: np.ndarray = series.dropna().to_numpy()  # type: ignore[type-arg]

        # Skip series with too few observations or zero variance (causes divide-by-zero in density).
        if len(values) < _min_hist_samples or np.std(values) == 0.0:
            continue

        ax.hist(
            values,
            bins=bins,
            alpha=_HIST_ALPHA,
            color=color,
            label=f"{bar_type} (hist)",
            density=True,
        )

        # KDE overlay — use scipy for a smooth curve.
        if len(values) > 1:
            kde: gaussian_kde = gaussian_kde(values)
            x_grid: np.ndarray = np.linspace(float(values.min()), float(values.max()), _KDE_POINTS)  # type: ignore[type-arg]
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5, label=f"{bar_type} (KDE)")

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distributions by Bar Type")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Bar duration boxplot
# ---------------------------------------------------------------------------


def create_bar_duration_boxplot(
    durations: dict[str, pd.Series],  # type: ignore[type-arg]
    *,
    width: float = _DEFAULT_WIDTH,
    height: float = _DEFAULT_HEIGHT,
) -> MplFigure:
    """Create a box-and-whisker plot of bar durations per bar type.

    Each box represents the distribution of durations for one bar type.

    Args:
        durations: Mapping from bar-type name to a ``Series`` of durations
            (e.g. in minutes).
        width: Figure width in inches. Defaults to 10.
        height: Figure height in inches. Defaults to 6.

    Returns:
        Matplotlib ``Figure`` with one boxplot per bar type.
    """
    fig: MplFigure
    ax: Axes
    fig, ax = plt.subplots(figsize=(width, height))

    labels: list[str] = list(durations.keys())
    data_arrays: list[np.ndarray] = [s.dropna().to_numpy() for s in durations.values()]  # type: ignore[type-arg]

    ax.boxplot(data_arrays, tick_labels=labels, patch_artist=True)

    ax.set_xlabel("Bar Type")
    ax.set_ylabel("Duration")
    ax.set_title("Bar Duration Distribution by Type")
    fig.tight_layout()

    return fig
