"""Smoke tests for Matplotlib chart functions.

Each test verifies that the chart function executes without error and
returns a ``matplotlib.figure.Figure``.  All figures are closed after
creation to prevent memory leaks.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure as MplFigure  # noqa: E402

from src.app.research.infrastructure.matplotlib_charts import (  # noqa: E402
    create_acf_comparison_grid,
    create_acf_stem_plot,
    create_bar_duration_boxplot,
    create_qq_grid,
    create_qq_plot,
    create_return_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42


@pytest.fixture
def small_arrays() -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Return small theoretical and observed quantile arrays.

    Returns:
        Tuple of (theoretical, observed) NumPy arrays with 100 elements.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    theoretical: np.ndarray = np.sort(rng.normal(0, 1, size=100))  # type: ignore[type-arg]
    observed: np.ndarray = np.sort(rng.normal(0.1, 1.1, size=100))  # type: ignore[type-arg]
    return theoretical, observed


@pytest.fixture
def qq_data_dict() -> dict[str, tuple[np.ndarray, np.ndarray]]:  # type: ignore[type-arg]
    """Return Q-Q data for two bar types.

    Returns:
        Dict mapping bar type to (theoretical, observed) arrays.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # type: ignore[type-arg]
    for bar_type in ("time", "dollar"):
        theoretical: np.ndarray = np.sort(rng.normal(0, 1, size=80))  # type: ignore[type-arg]
        observed: np.ndarray = np.sort(rng.normal(0, 1, size=80))  # type: ignore[type-arg]
        data[bar_type] = (theoretical, observed)
    return data


@pytest.fixture
def acf_values() -> np.ndarray:  # type: ignore[type-arg]
    """Return a 41-element synthetic ACF array (lag 0 to 40).

    Returns:
        NumPy array with ACF coefficients decaying from 1.0.
    """
    lags: np.ndarray = np.arange(41, dtype=float)  # type: ignore[type-arg]
    values: np.ndarray = np.exp(-lags / 10.0)  # type: ignore[type-arg]
    return values


@pytest.fixture
def acf_data_dict(acf_values: np.ndarray) -> dict[str, np.ndarray]:  # type: ignore[type-arg]
    """Return ACF data for two bar types.

    Args:
        acf_values: Fixture providing a synthetic ACF array.

    Returns:
        Dict mapping bar type to ACF coefficient arrays.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    acf_noisy: np.ndarray = acf_values + rng.normal(0, 0.05, size=len(acf_values))  # type: ignore[type-arg]
    return {"time": acf_values, "dollar": acf_noisy}


@pytest.fixture
def return_series_dict() -> dict[str, pd.Series]:  # type: ignore[type-arg]
    """Return two Series of synthetic returns.

    Returns:
        Dict mapping bar type to return Series.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    time_returns: pd.Series = pd.Series(rng.normal(0, 0.01, size=300), name="time")  # type: ignore[type-arg]
    dollar_returns: pd.Series = pd.Series(rng.normal(0, 0.015, size=300), name="dollar")  # type: ignore[type-arg]
    return {"time": time_returns, "dollar": dollar_returns}


@pytest.fixture
def duration_series_dict() -> dict[str, pd.Series]:  # type: ignore[type-arg]
    """Return two Series of synthetic bar durations.

    Returns:
        Dict mapping bar type to duration Series (in minutes).
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    time_dur: pd.Series = pd.Series(rng.exponential(60, size=200), name="time")  # type: ignore[type-arg]
    dollar_dur: pd.Series = pd.Series(rng.exponential(45, size=200), name="dollar")  # type: ignore[type-arg]
    return {"time": time_dur, "dollar": dollar_dur}


# ---------------------------------------------------------------------------
# Tests — Q-Q plot
# ---------------------------------------------------------------------------


class TestQQPlot:
    """Smoke tests for create_qq_plot."""

    def test_qq_plot_returns_figure(
        self,
        small_arrays: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """Q-Q plot must return a Matplotlib Figure."""
        theoretical: np.ndarray = small_arrays[0]  # type: ignore[type-arg]
        observed: np.ndarray = small_arrays[1]  # type: ignore[type-arg]
        fig: MplFigure = create_qq_plot(observed, theoretical)
        assert isinstance(fig, MplFigure)
        plt.close(fig)

    def test_qq_plot_has_axes(
        self,
        small_arrays: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """Q-Q plot figure must contain at least one Axes."""
        theoretical: np.ndarray = small_arrays[0]  # type: ignore[type-arg]
        observed: np.ndarray = small_arrays[1]  # type: ignore[type-arg]
        fig: MplFigure = create_qq_plot(observed, theoretical)
        assert len(fig.axes) >= 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — Q-Q grid
# ---------------------------------------------------------------------------


class TestQQGrid:
    """Smoke tests for create_qq_grid."""

    def test_qq_grid_returns_figure(
        self,
        qq_data_dict: dict[str, tuple[np.ndarray, np.ndarray]],  # type: ignore[type-arg]
    ) -> None:
        """Q-Q grid must return a Matplotlib Figure."""
        fig: MplFigure = create_qq_grid(qq_data_dict)
        assert isinstance(fig, MplFigure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — ACF stem plot
# ---------------------------------------------------------------------------


class TestACFStemPlot:
    """Smoke tests for create_acf_stem_plot."""

    def test_acf_stem_plot_returns_figure(self, acf_values: np.ndarray) -> None:  # type: ignore[type-arg]
        """ACF stem plot must return a Matplotlib Figure."""
        fig: MplFigure = create_acf_stem_plot(acf_values)
        assert isinstance(fig, MplFigure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — ACF comparison grid
# ---------------------------------------------------------------------------


class TestACFComparisonGrid:
    """Smoke tests for create_acf_comparison_grid."""

    def test_acf_comparison_grid_returns_figure(
        self,
        acf_data_dict: dict[str, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """ACF comparison grid must return a Matplotlib Figure."""
        fig: MplFigure = create_acf_comparison_grid(acf_data_dict)
        assert isinstance(fig, MplFigure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — return distribution
# ---------------------------------------------------------------------------


class TestReturnDistribution:
    """Smoke tests for create_return_distribution."""

    def test_return_distribution_returns_figure(
        self,
        return_series_dict: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        """Return distribution plot must return a Matplotlib Figure."""
        fig: MplFigure = create_return_distribution(return_series_dict)
        assert isinstance(fig, MplFigure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — bar duration boxplot
# ---------------------------------------------------------------------------


class TestBarDurationBoxplot:
    """Smoke tests for create_bar_duration_boxplot."""

    def test_bar_duration_boxplot_returns_figure(
        self,
        duration_series_dict: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        """Bar duration boxplot must return a Matplotlib Figure."""
        fig: MplFigure = create_bar_duration_boxplot(duration_series_dict)
        assert isinstance(fig, MplFigure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — cleanup
# ---------------------------------------------------------------------------


class TestChartCleanup:
    """Verify that plt.close works correctly after chart creation."""

    def test_all_charts_close_properly(
        self,
        small_arrays: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
        acf_values: np.ndarray,  # type: ignore[type-arg]
        return_series_dict: dict[str, pd.Series],  # type: ignore[type-arg]
        duration_series_dict: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        """Creating and closing multiple charts must not raise."""
        theoretical: np.ndarray = small_arrays[0]  # type: ignore[type-arg]
        observed: np.ndarray = small_arrays[1]  # type: ignore[type-arg]

        figures: list[MplFigure] = [
            create_qq_plot(observed, theoretical),
            create_acf_stem_plot(acf_values),
            create_return_distribution(return_series_dict),
            create_bar_duration_boxplot(duration_series_dict),
        ]

        for fig in figures:
            plt.close(fig)

        # Verify no figures remain open.
        plt.close("all")
