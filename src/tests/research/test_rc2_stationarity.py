"""Tests for RC2 Section 2 stationarity analysis service.

Covers cross-asset aggregation, feature classification (universally stationary,
universally non-stationary, mixed), summary table rendering, cross-asset matrix,
programmatic "Therefore" generation, and edge cases (single combo, single feature,
all stationary, all non-stationary, empty inputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.research.application.rc2_stationarity import (
    RC2StationarityAnalyzer,
    StationaritySummary,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 43
_N_ROWS: int = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stationary_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate a mean-reverting series that will pass ADF at alpha=0.05.

    Args:
        rng: NumPy random generator.
        n: Number of observations.

    Returns:
        1-D array of stationary values.
    """
    # Pure white noise + mild AR(1) to guarantee both ADF rejects unit-root
    # and KPSS fails to reject stationarity. The AR coefficient of 0.3 is far
    # from the unit-root boundary yet preserves enough structure for ADF power.
    values: np.ndarray = np.zeros(n)
    phi: float = 0.3
    noise: np.ndarray = rng.normal(0, 1.0, size=n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + noise[i]
    return values


def _make_non_stationary_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate a random walk series that will fail ADF.

    Args:
        rng: NumPy random generator.
        n: Number of observations.

    Returns:
        1-D array of non-stationary values (random walk).
    """
    return np.cumsum(rng.normal(0, 0.01, size=n))


def _build_feature_df(
    rng: np.random.Generator,
    feature_names: list[str],
    stationary_names: list[str],
    n: int = _N_ROWS,
) -> pd.DataFrame:
    """Build a DataFrame with a mix of stationary and non-stationary columns.

    Args:
        rng: NumPy random generator.
        feature_names: All column names.
        stationary_names: Subset of feature_names that should be stationary.
        n: Number of rows.

    Returns:
        DataFrame with the specified columns.
    """
    data: dict[str, np.ndarray] = {}
    for fname in feature_names:
        if fname in stationary_names:
            data[fname] = _make_stationary_series(rng, n)
        else:
            data[fname] = _make_non_stationary_series(rng, n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> RC2StationarityAnalyzer:
    """Return a fresh analyzer instance.

    Returns:
        RC2StationarityAnalyzer.
    """
    return RC2StationarityAnalyzer()


@pytest.fixture
def feature_names() -> list[str]:
    """Return test feature names with known stationarity behavior.

    Returns:
        List of 4 feature names.
    """
    return ["rsi_14", "macd_signal", "atr_14", "log_return"]


@pytest.fixture
def all_stationary_dfs(feature_names: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Return feature DataFrames where ALL features are stationary.

    Each combo uses an independent RNG seed so that stationarity results are
    deterministic and do not depend on the number of prior random draws.

    Args:
        feature_names: Feature column names.

    Returns:
        Dict mapping (asset, bar_type) to DataFrames with all-stationary columns.
    """
    combos: list[tuple[str, str]] = [
        ("BTCUSDT", "dollar"),
        ("ETHUSDT", "dollar"),
    ]
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for i, combo in enumerate(combos):
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED + i)
        result[combo] = _build_feature_df(rng, feature_names, stationary_names=feature_names)
    return result


@pytest.fixture
def mixed_dfs(feature_names: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Return feature DataFrames where some features are stationary, some not.

    - rsi_14, log_return: always stationary (mean-reverting)
    - atr_14: always non-stationary (random walk)
    - macd_signal: stationary in BTC, non-stationary in ETH

    Each combo uses an independent RNG seed for deterministic stationarity results.

    Args:
        feature_names: Feature column names.

    Returns:
        Dict mapping (asset, bar_type) to DataFrames with mixed stationarity.
    """
    btc_rng: np.random.Generator = np.random.default_rng(_RNG_SEED + 100)
    btc_df: pd.DataFrame = _build_feature_df(
        btc_rng,
        feature_names,
        stationary_names=["rsi_14", "log_return", "macd_signal"],
    )
    eth_rng: np.random.Generator = np.random.default_rng(_RNG_SEED + 200)
    eth_df: pd.DataFrame = _build_feature_df(
        eth_rng,
        feature_names,
        stationary_names=["rsi_14", "log_return"],
    )
    return {
        ("BTCUSDT", "dollar"): btc_df,
        ("ETHUSDT", "dollar"): eth_df,
    }


@pytest.fixture
def single_combo_dfs(feature_names: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Return a single (asset, bar_type) combination.

    Args:
        feature_names: Feature column names.

    Returns:
        Dict with one entry.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    df: pd.DataFrame = _build_feature_df(
        rng,
        feature_names,
        stationary_names=["rsi_14", "log_return"],
    )
    return {("BTCUSDT", "dollar"): df}


# ---------------------------------------------------------------------------
# TestStationaritySummaryConstruction
# ---------------------------------------------------------------------------


class TestStationaritySummaryConstruction:
    """Tests for building the StationaritySummary value object."""

    def test_all_stationary_features(
        self,
        analyzer: RC2StationarityAnalyzer,
        all_stationary_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """When all features are stationary, universally_stationary should contain all."""
        summary: StationaritySummary = analyzer.analyze_features(all_stationary_dfs, feature_names)
        assert len(summary.universally_stationary) == len(feature_names)
        assert len(summary.universally_non_stationary) == 0
        assert len(summary.mixed_features) == 0
        assert summary.n_stationary_pct == pytest.approx(100.0, abs=0.1)

    def test_mixed_stationarity(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Mixed features should be classified correctly."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        # rsi_14 and log_return should be universally stationary
        assert "rsi_14" in summary.universally_stationary
        assert "log_return" in summary.universally_stationary
        # atr_14 should be universally non-stationary
        assert "atr_14" in summary.universally_non_stationary
        # macd_signal should be mixed (stationary in BTC, not in ETH)
        assert "macd_signal" in summary.mixed_features

    def test_per_asset_bar_count_matches(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """per_asset_bar should have one report per input combination."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        assert len(summary.per_asset_bar) == len(mixed_dfs)

    def test_n_total_features(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """n_total_features matches input feature list length."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        assert summary.n_total_features == len(feature_names)

    def test_summary_is_frozen(
        self,
        analyzer: RC2StationarityAnalyzer,
        all_stationary_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """StationaritySummary must be immutable."""
        summary: StationaritySummary = analyzer.analyze_features(all_stationary_dfs, feature_names)
        with pytest.raises(Exception):  # noqa: B017, PT011
            summary.n_total_features = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRecommendedTransformations
# ---------------------------------------------------------------------------


class TestRecommendedTransformations:
    """Tests for the recommended_transformations field."""

    def test_atr_gets_pct_atr_suggestion(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Non-stationary atr_14 should get 'pct_atr' transformation suggestion."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        # atr_ prefix maps to pct_atr in _KNOWN_TRANSFORMATIONS
        assert "atr_14" in summary.recommended_transformations
        assert summary.recommended_transformations["atr_14"] == "pct_atr"

    def test_stationary_features_have_no_transformation(
        self,
        analyzer: RC2StationarityAnalyzer,
        all_stationary_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Universally stationary features should not appear in recommended_transformations."""
        summary: StationaritySummary = analyzer.analyze_features(all_stationary_dfs, feature_names)
        assert len(summary.recommended_transformations) == 0


# ---------------------------------------------------------------------------
# TestRenderSummaryTable
# ---------------------------------------------------------------------------


class TestRenderSummaryTable:
    """Tests for the summary table rendering."""

    def test_summary_table_shape(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Summary table should have one row per feature and expected columns."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_summary_table(summary)
        assert len(table) == len(feature_names)
        expected_cols: set[str] = {
            "feature_name",
            "n_stationary",
            "n_combos",
            "pct_stationary",
            "classification",
            "suggested_transformation",
        }
        assert set(table.columns) == expected_cols

    def test_summary_table_classifications(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Classifications in the table should match summary categories."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_summary_table(summary)

        rsi_row: pd.DataFrame = table[table["feature_name"] == "rsi_14"]
        assert rsi_row.iloc[0]["classification"] == "universally_stationary"

        atr_row: pd.DataFrame = table[table["feature_name"] == "atr_14"]
        assert atr_row.iloc[0]["classification"] == "universally_non_stationary"

        macd_row: pd.DataFrame = table[table["feature_name"] == "macd_signal"]
        assert macd_row.iloc[0]["classification"] == "mixed"

    def test_summary_table_pct_range(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """All pct_stationary values should be in [0, 100]."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_summary_table(summary)
        assert (table["pct_stationary"] >= 0.0).all()
        assert (table["pct_stationary"] <= 100.0).all()


# ---------------------------------------------------------------------------
# TestRenderCrossAssetTable
# ---------------------------------------------------------------------------


class TestRenderCrossAssetTable:
    """Tests for the cross-asset matrix rendering."""

    def test_cross_asset_table_shape(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Cross-asset table should have features as rows, combos as columns."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_cross_asset_table(summary)
        assert len(table) == len(feature_names)
        assert len(table.columns) == len(mixed_dfs)

    def test_cross_asset_table_column_labels(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Column labels should be 'asset|bar_type' format."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_cross_asset_table(summary)
        for col in table.columns:
            assert "|" in col

    def test_cross_asset_table_valid_classifications(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """All cell values should be valid classification strings."""
        valid_values: set[str] = {"stationary", "trend_stationary", "unit_root", "inconclusive"}
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        table: pd.DataFrame = analyzer.render_cross_asset_table(summary)
        for col in table.columns:
            for val in table[col]:
                assert val in valid_values, f"Unexpected classification: {val}"


# ---------------------------------------------------------------------------
# TestGenerateTherefore
# ---------------------------------------------------------------------------


class TestGenerateTherefore:
    """Tests for the programmatic 'Therefore' conclusion."""

    def test_therefore_contains_keyword(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Output must contain 'Therefore'."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        therefore: str = analyzer.generate_therefore(summary)
        assert "Therefore" in therefore

    def test_therefore_mentions_combo_count(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Output must mention the number of combinations."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        therefore: str = analyzer.generate_therefore(summary)
        n_combos: int = len(mixed_dfs)
        assert str(n_combos) in therefore

    def test_therefore_mentions_pct(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Output must mention the overall stationarity percentage."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        therefore: str = analyzer.generate_therefore(summary)
        assert "%" in therefore

    def test_therefore_lists_non_stationary_features(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Output must mention universally non-stationary features."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        therefore: str = analyzer.generate_therefore(summary)
        assert "atr_14" in therefore

    def test_therefore_implication_for_all_stationary(
        self,
        analyzer: RC2StationarityAnalyzer,
        all_stationary_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """When all features are stationary, implication should be optimistic."""
        summary: StationaritySummary = analyzer.analyze_features(all_stationary_dfs, feature_names)
        therefore: str = analyzer.generate_therefore(summary)
        assert "suitable" in therefore.lower() or "majority" in therefore.lower()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_combination(
        self,
        analyzer: RC2StationarityAnalyzer,
        single_combo_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Single (asset, bar_type) combo should work without error."""
        summary: StationaritySummary = analyzer.analyze_features(single_combo_dfs, feature_names)
        assert len(summary.per_asset_bar) == 1
        # With single combo, everything is either universally stationary or non-stationary
        assert len(summary.mixed_features) == 0

    def test_single_feature(
        self,
        analyzer: RC2StationarityAnalyzer,
    ) -> None:
        """Single feature across multiple combos should work."""
        rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
        names: list[str] = ["rsi_14"]
        dfs: dict[tuple[str, str], pd.DataFrame] = {
            ("BTCUSDT", "dollar"): _build_feature_df(rng, names, stationary_names=names),
            ("ETHUSDT", "dollar"): _build_feature_df(rng, names, stationary_names=names),
        }
        summary: StationaritySummary = analyzer.analyze_features(dfs, names)
        assert summary.n_total_features == 1
        assert (
            len(summary.universally_stationary) + len(summary.universally_non_stationary) + len(summary.mixed_features)
            == 1
        )

    def test_empty_feature_dfs_raises(
        self,
        analyzer: RC2StationarityAnalyzer,
        feature_names: list[str],
    ) -> None:
        """Empty feature_dfs dict should raise ValueError."""
        with pytest.raises(ValueError, match="feature_dfs must not be empty"):
            analyzer.analyze_features({}, feature_names)

    def test_empty_feature_names_raises(
        self,
        analyzer: RC2StationarityAnalyzer,
        all_stationary_dfs: dict[tuple[str, str], pd.DataFrame],
    ) -> None:
        """Empty feature_names list should raise ValueError."""
        with pytest.raises(ValueError, match="feature_names must not be empty"):
            analyzer.analyze_features(all_stationary_dfs, [])

    def test_custom_alpha(
        self,
        analyzer: RC2StationarityAnalyzer,
        single_combo_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """Custom alpha=0.01 (stricter) should produce fewer stationary features or equal."""
        summary_05: StationaritySummary = analyzer.analyze_features(single_combo_dfs, feature_names, alpha=0.05)
        summary_01: StationaritySummary = analyzer.analyze_features(single_combo_dfs, feature_names, alpha=0.01)
        # Stricter alpha should not increase stationarity rate
        # (fewer ADF rejections -> fewer "stationary" labels)
        assert summary_01.n_stationary_pct <= summary_05.n_stationary_pct + 0.1  # small tolerance for edge cases

    def test_categories_are_exhaustive(
        self,
        analyzer: RC2StationarityAnalyzer,
        mixed_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_names: list[str],
    ) -> None:
        """universally_stationary + universally_non_stationary + mixed = n_total_features."""
        summary: StationaritySummary = analyzer.analyze_features(mixed_dfs, feature_names)
        total: int = (
            len(summary.universally_stationary) + len(summary.universally_non_stationary) + len(summary.mixed_features)
        )
        assert total == summary.n_total_features
