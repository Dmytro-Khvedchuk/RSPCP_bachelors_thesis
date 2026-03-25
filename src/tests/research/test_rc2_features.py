"""Tests for RC2 Section 3 feature exploration analysis service.

Covers VIF computation, correlation matrix, feature rationale table, distribution
summaries, feature-target correlations, and edge cases (single feature, constant
columns, empty inputs, missing columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.research.application.rc2_features import RC2FeatureAnalyzer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RNG_SEED: int = 42
_N_ROWS: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(
    rng: np.random.Generator,
    n: int = _N_ROWS,
    n_features: int = 5,
    add_target: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a DataFrame with random features.

    Returns:
        Tuple of (DataFrame, feature_names).
    """
    feature_names: list[str] = [f"feat_{i}" for i in range(n_features)]
    data: dict[str, np.ndarray] = {}
    for fname in feature_names:
        data[fname] = rng.standard_normal(n)

    if add_target:
        data["fwd_logret_1"] = rng.standard_normal(n)

    return pd.DataFrame(data), feature_names


def _make_collinear_df(
    rng: np.random.Generator,
    n: int = _N_ROWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a DataFrame where feat_1 is nearly a linear combination of feat_0.

    Returns:
        Tuple of (DataFrame, feature_names).
    """
    feat_0: np.ndarray = rng.standard_normal(n)
    feat_1: np.ndarray = feat_0 * 0.99 + rng.standard_normal(n) * 0.01
    feat_2: np.ndarray = rng.standard_normal(n)
    feature_names: list[str] = ["feat_0", "feat_1", "feat_2"]
    df: pd.DataFrame = pd.DataFrame(
        {
            "feat_0": feat_0,
            "feat_1": feat_1,
            "feat_2": feat_2,
        }
    )
    return df, feature_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> RC2FeatureAnalyzer:
    return RC2FeatureAnalyzer()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(_RNG_SEED)


# ---------------------------------------------------------------------------
# VIF Tests
# ---------------------------------------------------------------------------


class TestComputeVIF:
    def test_returns_dataframe_with_expected_columns(
        self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator
    ) -> None:
        df, names = _make_feature_df(rng)
        result = analyzer.compute_vif(df, names)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "vif" in result.columns
        assert len(result) == len(names)

    def test_uncorrelated_features_have_low_vif(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n=2000, n_features=3)
        result = analyzer.compute_vif(df, names)
        # Independent features should have VIF close to 1
        for _, row in result.iterrows():
            assert row["vif"] < 5.0, f"Feature {row['feature']} has VIF {row['vif']}"

    def test_collinear_features_have_high_vif(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_collinear_df(rng)
        result = analyzer.compute_vif(df, names)
        # feat_0 and feat_1 are collinear -- both should have high VIF
        collinear_vifs = result[result["feature"].isin(["feat_0", "feat_1"])]["vif"]
        assert all(v > 10.0 for v in collinear_vifs), (
            f"Expected VIF > 10 for collinear features, got {collinear_vifs.tolist()}"
        )

    def test_sorted_by_vif_descending(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_collinear_df(rng)
        result = analyzer.compute_vif(df, names)
        vif_values = result["vif"].tolist()
        # Replace inf with a large number for comparison
        vif_finite = [v if np.isfinite(v) else 1e20 for v in vif_values]
        assert vif_finite == sorted(vif_finite, reverse=True)

    def test_single_feature_returns_vif(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, _ = _make_feature_df(rng, n_features=1)
        result = analyzer.compute_vif(df, ["feat_0"])
        assert len(result) == 1
        # Single feature VIF is always 1 (no other features to correlate with)
        assert np.isfinite(result["vif"].iloc[0])

    def test_constant_column_gets_inf_vif(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        n: int = 500
        df = pd.DataFrame(
            {
                "const": np.ones(n),
                "normal": rng.standard_normal(n),
            }
        )
        result = analyzer.compute_vif(df, ["const", "normal"])
        const_vif = result[result["feature"] == "const"]["vif"].iloc[0]
        # Constant column should have inf or very high VIF
        assert const_vif > 1e5 or np.isinf(const_vif)

    def test_empty_feature_names_raises(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, _ = _make_feature_df(rng)
        with pytest.raises(ValueError, match="feature_names must not be empty"):
            analyzer.compute_vif(df, [])

    def test_missing_columns_raises(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, _ = _make_feature_df(rng)
        with pytest.raises(ValueError, match="Features not found"):
            analyzer.compute_vif(df, ["nonexistent_col"])

    def test_handles_nan_values(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng)
        df.iloc[0, 0] = np.nan
        # Should not raise -- NaN is replaced with 0
        result = analyzer.compute_vif(df, names)
        assert len(result) == len(names)


# ---------------------------------------------------------------------------
# Correlation Matrix Tests
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_returns_square_matrix(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=4)
        result = analyzer.compute_correlation_matrix(df, names)
        assert result.shape == (4, 4)
        assert list(result.columns) == names
        assert list(result.index) == names

    def test_diagonal_is_one(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=3)
        result = analyzer.compute_correlation_matrix(df, names)
        for name in names:
            assert abs(result.loc[name, name] - 1.0) < 1e-10

    def test_symmetric_matrix(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng)
        result = analyzer.compute_correlation_matrix(df, names)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_collinear_features_high_correlation(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_collinear_df(rng)
        result = analyzer.compute_correlation_matrix(df, names)
        assert result.loc["feat_0", "feat_1"] > 0.95

    def test_empty_names_raises(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, _ = _make_feature_df(rng)
        with pytest.raises(ValueError, match="feature_names must not be empty"):
            analyzer.compute_correlation_matrix(df, [])


# ---------------------------------------------------------------------------
# Feature Rationale Table Tests
# ---------------------------------------------------------------------------


class TestBuildFeatureRationaleTable:
    def test_returns_23_features(self, analyzer: RC2FeatureAnalyzer) -> None:
        result = analyzer.build_feature_rationale_table()
        assert len(result) == 23

    def test_expected_columns(self, analyzer: RC2FeatureAnalyzer) -> None:
        result = analyzer.build_feature_rationale_table()
        expected_cols = {
            "Feature",
            "Group",
            "Formula",
            "Economic Rationale",
            "Stationarity Exp.",
            "Transformation",
            "Reference",
        }
        assert set(result.columns) == expected_cols

    def test_all_groups_present(self, analyzer: RC2FeatureAnalyzer) -> None:
        result = analyzer.build_feature_rationale_table()
        groups = set(result["Group"].unique())
        expected_groups = {"returns", "volatility", "momentum", "volume", "statistical"}
        assert groups == expected_groups

    def test_no_empty_rationales(self, analyzer: RC2FeatureAnalyzer) -> None:
        result = analyzer.build_feature_rationale_table()
        for _, row in result.iterrows():
            assert len(str(row["Economic Rationale"])) > 10
            assert len(str(row["Formula"])) > 3


# ---------------------------------------------------------------------------
# Feature Distribution Tests
# ---------------------------------------------------------------------------


class TestComputeFeatureDistributions:
    def test_returns_expected_columns(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=3)
        result = analyzer.compute_feature_distributions(df, names, kept_features=names[:1])
        expected_cols = {
            "feature",
            "status",
            "mean",
            "std",
            "median",
            "skew",
            "kurtosis",
            "min",
            "max",
            "q25",
            "q75",
            "n_obs",
        }
        assert set(result.columns) == expected_cols

    def test_kept_dropped_labels(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=4)
        kept = names[:2]
        result = analyzer.compute_feature_distributions(df, names, kept_features=kept)
        kept_rows = result[result["status"] == "kept"]
        dropped_rows = result[result["status"] == "dropped"]
        assert len(kept_rows) == 2
        assert len(dropped_rows) == 2

    def test_empty_feature_names_raises(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, _ = _make_feature_df(rng)
        with pytest.raises(ValueError, match="feature_names must not be empty"):
            analyzer.compute_feature_distributions(df, [], [])

    def test_all_kept(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=3)
        result = analyzer.compute_feature_distributions(df, names, kept_features=names)
        assert all(result["status"] == "kept")

    def test_all_dropped(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, n_features=3)
        result = analyzer.compute_feature_distributions(df, names, kept_features=[])
        assert all(result["status"] == "dropped")


# ---------------------------------------------------------------------------
# Feature-Target Correlation Tests
# ---------------------------------------------------------------------------


class TestComputeFeatureTargetCorrelations:
    def test_returns_expected_columns(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, add_target=True)
        result = analyzer.compute_feature_target_correlations(df, names, "fwd_logret_1")
        expected_cols = {"feature", "pearson_r", "spearman_r", "abs_pearson_r"}
        assert set(result.columns) == expected_cols

    def test_sorted_by_abs_pearson_descending(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng, add_target=True)
        result = analyzer.compute_feature_target_correlations(df, names, "fwd_logret_1")
        abs_vals = result["abs_pearson_r"].tolist()
        # Filter out NaN for comparison
        finite_vals = [v for v in abs_vals if np.isfinite(v)]
        assert finite_vals == sorted(finite_vals, reverse=True)

    def test_missing_target_raises(self, analyzer: RC2FeatureAnalyzer, rng: np.random.Generator) -> None:
        df, names = _make_feature_df(rng)
        with pytest.raises(ValueError, match="Target column"):
            analyzer.compute_feature_target_correlations(df, names, "nonexistent")

    def test_perfect_correlation(
        self,
        analyzer: RC2FeatureAnalyzer,
    ) -> None:
        n: int = 200
        x: np.ndarray = np.linspace(0, 10, n)
        df = pd.DataFrame({"feat_0": x, "target": x * 2 + 1})
        result = analyzer.compute_feature_target_correlations(df, ["feat_0"], "target")
        assert result["pearson_r"].iloc[0] > 0.99
