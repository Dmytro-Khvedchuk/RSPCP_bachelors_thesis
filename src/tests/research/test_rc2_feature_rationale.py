"""Tests for RC2 Section 3 feature rationale table and utility functions.

Covers the rationale table construction, lookup functions, VIF cluster
expectations, sign summaries, the Section 3 "Therefore" paragraph, and
edge cases for feature name matching.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.app.research.application.rc2_feature_rationale import (
    FeatureRationale,
    build_feature_rationale_table,
    build_sign_expectation_summary,
    build_vif_expectation_table,
    generate_section3_therefore,
    get_all_feature_names,
    get_expected_vif_clusters,
    get_feature_rationale,
    get_group_rationales,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPECTED_N_FEATURES: int = 23

_ALL_GROUPS: frozenset[str] = frozenset({"returns", "volatility", "momentum", "volume", "statistical"})

_EXPECTED_FEATURE_NAMES: tuple[str, ...] = (
    "logret_1",
    "logret_4",
    "logret_12",
    "logret_24",
    "rv_12",
    "rv_24",
    "rv_48",
    "gk_vol_24",
    "park_vol_24",
    "atr_14",
    "ema_xover_8_21",
    "rsi_14",
    "roc_1",
    "roc_4",
    "roc_12",
    "vol_zscore_24",
    "obv_slope_14",
    "amihud_24",
    "ret_zscore_24",
    "bbpctb_20_2.0",
    "bbwidth_20_2.0",
    "slope_14",
    "hurst_100",
)


# ---------------------------------------------------------------------------
# TestBuildFeatureRationaleTable
# ---------------------------------------------------------------------------


class TestBuildFeatureRationaleTable:
    """Tests for the main rationale table builder."""

    def test_returns_dataframe(self) -> None:
        """build_feature_rationale_table() must return a Pandas DataFrame."""
        df = build_feature_rationale_table()
        assert isinstance(df, pd.DataFrame)

    def test_has_23_rows(self) -> None:
        """Rationale table must have exactly 23 rows (one per feature)."""
        df = build_feature_rationale_table()
        assert len(df) == _EXPECTED_N_FEATURES

    def test_has_required_columns(self) -> None:
        """All expected columns must be present."""
        df = build_feature_rationale_table()
        required_cols = {
            "feature_name",
            "group",
            "economic_intuition",
            "literature_ref",
            "expected_sign",
            "sign_rationale",
            "stationarity_expectation",
            "vif_cluster",
            "is_transformation_based",
        }
        assert required_cols.issubset(set(df.columns))

    def test_all_feature_names_match_expected(self) -> None:
        """Feature names in the table must match the expected set."""
        df = build_feature_rationale_table()
        actual_names = set(df["feature_name"].tolist())
        expected_names = set(_EXPECTED_FEATURE_NAMES)
        assert actual_names == expected_names

    def test_all_groups_are_valid(self) -> None:
        """Every feature must belong to one of the 5 defined groups."""
        df = build_feature_rationale_table()
        actual_groups = set(df["group"].unique())
        assert actual_groups.issubset(_ALL_GROUPS)
        # Every group should have at least one feature
        assert actual_groups == _ALL_GROUPS

    def test_no_empty_intuitions(self) -> None:
        """Every feature must have a non-empty economic intuition."""
        df = build_feature_rationale_table()
        for _, row in df.iterrows():
            assert len(str(row["economic_intuition"])) > 10, (
                f"Feature {row['feature_name']} has too-short economic_intuition"
            )

    def test_no_empty_literature_refs(self) -> None:
        """Every feature must have a non-empty literature reference."""
        df = build_feature_rationale_table()
        for _, row in df.iterrows():
            assert len(str(row["literature_ref"])) > 5, f"Feature {row['feature_name']} has too-short literature_ref"

    def test_expected_sign_values_valid(self) -> None:
        """Expected sign must be one of the valid values."""
        valid_signs = {"positive", "negative", "ambiguous", "unsigned"}
        df = build_feature_rationale_table()
        actual_signs = set(df["expected_sign"].unique())
        assert actual_signs.issubset(valid_signs)

    def test_stationarity_values_valid(self) -> None:
        """Stationarity expectation must be one of the valid values."""
        valid_expectations = {"stationary", "likely_stationary", "likely_non_stationary"}
        df = build_feature_rationale_table()
        actual_expectations = set(df["stationarity_expectation"].unique())
        assert actual_expectations.issubset(valid_expectations)

    def test_index_is_feature_name(self) -> None:
        """DataFrame index must be set to feature_name."""
        df = build_feature_rationale_table()
        assert df.index.name == "feature_name"


# ---------------------------------------------------------------------------
# TestGetFeatureRationale
# ---------------------------------------------------------------------------


class TestGetFeatureRationale:
    """Tests for single-feature lookup."""

    def test_known_feature(self) -> None:
        """Looking up a known feature must return a FeatureRationale."""
        result = get_feature_rationale("logret_1")
        assert result is not None
        assert isinstance(result, FeatureRationale)
        assert result.feature_name == "logret_1"
        assert result.group == "returns"

    def test_unknown_feature_returns_none(self) -> None:
        """Looking up an unknown feature must return None."""
        result = get_feature_rationale("nonexistent_feature_xyz")
        assert result is None

    def test_each_feature_is_retrievable(self) -> None:
        """Every expected feature name must be retrievable."""
        for fname in _EXPECTED_FEATURE_NAMES:
            result = get_feature_rationale(fname)
            assert result is not None, f"Feature {fname} not found in rationale"
            assert result.feature_name == fname

    def test_hurst_has_regime_cluster(self) -> None:
        """hurst_100 should be in the 'regime' VIF cluster."""
        result = get_feature_rationale("hurst_100")
        assert result is not None
        assert result.vif_cluster == "regime"

    def test_volatility_features_are_unsigned(self) -> None:
        """All volatility features should have 'unsigned' expected sign."""
        vol_features = ["rv_12", "rv_24", "rv_48", "gk_vol_24", "park_vol_24", "atr_14"]
        for fname in vol_features:
            result = get_feature_rationale(fname)
            assert result is not None
            assert result.expected_sign == "unsigned", (
                f"{fname} should have unsigned expected sign, got {result.expected_sign}"
            )


# ---------------------------------------------------------------------------
# TestGetGroupRationales
# ---------------------------------------------------------------------------


class TestGetGroupRationales:
    """Tests for group-level lookup."""

    def test_returns_group_has_4_features(self) -> None:
        """The 'returns' group should have exactly 4 features."""
        results = get_group_rationales("returns")
        assert len(results) == 4

    def test_volatility_group_has_6_features(self) -> None:
        """The 'volatility' group should have exactly 6 features."""
        results = get_group_rationales("volatility")
        assert len(results) == 6

    def test_momentum_group_has_5_features(self) -> None:
        """The 'momentum' group should have exactly 5 features."""
        results = get_group_rationales("momentum")
        assert len(results) == 5

    def test_volume_group_has_3_features(self) -> None:
        """The 'volume' group should have exactly 3 features."""
        results = get_group_rationales("volume")
        assert len(results) == 3

    def test_statistical_group_has_5_features(self) -> None:
        """The 'statistical' group should have exactly 5 features."""
        results = get_group_rationales("statistical")
        assert len(results) == 5

    def test_nonexistent_group_returns_empty(self) -> None:
        """Looking up a nonexistent group should return an empty list."""
        results = get_group_rationales("nonexistent_group")
        assert results == []

    def test_group_counts_sum_to_23(self) -> None:
        """Total features across all groups must equal 23."""
        total = sum(len(get_group_rationales(g)) for g in _ALL_GROUPS)
        assert total == _EXPECTED_N_FEATURES


# ---------------------------------------------------------------------------
# TestGetExpectedVIFClusters
# ---------------------------------------------------------------------------


class TestGetExpectedVIFClusters:
    """Tests for VIF cluster expectations."""

    def test_returns_dict(self) -> None:
        """get_expected_vif_clusters() must return a dict."""
        clusters = get_expected_vif_clusters()
        assert isinstance(clusters, dict)

    def test_volatility_cluster_is_largest(self) -> None:
        """The 'volatility' cluster should have the most features."""
        clusters = get_expected_vif_clusters()
        assert "volatility" in clusters
        vol_count = len(clusters["volatility"])
        for name, members in clusters.items():
            if name != "volatility":
                assert len(members) <= vol_count, (
                    f"Cluster '{name}' ({len(members)}) should not exceed 'volatility' ({vol_count})"
                )

    def test_all_features_assigned_to_cluster(self) -> None:
        """Every feature must be in exactly one VIF cluster."""
        clusters = get_expected_vif_clusters()
        all_features_in_clusters = [f for members in clusters.values() for f in members]
        assert len(all_features_in_clusters) == _EXPECTED_N_FEATURES
        # No duplicates
        assert len(set(all_features_in_clusters)) == _EXPECTED_N_FEATURES

    def test_returns_short_cluster_has_roc_and_logret(self) -> None:
        """returns_short cluster should contain both logret_1 and roc_1."""
        clusters = get_expected_vif_clusters()
        assert "returns_short" in clusters
        short_members = clusters["returns_short"]
        assert "logret_1" in short_members
        assert "roc_1" in short_members


# ---------------------------------------------------------------------------
# TestGetAllFeatureNames
# ---------------------------------------------------------------------------


class TestGetAllFeatureNames:
    """Tests for the canonical feature name list."""

    def test_returns_tuple(self) -> None:
        """get_all_feature_names() must return a tuple."""
        result = get_all_feature_names()
        assert isinstance(result, tuple)

    def test_has_23_names(self) -> None:
        """Must return exactly 23 feature names."""
        result = get_all_feature_names()
        assert len(result) == _EXPECTED_N_FEATURES

    def test_matches_expected_set(self) -> None:
        """Feature names must match the expected set."""
        result = get_all_feature_names()
        assert set(result) == set(_EXPECTED_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# TestBuildSignExpectationSummary
# ---------------------------------------------------------------------------


class TestBuildSignExpectationSummary:
    """Tests for the sign expectation summary."""

    def test_returns_dataframe(self) -> None:
        """build_sign_expectation_summary() must return a DataFrame."""
        df = build_sign_expectation_summary()
        assert isinstance(df, pd.DataFrame)

    def test_has_all_sign_categories(self) -> None:
        """All four sign categories must be present."""
        df = build_sign_expectation_summary()
        actual_signs = set(df["expected_sign"].tolist())
        expected_signs = {"positive", "negative", "ambiguous", "unsigned"}
        assert actual_signs == expected_signs

    def test_counts_sum_to_23(self) -> None:
        """Total count across all sign categories must equal 23."""
        df = build_sign_expectation_summary()
        total = df["count"].sum()
        assert total == _EXPECTED_N_FEATURES


# ---------------------------------------------------------------------------
# TestBuildVIFExpectationTable
# ---------------------------------------------------------------------------


class TestBuildVIFExpectationTable:
    """Tests for the VIF expectation table."""

    def test_returns_dataframe(self) -> None:
        """build_vif_expectation_table() must return a DataFrame."""
        df = build_vif_expectation_table()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        """Must have vif_cluster, n_features, features, expected_high_vif."""
        df = build_vif_expectation_table()
        required = {"vif_cluster", "n_features", "features", "expected_high_vif"}
        assert required.issubset(set(df.columns))

    def test_single_feature_clusters_not_high_vif(self) -> None:
        """Clusters with only 1 feature should not expect high VIF."""
        df = build_vif_expectation_table()
        single_clusters = df[df["n_features"] == 1]
        for _, row in single_clusters.iterrows():
            assert row["expected_high_vif"] is False or not row["expected_high_vif"]


# ---------------------------------------------------------------------------
# TestGenerateSection3Therefore
# ---------------------------------------------------------------------------


class TestGenerateSection3Therefore:
    """Tests for the Section 3 conclusion generator."""

    def test_returns_string(self) -> None:
        """generate_section3_therefore() must return a string."""
        result = generate_section3_therefore()
        assert isinstance(result, str)

    def test_mentions_23_features(self) -> None:
        """The conclusion must mention 23 features."""
        result = generate_section3_therefore()
        assert "23" in result

    def test_mentions_da_gap(self) -> None:
        """The conclusion must mention the DA gap argument."""
        result = generate_section3_therefore()
        lower = result.lower()
        assert "break-even" in lower or "directional accuracy" in lower

    def test_mentions_recommendation_system(self) -> None:
        """The conclusion must reference the ML recommendation system."""
        result = generate_section3_therefore()
        lower = result.lower()
        assert "recommendation" in lower or "combining" in lower

    def test_mentions_momentum_and_mean_reversion(self) -> None:
        """The conclusion must mention both directional hypotheses."""
        result = generate_section3_therefore()
        lower = result.lower()
        assert "momentum" in lower
        assert "mean reversion" in lower

    def test_starts_with_therefore(self) -> None:
        """The conclusion must start with 'Therefore'."""
        result = generate_section3_therefore()
        assert result.startswith("**Therefore")


# ---------------------------------------------------------------------------
# TestFeatureRationaleImmutability
# ---------------------------------------------------------------------------


class TestFeatureRationaleImmutability:
    """Tests that FeatureRationale objects are frozen."""

    def test_feature_rationale_is_frozen(self) -> None:
        """FeatureRationale must be immutable (frozen=True)."""
        rationale = get_feature_rationale("logret_1")
        assert rationale is not None
        with pytest.raises(Exception):  # noqa: B017, PT011
            rationale.feature_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestConsistencyWithIndicatorConfig
# ---------------------------------------------------------------------------


class TestConsistencyWithIndicatorConfig:
    """Tests that the rationale table is consistent with indicator defaults."""

    def test_return_horizons_match(self) -> None:
        """Logret features should match default return_horizons (1, 4, 12, 24)."""
        returns = get_group_rationales("returns")
        horizons = sorted(int(r.feature_name.split("_")[1]) for r in returns)
        assert horizons == [1, 4, 12, 24]

    def test_roc_periods_match(self) -> None:
        """Roc features should match default roc_periods (1, 4, 12)."""
        momentum = get_group_rationales("momentum")
        roc_features = [r for r in momentum if r.feature_name.startswith("roc_")]
        roc_periods = sorted(int(r.feature_name.split("_")[1]) for r in roc_features)
        assert roc_periods == [1, 4, 12]

    def test_rv_windows_match(self) -> None:
        """Rv features should match default realized_vol_windows (12, 24, 48)."""
        vol = get_group_rationales("volatility")
        rv_features = [r for r in vol if r.feature_name.startswith("rv_")]
        rv_windows = sorted(int(r.feature_name.split("_")[1]) for r in rv_features)
        assert rv_windows == [12, 24, 48]
