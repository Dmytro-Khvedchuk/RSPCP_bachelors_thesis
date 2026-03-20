"""Unit tests for RC2 research domain value objects.

Tests cover the feature rationale table construction, stationarity expectation
enums, section summary validation, surprise tracking, and helper functions.
"""

from __future__ import annotations

import pytest

from src.app.research.domain.rc2_value_objects import (
    FeatureRationale,
    StationarityExpectation,
    StationaritySectionSummary,
    StationaritySurprise,
    build_default_feature_rationales,
    count_expected_nonstationary,
    get_features_needing_transformation,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPECTED_FEATURE_COUNT: int = 23
"""The default indicator config produces exactly 23 features."""

_EXPECTED_GROUPS: frozenset[str] = frozenset({"returns", "volatility", "momentum", "volume", "statistical"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_rationales() -> tuple[FeatureRationale, ...]:
    """Build the default 23-feature rationale table.

    Returns:
        Tuple of FeatureRationale objects.
    """
    return build_default_feature_rationales()


@pytest.fixture
def valid_summary() -> StationaritySectionSummary:
    """Build a minimal valid StationaritySectionSummary.

    Returns:
        Valid summary with 20 stationary, 2 unit root, 1 inconclusive.
    """
    return StationaritySectionSummary(
        n_features_total=23,
        n_stationary=20,
        n_trend_stationary=0,
        n_unit_root=2,
        n_inconclusive=1,
        n_expected_nonstationary=3,
        n_surprises=0,
        surprises=(),
        pass_rate=20 / 23,
        all_expected_nonstationary_confirmed=True,
        downstream_validity=True,
    )


# ---------------------------------------------------------------------------
# TestFeatureRationale -- construction and properties
# ---------------------------------------------------------------------------


class TestFeatureRationale:
    """Tests for individual FeatureRationale value objects."""

    def test_frozen(self) -> None:
        """FeatureRationale must be immutable."""
        rationale = FeatureRationale(
            feature_name="logret_1",
            group="returns",
            formula_summary="ln(C_t / C_{t-1})",
            economic_rationale="test",
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="test",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            rationale.feature_name = "changed"  # type: ignore[misc]

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields transformation and literature_reference default to None."""
        rationale = FeatureRationale(
            feature_name="logret_1",
            group="returns",
            formula_summary="test",
            economic_rationale="test",
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="test",
        )
        assert rationale.transformation_if_nonstationary is None
        assert rationale.literature_reference is None

    def test_with_transformation(self) -> None:
        """Features with transformations must store them correctly."""
        rationale = FeatureRationale(
            feature_name="atr_14",
            group="volatility",
            formula_summary="ewm_mean(TR, 1/14)",
            economic_rationale="price range",
            stationarity_expectation=StationarityExpectation.NON_STATIONARY,
            stationarity_reasoning="absolute units",
            transformation_if_nonstationary="pct_atr (ATR / close)",
        )
        assert rationale.transformation_if_nonstationary is not None
        assert "pct_atr" in rationale.transformation_if_nonstationary


# ---------------------------------------------------------------------------
# TestBuildDefaultFeatureRationales -- the 23-feature table
# ---------------------------------------------------------------------------


class TestBuildDefaultFeatureRationales:
    """Tests for the default feature rationale table builder."""

    def test_produces_expected_count(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Default rationale table must have exactly 23 entries."""
        assert len(default_rationales) == _EXPECTED_FEATURE_COUNT

    def test_all_feature_names_unique(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """All feature names in the rationale table must be unique."""
        names = [r.feature_name for r in default_rationales]
        assert len(names) == len(set(names))

    def test_all_five_groups_represented(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """All five feature groups must be represented."""
        groups = {r.group for r in default_rationales}
        assert groups == _EXPECTED_GROUPS

    def test_returns_group_has_four_features(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Returns group must have exactly 4 features."""
        returns = [r for r in default_rationales if r.group == "returns"]
        assert len(returns) == 4

    def test_volatility_group_has_six_features(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Volatility group must have exactly 6 features."""
        vol = [r for r in default_rationales if r.group == "volatility"]
        assert len(vol) == 6

    def test_momentum_group_has_five_features(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Momentum group must have exactly 5 features."""
        momentum = [r for r in default_rationales if r.group == "momentum"]
        assert len(momentum) == 5

    def test_volume_group_has_three_features(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Volume group must have exactly 3 features."""
        volume = [r for r in default_rationales if r.group == "volume"]
        assert len(volume) == 3

    def test_statistical_group_has_five_features(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Statistical group must have exactly 5 features."""
        stat = [r for r in default_rationales if r.group == "statistical"]
        assert len(stat) == 5

    def test_all_returns_expected_stationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """All return features must be expected stationary."""
        returns = [r for r in default_rationales if r.group == "returns"]
        for r in returns:
            assert r.stationarity_expectation == StationarityExpectation.STATIONARY

    def test_atr_expected_nonstationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """ATR must be expected non-stationary with a transformation."""
        atr = [r for r in default_rationales if r.feature_name == "atr_14"]
        assert len(atr) == 1
        assert atr[0].stationarity_expectation == StationarityExpectation.NON_STATIONARY
        assert atr[0].transformation_if_nonstationary is not None

    def test_slope_expected_nonstationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Price slope must be expected non-stationary with a transformation."""
        slope = [r for r in default_rationales if r.feature_name == "slope_14"]
        assert len(slope) == 1
        assert slope[0].stationarity_expectation == StationarityExpectation.NON_STATIONARY
        assert slope[0].transformation_if_nonstationary is not None

    def test_amihud_expected_borderline(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Amihud must be expected borderline with a transformation."""
        amihud = [r for r in default_rationales if r.feature_name == "amihud_24"]
        assert len(amihud) == 1
        assert amihud[0].stationarity_expectation == StationarityExpectation.BORDERLINE
        assert amihud[0].transformation_if_nonstationary is not None

    def test_rsi_expected_stationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """RSI must be expected stationary (bounded oscillator)."""
        rsi = [r for r in default_rationales if r.feature_name == "rsi_14"]
        assert len(rsi) == 1
        assert rsi[0].stationarity_expectation == StationarityExpectation.STATIONARY

    def test_hurst_expected_stationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Hurst exponent must be expected stationary (bounded [0, 1])."""
        hurst = [r for r in default_rationales if r.feature_name == "hurst_100"]
        assert len(hurst) == 1
        assert hurst[0].stationarity_expectation == StationarityExpectation.STATIONARY

    def test_all_have_nonempty_economic_rationale(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Every feature must have a non-empty economic rationale."""
        for r in default_rationales:
            assert len(r.economic_rationale) > 10, f"{r.feature_name} has empty rationale"

    def test_all_have_nonempty_stationarity_reasoning(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Every feature must have a non-empty stationarity reasoning."""
        for r in default_rationales:
            assert len(r.stationarity_reasoning) > 5, f"{r.feature_name} has empty reasoning"

    def test_all_have_nonempty_formula(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Every feature must have a non-empty formula summary."""
        for r in default_rationales:
            assert len(r.formula_summary) > 3, f"{r.feature_name} has empty formula"

    def test_nonstationary_features_have_transformations(
        self, default_rationales: tuple[FeatureRationale, ...]
    ) -> None:
        """All non-stationary and borderline features must have transformations."""
        for r in default_rationales:
            if r.stationarity_expectation in {
                StationarityExpectation.NON_STATIONARY,
                StationarityExpectation.BORDERLINE,
            }:
                assert r.transformation_if_nonstationary is not None, (
                    f"{r.feature_name}: expected non-stationary/borderline but has no transformation"
                )

    def test_stationary_features_have_no_transformations(
        self, default_rationales: tuple[FeatureRationale, ...]
    ) -> None:
        """Stationary features should not have transformations pre-registered."""
        for r in default_rationales:
            if r.stationarity_expectation == StationarityExpectation.STATIONARY:
                assert r.transformation_if_nonstationary is None, (
                    f"{r.feature_name}: expected stationary "
                    f"but has transformation '{r.transformation_if_nonstationary}'"
                )

    def test_feature_names_match_indicator_naming_convention(
        self, default_rationales: tuple[FeatureRationale, ...]
    ) -> None:
        """Feature names must follow the {indicator}_{param} convention from indicators.py."""
        for r in default_rationales:
            # All feature names should contain an underscore
            assert "_" in r.feature_name, f"{r.feature_name} doesn't follow naming convention"


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for count_expected_nonstationary and get_features_needing_transformation."""

    def test_count_expected_nonstationary(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Must count ATR (non_stationary), slope (non_stationary), amihud (borderline) = 3."""
        count = count_expected_nonstationary(default_rationales)
        assert count == 3  # atr_14, slope_14, amihud_24

    def test_count_expected_nonstationary_empty(self) -> None:
        """Empty tuple returns 0."""
        count = count_expected_nonstationary(())
        assert count == 0

    def test_get_features_needing_transformation(self, default_rationales: tuple[FeatureRationale, ...]) -> None:
        """Must return exactly the 3 features with pre-registered transformations."""
        needing = get_features_needing_transformation(default_rationales)
        assert len(needing) == 3
        names = {r.feature_name for r in needing}
        assert names == {"atr_14", "slope_14", "amihud_24"}

    def test_get_features_needing_transformation_empty(self) -> None:
        """Empty tuple returns empty tuple."""
        needing = get_features_needing_transformation(())
        assert len(needing) == 0


# ---------------------------------------------------------------------------
# TestStationarityExpectation -- enum values
# ---------------------------------------------------------------------------


class TestStationarityExpectation:
    """Tests for the StationarityExpectation enum."""

    def test_three_values(self) -> None:
        """Enum must have exactly 3 members."""
        assert len(StationarityExpectation) == 3

    def test_string_values(self) -> None:
        """Enum members must be lowercase strings for serialization."""
        assert StationarityExpectation.STATIONARY.value == "stationary"
        assert StationarityExpectation.NON_STATIONARY.value == "non_stationary"
        assert StationarityExpectation.BORDERLINE.value == "borderline"

    def test_is_str_subclass(self) -> None:
        """Enum must be a str subclass for Pydantic serialization."""
        assert isinstance(StationarityExpectation.STATIONARY, str)


# ---------------------------------------------------------------------------
# TestStationaritySurprise
# ---------------------------------------------------------------------------


class TestStationaritySurprise:
    """Tests for the StationaritySurprise value object."""

    def test_frozen(self) -> None:
        """StationaritySurprise must be immutable."""
        surprise = StationaritySurprise(
            feature_name="rv_12",
            expected=StationarityExpectation.STATIONARY,
            observed_classification="unit_root",
            asset="BTCUSDT",
            bar_type="dollar",
            explanation="Unexpected long memory",
            action_taken="first_difference",
            is_post_hoc=True,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            surprise.feature_name = "changed"  # type: ignore[misc]

    def test_post_hoc_flag(self) -> None:
        """Post-hoc flag must be stored correctly."""
        pre_registered = StationaritySurprise(
            feature_name="atr_14",
            expected=StationarityExpectation.NON_STATIONARY,
            observed_classification="unit_root",
            asset="BTCUSDT",
            bar_type="dollar",
            explanation="As expected",
            action_taken="pct_atr",
            is_post_hoc=False,
        )
        assert not pre_registered.is_post_hoc

        post_hoc = StationaritySurprise(
            feature_name="rv_12",
            expected=StationarityExpectation.STATIONARY,
            observed_classification="unit_root",
            asset="BTCUSDT",
            bar_type="dollar",
            explanation="Surprising",
            action_taken="first_difference",
            is_post_hoc=True,
        )
        assert post_hoc.is_post_hoc


# ---------------------------------------------------------------------------
# TestStationaritySectionSummary
# ---------------------------------------------------------------------------


class TestStationaritySectionSummary:
    """Tests for StationaritySectionSummary validation."""

    def test_valid_summary_constructs(self, valid_summary: StationaritySectionSummary) -> None:
        """A well-formed summary must construct without error."""
        assert valid_summary.n_features_total == 23
        assert valid_summary.n_stationary == 20
        assert valid_summary.downstream_validity is True

    def test_frozen(self, valid_summary: StationaritySectionSummary) -> None:
        """Summary must be immutable."""
        with pytest.raises(Exception):  # noqa: B017, PT011
            valid_summary.n_stationary = 99  # type: ignore[misc]

    def test_counts_must_sum_to_total(self) -> None:
        """Classification counts must sum to n_features_total."""
        with pytest.raises(ValueError, match="must sum to"):
            StationaritySectionSummary(
                n_features_total=23,
                n_stationary=20,
                n_trend_stationary=0,
                n_unit_root=2,
                n_inconclusive=5,  # 20+0+2+5 = 27 != 23
                n_expected_nonstationary=3,
                n_surprises=0,
                surprises=(),
                pass_rate=0.87,
                all_expected_nonstationary_confirmed=True,
                downstream_validity=True,
            )

    def test_surprise_count_must_match_tuple_length(self) -> None:
        """n_surprises must equal len(surprises)."""
        with pytest.raises(ValueError, match="must equal"):
            StationaritySectionSummary(
                n_features_total=23,
                n_stationary=20,
                n_trend_stationary=0,
                n_unit_root=2,
                n_inconclusive=1,
                n_expected_nonstationary=3,
                n_surprises=2,  # but surprises tuple is empty
                surprises=(),
                pass_rate=0.87,
                all_expected_nonstationary_confirmed=True,
                downstream_validity=True,
            )

    def test_with_surprises(self) -> None:
        """Summary with surprise entries must validate correctly."""
        surprise = StationaritySurprise(
            feature_name="rv_12",
            expected=StationarityExpectation.STATIONARY,
            observed_classification="unit_root",
            asset="BTCUSDT",
            bar_type="dollar",
            explanation="Unexpected",
            action_taken="first_difference",
            is_post_hoc=True,
        )
        summary = StationaritySectionSummary(
            n_features_total=23,
            n_stationary=19,
            n_trend_stationary=0,
            n_unit_root=3,
            n_inconclusive=1,
            n_expected_nonstationary=3,
            n_surprises=1,
            surprises=(surprise,),
            pass_rate=19 / 23,
            all_expected_nonstationary_confirmed=False,
            downstream_validity=True,
        )
        assert summary.n_surprises == 1
        assert len(summary.surprises) == 1
        assert not summary.all_expected_nonstationary_confirmed

    def test_all_stationary(self) -> None:
        """Edge case: all features stationary."""
        summary = StationaritySectionSummary(
            n_features_total=23,
            n_stationary=23,
            n_trend_stationary=0,
            n_unit_root=0,
            n_inconclusive=0,
            n_expected_nonstationary=0,
            n_surprises=0,
            surprises=(),
            pass_rate=1.0,
            all_expected_nonstationary_confirmed=True,
            downstream_validity=True,
        )
        assert summary.pass_rate == 1.0

    def test_all_nonstationary(self) -> None:
        """Edge case: all features non-stationary."""
        summary = StationaritySectionSummary(
            n_features_total=23,
            n_stationary=0,
            n_trend_stationary=0,
            n_unit_root=23,
            n_inconclusive=0,
            n_expected_nonstationary=23,
            n_surprises=0,
            surprises=(),
            pass_rate=0.0,
            all_expected_nonstationary_confirmed=True,
            downstream_validity=False,
        )
        assert summary.pass_rate == 0.0
        assert not summary.downstream_validity

    def test_single_feature(self) -> None:
        """Edge case: single feature."""
        summary = StationaritySectionSummary(
            n_features_total=1,
            n_stationary=1,
            n_trend_stationary=0,
            n_unit_root=0,
            n_inconclusive=0,
            n_expected_nonstationary=0,
            n_surprises=0,
            surprises=(),
            pass_rate=1.0,
            all_expected_nonstationary_confirmed=True,
            downstream_validity=True,
        )
        assert summary.n_features_total == 1

    def test_pass_rate_bounds(self) -> None:
        """pass_rate must be in [0, 1]."""
        with pytest.raises(Exception):  # noqa: B017, PT011
            StationaritySectionSummary(
                n_features_total=23,
                n_stationary=20,
                n_trend_stationary=0,
                n_unit_root=2,
                n_inconclusive=1,
                n_expected_nonstationary=3,
                n_surprises=0,
                surprises=(),
                pass_rate=1.5,  # out of bounds
                all_expected_nonstationary_confirmed=True,
                downstream_validity=True,
            )
