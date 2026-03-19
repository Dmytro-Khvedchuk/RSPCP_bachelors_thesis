"""Unit tests for features domain entities.

Tests construction, immutability, and field constraints for
FeatureValidationResult, InteractionTestResult, and ValidationReport.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.app.features.domain.entities import (
    FeatureValidationResult,
    InteractionTestResult,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validation_result(**overrides: object) -> FeatureValidationResult:
    """Build a minimal valid FeatureValidationResult.

    Args:
        **overrides: Field overrides to apply.

    Returns:
        Configured FeatureValidationResult instance.
    """
    defaults: dict[str, object] = {
        "feature_name": "logret_1",
        "mi_score": 0.05,
        "mi_pvalue": 0.03,
        "fdr_corrected_p": 0.04,
        "mi_significant": True,
        "directional_accuracy": 0.55,
        "da_null_mean": 0.50,
        "da_pvalue": 0.04,
        "da_beats_null": True,
        "dc_mae": 0.002,
        "dc_mae_null_mean": 0.003,
        "stability_score": 0.75,
        "is_stable": True,
        "group": "returns",
        "keep": True,
    }
    defaults.update(overrides)
    return FeatureValidationResult(**defaults)  # type: ignore[arg-type]


def _make_interaction_result(**overrides: object) -> InteractionTestResult:
    """Build a minimal valid InteractionTestResult.

    Args:
        **overrides: Field overrides to apply.

    Returns:
        Configured InteractionTestResult instance.
    """
    defaults: dict[str, object] = {
        "group_name": "returns",
        "features_in_group": ("logret_1", "logret_4"),
        "combined_r2": 0.08,
        "sum_individual_r2": 0.06,
        "interaction_ratio": 1.3,
        "has_interaction": True,
        "is_redundant": False,
    }
    defaults.update(overrides)
    return InteractionTestResult(**defaults)  # type: ignore[arg-type]


def _make_validation_report(**overrides: object) -> ValidationReport:
    """Build a minimal valid ValidationReport.

    Args:
        **overrides: Field overrides to apply.

    Returns:
        Configured ValidationReport instance.
    """
    result1: FeatureValidationResult = _make_validation_result(feature_name="feat_a", keep=True)
    result2: FeatureValidationResult = _make_validation_result(feature_name="feat_b", keep=False, mi_significant=False)
    interaction: InteractionTestResult = _make_interaction_result()

    defaults: dict[str, object] = {
        "feature_results": (result1, result2),
        "interaction_results": (interaction,),
        "n_features_total": 2,
        "n_features_kept": 1,
        "n_features_dropped": 1,
        "kept_feature_names": ("feat_a",),
        "dropped_feature_names": ("feat_b",),
        "fallback_triggered": False,
        "stability_skipped": False,
    }
    defaults.update(overrides)
    return ValidationReport(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FeatureValidationResult tests
# ---------------------------------------------------------------------------


class TestFeatureValidationResult:
    """Tests for the FeatureValidationResult entity."""

    def test_valid_construction(self) -> None:
        """FeatureValidationResult should construct successfully with valid inputs."""
        result: FeatureValidationResult = _make_validation_result()
        assert result.feature_name == "logret_1"
        assert result.keep is True
        assert result.mi_score >= 0.0

    def test_immutable_frozen_model(self) -> None:
        """FeatureValidationResult is frozen — mutation must raise."""
        result: FeatureValidationResult = _make_validation_result()
        with pytest.raises(ValidationError):
            result.keep = False  # type: ignore[misc]

    def test_mi_score_negative_raises(self) -> None:
        """mi_score < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(mi_score=-0.1)

    def test_mi_pvalue_out_of_range_raises(self) -> None:
        """mi_pvalue > 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(mi_pvalue=1.5)

    def test_fdr_corrected_p_out_of_range_raises(self) -> None:
        """fdr_corrected_p > 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(fdr_corrected_p=2.0)

    def test_directional_accuracy_out_of_range_raises(self) -> None:
        """directional_accuracy > 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(directional_accuracy=1.5)

    def test_dc_mae_negative_raises(self) -> None:
        """dc_mae < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(dc_mae=-0.001)

    def test_stability_score_out_of_range_raises(self) -> None:
        """stability_score > 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_validation_result(stability_score=1.1)

    def test_keep_false_construction(self) -> None:
        """FeatureValidationResult with keep=False should construct successfully."""
        result: FeatureValidationResult = _make_validation_result(keep=False)
        assert result.keep is False

    def test_group_field_stored(self) -> None:
        """The group field should be stored as provided."""
        result: FeatureValidationResult = _make_validation_result(group="volatility")
        assert result.group == "volatility"

    def test_dc_mae_null_mean_accepts_nan(self) -> None:
        """dc_mae_null_mean should accept NaN (DC-MAE is now an observed diagnostic only)."""
        import math

        result: FeatureValidationResult = _make_validation_result(dc_mae_null_mean=float("nan"))
        assert math.isnan(result.dc_mae_null_mean)


# ---------------------------------------------------------------------------
# InteractionTestResult tests
# ---------------------------------------------------------------------------


class TestInteractionTestResult:
    """Tests for the InteractionTestResult entity."""

    def test_valid_construction(self) -> None:
        """InteractionTestResult should construct successfully with valid inputs."""
        result: InteractionTestResult = _make_interaction_result()
        assert result.group_name == "returns"
        assert result.has_interaction is True

    def test_immutable_frozen_model(self) -> None:
        """InteractionTestResult is frozen — mutation must raise."""
        result: InteractionTestResult = _make_interaction_result()
        with pytest.raises(ValidationError):
            result.has_interaction = False  # type: ignore[misc]

    def test_no_interaction_construction(self) -> None:
        """has_interaction=False should construct successfully."""
        result: InteractionTestResult = _make_interaction_result(
            has_interaction=False,
            is_redundant=True,
        )
        assert result.has_interaction is False
        assert result.is_redundant is True

    def test_features_in_group_stored_as_tuple(self) -> None:
        """features_in_group should be stored as a tuple."""
        result: InteractionTestResult = _make_interaction_result(
            features_in_group=("feat_x", "feat_y", "feat_z"),
        )
        assert isinstance(result.features_in_group, tuple)
        assert len(result.features_in_group) == 3


# ---------------------------------------------------------------------------
# ValidationReport tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for the ValidationReport aggregate entity."""

    def test_valid_construction(self) -> None:
        """ValidationReport should construct successfully with valid inputs."""
        report: ValidationReport = _make_validation_report()
        assert report.n_features_total == 2
        assert report.n_features_kept == 1
        assert report.n_features_dropped == 1

    def test_immutable_frozen_model(self) -> None:
        """ValidationReport is frozen — mutation must raise."""
        report: ValidationReport = _make_validation_report()
        with pytest.raises(ValidationError):
            report.n_features_kept = 99  # type: ignore[misc]

    def test_fallback_triggered_false(self) -> None:
        """fallback_triggered=False should be stored correctly."""
        report: ValidationReport = _make_validation_report(fallback_triggered=False)
        assert report.fallback_triggered is False

    def test_fallback_triggered_true(self) -> None:
        """fallback_triggered=True should be stored correctly."""
        report: ValidationReport = _make_validation_report(fallback_triggered=True)
        assert report.fallback_triggered is True

    def test_stability_skipped_true(self) -> None:
        """stability_skipped=True should be stored correctly."""
        report: ValidationReport = _make_validation_report(stability_skipped=True)
        assert report.stability_skipped is True

    def test_feature_results_stored_as_tuple(self) -> None:
        """feature_results should be a tuple of FeatureValidationResult objects."""
        report: ValidationReport = _make_validation_report()
        assert isinstance(report.feature_results, tuple)
        assert all(isinstance(r, FeatureValidationResult) for r in report.feature_results)

    def test_interaction_results_stored_as_tuple(self) -> None:
        """interaction_results should be a tuple of InteractionTestResult objects."""
        report: ValidationReport = _make_validation_report()
        assert isinstance(report.interaction_results, tuple)
        assert all(isinstance(r, InteractionTestResult) for r in report.interaction_results)

    def test_kept_feature_names_as_tuple(self) -> None:
        """kept_feature_names should be a tuple of strings."""
        report: ValidationReport = _make_validation_report()
        assert isinstance(report.kept_feature_names, tuple)

    def test_empty_interaction_results(self) -> None:
        """ValidationReport with no interaction results should be valid."""
        report: ValidationReport = _make_validation_report(interaction_results=())
        assert report.interaction_results == ()

    def test_all_features_dropped(self) -> None:
        """ValidationReport with zero kept features should be valid."""
        result: FeatureValidationResult = _make_validation_result(feature_name="x", keep=False)
        report: ValidationReport = ValidationReport(
            feature_results=(result,),
            interaction_results=(),
            n_features_total=1,
            n_features_kept=0,
            n_features_dropped=1,
            kept_feature_names=(),
            dropped_feature_names=("x",),
            fallback_triggered=False,
            stability_skipped=True,
        )
        assert report.n_features_kept == 0
