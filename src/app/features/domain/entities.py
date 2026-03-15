"""Feature validation domain entities -- per-feature, interaction, and report models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel
from pydantic import Field as PydanticField


class FeatureValidationResult(BaseModel, frozen=True):
    """Per-feature validation result across MI, Ridge, and temporal tests.

    The ``keep`` flag is ``True`` only when the feature passes all three
    independent gates: MI significance after BH correction, directional
    accuracy beating the null, and temporal stability.

    Attributes:
        feature_name: Column name of the validated feature.
        mi_score: Raw mutual information in nats.
        mi_pvalue: Empirical p-value from the MI permutation test.
        fdr_corrected_p: Benjamini-Hochberg corrected p-value.
        mi_significant: Whether corrected p-value < alpha.
        directional_accuracy: Fraction of predictions with correct sign.
        da_null_mean: Mean DA from Ridge null distribution.
        da_pvalue: Empirical p-value from Ridge DA null distribution.
        da_beats_null: Whether DA empirical p-value < alpha.
        dc_mae: Direction-conditional MAE (only correct-sign predictions).
        dc_mae_null_mean: Mean DC-MAE from Ridge null distribution.
        stability_score: Fraction of temporal windows where MI is significant.
        is_stable: Whether stability_score >= threshold.
        group: Feature group name from prefix mapping.
        keep: Passes all gates (mi_significant AND da_beats_null AND is_stable).
    """

    feature_name: str

    mi_score: Annotated[float, PydanticField(ge=0)]
    mi_pvalue: Annotated[float, PydanticField(ge=0, le=1)]
    fdr_corrected_p: Annotated[float, PydanticField(ge=0, le=1)]
    mi_significant: bool

    directional_accuracy: Annotated[float, PydanticField(ge=0, le=1)]
    da_null_mean: Annotated[float, PydanticField(ge=0, le=1)]
    da_pvalue: Annotated[float, PydanticField(ge=0, le=1)]
    da_beats_null: bool

    dc_mae: Annotated[float, PydanticField(ge=0)]
    dc_mae_null_mean: Annotated[float, PydanticField(ge=0)]

    stability_score: Annotated[float, PydanticField(ge=0, le=1)]
    is_stable: bool

    group: str
    keep: bool


class InteractionTestResult(BaseModel, frozen=True):
    """Per-group interaction test result.

    Compares group-level Ridge R-squared with the sum of individual
    feature R-squared values to detect synergy or redundancy.

    Attributes:
        group_name: Feature group identifier.
        features_in_group: Tuple of feature column names in this group.
        combined_r2: R-squared using all group features together.
        sum_individual_r2: Sum of individual single-feature R-squared values.
        interaction_ratio: ``combined_r2 / (sum_individual_r2 + eps)``.
        has_interaction: Whether the group exhibits synergy (combined > sum).
        is_redundant: Whether combined R-squared is near the maximum individual
            R-squared (no added value from combining).
    """

    group_name: str
    features_in_group: tuple[str, ...]
    combined_r2: float
    sum_individual_r2: float
    interaction_ratio: float
    has_interaction: bool
    is_redundant: bool


class ValidationReport(BaseModel, frozen=True):
    """Aggregate validation report across all features and groups.

    Attributes:
        feature_results: Per-feature validation results.
        interaction_results: Per-group interaction test results.
        n_features_total: Total number of features validated.
        n_features_kept: Number of features passing all gates.
        n_features_dropped: Number of features failing at least one gate.
        kept_feature_names: Sorted names of kept features.
        dropped_feature_names: Sorted names of dropped features.
        fallback_triggered: Whether the minimum-features-kept fallback was used.
        stability_skipped: Whether temporal stability was skipped (no timestamp column).
    """

    feature_results: tuple[FeatureValidationResult, ...]
    interaction_results: tuple[InteractionTestResult, ...]
    n_features_total: int
    n_features_kept: int
    n_features_dropped: int
    kept_feature_names: tuple[str, ...]
    dropped_feature_names: tuple[str, ...]
    fallback_triggered: bool
    stability_skipped: bool
