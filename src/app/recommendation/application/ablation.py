"""Ablation analysis — structured feature group removal with Diebold-Mariano testing."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

from src.app.recommendation.application.metrics import (
    RecommendationMetrics,
    compute_recommendation_metrics,
)
from src.app.recommendation.domain.protocols import IRecommender
from src.app.recommendation.domain.value_objects import Recommendation


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


class FeatureGroup(StrEnum):
    """Feature groups for structured ablation.

    Each group corresponds to a set of columns produced by
    :class:`RecommenderFeatureBuilder`.  Ablation removes one group at
    a time and retrains to measure its marginal contribution.

    Attributes:
        CLASSIFIER: Classifier output features (direction, confidence,
            rolling accuracy).
        REGRESSOR: Regressor output features (predicted return, std,
            quantile spread, CI width).
        REGIME: Regime indicator features (vol regime, MI significance,
            rolling permutation entropy).
    """

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    REGIME = "regime"


# Mapping from feature group to the column name prefixes / exact names
# that identify features belonging to that group.
_GROUP_PREFIXES: dict[FeatureGroup, list[str]] = {
    FeatureGroup.CLASSIFIER: [
        "clf_direction",
        "clf_confidence",
        "clf_rolling_accuracy",
        "forecast_agreement",
        "conviction_score",
    ],
    FeatureGroup.REGRESSOR: [
        "reg_predicted_return",
        "reg_prediction_std",
        "reg_quantile_spread",
        "reg_ci_width",
    ],
    FeatureGroup.REGIME: [
        "vol_regime",
        "mi_significant_regime",
        "rolling_perm_entropy",
    ],
}


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


class AblationResult(BaseModel, frozen=True):
    """Result from ablating a single feature group.

    Contains the ablated group identity, performance metrics of the
    retrained model, and a Diebold-Mariano test comparing the ablated
    model against the full model.

    Attributes:
        group: The feature group that was removed.
        n_features_removed: Number of feature columns removed.
        n_features_remaining: Number of feature columns retained.
        metrics: Recommendation metrics of the ablated model.
        dm_statistic: Diebold-Mariano test statistic (positive means
            ablated model is worse).
        dm_p_value: Two-sided p-value from the DM test.
        mean_loss_full: Mean squared-error loss of the full model.
        mean_loss_ablated: Mean squared-error loss of the ablated model.
    """

    group: FeatureGroup
    n_features_removed: Annotated[int, PydanticField(ge=0)]
    n_features_remaining: Annotated[int, PydanticField(ge=0)]
    metrics: RecommendationMetrics
    dm_statistic: Annotated[float | None, PydanticField(default=None)]
    dm_p_value: Annotated[float | None, PydanticField(default=None)]
    mean_loss_full: Annotated[float | None, PydanticField(default=None)]
    mean_loss_ablated: Annotated[float | None, PydanticField(default=None)]


class AblationSummary(BaseModel, frozen=True):
    """Aggregated ablation results across all feature groups.

    Attributes:
        full_model_metrics: Metrics from the full (non-ablated) model.
        ablation_results: Per-group ablation results.
        n_groups_tested: Number of feature groups ablated.
    """

    full_model_metrics: RecommendationMetrics
    ablation_results: list[AblationResult]
    n_groups_tested: Annotated[int, PydanticField(ge=0)]


# ---------------------------------------------------------------------------
# Public API — run_ablation
# ---------------------------------------------------------------------------


def run_ablation(  # noqa: PLR0913, PLR0917
    recommender_factory: type[IRecommender],
    x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_test: np.ndarray[tuple[int], np.dtype[np.float64]],
    feature_names: list[str],
    full_predictions: list[Recommendation],
    full_metrics: RecommendationMetrics,
    *,
    groups: list[FeatureGroup] | None = None,
    lo_correction_lags: int = 6,
    periods_per_year: float = 365.25 * 24.0,
) -> AblationSummary:
    """Run structured ablation: remove one feature group at a time, retrain, evaluate.

    For each feature group:

    1. Identify columns belonging to the group via prefix matching.
    2. Remove those columns from train and test feature matrices.
    3. Retrain a fresh recommender on the reduced feature set.
    4. Predict on the test set and compute recommendation metrics.
    5. Run a Diebold-Mariano test comparing ablated vs full model.

    This directly tests hypothesis H3: "Does combining tracks add value?"

    Args:
        recommender_factory: Callable that creates a fresh ``IRecommender``
            instance (e.g. ``GradientBoostingRecommender``).
        x_train: Full training feature matrix ``(n_train, n_features)``.
        y_train: Training labels ``(n_train,)``.
        x_test: Full test feature matrix ``(n_test, n_features)``.
        y_test: Test labels ``(n_test,)``.
        feature_names: Column names matching ``x_train``/``x_test`` columns.
        full_predictions: Predictions from the full (non-ablated) model
            on the test set.
        full_metrics: Pre-computed metrics from the full model.
        groups: Feature groups to ablate. Defaults to all groups.
        lo_correction_lags: Autocorrelation lags for Lo Sharpe correction.
        periods_per_year: Decision periods per year for Sharpe computation.

    Returns:
        Frozen :class:`AblationSummary` with per-group results.

    Raises:
        ValueError: If ``feature_names`` length does not match feature matrix
            width or if ``full_predictions`` and ``y_test`` have different
            lengths.
    """
    n_features: int = x_train.shape[1]
    if len(feature_names) != n_features:
        msg: str = f"feature_names length ({len(feature_names)}) must match x_train width ({n_features})"
        raise ValueError(msg)

    if len(full_predictions) != len(y_test):
        msg = f"full_predictions ({len(full_predictions)}) and y_test ({len(y_test)}) must have the same length"
        raise ValueError(msg)

    target_groups: list[FeatureGroup] = groups if groups is not None else list(FeatureGroup)
    ablation_results: list[AblationResult] = []

    # Pre-compute full model squared errors for DM test
    full_errors: np.ndarray[tuple[int], np.dtype[np.float64]] = _squared_errors(
        full_predictions,
        y_test,
    )

    for group in target_groups:
        result: AblationResult = _ablate_single_group(
            group=group,
            recommender_factory=recommender_factory,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            feature_names=feature_names,
            full_errors=full_errors,
            lo_correction_lags=lo_correction_lags,
            periods_per_year=periods_per_year,
        )
        ablation_results.append(result)

    logger.info(
        "Ablation complete: {} groups tested",
        len(ablation_results),
    )

    return AblationSummary(
        full_model_metrics=full_metrics,
        ablation_results=ablation_results,
        n_groups_tested=len(ablation_results),
    )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _ablate_single_group(  # noqa: PLR0913, PLR0914
    *,
    group: FeatureGroup,
    recommender_factory: type[IRecommender],
    x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_test: np.ndarray[tuple[int], np.dtype[np.float64]],
    feature_names: list[str],
    full_errors: np.ndarray[tuple[int], np.dtype[np.float64]],
    lo_correction_lags: int,
    periods_per_year: float,
) -> AblationResult:
    """Ablate a single feature group and evaluate.

    Args:
        group: Feature group to remove.
        recommender_factory: Factory for fresh recommender instances.
        x_train: Full training features.
        y_train: Training labels.
        x_test: Full test features.
        y_test: Test labels.
        feature_names: Column names for the feature matrices.
        full_errors: Squared errors from the full model for DM test.
        lo_correction_lags: Lo correction lags.
        periods_per_year: Periods per year for Sharpe.

    Returns:
        Single ablation result.
    """
    # Identify columns to keep (those NOT in this group)
    remove_set: set[str] = set(_GROUP_PREFIXES.get(group, []))
    keep_indices: list[int] = [i for i, name in enumerate(feature_names) if name not in remove_set]
    n_removed: int = len(feature_names) - len(keep_indices)
    n_remaining: int = len(keep_indices)

    logger.info(
        "Ablating group '{}': removing {} features, {} remaining",
        group.value,
        n_removed,
        n_remaining,
    )

    # Handle edge case: no features removed (group not present)
    if n_removed == 0:
        logger.warning(
            "Group '{}' has no matching features — ablation is a no-op",
            group.value,
        )

    # Handle edge case: all features removed
    if n_remaining == 0:
        logger.warning(
            "Group '{}' ablation removes ALL features — returning empty metrics",
            group.value,
        )
        empty_metrics: RecommendationMetrics = RecommendationMetrics(  # ty: ignore[missing-argument]
            n_decisions=len(y_test),
            n_deployed=0,
        )
        return AblationResult(  # ty: ignore[missing-argument]
            group=group,
            n_features_removed=n_removed,
            n_features_remaining=0,
            metrics=empty_metrics,
        )

    # Slice feature matrices to keep only non-ablated columns
    keep_arr: np.ndarray[tuple[int], np.dtype[np.intp]] = np.array(
        keep_indices,
        dtype=np.intp,
    )
    x_train_ablated: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_train[:, keep_arr]
    x_test_ablated: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_test[:, keep_arr]

    # Retrain fresh recommender on reduced features
    ablated_model: IRecommender = recommender_factory()  # type: ignore[call-arg]
    ablated_model.fit(x_train_ablated, y_train)

    # Predict on test set
    ablated_predictions: list[Recommendation] = ablated_model.predict(x_test_ablated)

    # Compute ablated model metrics
    ablated_metrics: RecommendationMetrics = compute_recommendation_metrics(
        ablated_predictions,
        y_test.tolist(),
        lo_correction_lags=lo_correction_lags,
        periods_per_year=periods_per_year,
    )

    # Diebold-Mariano test: full vs ablated
    ablated_errors: np.ndarray[tuple[int], np.dtype[np.float64]] = _squared_errors(
        ablated_predictions,
        y_test,
    )
    dm_stat: float | None
    dm_p: float | None
    mean_loss_full: float | None
    mean_loss_ablated: float | None
    dm_stat, dm_p, mean_loss_full, mean_loss_ablated = _diebold_mariano_test(
        full_errors,
        ablated_errors,
    )

    logger.info(
        "Ablation '{}': DM_stat={} p_value={} | loss_full={} loss_ablated={}",
        group.value,
        f"{dm_stat:.4f}" if dm_stat is not None else "N/A",
        f"{dm_p:.4f}" if dm_p is not None else "N/A",
        f"{mean_loss_full:.6f}" if mean_loss_full is not None else "N/A",
        f"{mean_loss_ablated:.6f}" if mean_loss_ablated is not None else "N/A",
    )

    return AblationResult(
        group=group,
        n_features_removed=n_removed,
        n_features_remaining=n_remaining,
        metrics=ablated_metrics,
        dm_statistic=dm_stat,
        dm_p_value=dm_p,
        mean_loss_full=mean_loss_full,
        mean_loss_ablated=mean_loss_ablated,
    )


def _squared_errors(
    predictions: list[Recommendation],
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute squared prediction errors.

    Uses ``predicted_strategy_return`` from each recommendation against
    the realised return.

    Args:
        predictions: Model recommendations.
        actuals: Realised returns array.

    Returns:
        Array of squared errors.
    """
    predicted: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        [r.predicted_strategy_return for r in predictions],
        dtype=np.float64,
    )
    errors: np.ndarray[tuple[int], np.dtype[np.float64]] = (actuals - predicted) ** 2
    return errors


def _diebold_mariano_test(
    errors_full: np.ndarray[tuple[int], np.dtype[np.float64]],
    errors_ablated: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_lag: int | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Diebold-Mariano test comparing two models' forecast accuracy.

    Tests ``H_0``: equal predictive accuracy (``E[d_t] = 0``), where
    ``d_t = L(e_full_t) - L(e_ablated_t)`` and ``L`` is squared error.

    A **positive** DM statistic means the full model has **larger** loss
    than the ablated model (i.e. the ablated group was hurting).  A
    **negative** statistic means the full model is better (the ablated
    group was helping).

    Uses Newey-West HAC variance estimator to handle autocorrelated
    loss differentials.

    Args:
        errors_full: Squared errors from the full model.
        errors_ablated: Squared errors from the ablated model.
        max_lag: Maximum lag for Newey-West estimator.  Defaults to
            ``int(n^(1/3))`` following Andrews (1991).

    Returns:
        Tuple of ``(dm_statistic, p_value, mean_loss_full, mean_loss_ablated)``.
        All ``None`` if insufficient data.
    """
    n: int = len(errors_full)
    if n < 2:  # noqa: PLR2004
        return None, None, None, None

    # Loss differential: d_t = L(full) - L(ablated)
    d: np.ndarray[tuple[int], np.dtype[np.float64]] = errors_full - errors_ablated
    d_bar: float = float(np.mean(d))
    mean_loss_full: float = float(np.mean(errors_full))
    mean_loss_ablated: float = float(np.mean(errors_ablated))

    # Newey-West HAC variance estimator
    if max_lag is None:
        max_lag = int(np.power(n, 1.0 / 3.0))
    max_lag = max(max_lag, 0)

    d_demeaned: np.ndarray[tuple[int], np.dtype[np.float64]] = d - d_bar
    gamma_0: float = float(np.dot(d_demeaned, d_demeaned)) / n

    nw_sum: float = 0.0
    for k in range(1, max_lag + 1):
        gamma_k: float = float(np.dot(d_demeaned[k:], d_demeaned[:-k])) / n
        bartlett_weight: float = 1.0 - k / (max_lag + 1)
        nw_sum += 2.0 * bartlett_weight * gamma_k

    var_d: float = (gamma_0 + nw_sum) / n
    if var_d <= 0.0:
        return None, None, mean_loss_full, mean_loss_ablated

    dm_stat: float = d_bar / np.sqrt(var_d)

    # Two-sided p-value from t-distribution with n-1 df
    p_value: float = float(
        2.0 * scipy_stats.t.sf(abs(dm_stat), df=n - 1),
    )

    return dm_stat, p_value, mean_loss_full, mean_loss_ablated
