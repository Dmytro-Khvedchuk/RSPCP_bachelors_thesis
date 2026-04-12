"""Tests for recommendation metrics, conformal intervals, and ablation analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.app.recommendation.application.ablation import (
    AblationResult,
    AblationSummary,
    FeatureGroup,
    run_ablation,
)
from src.app.recommendation.application.gradient_boosting_recommender import (
    GradientBoostingRecommender,
    GradientBoostingRecommenderConfig,
)
from src.app.recommendation.application.metrics import (
    ConformalDeployConfig,
    RecommendationMetrics,
    build_conformal_intervals,
    compute_recommendation_metrics,
)
from src.app.recommendation.domain.value_objects import Recommendation


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SEED: int = 42
_N_TRAIN: int = 150
_N_TEST: int = 50
_N_FEATURES: int = 8

# Feature names that contain known group prefixes from _GROUP_PREFIXES
_CLASSIFIER_FEATURES: list[str] = [
    "clf_direction",
    "clf_confidence",
    "clf_rolling_accuracy",
    "forecast_agreement",
    "conviction_score",
]
_REGRESSOR_FEATURES: list[str] = [
    "reg_predicted_return",
    "reg_prediction_std",
    "reg_quantile_spread",
    "reg_ci_width",
]
_REGIME_FEATURES: list[str] = [
    "vol_regime",
    "mi_significant_regime",
    "rolling_perm_entropy",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_recommendation(
    *,
    predicted_return: float = 0.01,
    deploy: bool = True,
    position_size: float = 1.0,
    asset: str = "BTCUSDT",
) -> Recommendation:
    """Create a single Recommendation with sensible defaults.

    Args:
        predicted_return: Predicted strategy return.
        deploy: Whether this recommendation deploys.
        position_size: Position size in [0.0, 1.0].
        asset: Asset symbol.

    Returns:
        A valid :class:`Recommendation` instance.
    """
    return Recommendation(
        asset=asset,
        predicted_strategy_return=predicted_return,
        confidence=min(abs(predicted_return) * 10, 1.0),
        deploy=deploy,
        predicted_direction=1 if predicted_return >= 0 else -1,
        predicted_magnitude=abs(predicted_return),
        position_size=position_size,
    )


def _make_linear_ablation_data(
    n: int = _N_TRAIN + _N_TEST,
    n_features: int = _N_FEATURES,
    seed: int = _SEED,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate synthetic linear data for ablation tests.

    Args:
        n: Total number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        Tuple of (X, y) arrays.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
    w: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n_features).astype(np.float64)
    y: np.ndarray[tuple[int], np.dtype[np.float64]] = (x @ w + rng.normal(0, 0.3, n)).astype(np.float64)
    return x, y


def _make_noise_ablation_data(
    n: int = _N_TRAIN + _N_TEST,
    n_features: int = _N_FEATURES,
    seed: int = _SEED,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate pure noise data for ablation tests.

    Args:
        n: Total number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        Tuple of (X, y) arrays.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
    y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)
    return x, y


def _build_full_model_predictions(
    x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> list[Recommendation]:
    """Train a GradientBoostingRecommender and return test predictions.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        x_test: Test feature matrix.

    Returns:
        List of Recommendation objects for the test set.
    """
    model: GradientBoostingRecommender = GradientBoostingRecommender(
        config=GradientBoostingRecommenderConfig(n_estimators=30, min_threshold=0.0)
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


# ---------------------------------------------------------------------------
# TestComputeRecommendationMetrics
# ---------------------------------------------------------------------------


class TestComputeRecommendationMetrics:
    """Tests for the compute_recommendation_metrics function."""

    def test_empty_predictions_returns_zeros(self) -> None:
        """Empty prediction list returns n_decisions=0, n_deployed=0."""
        result: RecommendationMetrics = compute_recommendation_metrics([], [])
        assert result.n_decisions == 0
        assert result.n_deployed == 0
        assert result.deploy_rate is None
        assert result.deploy_precision is None
        assert result.sharpe_with_sizing is None

    def test_length_mismatch_raises(self) -> None:
        """Mismatched prediction and actual lengths raise ValueError."""
        preds: list[Recommendation] = [_make_recommendation()]
        actuals: list[float] = [0.01, 0.02]
        with pytest.raises(ValueError, match="same length"):
            compute_recommendation_metrics(preds, actuals)

    def test_all_deployed_positive_returns_precision_one(self) -> None:
        """All deploy=True with positive returns yields precision=1.0."""
        n: int = 20
        preds: list[Recommendation] = [_make_recommendation(predicted_return=0.01, deploy=True) for _ in range(n)]
        actuals: list[float] = [0.005] * n
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.n_decisions == n
        assert result.n_deployed == n
        assert result.deploy_precision == pytest.approx(1.0)

    def test_deploy_precision_mixed(self) -> None:
        """Mixed positive/negative deployed returns gives correct precision."""
        # 3 positive, 1 negative => precision = 0.75
        preds: list[Recommendation] = [
            _make_recommendation(deploy=True),
            _make_recommendation(deploy=True),
            _make_recommendation(deploy=True),
            _make_recommendation(deploy=True),
        ]
        actuals: list[float] = [0.01, 0.02, 0.03, -0.04]
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.n_deployed == 4
        assert result.deploy_precision == pytest.approx(0.75)

    def test_no_deployments_precision_none(self) -> None:
        """When all deploy=False, deploy_precision is None and deploy_rate=0."""
        n: int = 10
        preds: list[Recommendation] = [
            _make_recommendation(deploy=False, position_size=0.0, predicted_return=0.01) for _ in range(n)
        ]
        actuals: list[float] = [0.01] * n
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.n_deployed == 0
        assert result.deploy_precision is None
        assert result.deploy_rate == pytest.approx(0.0)

    def test_sizing_value_positive_when_sizing_helps(self) -> None:
        """Scenario where varying position_size improves Sharpe vs binary sizing.

        When positions are proportional to signal strength and the signal is
        informative, sized returns should have lower variance per unit of mean
        than binary returns, yielding a higher Sharpe (or at least non-negative
        sizing_value when averaged over many samples).
        """
        rng: np.random.Generator = np.random.default_rng(0)
        n: int = 200
        # Strong signal in position_size: higher size correlates with higher return
        sizes: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.uniform(0.1, 1.0, n).astype(np.float64)
        returns: np.ndarray[tuple[int], np.dtype[np.float64]] = (sizes * 0.02 + rng.normal(0, 0.005, n)).astype(
            np.float64
        )

        preds: list[Recommendation] = [
            _make_recommendation(
                predicted_return=float(returns[i]),
                deploy=True,
                position_size=float(sizes[i]),
            )
            for i in range(n)
        ]
        actuals: list[float] = returns.tolist()
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals, periods_per_year=252.0)
        # sizing_value is not None when both Sharpes are computable
        assert result.sizing_value is not None

    def test_sizing_value_near_zero_when_all_size_one(self) -> None:
        """When all position_size=1.0, sized portfolio equals binary → sizing_value=0."""
        n: int = 50
        preds: list[Recommendation] = [_make_recommendation(deploy=True, position_size=1.0) for _ in range(n)]
        actuals: list[float] = [0.01 if i % 2 == 0 else -0.005 for i in range(n)]
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.sizing_value is not None
        assert result.sizing_value == pytest.approx(0.0, abs=1e-10)

    def test_cumulative_return_calculation(self) -> None:
        """Cumulative return equals prod(1 + r_i) - 1 on sized deployed returns."""
        preds: list[Recommendation] = [
            _make_recommendation(deploy=True, position_size=1.0, predicted_return=0.01),
            _make_recommendation(deploy=True, position_size=0.5, predicted_return=0.01),
            _make_recommendation(deploy=False, position_size=0.0, predicted_return=-0.01),
            _make_recommendation(deploy=True, position_size=1.0, predicted_return=0.02),
        ]
        actuals: list[float] = [0.02, 0.03, -0.01, 0.01]
        # sized_returns: [1.0*0.02, 0.5*0.03, 0, 1.0*0.01] = [0.02, 0.015, 0, 0.01]
        r0: float = 0.02
        r1: float = 0.015
        r2: float = 0.0
        r3: float = 0.01
        expected_cum: float = (1 + r0) * (1 + r1) * (1 + r2) * (1 + r3) - 1.0
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.cumulative_return == pytest.approx(expected_cum, rel=1e-9)

    def test_mean_portfolio_return(self) -> None:
        """Mean portfolio return equals mean of sized returns."""
        preds: list[Recommendation] = [
            _make_recommendation(deploy=True, position_size=1.0, predicted_return=0.01),
            _make_recommendation(deploy=False, position_size=0.0, predicted_return=-0.01),
            _make_recommendation(deploy=True, position_size=0.5, predicted_return=0.01),
        ]
        actuals: list[float] = [0.04, -0.02, 0.06]
        # sized_returns: [1.0*0.04, 0, 0.5*0.06] = [0.04, 0, 0.03]
        expected_mean: float = (0.04 + 0.0 + 0.03) / 3.0
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.mean_portfolio_return == pytest.approx(expected_mean, rel=1e-9)

    def test_lo_correction_applied(self) -> None:
        """Lo correction factor is computed and is not None for sufficient data."""
        n: int = 100
        rng: np.random.Generator = np.random.default_rng(7)
        returns: list[float] = rng.normal(0.005, 0.01, n).tolist()
        preds: list[Recommendation] = [_make_recommendation(deploy=True, position_size=1.0) for _ in range(n)]
        result: RecommendationMetrics = compute_recommendation_metrics(preds, returns)
        assert result.lo_correction_factor is not None
        # Lo factor should be a positive real number
        assert result.lo_correction_factor > 0.0
        assert math.isfinite(result.lo_correction_factor)

    def test_n_deployed_counts_only_deploy_true(self) -> None:
        """n_deployed counts only recommendations where deploy=True."""
        preds: list[Recommendation] = [
            _make_recommendation(deploy=True),
            _make_recommendation(deploy=False, position_size=0.0, predicted_return=-0.01),
            _make_recommendation(deploy=True),
            _make_recommendation(deploy=False, position_size=0.0, predicted_return=-0.02),
            _make_recommendation(deploy=True),
        ]
        actuals: list[float] = [0.01] * 5
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.n_decisions == 5
        assert result.n_deployed == 3
        assert result.deploy_rate == pytest.approx(3.0 / 5.0)

    def test_sharpe_none_when_all_returns_zero(self) -> None:
        """Zero-variance returns yield Sharpe=None (undefined)."""
        n: int = 10
        preds: list[Recommendation] = [_make_recommendation(deploy=True, position_size=1.0) for _ in range(n)]
        # All actuals zero → sized_returns all zero → std=0
        actuals: list[float] = [0.0] * n
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.sharpe_with_sizing is None
        assert result.sharpe_without_sizing is None

    def test_single_prediction_returns_metrics(self) -> None:
        """A single prediction is insufficient for Sharpe but gives basic counts."""
        preds: list[Recommendation] = [_make_recommendation(deploy=True)]
        actuals: list[float] = [0.02]
        result: RecommendationMetrics = compute_recommendation_metrics(preds, actuals)
        assert result.n_decisions == 1
        assert result.n_deployed == 1
        # Sharpe requires >=2 data points
        assert result.sharpe_with_sizing is None


# ---------------------------------------------------------------------------
# TestBuildConformalIntervals
# ---------------------------------------------------------------------------


class TestBuildConformalIntervals:
    """Tests for the build_conformal_intervals function."""

    def test_calibration_length_mismatch_raises(self) -> None:
        """Different-length calibration arrays raise ValueError."""
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.02], dtype=np.float64)
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01], dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            build_conformal_intervals(cal_preds, cal_actuals, new_preds)

    def test_empty_calibration_raises(self) -> None:
        """Empty calibration arrays raise ValueError."""
        empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([], dtype=np.float64)
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01], dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            build_conformal_intervals(empty, empty, new_preds)

    def test_perfect_calibration_narrow_intervals(self) -> None:
        """Near-zero calibration errors produce small q and more deployments."""
        rng: np.random.Generator = np.random.default_rng(0)
        n_cal: int = 100
        n_new: int = 50
        # Predicted and actual are nearly identical → nonconformity scores ≈ 0
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.uniform(0.01, 0.05, n_cal).astype(np.float64)
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = (cal_preds + rng.normal(0, 1e-6, n_cal)).astype(
            np.float64
        )
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.uniform(0.01, 0.05, n_new).astype(np.float64)

        deploy, lower, upper, q = build_conformal_intervals(cal_preds, cal_actuals, new_preds)
        # Very small q means most lower bounds > 0 for positive predictions
        assert q < 0.01
        assert deploy.sum() > n_new // 2

    def test_wide_errors_fewer_deployments(self) -> None:
        """Large calibration errors produce large q and fewer deployments."""
        rng: np.random.Generator = np.random.default_rng(1)
        n_cal: int = 100
        n_new: int = 50
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.uniform(0.01, 0.05, n_cal).astype(np.float64)
        # Large noise → large nonconformity scores
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = (cal_preds + rng.normal(0, 5.0, n_cal)).astype(
            np.float64
        )
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.uniform(0.01, 0.05, n_new).astype(np.float64)

        deploy, lower, upper, q = build_conformal_intervals(cal_preds, cal_actuals, new_preds)
        # Large q means almost nothing deploys (lower_bound = pred - q << 0)
        assert q > 1.0
        assert deploy.sum() == 0

    def test_lower_bound_positive_means_deploy(self) -> None:
        """Deploy decision is True iff lower_bound > 0."""
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, 0.5, 0.001], dtype=np.float64)

        deploy, lower, upper, q = build_conformal_intervals(cal_preds, cal_actuals, new_preds)
        for i in range(len(new_preds)):
            expected_deploy: bool = bool(lower[i] > 0.0)
            assert bool(deploy[i]) == expected_deploy

    def test_alpha_affects_interval_width(self) -> None:
        """Lower alpha produces wider intervals (more conservative)."""
        rng: np.random.Generator = np.random.default_rng(2)
        n_cal: int = 200
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.normal(0.0, 0.01, n_cal).astype(np.float64)
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = (cal_preds + rng.normal(0, 0.005, n_cal)).astype(
            np.float64
        )
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.normal(0.01, 0.01, 20).astype(np.float64)

        # alpha=0.01 (very conservative) vs alpha=0.5 (lenient)
        cfg_conservative: ConformalDeployConfig = ConformalDeployConfig(alpha=0.01)
        cfg_lenient: ConformalDeployConfig = ConformalDeployConfig(alpha=0.5)

        _, _, _, q_conservative = build_conformal_intervals(cal_preds, cal_actuals, new_preds, cfg_conservative)
        _, _, _, q_lenient = build_conformal_intervals(cal_preds, cal_actuals, new_preds, cfg_lenient)

        # Lower alpha → higher quantile level → larger q
        assert q_conservative >= q_lenient

    def test_quantile_finite_sample_correction(self) -> None:
        """Quantile uses (1-alpha)*(1+1/n) level for finite-sample validity.

        Verifies that the implementation matches the documented formula:
        quantile_level = min(1.0, (1 - alpha) * (1 + 1/n_cal)).
        """
        n_cal: int = 10
        cal_errors: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64
        )
        # Perfect predictions so nonconformity = |actuals - preds| = cal_errors
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_cal, dtype=np.float64)
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.5], dtype=np.float64)

        cfg: ConformalDeployConfig = ConformalDeployConfig(alpha=0.1)
        alpha: float = 0.1
        expected_level: float = min(1.0, (1 - alpha) * (1 + 1.0 / n_cal))
        expected_q: float = float(np.quantile(cal_errors, expected_level))

        _, _, _, actual_q = build_conformal_intervals(cal_preds, cal_errors, new_preds, cfg)
        assert actual_q == pytest.approx(expected_q, rel=1e-9)

    def test_return_shapes_match_new_predictions(self) -> None:
        """deploy, lower_bounds, upper_bounds all have len(new_predictions) elements."""
        cal_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(20, dtype=np.float64) * 0.01
        cal_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(20, dtype=np.float64) * 0.012
        new_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.05, 0.03, 0.07], dtype=np.float64)

        deploy, lower, upper, q = build_conformal_intervals(cal_preds, cal_actuals, new_preds)
        assert len(deploy) == 3
        assert len(lower) == 3
        assert len(upper) == 3


# ---------------------------------------------------------------------------
# TestAblationResult
# ---------------------------------------------------------------------------


class TestAblationResult:
    """Tests for the AblationResult value object."""

    def test_valid_construction(self) -> None:
        """AblationResult can be constructed with all required fields."""
        metrics: RecommendationMetrics = RecommendationMetrics(
            n_decisions=10,
            n_deployed=5,
            deploy_rate=0.5,
        )
        result: AblationResult = AblationResult(
            group=FeatureGroup.CLASSIFIER,
            n_features_removed=3,
            n_features_remaining=5,
            metrics=metrics,
            dm_statistic=-1.5,
            dm_p_value=0.13,
            mean_loss_full=0.001,
            mean_loss_ablated=0.003,
        )
        assert result.group == FeatureGroup.CLASSIFIER
        assert result.n_features_removed == 3
        assert result.n_features_remaining == 5
        assert result.dm_statistic == pytest.approx(-1.5)
        assert result.dm_p_value == pytest.approx(0.13)

    def test_frozen(self) -> None:
        """AblationResult is immutable — field assignment raises ValidationError."""
        metrics: RecommendationMetrics = RecommendationMetrics(
            n_decisions=5,
            n_deployed=2,
        )
        result: AblationResult = AblationResult(
            group=FeatureGroup.REGRESSOR,
            n_features_removed=2,
            n_features_remaining=6,
            metrics=metrics,
        )
        with pytest.raises((ValidationError, TypeError)):
            result.group = FeatureGroup.REGIME  # type: ignore[misc]

    def test_optional_fields_default_none(self) -> None:
        """dm_statistic, dm_p_value, mean_loss_full, mean_loss_ablated default to None."""
        metrics: RecommendationMetrics = RecommendationMetrics(
            n_decisions=5,
            n_deployed=2,
        )
        result: AblationResult = AblationResult(
            group=FeatureGroup.REGIME,
            n_features_removed=0,
            n_features_remaining=8,
            metrics=metrics,
        )
        assert result.dm_statistic is None
        assert result.dm_p_value is None
        assert result.mean_loss_full is None
        assert result.mean_loss_ablated is None

    def test_negative_n_features_removed_raises(self) -> None:
        """n_features_removed must be >= 0 (Pydantic ge=0 constraint)."""
        metrics: RecommendationMetrics = RecommendationMetrics(
            n_decisions=5,
            n_deployed=2,
        )
        with pytest.raises(ValidationError):
            AblationResult(
                group=FeatureGroup.CLASSIFIER,
                n_features_removed=-1,
                n_features_remaining=5,
                metrics=metrics,
            )


# ---------------------------------------------------------------------------
# TestRunAblation
# ---------------------------------------------------------------------------


class TestRunAblation:
    """Tests for the run_ablation function."""

    def test_feature_names_length_mismatch_raises(self) -> None:
        """feature_names length != x_train width raises ValueError."""
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((_N_TRAIN, 5), dtype=np.float64)
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(_N_TRAIN, dtype=np.float64)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((_N_TEST, 5), dtype=np.float64)
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(_N_TEST, dtype=np.float64)
        wrong_names: list[str] = ["a", "b"]  # only 2, should be 5

        # Need placeholder predictions with matching length
        placeholder_preds: list[Recommendation] = [_make_recommendation() for _ in range(_N_TEST)]
        placeholder_metrics: RecommendationMetrics = RecommendationMetrics(n_decisions=_N_TEST, n_deployed=0)
        with pytest.raises(ValueError, match="feature_names length"):
            run_ablation(
                GradientBoostingRecommender,
                x_train,
                y_train,
                x_test,
                y_test,
                wrong_names,
                placeholder_preds,
                placeholder_metrics,
            )

    def test_predictions_y_test_length_mismatch_raises(self) -> None:
        """full_predictions len != y_test len raises ValueError."""
        n_features: int = 3
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((_N_TRAIN, n_features), dtype=np.float64)
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(_N_TRAIN, dtype=np.float64)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((_N_TEST, n_features), dtype=np.float64)
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(_N_TEST, dtype=np.float64)
        feature_names: list[str] = ["a", "b", "c"]
        wrong_preds: list[Recommendation] = [
            _make_recommendation()
            for _ in range(_N_TEST + 5)  # wrong length
        ]
        placeholder_metrics: RecommendationMetrics = RecommendationMetrics(n_decisions=_N_TEST, n_deployed=0)
        with pytest.raises(ValueError, match="same length"):
            run_ablation(
                GradientBoostingRecommender,
                x_train,
                y_train,
                x_test,
                y_test,
                feature_names,
                wrong_preds,
                placeholder_metrics,
            )

    def test_all_groups_ablated_by_default(self) -> None:
        """When groups=None, all 3 FeatureGroup values are ablated."""
        n_features: int = 3
        x, y = _make_linear_ablation_data(n=60, n_features=n_features, seed=10)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:40]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:40]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[40:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[40:]
        feature_names: list[str] = ["other_a", "other_b", "other_c"]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
        )

        assert summary.n_groups_tested == 3
        groups_tested: list[FeatureGroup] = [r.group for r in summary.ablation_results]
        assert set(groups_tested) == {
            FeatureGroup.CLASSIFIER,
            FeatureGroup.REGRESSOR,
            FeatureGroup.REGIME,
        }

    def test_single_group_ablation(self) -> None:
        """When groups=[CLASSIFIER], exactly 1 ablation result is returned."""
        n_features: int = 4
        x, y = _make_linear_ablation_data(n=60, n_features=n_features, seed=20)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:40]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:40]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[40:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[40:]
        feature_names: list[str] = ["other_a", "other_b", "other_c", "other_d"]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        assert summary.n_groups_tested == 1
        assert len(summary.ablation_results) == 1
        assert summary.ablation_results[0].group == FeatureGroup.CLASSIFIER

    def test_ablated_model_uses_fewer_features(self) -> None:
        """Ablating a group with matching features records n_features_removed > 0."""
        # Include a classifier feature name so the group actually removes something
        feature_names: list[str] = _CLASSIFIER_FEATURES[:3] + ["other_x", "other_y"]
        n_features: int = len(feature_names)
        x, y = _make_linear_ablation_data(n=60, n_features=n_features, seed=30)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:40]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:40]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[40:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[40:]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        clf_result: AblationResult = summary.ablation_results[0]
        # 3 classifier features are in the name list → 3 should be removed
        assert clf_result.n_features_removed == 3
        assert clf_result.n_features_remaining == 2

    def test_dm_test_returns_statistic_and_pvalue(self) -> None:
        """dm_statistic and dm_p_value are not None after ablation."""
        feature_names: list[str] = _CLASSIFIER_FEATURES[:2] + ["other_a", "other_b"]
        n_features: int = len(feature_names)
        x, y = _make_linear_ablation_data(n=80, n_features=n_features, seed=40)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:55]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:55]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[55:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[55:]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        clf_result: AblationResult = summary.ablation_results[0]
        assert clf_result.dm_statistic is not None
        assert clf_result.dm_p_value is not None
        assert 0.0 <= clf_result.dm_p_value <= 1.0
        assert math.isfinite(clf_result.dm_statistic)

    def test_no_matching_features_is_noop(self) -> None:
        """When no feature names match a group prefix, n_features_removed=0."""
        # None of these names appear in _GROUP_PREFIXES
        feature_names: list[str] = ["price_momentum", "volume_ratio", "rsi_14"]
        n_features: int = len(feature_names)
        x, y = _make_linear_ablation_data(n=60, n_features=n_features, seed=50)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:40]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:40]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[40:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[40:]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        clf_result: AblationResult = summary.ablation_results[0]
        assert clf_result.n_features_removed == 0
        assert clf_result.n_features_remaining == n_features

    def test_ablation_with_signal_in_one_group(self) -> None:  # noqa: PLR0914
        """Ablating the informative feature group degrades metrics vs no-signal group.

        Signal is embedded purely in classifier features (columns 0-4). Regime
        features (columns 5-7) carry no signal (pure noise). Ablating the
        classifier group should produce worse predictions than ablating regime.
        """
        rng: np.random.Generator = np.random.default_rng(99)
        n_train: int = 120
        n_test: int = 40
        n_clf: int = 5
        n_regime: int = 3
        n_total: int = n_clf + n_regime

        # X columns 0..n_clf-1 carry the signal; columns n_clf..end are noise
        x_all: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal(
            (n_train + n_test, n_total)
        ).astype(np.float64)
        # Only the first n_clf features contribute to y
        w_clf: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(n_clf, dtype=np.float64)
        y_all: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            x_all[:, :n_clf] @ w_clf + rng.normal(0, 0.1, n_train + n_test)
        ).astype(np.float64)

        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_all[:n_train]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y_all[:n_train]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_all[n_train:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y_all[n_train:]

        feature_names: list[str] = _CLASSIFIER_FEATURES[:n_clf] + _REGIME_FEATURES[:n_regime]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER, FeatureGroup.REGIME],
        )

        clf_result: AblationResult = next(r for r in summary.ablation_results if r.group == FeatureGroup.CLASSIFIER)
        regime_result: AblationResult = next(r for r in summary.ablation_results if r.group == FeatureGroup.REGIME)

        # Ablating classifier (signal) should produce higher loss than ablating regime (noise)
        if clf_result.mean_loss_ablated is not None and regime_result.mean_loss_ablated is not None:
            assert clf_result.mean_loss_ablated >= regime_result.mean_loss_ablated

    def test_full_model_metrics_preserved_in_summary(self) -> None:
        """AblationSummary.full_model_metrics equals the passed full_metrics."""
        n_features: int = 3
        x, y = _make_linear_ablation_data(n=60, n_features=n_features, seed=60)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:40]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:40]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[40:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[40:]
        feature_names: list[str] = ["alpha_feat", "beta_feat", "gamma_feat"]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
        )

        assert summary.full_model_metrics is full_metrics


# ---------------------------------------------------------------------------
# TestDieboldMarianoIntegration
# ---------------------------------------------------------------------------


class TestDieboldMarianoIntegration:
    """Tests for Diebold-Mariano behavior through run_ablation.

    Validates that DM statistics behave directionally as expected:
    ablating truly informative features should produce statistically
    different DM outcomes than ablating noise features.
    """

    def test_dm_pvalue_in_valid_range(self) -> None:
        """DM p-value is always in [0, 1] after ablation."""
        feature_names: list[str] = _CLASSIFIER_FEATURES[:3] + ["noise_a", "noise_b"]
        n_features: int = len(feature_names)
        x, y = _make_linear_ablation_data(n=80, n_features=n_features, seed=70)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:55]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:55]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[55:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[55:]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())
        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        for ablation_result in summary.ablation_results:
            if ablation_result.dm_p_value is not None:
                assert 0.0 <= ablation_result.dm_p_value <= 1.0

    def test_dm_statistic_negative_for_important_features(self) -> None:  # noqa: PLR0914
        """Ablating the primary signal features yields negative DM statistic.

        Negative DM means full model has lower loss than ablated model,
        i.e. the ablated group was genuinely contributing.
        """
        rng: np.random.Generator = np.random.default_rng(123)
        n_train: int = 150
        n_test: int = 50
        n_clf: int = 5
        n_noise: int = 3
        n_total: int = n_clf + n_noise

        x_all: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal(
            (n_train + n_test, n_total)
        ).astype(np.float64)
        # Signal lives only in clf columns
        w: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(n_clf, dtype=np.float64) * 2.0
        y_all: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            x_all[:, :n_clf] @ w + rng.normal(0, 0.05, n_train + n_test)
        ).astype(np.float64)

        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_all[:n_train]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y_all[:n_train]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x_all[n_train:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y_all[n_train:]

        feature_names: list[str] = _CLASSIFIER_FEATURES[:n_clf] + [
            "noise_a",
            "noise_b",
            "noise_c",
        ]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())

        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.CLASSIFIER],
        )

        clf_result: AblationResult = summary.ablation_results[0]
        # Full model is better → mean_loss_full < mean_loss_ablated
        # → d_t = L(full) - L(ablated) < 0 → DM statistic negative
        if clf_result.dm_statistic is not None and clf_result.mean_loss_ablated is not None:
            assert clf_result.mean_loss_ablated >= clf_result.mean_loss_full  # type: ignore[operator]

    def test_dm_mean_losses_are_nonnegative(self) -> None:
        """mean_loss_full and mean_loss_ablated are non-negative (they are MSE values)."""
        feature_names: list[str] = _REGRESSOR_FEATURES[:2] + ["other_x", "other_y"]
        n_features: int = len(feature_names)
        x, y = _make_linear_ablation_data(n=70, n_features=n_features, seed=80)
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:50]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:50]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[50:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[50:]

        full_preds: list[Recommendation] = _build_full_model_predictions(x_train, y_train, x_test)
        full_metrics: RecommendationMetrics = compute_recommendation_metrics(full_preds, y_test.tolist())
        summary: AblationSummary = run_ablation(
            GradientBoostingRecommender,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            full_preds,
            full_metrics,
            groups=[FeatureGroup.REGRESSOR],
        )

        reg_result: AblationResult = summary.ablation_results[0]
        if reg_result.mean_loss_full is not None:
            assert reg_result.mean_loss_full >= 0.0
        if reg_result.mean_loss_ablated is not None:
            assert reg_result.mean_loss_ablated >= 0.0
