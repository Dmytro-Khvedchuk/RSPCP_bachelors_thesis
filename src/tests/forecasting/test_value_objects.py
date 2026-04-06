"""Unit tests for forecasting domain value objects and config validators.

Tests PointPrediction, QuantilePrediction, VolatilityForecast,
ConformalInterval, ReliabilityDiagramResult, ResidualDiagnostics,
RegimeCoverage containers and all model config validators.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.domain.value_objects import (
    ConformalInterval,
    GARCHConfig,
    GradientBoostingConfig,
    GRUConfig,
    HARRVConfig,
    PointPrediction,
    QuantilePrediction,
    RegimeCoverage,
    ReliabilityDiagramResult,
    ResidualDiagnostics,
    RidgeConfig,
    VolatilityForecast,
)


# ---------------------------------------------------------------------------
# PointPrediction
# ---------------------------------------------------------------------------


class TestPointPrediction:
    def test_valid_creation(self) -> None:
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        std = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        pred = PointPrediction(mean=mean, std=std)

        assert pred.mean.shape == (3,)
        assert pred.std.shape == (3,)

    def test_shape_mismatch_raises(self) -> None:
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        std = np.array([0.1, 0.2], dtype=np.float64)

        with pytest.raises(ValueError, match="mean shape.*std shape"):
            PointPrediction(mean=mean, std=std)


# ---------------------------------------------------------------------------
# QuantilePrediction
# ---------------------------------------------------------------------------


class TestQuantilePrediction:
    def test_valid_creation(self) -> None:
        quantiles = (0.05, 0.50, 0.95)
        values = np.ones((10, 3), dtype=np.float64)
        qp = QuantilePrediction(quantiles=quantiles, values=values)

        assert qp.quantiles == (0.05, 0.50, 0.95)
        assert qp.values.shape == (10, 3)

    def test_wrong_ndim_raises(self) -> None:
        quantiles = (0.50,)
        values_1d = np.ones(10, dtype=np.float64)

        with pytest.raises(ValueError, match="must be 2-D"):
            QuantilePrediction(quantiles=quantiles, values=values_1d)

    def test_column_count_mismatch_raises(self) -> None:
        quantiles = (0.05, 0.50, 0.95)
        values = np.ones((10, 2), dtype=np.float64)  # 2 columns but 3 quantiles

        with pytest.raises(ValueError, match="columns"):
            QuantilePrediction(quantiles=quantiles, values=values)

    def test_non_ascending_quantiles_raises(self) -> None:
        quantiles = (0.50, 0.25, 0.95)  # not sorted
        values = np.ones((10, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="strictly ascending"):
            QuantilePrediction(quantiles=quantiles, values=values)

    def test_duplicate_quantiles_raises(self) -> None:
        quantiles = (0.50, 0.50, 0.95)
        values = np.ones((10, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="strictly ascending"):
            QuantilePrediction(quantiles=quantiles, values=values)


# ---------------------------------------------------------------------------
# VolatilityForecast
# ---------------------------------------------------------------------------


class TestVolatilityForecast:
    def test_valid_creation(self) -> None:
        vol = np.array([0.1, 0.2], dtype=np.float64)
        var = np.array([0.01, 0.04], dtype=np.float64)
        forecast = VolatilityForecast(predicted_vol=vol, predicted_var=var)

        assert forecast.predicted_vol.shape == (2,)
        assert forecast.predicted_var.shape == (2,)

    def test_shape_mismatch_raises(self) -> None:
        vol = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        var = np.array([0.01, 0.04], dtype=np.float64)

        with pytest.raises(ValueError, match="predicted_vol shape.*predicted_var shape"):
            VolatilityForecast(predicted_vol=vol, predicted_var=var)


# ---------------------------------------------------------------------------
# ConformalInterval
# ---------------------------------------------------------------------------


class TestConformalInterval:
    def test_valid_without_coverage(self) -> None:
        lower = np.array([-1.0, -2.0], dtype=np.float64)
        upper = np.array([1.0, 2.0], dtype=np.float64)
        ci = ConformalInterval(lower=lower, upper=upper)

        assert ci.lower.shape == (2,)
        assert ci.upper.shape == (2,)
        assert ci.coverage is None

    def test_valid_with_coverage(self) -> None:
        lower = np.array([-1.0], dtype=np.float64)
        upper = np.array([1.0], dtype=np.float64)
        ci = ConformalInterval(lower=lower, upper=upper, coverage=0.92)

        assert ci.coverage == 0.92

    def test_shape_mismatch_raises(self) -> None:
        lower = np.array([-1.0, -2.0], dtype=np.float64)
        upper = np.array([1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="lower shape.*upper shape"):
            ConformalInterval(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# ReliabilityDiagramResult
# ---------------------------------------------------------------------------


class TestReliabilityDiagramResult:
    def test_valid_creation(self) -> None:
        expected = np.array([0.25, 0.50, 0.75], dtype=np.float64)
        observed = np.array([0.24, 0.49, 0.76], dtype=np.float64)
        result = ReliabilityDiagramResult(
            expected_coverage=expected,
            observed_coverage=observed,
            n_samples=100,
        )

        assert result.expected_coverage.shape == (3,)
        assert result.observed_coverage.shape == (3,)
        assert result.n_samples == 100

    def test_shape_mismatch_raises(self) -> None:
        expected = np.array([0.25, 0.50], dtype=np.float64)
        observed = np.array([0.24, 0.49, 0.76], dtype=np.float64)

        with pytest.raises(ValueError, match="expected_coverage shape.*observed_coverage shape"):
            ReliabilityDiagramResult(
                expected_coverage=expected,
                observed_coverage=observed,
                n_samples=100,
            )


# ---------------------------------------------------------------------------
# ResidualDiagnostics (frozen model — just test creation)
# ---------------------------------------------------------------------------


class TestResidualDiagnostics:
    def test_valid_creation(self) -> None:
        result = ResidualDiagnostics(
            shapiro_stat=0.98,
            shapiro_pvalue=0.15,
            breusch_pagan_stat=1.2,
            breusch_pagan_pvalue=0.27,
            mean_residual=0.001,
            std_residual=1.02,
            is_normal=True,
            is_homoscedastic=True,
        )

        assert result.shapiro_stat == 0.98
        assert result.is_normal is True
        assert result.is_homoscedastic is True

    def test_frozen_model_immutable(self) -> None:
        result = ResidualDiagnostics(
            shapiro_stat=0.98,
            shapiro_pvalue=0.15,
            breusch_pagan_stat=1.2,
            breusch_pagan_pvalue=0.27,
            mean_residual=0.001,
            std_residual=1.02,
            is_normal=True,
            is_homoscedastic=True,
        )

        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            result.shapiro_stat = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RegimeCoverage (frozen model — just test creation)
# ---------------------------------------------------------------------------


class TestRegimeCoverage:
    def test_valid_creation(self) -> None:
        result = RegimeCoverage(
            overall_coverage=0.90,
            high_vol_coverage=0.82,
            low_vol_coverage=0.96,
            high_vol_count=50,
            low_vol_count=50,
            vol_threshold=1.5,
        )

        assert result.overall_coverage == 0.90
        assert result.high_vol_coverage == 0.82
        assert result.low_vol_coverage == 0.96


# ---------------------------------------------------------------------------
# RidgeConfig
# ---------------------------------------------------------------------------


class TestRidgeConfig:
    def test_default_works(self) -> None:
        config = RidgeConfig()
        assert config.alpha == 1.0
        assert config.use_huber is False
        assert config.huber_epsilon == 1.35
        assert config.random_seed == 42

    def test_custom_alpha(self) -> None:
        config = RidgeConfig(alpha=0.5)
        assert config.alpha == 0.5

    def test_huber_epsilon_must_be_gt_one(self) -> None:
        with pytest.raises(ValueError, match="greater than 1"):
            RidgeConfig(huber_epsilon=0.5)

    def test_huber_epsilon_exactly_one_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than 1"):
            RidgeConfig(huber_epsilon=1.0)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            RidgeConfig(alpha=0.0)

        with pytest.raises(ValueError, match="greater than 0"):
            RidgeConfig(alpha=-1.0)


# ---------------------------------------------------------------------------
# GradientBoostingConfig
# ---------------------------------------------------------------------------


class TestGradientBoostingConfig:
    def test_default_works(self) -> None:
        config = GradientBoostingConfig()
        assert config.quantiles == (0.05, 0.25, 0.50, 0.75, 0.95)
        assert config.n_estimators == 500

    def test_quantile_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            GradientBoostingConfig(quantiles=(0.0, 0.50, 0.95))

        with pytest.raises(ValueError, match="must be in"):
            GradientBoostingConfig(quantiles=(0.05, 0.50, 1.0))

    def test_quantile_non_ascending_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            GradientBoostingConfig(quantiles=(0.95, 0.50, 0.05))

    def test_quantile_duplicates_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            GradientBoostingConfig(quantiles=(0.50, 0.50))


# ---------------------------------------------------------------------------
# GRUConfig
# ---------------------------------------------------------------------------


class TestGRUConfig:
    def test_default_works(self) -> None:
        config = GRUConfig()
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.dropout == 0.2
        assert config.sequence_length == 20
        assert config.mc_samples == 50

    def test_boundary_values(self) -> None:
        config = GRUConfig(
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            sequence_length=2,
            n_epochs=1,
            batch_size=1,
            mc_samples=2,
            patience=1,
        )
        assert config.hidden_size == 8
        assert config.dropout == 0.0
        assert config.sequence_length == 2
        assert config.mc_samples == 2

    def test_invalid_hidden_size_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 8"):
            GRUConfig(hidden_size=4)  # min is 8

    def test_invalid_sequence_length_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 2"):
            GRUConfig(sequence_length=1)  # min is 2


# ---------------------------------------------------------------------------
# HARRVConfig
# ---------------------------------------------------------------------------


class TestHARRVConfig:
    def test_default_works(self) -> None:
        config = HARRVConfig()
        assert config.daily_lag == 1
        assert config.weekly_lag == 5
        assert config.monthly_lag == 22
        assert config.fit_intercept is True

    def test_custom_lags(self) -> None:
        config = HARRVConfig(daily_lag=2, weekly_lag=10, monthly_lag=30)
        assert config.daily_lag == 2
        assert config.weekly_lag == 10
        assert config.monthly_lag == 30


# ---------------------------------------------------------------------------
# GARCHConfig
# ---------------------------------------------------------------------------


class TestGARCHConfig:
    def test_default_works(self) -> None:
        config = GARCHConfig()
        assert config.p == 1
        assert config.q == 1
        assert config.mean_model == "Constant"
        assert config.dist == "t"
        assert config.rescale is True

    def test_custom_distribution(self) -> None:
        config = GARCHConfig(dist="normal")
        assert config.dist == "normal"

    def test_custom_orders(self) -> None:
        config = GARCHConfig(p=2, q=2)
        assert config.p == 2
        assert config.q == 2
