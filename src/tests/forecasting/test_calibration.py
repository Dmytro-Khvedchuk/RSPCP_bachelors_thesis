"""Unit tests for calibration and conformal prediction.

Tests the Adaptive Conformal Inference wrapper, reliability diagrams,
residual diagnostics, and per-regime coverage analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.calibration import (
    AdaptiveConformalPredictor,
    compute_regime_coverage,
    compute_reliability_diagram,
    compute_residual_diagnostics,
)
from src.app.forecasting.domain.value_objects import (
    ACIConfig,
    ConformalInterval,
    QuantilePrediction,
    RegimeCoverage,
    ReliabilityDiagramResult,
    ResidualDiagnostics,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_aci_config(**overrides: object) -> ACIConfig:
    defaults: dict[str, object] = {
        "target_coverage": 0.90,
        "gamma": 0.005,
        "initial_alpha": None,
        "min_alpha": 0.01,
        "max_alpha": 0.50,
    }
    defaults.update(overrides)
    return ACIConfig(**defaults)  # type: ignore[arg-type]


def make_well_behaved_data(
    n: int = 500,
    seed: int = 42,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate predictions, actuals, and residuals from a Gaussian model.

    Returns (predictions, actuals, residuals) where residuals ~ N(0, 1).
    """
    rng = np.random.default_rng(seed)
    predictions = rng.standard_normal(n).astype(np.float64)
    noise = rng.standard_normal(n).astype(np.float64)
    actuals = (predictions + noise).astype(np.float64)
    residuals = (actuals - predictions).astype(np.float64)
    return predictions, actuals, residuals


@pytest.fixture
def aci_config() -> ACIConfig:
    return make_aci_config()


@pytest.fixture
def well_behaved_data() -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    return make_well_behaved_data()


# ---------------------------------------------------------------------------
# ACIConfig validation tests
# ---------------------------------------------------------------------------


class TestACIConfig:
    def test_default_config_valid(self) -> None:
        config = ACIConfig()
        assert config.target_coverage == 0.90
        assert config.gamma == 0.005
        assert config.initial_alpha is None
        assert config.min_alpha == 0.01
        assert config.max_alpha == 0.50

    def test_custom_initial_alpha(self) -> None:
        config = ACIConfig(initial_alpha=0.15)
        assert config.initial_alpha == 0.15

    def test_invalid_min_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_alpha"):
            ACIConfig(min_alpha=0.6, max_alpha=0.3)

    def test_initial_alpha_outside_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_alpha"):
            ACIConfig(initial_alpha=0.005, min_alpha=0.01, max_alpha=0.50)


# ---------------------------------------------------------------------------
# AdaptiveConformalPredictor tests
# ---------------------------------------------------------------------------


class TestAdaptiveConformalPredictor:
    def test_calibrate_stores_sorted_scores(self, aci_config: ACIConfig) -> None:
        predictor = AdaptiveConformalPredictor(aci_config)
        residuals = np.array([3.0, -1.0, 2.0, -4.0, 0.5], dtype=np.float64)
        predictor.calibrate(residuals)

        # Scores should be sorted absolute residuals
        assert predictor._scores is not None
        expected_scores = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(predictor._scores, expected_scores)

    def test_calibrate_empty_raises(self, aci_config: ACIConfig) -> None:
        predictor = AdaptiveConformalPredictor(aci_config)
        with pytest.raises(ValueError, match="at least one sample"):
            predictor.calibrate(np.array([], dtype=np.float64))

    def test_predict_before_calibrate_raises(self, aci_config: ACIConfig) -> None:
        predictor = AdaptiveConformalPredictor(aci_config)
        preds = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict_interval(preds)

    def test_predict_empty_raises(self, aci_config: ACIConfig) -> None:
        predictor = AdaptiveConformalPredictor(aci_config)
        predictor.calibrate(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        with pytest.raises(ValueError, match="at least one sample"):
            predictor.predict_interval(np.array([], dtype=np.float64))

    def test_predict_actuals_shape_mismatch_raises(self, aci_config: ACIConfig) -> None:
        predictor = AdaptiveConformalPredictor(aci_config)
        predictor.calibrate(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        preds = np.array([1.0, 2.0], dtype=np.float64)
        actuals = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="actuals length"):
            predictor.predict_interval(preds, actuals)

    def test_batch_mode_no_actuals(self, aci_config: ACIConfig) -> None:
        """Without actuals, intervals should be produced but alpha stays fixed."""
        predictor = AdaptiveConformalPredictor(aci_config)
        residuals = np.linspace(-2, 2, 100).astype(np.float64)
        predictor.calibrate(residuals)

        alpha_before = predictor.alpha_t
        preds = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        result = predictor.predict_interval(preds)

        assert isinstance(result, ConformalInterval)
        assert result.lower.shape == (3,)
        assert result.upper.shape == (3,)
        assert result.coverage is None
        # Alpha should not change in batch mode
        assert predictor.alpha_t == alpha_before
        # Only the initial value should be in history
        assert len(predictor.alpha_history) == 1

    def test_online_mode_adapts_alpha(self) -> None:
        """With actuals provided, alpha_t should adapt over time."""
        config = make_aci_config(gamma=0.05)
        predictor = AdaptiveConformalPredictor(config)
        residuals = np.linspace(-2, 2, 100).astype(np.float64)
        predictor.calibrate(residuals)

        rng = np.random.default_rng(99)
        n_test = 50
        preds = np.zeros(n_test, dtype=np.float64)
        actuals = rng.standard_normal(n_test).astype(np.float64)

        alpha_before = predictor.alpha_t
        result = predictor.predict_interval(preds, actuals)

        assert result.coverage is not None
        # Alpha history should have grown: initial + n_test updates
        assert len(predictor.alpha_history) == 1 + n_test
        # At least one alpha value should differ from the initial
        alphas = predictor.alpha_history
        assert not all(a == alpha_before for a in alphas)

    def test_conformal_coverage_batch_mode(self) -> None:
        """Batch-mode (fixed alpha) conformal intervals achieve ~90% coverage on Gaussian data."""
        config = make_aci_config(target_coverage=0.90, gamma=0.005)
        predictor = AdaptiveConformalPredictor(config)

        rng = np.random.default_rng(123)
        n_cal = 500
        n_test = 1000

        # Calibration: residuals ~ N(0, 1)
        cal_residuals = rng.standard_normal(n_cal).astype(np.float64)
        predictor.calibrate(cal_residuals)

        # Test: predictions + N(0,1) noise — batch mode (no actuals)
        test_preds = rng.standard_normal(n_test).astype(np.float64)
        test_actuals = (test_preds + rng.standard_normal(n_test)).astype(np.float64)

        result = predictor.predict_interval(test_preds)

        # Manually compute coverage
        inside = (test_actuals >= result.lower) & (test_actuals <= result.upper)
        coverage = float(np.mean(inside))
        # Coverage should be approximately 90% (allow +-5% margin for finite sample)
        assert 0.85 < coverage < 0.95, f"Coverage {coverage:.4f} outside expected range"

    def test_alpha_clamping(self) -> None:
        """Alpha should stay within [min_alpha, max_alpha] bounds."""
        config = make_aci_config(gamma=0.5, min_alpha=0.05, max_alpha=0.40)
        predictor = AdaptiveConformalPredictor(config)
        residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        predictor.calibrate(residuals)

        # Push alpha down: all actuals inside → err_t = 0 repeatedly
        preds = np.zeros(20, dtype=np.float64)
        actuals = np.zeros(20, dtype=np.float64)  # always inside interval
        predictor.predict_interval(preds, actuals)

        # Alpha must not drop below min_alpha
        for alpha in predictor.alpha_history:
            assert alpha >= config.min_alpha
            assert alpha <= config.max_alpha

    def test_online_mode_achieves_coverage(self) -> None:
        """With online ACI, coverage should be at least the target level.

        ACI's update rule pushes alpha down when there are no misses,
        which widens intervals and causes over-coverage (conservative).
        This is the expected behaviour: ACI guarantees asymptotic
        marginal coverage >= target_coverage.  We verify that coverage
        stays above 85% and that the adaptation mechanism is active.
        """
        config = make_aci_config(target_coverage=0.90, gamma=0.005)
        predictor = AdaptiveConformalPredictor(config)

        rng = np.random.default_rng(42)
        n_cal = 1000
        n_test = 2000

        # Calibrate on N(0,1) residuals
        cal_residuals = rng.standard_normal(n_cal).astype(np.float64)
        predictor.calibrate(cal_residuals)

        # Test: exchangeable IID data (same distribution as calibration)
        test_preds = np.zeros(n_test, dtype=np.float64)
        test_actuals = rng.standard_normal(n_test).astype(np.float64)

        result = predictor.predict_interval(test_preds, test_actuals)

        assert result.coverage is not None
        # ACI should achieve at least the target coverage (may over-cover due to
        # conservative alpha drift)
        assert result.coverage >= 0.85, f"Online ACI coverage {result.coverage:.4f} below 85%"
        # Alpha should have adapted (history grew)
        assert len(predictor.alpha_history) == 1 + n_test

    def test_intervals_symmetric_around_predictions(self, aci_config: ACIConfig) -> None:
        """Intervals should be symmetric: lower = pred - q, upper = pred + q."""
        predictor = AdaptiveConformalPredictor(aci_config)
        residuals = np.linspace(-3, 3, 200).astype(np.float64)
        predictor.calibrate(residuals)

        preds = np.array([5.0, -3.0, 0.0], dtype=np.float64)
        result = predictor.predict_interval(preds)

        widths_lower = preds - result.lower
        widths_upper = result.upper - preds
        np.testing.assert_array_almost_equal(widths_lower, widths_upper)


# ---------------------------------------------------------------------------
# Reliability diagram tests
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    def test_perfect_calibration_on_gaussian(self) -> None:
        """Quantile predictions from the true distribution yield diagonal reliability."""
        rng = np.random.default_rng(42)
        n = 5000
        quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)

        # True model: y ~ N(mu, 1)
        mu = rng.standard_normal(n).astype(np.float64)
        actuals = (mu + rng.standard_normal(n)).astype(np.float64)

        # Oracle quantile predictions from the true N(mu, 1) distribution
        from scipy.stats import norm

        values = np.column_stack([norm.ppf(q, loc=mu, scale=1.0) for q in quantiles]).astype(np.float64)

        qp = QuantilePrediction(quantiles=quantiles, values=values)
        result = compute_reliability_diagram(qp, actuals)

        assert isinstance(result, ReliabilityDiagramResult)
        assert result.n_samples == n
        # Observed coverage should be close to expected (within +-3%)
        np.testing.assert_allclose(result.observed_coverage, result.expected_coverage, atol=0.03)

    def test_actuals_shape_mismatch_raises(self) -> None:
        quantiles = (0.25, 0.50, 0.75)
        values = np.ones((10, 3), dtype=np.float64)
        qp = QuantilePrediction(quantiles=quantiles, values=values)
        with pytest.raises(ValueError, match="actuals length"):
            compute_reliability_diagram(qp, np.ones(5, dtype=np.float64))

    def test_single_quantile(self) -> None:
        """Reliability diagram works with a single quantile level."""
        rng = np.random.default_rng(7)
        n = 200
        values = rng.standard_normal((n, 1)).astype(np.float64)
        actuals = rng.standard_normal(n).astype(np.float64)
        qp = QuantilePrediction(quantiles=(0.50,), values=values)
        result = compute_reliability_diagram(qp, actuals)
        assert result.expected_coverage.shape == (1,)
        assert result.observed_coverage.shape == (1,)


# ---------------------------------------------------------------------------
# Residual diagnostics tests
# ---------------------------------------------------------------------------


class TestResidualDiagnostics:
    def test_gaussian_residuals_are_normal(self) -> None:
        """Shapiro-Wilk should not reject normality for Gaussian residuals."""
        rng = np.random.default_rng(42)
        n = 500
        predictions = rng.standard_normal(n).astype(np.float64)
        residuals = rng.standard_normal(n).astype(np.float64)

        result = compute_residual_diagnostics(residuals, predictions)

        assert isinstance(result, ResidualDiagnostics)
        assert result.is_normal is True
        assert result.shapiro_pvalue > 0.05
        assert abs(result.mean_residual) < 0.2
        assert result.std_residual > 0.0

    def test_heavy_tailed_residuals_not_normal(self) -> None:
        """Student-t(3) residuals should fail normality test."""
        rng = np.random.default_rng(42)
        n = 500
        predictions = np.zeros(n, dtype=np.float64)
        residuals = (rng.standard_t(df=3, size=n)).astype(np.float64)

        result = compute_residual_diagnostics(residuals, predictions)
        assert result.is_normal is False

    def test_homoscedastic_residuals(self) -> None:
        """Constant-variance residuals should pass Breusch-Pagan."""
        rng = np.random.default_rng(42)
        n = 500
        predictions = rng.standard_normal(n).astype(np.float64)
        residuals = rng.standard_normal(n).astype(np.float64)  # constant variance

        result = compute_residual_diagnostics(residuals, predictions)
        assert result.is_homoscedastic is True

    def test_heteroscedastic_residuals(self) -> None:
        """Residuals with variance proportional to |prediction| should fail BP."""
        rng = np.random.default_rng(42)
        n = 500
        predictions = np.linspace(0.1, 10.0, n).astype(np.float64)
        # Variance grows with prediction magnitude
        residuals = (rng.standard_normal(n) * predictions).astype(np.float64)

        result = compute_residual_diagnostics(residuals, predictions)
        assert result.is_homoscedastic is False

    def test_empty_residuals_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            compute_residual_diagnostics(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="predictions length"):
            compute_residual_diagnostics(
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
            )

    def test_two_samples_graceful(self) -> None:
        """With n < 3, diagnostics should return NaN stats gracefully."""
        residuals = np.array([0.5, -0.5], dtype=np.float64)
        predictions = np.array([1.0, 2.0], dtype=np.float64)
        result = compute_residual_diagnostics(residuals, predictions)
        assert np.isnan(result.shapiro_stat)
        assert np.isnan(result.breusch_pagan_stat)
        assert result.is_normal is False
        assert result.is_homoscedastic is False

    def test_large_sample_subsamples_shapiro(self) -> None:
        """n > 5000 should still produce valid Shapiro results (via subsample)."""
        rng = np.random.default_rng(42)
        n = 6000
        predictions = rng.standard_normal(n).astype(np.float64)
        residuals = rng.standard_normal(n).astype(np.float64)

        result = compute_residual_diagnostics(residuals, predictions)
        assert not np.isnan(result.shapiro_stat)
        assert 0.0 <= result.shapiro_pvalue <= 1.0


# ---------------------------------------------------------------------------
# Regime coverage tests
# ---------------------------------------------------------------------------


class TestRegimeCoverage:
    def test_perfect_coverage_everywhere(self) -> None:
        """If every actual is inside the interval, coverage should be 1.0 everywhere."""
        n = 100
        actuals = np.zeros(n, dtype=np.float64)
        lower = np.full(n, -1.0, dtype=np.float64)
        upper = np.full(n, 1.0, dtype=np.float64)
        vol = np.linspace(0.1, 2.0, n).astype(np.float64)

        result = compute_regime_coverage(lower, upper, actuals, vol)

        assert isinstance(result, RegimeCoverage)
        assert result.overall_coverage == 1.0
        assert result.high_vol_coverage == 1.0
        assert result.low_vol_coverage == 1.0
        assert result.high_vol_count + result.low_vol_count == n

    def test_zero_coverage_everywhere(self) -> None:
        """If every actual is outside the interval, coverage should be 0.0."""
        n = 100
        actuals = np.full(n, 10.0, dtype=np.float64)
        lower = np.full(n, -1.0, dtype=np.float64)
        upper = np.full(n, 1.0, dtype=np.float64)
        vol = np.linspace(0.1, 2.0, n).astype(np.float64)

        result = compute_regime_coverage(lower, upper, actuals, vol)
        assert result.overall_coverage == 0.0

    def test_coverage_differs_by_regime(self) -> None:
        """Coverage degradation during high-vol periods should be detectable."""
        rng = np.random.default_rng(42)
        n = 1000

        vol = np.concatenate(
            [
                np.full(500, 0.5, dtype=np.float64),  # low vol
                np.full(500, 2.0, dtype=np.float64),  # high vol
            ]
        )
        # Low vol: noise ~ N(0, 0.5) → mostly inside +-2
        # High vol: noise ~ N(0, 3.0) → often outside +-2
        noise_low = rng.normal(0, 0.5, size=500).astype(np.float64)
        noise_high = rng.normal(0, 3.0, size=500).astype(np.float64)
        actuals = np.concatenate([noise_low, noise_high]).astype(np.float64)
        lower = np.full(n, -2.0, dtype=np.float64)
        upper = np.full(n, 2.0, dtype=np.float64)

        result = compute_regime_coverage(lower, upper, actuals, vol)

        # Low-vol coverage should be higher than high-vol coverage
        assert result.low_vol_coverage > result.high_vol_coverage
        assert result.vol_threshold > 0

    def test_empty_arrays_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_regime_coverage(empty, empty, empty, empty)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_regime_coverage(
                np.array([1.0], dtype=np.float64),
                np.array([2.0], dtype=np.float64),
                np.array([1.5, 1.6], dtype=np.float64),
                np.array([0.1], dtype=np.float64),
            )

    def test_single_sample(self) -> None:
        """Gracefully handle a single sample."""
        result = compute_regime_coverage(
            lower=np.array([-1.0], dtype=np.float64),
            upper=np.array([1.0], dtype=np.float64),
            actuals=np.array([0.0], dtype=np.float64),
            volatility=np.array([0.5], dtype=np.float64),
        )
        assert result.overall_coverage == 1.0
        # With one sample at the median, it goes to low_vol (not >)
        assert result.low_vol_count == 1
        assert result.high_vol_count == 0


# ---------------------------------------------------------------------------
# Integration: ACI + regime coverage on well-behaved data
# ---------------------------------------------------------------------------


class TestACIIntegration:
    def test_aci_then_regime_coverage(self) -> None:
        """End-to-end: calibrate ACI, predict intervals, check regime coverage."""
        rng = np.random.default_rng(77)
        n_cal = 300
        n_test = 500

        # Calibration residuals ~ N(0, 1)
        cal_residuals = rng.standard_normal(n_cal).astype(np.float64)

        config = make_aci_config(target_coverage=0.90, gamma=0.005)
        predictor = AdaptiveConformalPredictor(config)
        predictor.calibrate(cal_residuals)

        # Test data with two regimes
        test_preds = np.zeros(n_test, dtype=np.float64)
        vol = np.concatenate(
            [
                np.full(250, 0.3, dtype=np.float64),  # low vol
                np.full(250, 1.5, dtype=np.float64),  # high vol
            ]
        )
        noise_low = rng.normal(0, 0.3, size=250).astype(np.float64)
        noise_high = rng.normal(0, 1.5, size=250).astype(np.float64)
        test_actuals = np.concatenate([noise_low, noise_high]).astype(np.float64)

        interval = predictor.predict_interval(test_preds, test_actuals)

        regime = compute_regime_coverage(interval.lower, interval.upper, test_actuals, vol)

        # Low-vol regime should have higher coverage than high-vol
        assert regime.low_vol_coverage > regime.high_vol_coverage
        assert regime.high_vol_count > 0
        assert regime.low_vol_count > 0
