"""Tests for DistributionAnalyzer against synthetic return distributions.

Verifies Jarque-Bera normality testing, Student-t MLE fitting,
AIC/BIC model comparison, KS distance, and tier-gated behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.distribution import DistributionAnalyzer
from src.app.profiling.domain.value_objects import (
    DistributionConfig,
    SampleTier,
)
from src.tests.profiling.conftest import (
    make_normal_returns,
    make_skewed_returns,
    make_student_t_returns,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_config() -> DistributionConfig:
    """Return a default DistributionConfig for tests."""
    return DistributionConfig()


def _make_analyzer() -> DistributionAnalyzer:
    """Return a fresh DistributionAnalyzer instance."""
    return DistributionAnalyzer()


# ---------------------------------------------------------------------------
# Tier A — Normal data
# ---------------------------------------------------------------------------


class TestAnalyzeNormalDataTierA:
    """Tests for Tier A analysis on normally distributed data."""

    def test_jb_does_not_reject_normality(self) -> None:
        """N(0, 0.01) data should not be rejected as non-normal by JB test."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.jb_pvalue > 0.05
        assert profile.is_normal is True

    def test_student_t_nu_is_high_for_normal(self) -> None:
        """Normal data should yield high Student-t nu (approaching Gaussian)."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.student_t_nu is not None
        assert profile.student_t_nu > 20

    def test_aic_bic_fields_populated(self) -> None:
        """Tier A analysis should populate all AIC/BIC fields."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.aic_normal is not None
        assert profile.aic_student_t is not None
        assert profile.bic_normal is not None
        assert profile.bic_student_t is not None
        assert profile.best_fit is not None

    def test_ks_statistic_is_small_for_normal(self) -> None:
        """KS D_n against fitted Normal should be small for Normal data."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.ks_statistic is not None
        assert profile.ks_statistic < 0.05


# ---------------------------------------------------------------------------
# Tier A — Fat-tailed data
# ---------------------------------------------------------------------------


class TestAnalyzeFatTailedDataTierA:
    """Tests for Tier A analysis on Student-t(5) distributed data."""

    def test_jb_rejects_normality(self) -> None:
        """Student-t(5) data should be rejected as non-normal by JB test."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.jb_pvalue < 0.05
        assert profile.is_normal is False

    def test_fitted_nu_near_true_value(self) -> None:
        """Fitted Student-t nu should be near the true value of 5."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.student_t_nu is not None
        assert 3.0 <= profile.student_t_nu <= 8.0

    def test_aic_prefers_student_t(self) -> None:
        """AIC should prefer Student-t over Normal for fat-tailed data."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.aic_student_t is not None
        assert profile.aic_normal is not None
        assert profile.aic_student_t < profile.aic_normal
        assert profile.best_fit == "student_t"


# ---------------------------------------------------------------------------
# Tier C — descriptive stats only
# ---------------------------------------------------------------------------


class TestAnalyzeTierC:
    """Tests for Tier C analysis (descriptive stats only)."""

    def test_descriptive_stats_populated(self) -> None:
        """Tier C should still produce descriptive statistics."""
        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, _default_config())

        assert profile.n_observations == 100
        # mean_return may be near zero for normal data -- just check it's finite
        assert math.isfinite(profile.mean_return)
        assert profile.std_return > 0.0

    def test_student_t_fields_are_none(self) -> None:
        """Tier C should not populate Student-t fitting fields."""
        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, _default_config())

        assert profile.student_t_nu is None
        assert profile.student_t_loc is None
        assert profile.student_t_scale is None

    def test_aic_bic_are_none(self) -> None:
        """Tier C should not populate AIC/BIC fields."""
        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, _default_config())

        assert profile.aic_normal is None
        assert profile.aic_student_t is None
        assert profile.bic_normal is None
        assert profile.bic_student_t is None
        assert profile.best_fit is None

    def test_ks_statistic_is_none(self) -> None:
        """Tier C should not populate KS statistic."""
        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, _default_config())

        assert profile.ks_statistic is None


# ---------------------------------------------------------------------------
# Tier B — same depth as Tier A
# ---------------------------------------------------------------------------


class TestAnalyzeTierB:
    """Tests for Tier B analysis (same fitting depth as Tier A)."""

    def test_student_t_fields_populated(self) -> None:
        """Tier B should fit Student-t distribution like Tier A."""
        returns = make_normal_returns(n=1000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B, _default_config())

        assert profile.student_t_nu is not None
        assert profile.student_t_loc is not None
        assert profile.student_t_scale is not None
        assert profile.aic_normal is not None
        assert profile.aic_student_t is not None
        assert profile.ks_statistic is not None


# ---------------------------------------------------------------------------
# Skewness tests
# ---------------------------------------------------------------------------


class TestSkewness:
    """Tests for skewness computation on known distributions."""

    def test_symmetric_data_near_zero_skewness(self) -> None:
        """Symmetric (Normal) data should have skewness near zero."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert abs(profile.skewness) < 0.1

    def test_left_skewed_data_negative_skewness(self) -> None:
        """Left-skewed data should have negative skewness."""
        returns = make_skewed_returns(n=5000, skew_direction="left", seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.skewness < 0.0


# ---------------------------------------------------------------------------
# Excess kurtosis tests
# ---------------------------------------------------------------------------


class TestExcessKurtosis:
    """Tests for excess kurtosis computation."""

    def test_normal_data_kurtosis_near_zero(self) -> None:
        """Normal data should have excess kurtosis near 0."""
        returns = make_normal_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert abs(profile.excess_kurtosis) < 0.3

    def test_fat_tailed_data_high_kurtosis(self) -> None:
        """Student-t(3) data should have very high excess kurtosis."""
        returns = make_student_t_returns(n=5000, nu=3.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        # Student-t(3) theoretical excess kurtosis = infinity,
        # but finite sample estimate should be large (> 3)
        assert profile.excess_kurtosis > 3.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_constant_returns_graceful(self) -> None:
        """Constant returns (std=0) should be handled gracefully."""
        returns = pd.Series(np.zeros(100), dtype=np.float64, name="log_return")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.std_return == 0.0
        assert profile.jb_stat == 0.0
        assert profile.jb_pvalue == 1.0
        assert profile.is_normal is True
        # Student-t fitting is skipped when std=0
        assert profile.student_t_nu is None

    def test_too_few_samples_for_jb(self) -> None:
        """Fewer than min_samples_jb should skip JB test gracefully."""
        returns = pd.Series([0.01, -0.02], dtype=np.float64, name="log_return")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.jb_stat == 0.0
        assert profile.jb_pvalue == 1.0

    def test_too_few_samples_for_fit(self) -> None:
        """Fewer than min_samples_fit should skip Student-t fitting."""
        returns = make_normal_returns(n=20, seed=42)
        config = DistributionConfig(min_samples_fit=30)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, config)

        assert profile.student_t_nu is None
        assert profile.aic_normal is None


# ---------------------------------------------------------------------------
# QQ data against Student-t
# ---------------------------------------------------------------------------


class TestQQDataStudentT:
    """Tests for compute_qq_data_student_t."""

    def test_returns_two_equal_length_arrays(self) -> None:
        """QQ data should return two arrays of equal length."""
        returns = make_student_t_returns(n=500, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.student_t_nu is not None
        theoretical, ordered = _make_analyzer().compute_qq_data_student_t(
            returns, profile.student_t_nu, profile.student_t_loc or 0.0, profile.student_t_scale or 1.0
        )

        assert len(theoretical) == len(ordered)
        assert len(theoretical) == len(returns)

    def test_good_fit_points_near_diagonal(self) -> None:
        """For Student-t data with correct params, QQ points should be near the diagonal."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.student_t_nu is not None
        theoretical, ordered = _make_analyzer().compute_qq_data_student_t(
            returns, profile.student_t_nu, profile.student_t_loc or 0.0, profile.student_t_scale or 1.0
        )

        # Compute correlation between theoretical and ordered quantiles
        # A good fit should give R > 0.99
        correlation = np.corrcoef(theoretical, ordered)[0, 1]
        assert correlation > 0.99

    def test_too_few_samples_returns_empty(self) -> None:
        """Fewer than 3 samples should return empty arrays."""
        returns = pd.Series([0.01, -0.01], dtype=np.float64, name="log_return")
        theoretical, ordered = _make_analyzer().compute_qq_data_student_t(returns, 5.0, 0.0, 0.01)

        assert len(theoretical) == 0
        assert len(ordered) == 0


# ---------------------------------------------------------------------------
# KS statistic range
# ---------------------------------------------------------------------------


class TestKSStatistic:
    """Tests for KS D_n statistic validity."""

    def test_ks_in_zero_one_range(self) -> None:
        """KS D_n statistic should always be in [0, 1]."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.ks_statistic is not None
        assert 0.0 <= profile.ks_statistic <= 1.0


# ---------------------------------------------------------------------------
# AIC / BIC consistency
# ---------------------------------------------------------------------------


class TestAICBICConsistency:
    """Tests for AIC / BIC model selection consistency."""

    def test_both_prefer_student_t_for_fat_tails(self) -> None:
        """Both AIC and BIC should prefer Student-t for fat-tailed data."""
        returns = make_student_t_returns(n=5000, nu=5.0, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, _default_config())

        assert profile.aic_student_t is not None
        assert profile.aic_normal is not None
        assert profile.bic_student_t is not None
        assert profile.bic_normal is not None

        assert profile.aic_student_t < profile.aic_normal
        assert profile.bic_student_t < profile.bic_normal


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    """Tests for reproducibility of analysis results."""

    def test_same_input_same_output(self) -> None:
        """Identical input should produce identical output."""
        returns = make_student_t_returns(n=1000, nu=5.0, seed=123)
        config = _default_config()
        analyzer = _make_analyzer()

        profile_1 = analyzer.analyze(returns, "BTCUSDT", "dollar", SampleTier.A, config)
        profile_2 = analyzer.analyze(returns, "BTCUSDT", "dollar", SampleTier.A, config)

        assert profile_1 == profile_2


# ---------------------------------------------------------------------------
# Value object construction
# ---------------------------------------------------------------------------


class TestDistributionProfile:
    """Tests for DistributionProfile value object construction."""

    def test_frozen_immutability(self) -> None:
        """DistributionProfile should be immutable."""
        from pydantic import ValidationError

        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, _default_config())

        with pytest.raises(ValidationError):
            profile.mean_return = 999.0  # type: ignore[misc]

    def test_metadata_fields(self) -> None:
        """Profile should carry correct asset, bar_type, and tier metadata."""
        returns = make_normal_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "ETHUSDT", "volume", SampleTier.C, _default_config())

        assert profile.asset == "ETHUSDT"
        assert profile.bar_type == "volume"
        assert profile.tier == SampleTier.C
        assert profile.n_observations == 100


class TestDistributionConfig:
    """Tests for DistributionConfig value object construction."""

    def test_default_values(self) -> None:
        """Default config should have sensible defaults."""
        config = DistributionConfig()

        assert config.jb_alpha == 0.05
        assert config.price_col == "close"
        assert config.min_samples_jb == 3
        assert config.min_samples_fit == 30

    def test_frozen_immutability(self) -> None:
        """DistributionConfig should be immutable."""
        from pydantic import ValidationError

        config = DistributionConfig()

        with pytest.raises(ValidationError):
            config.jb_alpha = 0.1  # type: ignore[misc]
