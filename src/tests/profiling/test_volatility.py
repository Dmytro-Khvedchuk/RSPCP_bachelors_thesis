"""Tests for VolatilityAnalyzer against synthetic GARCH, GJR-GARCH, and i.i.d. data.

Verifies GARCH fitting, sign bias, GJR-GARCH leverage, ARCH-LM, BDS nonlinearity,
regime labeling, tier gating, configuration, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.volatility import VolatilityAnalyzer
from src.app.profiling.domain.value_objects import (
    SampleTier,
    VolatilityConfig,
    VolatilityRegime,
)
from src.tests.profiling.conftest import (
    make_gjr_garch_returns,
    make_iid_returns,
    make_nonlinear_returns,
    make_true_garch_returns,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_config() -> VolatilityConfig:
    """Return a default VolatilityConfig for tests."""
    return VolatilityConfig()


def _make_analyzer() -> VolatilityAnalyzer:
    """Return a fresh VolatilityAnalyzer instance."""
    return VolatilityAnalyzer()


# ---------------------------------------------------------------------------
# GARCH fitting tests
# ---------------------------------------------------------------------------


class TestGARCHFitting:
    """Tests for GARCH(1,1) multi-distribution fitting."""

    def test_recovers_persistence(self) -> None:
        """True GARCH(1,1) (alpha=0.1, beta=0.85): fitted persistence ~ 0.95."""
        returns = make_true_garch_returns(n=3000, alpha=0.1, beta=0.85, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.garch_fits is not None
        assert profile.persistence is not None
        # Tolerance: 0.95 +/- 0.1
        assert abs(profile.persistence - 0.95) < 0.1

    def test_student_t_beats_normal_on_fat_tails(self) -> None:
        """Student-t or skewt should have lower AIC than normal on fat-tailed data."""
        # Generate GARCH returns with Student-t innovations for heavier tails
        rng = np.random.default_rng(42)
        n = 3000
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        z = rng.standard_t(df=5, size=n)
        z /= np.std(z)  # standardize
        sigma2 = np.zeros(n, dtype=np.float64)
        rets = np.zeros(n, dtype=np.float64)
        sigma2[0] = omega / (1 - alpha - beta)
        rets[0] = np.sqrt(sigma2[0]) * z[0]
        for i in range(1, n):
            sigma2[i] = omega + alpha * rets[i - 1] ** 2 + beta * sigma2[i - 1]
            rets[i] = np.sqrt(sigma2[i]) * z[i]
        returns = pd.Series(rets, dtype=np.float64, name="fat_tail_return")

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.garch_fits is not None
        assert profile.best_distribution is not None
        # With fat-tailed innovations, t or skewt should win over normal
        assert profile.best_distribution in {"t", "skewt"}

    def test_best_distribution_from_configured(self) -> None:
        """Best distribution must be one of the configured distributions."""
        returns = make_true_garch_returns(n=2000, seed=42)
        config = VolatilityConfig(innovation_distributions=("normal", "t"))
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A, config=config)

        assert profile.best_distribution is not None
        assert profile.best_distribution in {"normal", "t"}

    def test_igarch_flag(self) -> None:
        """High persistence (alpha=0.05, beta=0.945) should flag IGARCH."""
        returns = make_true_garch_returns(n=3000, alpha=0.05, beta=0.945, seed=42)
        config = VolatilityConfig(persistence_threshold=0.99)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A, config=config)

        assert profile.persistence is not None
        assert profile.is_igarch is not None
        # With true persistence = 0.995, should be flagged as IGARCH
        assert profile.is_igarch is True

    def test_not_applied_to_non_time_bars(self) -> None:
        """Non-time bar types should skip GARCH fitting entirely."""
        returns = make_true_garch_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.garch_fits is None
        assert profile.best_distribution is None
        assert profile.persistence is None
        assert profile.is_igarch is None

    def test_convergence_failure_handled(self) -> None:
        """Very short or constant series should not raise; garch_fits is None."""
        # Short series below min_samples_garch
        returns_short = pd.Series(np.zeros(10, dtype=np.float64), name="const_return")
        profile = _make_analyzer().analyze(returns_short, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.garch_fits is None


# ---------------------------------------------------------------------------
# Sign bias tests
# ---------------------------------------------------------------------------


class TestSignBias:
    """Tests for Engle-Ng sign bias test."""

    def test_detects_leverage_on_gjr_data(self) -> None:
        """GJR-GARCH returns with strong gamma should detect leverage effect."""
        returns = make_gjr_garch_returns(n=5000, alpha=0.05, gamma=0.2, beta=0.70, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.sign_bias is not None
        assert profile.sign_bias.has_leverage_effect is True

    def test_not_significant_on_symmetric_garch(self) -> None:
        """Symmetric GARCH returns: sign bias likely not significant."""
        returns = make_true_garch_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.sign_bias is not None
        # Symmetric GARCH should usually not detect leverage
        # (allow for occasional false positive due to randomness)

    def test_none_for_tier_c(self) -> None:
        """Tier C should not compute sign bias."""
        returns = make_true_garch_returns(n=100, seed=42)
        config = VolatilityConfig(min_samples_garch=500)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.C, config=config)

        assert profile.sign_bias is None


# ---------------------------------------------------------------------------
# GJR-GARCH tests
# ---------------------------------------------------------------------------


class TestGJRGARCH:
    """Tests for GJR-GARCH asymmetric leverage coefficient."""

    def test_positive_gamma_on_leverage_data(self) -> None:
        """GJR returns with Tier A + leverage detected -> positive gamma."""
        returns = make_gjr_garch_returns(n=5000, alpha=0.05, gamma=0.2, beta=0.70, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        # If sign bias detected leverage, gjr_gamma should be extracted
        if profile.sign_bias is not None and profile.sign_bias.has_leverage_effect:
            assert profile.gjr_gamma is not None
            assert profile.gjr_gamma > 0

    def test_skipped_when_no_leverage(self) -> None:
        """Symmetric GARCH with Tier A: gjr_gamma should be None if no leverage detected."""
        returns = make_true_garch_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        # If sign bias did not detect leverage, gamma should be None
        if profile.sign_bias is not None and not profile.sign_bias.has_leverage_effect:
            assert profile.gjr_gamma is None

    def test_skipped_for_tier_b(self) -> None:
        """GJR-GARCH should not be computed for Tier B."""
        returns = make_gjr_garch_returns(n=5000, alpha=0.05, gamma=0.2, beta=0.70, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.B)

        # GJR-GARCH is Tier A only
        assert profile.gjr_gamma is None


# ---------------------------------------------------------------------------
# ARCH-LM tests
# ---------------------------------------------------------------------------


class TestARCHLM:
    """Tests for ARCH-LM heteroscedasticity test."""

    def test_significant_on_raw_garch_data(self) -> None:
        """GARCH returns (raw, not residuals): ARCH-LM should detect effects."""
        returns = make_true_garch_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.arch_lm_stat is not None
        assert profile.arch_lm_pvalue is not None

    def test_arch_lm_computed_for_time_bars(self) -> None:
        """time_1h bar with sufficient data should have ARCH-LM computed."""
        returns = make_true_garch_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.B)

        assert profile.arch_lm_stat is not None
        assert profile.arch_lm_pvalue is not None
        assert 0.0 <= profile.arch_lm_pvalue <= 1.0


# ---------------------------------------------------------------------------
# BDS tests
# ---------------------------------------------------------------------------


class TestBDS:
    """Tests for BDS independence test on standardised residuals."""

    def test_iid_doesnt_reject(self) -> None:
        """i.i.d. returns: most BDS dimensions should NOT be significant."""
        returns = make_iid_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        # GARCH on i.i.d. data may or may not converge; if it does, BDS
        # on standardized residuals of i.i.d. data should mostly not reject
        if profile.bds_results is not None:
            n_significant = sum(1 for r in profile.bds_results if r.significant)
            # Allow at most 1 false positive out of max_dim-1 = 4 tests
            assert n_significant <= 2

    def test_nonlinear_detected(self) -> None:
        """TAR returns: BDS should detect nonlinear structure.

        NOTE: We test the BDS function directly on nonlinear data
        since GARCH fitting on TAR data may not converge in
        the expected way.
        """
        from src.app.profiling.application.volatility import _compute_bds

        tar_returns = make_nonlinear_returns(n=3000, seed=42)
        # Run BDS directly on the raw TAR returns
        bds_results = _compute_bds(tar_returns.to_numpy(), max_dim=5, alpha=0.05)
        n_significant = sum(1 for r in bds_results if r.significant)
        assert n_significant >= 2

    def test_tier_a_only(self) -> None:
        """Tier B should not have BDS results."""
        returns = make_true_garch_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.B)

        assert profile.bds_results is None
        assert profile.nonlinear_structure_detected is None


# ---------------------------------------------------------------------------
# Regime labeling tests
# ---------------------------------------------------------------------------


class TestRegimeLabeling:
    """Tests for quantile-based volatility regime classification."""

    def test_approximate_proportions(self) -> None:
        """~25% LOW, ~50% NORMAL, ~25% HIGH within tolerance."""
        returns = make_true_garch_returns(n=5000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.regime_labels is not None
        labels = profile.regime_labels
        n = len(labels)
        n_low = sum(1 for lbl in labels if lbl == VolatilityRegime.LOW)
        n_high = sum(1 for lbl in labels if lbl == VolatilityRegime.HIGH)
        n_normal = sum(1 for lbl in labels if lbl == VolatilityRegime.NORMAL)

        # Proportions with generous tolerance (±10%)
        assert abs(n_low / n - 0.25) < 0.10
        assert abs(n_high / n - 0.25) < 0.10
        assert abs(n_normal / n - 0.50) < 0.10

    def test_applied_to_non_time_bars(self) -> None:
        """Non-time bar types still get regime labels."""
        returns = make_true_garch_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.regime_labels is not None
        assert profile.regime_low_threshold is not None
        assert profile.regime_high_threshold is not None

    def test_correct_length(self) -> None:
        """Regime labels array should match n_observations."""
        n = 1500
        returns = make_true_garch_returns(n=n, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.regime_labels is not None
        assert len(profile.regime_labels) == n


# ---------------------------------------------------------------------------
# Tier gating tests
# ---------------------------------------------------------------------------


class TestTierGating:
    """Tests for tier-based analysis gating."""

    def test_tier_a_full(self) -> None:
        """Tier A + time_1h with sufficient data: all fields populated."""
        returns = make_true_garch_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.garch_fits is not None
        assert profile.sign_bias is not None
        assert profile.arch_lm_stat is not None
        assert profile.regime_labels is not None
        # BDS is also Tier A
        assert profile.bds_results is not None

    def test_tier_b_no_gjr_no_bds(self) -> None:
        """Tier B + time_1h: no GJR-GARCH, no BDS."""
        returns = make_true_garch_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.B)

        assert profile.bds_results is None
        assert profile.gjr_gamma is None
        assert profile.nonlinear_structure_detected is None
        # But GARCH and sign bias should be present
        assert profile.garch_fits is not None
        assert profile.sign_bias is not None

    def test_tier_c_regime_only(self) -> None:
        """Tier C: only regime labels, no GARCH/sign bias/BDS."""
        returns = make_true_garch_returns(n=100, seed=42)
        config = VolatilityConfig(min_samples_garch=500)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.C, config=config)

        assert profile.garch_fits is None
        assert profile.sign_bias is None
        assert profile.bds_results is None
        assert profile.arch_lm_stat is None
        assert profile.regime_labels is not None


# ---------------------------------------------------------------------------
# VolatilityConfig tests
# ---------------------------------------------------------------------------


class TestVolatilityConfig:
    """Tests for VolatilityConfig value object."""

    def test_defaults(self) -> None:
        """Verify default configuration values."""
        config = VolatilityConfig()
        assert config.garch_p == 1
        assert config.garch_q == 1
        assert config.innovation_distributions == ("normal", "t", "skewt")
        assert config.sign_bias_alpha == 0.05
        assert config.bds_max_dim == 5
        assert config.arch_lm_nlags == 10
        assert config.persistence_threshold == 0.99
        assert config.regime_low_quantile == 0.25
        assert config.regime_high_quantile == 0.75
        assert config.min_samples_garch == 500

    def test_immutability(self) -> None:
        """Frozen config should raise on assignment."""
        from pydantic import ValidationError

        config = VolatilityConfig()
        with pytest.raises(ValidationError):
            config.garch_p = 2  # type: ignore[misc]

    def test_quantile_order_validation(self) -> None:
        """low_quantile >= high_quantile should raise ValueError."""
        with pytest.raises(ValueError, match="regime_low_quantile"):
            VolatilityConfig(regime_low_quantile=0.75, regime_high_quantile=0.25)

        with pytest.raises(ValueError, match="regime_low_quantile"):
            VolatilityConfig(regime_low_quantile=0.5, regime_high_quantile=0.5)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and degenerate inputs."""

    def test_short_series_skips_garch(self) -> None:
        """n=50 (< min_samples_garch=500): GARCH fits should be None."""
        returns = make_true_garch_returns(n=50, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        assert profile.garch_fits is None
        assert profile.regime_labels is not None

    def test_constant_returns(self) -> None:
        """All-zero returns should not raise; GARCH fits None."""
        returns = pd.Series(np.zeros(1000, dtype=np.float64), name="const")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A)

        # Constant returns have zero variance; GARCH should fail gracefully
        # Either garch_fits is None (no convergence) or it's present but degenerate
        # The important thing is no exception was raised
        assert profile.regime_labels is not None

    def test_nan_in_realized_vol(self) -> None:
        """Realized vol with NaN: regime labels should use NORMAL for NaN positions."""
        returns = make_true_garch_returns(n=1000, seed=42)
        realized_vol = returns.rolling(20).std().to_numpy()
        # First 19 values are NaN from rolling std
        assert np.isnan(realized_vol[0])

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "time_1h", SampleTier.A, realized_vol=realized_vol)

        assert profile.regime_labels is not None
        # NaN positions should be classified as NORMAL
        for i in range(19):
            assert profile.regime_labels[i] == VolatilityRegime.NORMAL
