"""Tests for PredictabilityAnalyzer against synthetic deterministic and random data.

Verifies permutation entropy, Kish effective sample size, minimum detectable
effect, break-even directional accuracy, signal-to-noise ratio, tier gating,
configuration, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.predictability import PredictabilityAnalyzer
from src.app.profiling.domain.value_objects import (
    PredictabilityConfig,
    SampleTier,
)
from src.tests.profiling.conftest import (
    make_deterministic_returns,
    make_predictable_features,
    make_random_walk_predictability_returns,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_config() -> PredictabilityConfig:
    """Return a default PredictabilityConfig for tests."""
    return PredictabilityConfig()


def _make_analyzer() -> PredictabilityAnalyzer:
    """Return a fresh PredictabilityAnalyzer instance."""
    return PredictabilityAnalyzer()


# ---------------------------------------------------------------------------
# Permutation entropy tests
# ---------------------------------------------------------------------------


class TestPermutationEntropy:
    """Tests for permutation entropy computation."""

    def test_high_entropy_for_random_walk(self) -> None:
        """Random walk returns should have H_norm > 0.9 for all dimensions."""
        returns = make_random_walk_predictability_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.permutation_entropies is not None
        for pe_result in profile.permutation_entropies:
            assert pe_result.normalized_entropy > 0.9, (
                f"d={pe_result.dimension}: H_norm={pe_result.normalized_entropy:.4f} should be > 0.9"
            )

    def test_lower_entropy_for_deterministic(self) -> None:
        """Deterministic pattern returns should have lower H_norm than random walk."""
        rw_returns = make_random_walk_predictability_returns(n=3000, seed=42)
        det_returns = make_deterministic_returns(n=3000, seed=42)

        rw_profile = _make_analyzer().analyze(rw_returns, "BTCUSDT", "dollar", SampleTier.A)
        det_profile = _make_analyzer().analyze(det_returns, "BTCUSDT", "dollar", SampleTier.A)

        assert rw_profile.permutation_entropies is not None
        assert det_profile.permutation_entropies is not None

        for rw_pe, det_pe in zip(
            rw_profile.permutation_entropies,
            det_profile.permutation_entropies,
            strict=True,
        ):
            assert det_pe.normalized_entropy < rw_pe.normalized_entropy, (
                f"d={det_pe.dimension}: deterministic H_norm={det_pe.normalized_entropy:.4f} "
                f"should be < random walk H_norm={rw_pe.normalized_entropy:.4f}"
            )

    def test_dimensions_match_config(self) -> None:
        """Number of PE results should equal len(config.pe_dimensions)."""
        config = PredictabilityConfig(pe_dimensions=(3, 5))
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, config=config)

        assert profile.permutation_entropies is not None
        assert len(profile.permutation_entropies) == 2
        assert profile.permutation_entropies[0].dimension == 3
        assert profile.permutation_entropies[1].dimension == 5

    def test_complexity_positive(self) -> None:
        """JS complexity C should be >= 0 for all dimensions."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.permutation_entropies is not None
        for pe_result in profile.permutation_entropies:
            assert pe_result.js_complexity >= 0, (
                f"d={pe_result.dimension}: C={pe_result.js_complexity:.4f} should be >= 0"
            )

    def test_none_for_tier_c(self) -> None:
        """Tier C should have no permutation entropy results."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C)

        assert profile.permutation_entropies is None


# ---------------------------------------------------------------------------
# Kish effective sample size tests
# ---------------------------------------------------------------------------


class TestKishNeff:
    """Tests for Kish effective sample size computation."""

    def test_iid_neff_close_to_n(self) -> None:
        """i.i.d. returns should have n_eff_ratio close to 1.0."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.n_eff_ratio is not None
        assert profile.n_eff_ratio > 0.7, f"n_eff_ratio={profile.n_eff_ratio:.3f} should be > 0.7 for i.i.d. data"

    def test_autocorrelated_neff_lower(self) -> None:
        """AR(1) returns should have lower n_eff_ratio than i.i.d."""
        # Create autocorrelated returns: AR(1) with phi=0.5
        rng = np.random.default_rng(42)
        n = 2000
        noise = rng.normal(0, 0.01, size=n)
        ar1_data = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ar1_data[i] = 0.5 * ar1_data[i - 1] + noise[i]
        ar1_returns = pd.Series(ar1_data, dtype=np.float64, name="ar1")

        profile = _make_analyzer().analyze(ar1_returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.n_eff_ratio is not None
        assert profile.n_eff_ratio < 0.8, f"n_eff_ratio={profile.n_eff_ratio:.3f} should be < 0.8 for AR(1) phi=0.5"

    def test_none_for_tier_c(self) -> None:
        """Tier C should have no n_eff."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C)

        assert profile.n_eff is None
        assert profile.n_eff_ratio is None


# ---------------------------------------------------------------------------
# MDE directional accuracy tests
# ---------------------------------------------------------------------------


class TestMDE:
    """Tests for minimum detectable effect for directional accuracy."""

    def test_mde_decreases_with_larger_neff(self) -> None:
        """Larger effective sample size should yield a smaller MDE (lower mde_da)."""
        small_returns = make_random_walk_predictability_returns(n=500, seed=42)
        large_returns = make_random_walk_predictability_returns(n=5000, seed=42)

        small_profile = _make_analyzer().analyze(small_returns, "BTCUSDT", "dollar", SampleTier.B)
        large_profile = _make_analyzer().analyze(large_returns, "BTCUSDT", "dollar", SampleTier.A)

        assert small_profile.mde_da is not None
        assert large_profile.mde_da is not None
        assert large_profile.mde_da < small_profile.mde_da, (
            f"Large N mde_da={large_profile.mde_da:.4f} should be < small N mde_da={small_profile.mde_da:.4f}"
        )

    def test_mde_above_50_percent(self) -> None:
        """MDE DA should always be above 0.5."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.mde_da is not None
        assert profile.mde_da > 0.5

    def test_none_for_tier_c(self) -> None:
        """Tier C should have no MDE DA."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C)

        assert profile.mde_da is None


# ---------------------------------------------------------------------------
# Break-even DA tests
# ---------------------------------------------------------------------------


class TestBreakevenDA:
    """Tests for break-even directional accuracy from transaction costs."""

    def test_breakeven_da_realistic_range(self) -> None:
        """For typical crypto returns, breakeven_da should be in [0.5, 0.7]."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.breakeven_da is not None
        assert 0.5 <= profile.breakeven_da <= 0.7, f"breakeven_da={profile.breakeven_da:.4f} should be in [0.5, 0.7]"

    def test_zero_returns_gives_max_da(self) -> None:
        """Near-zero mean absolute return should give breakeven_da = 1.0."""
        returns = pd.Series(np.zeros(500, dtype=np.float64), name="zero_return")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B)

        assert profile.breakeven_da is not None
        assert profile.breakeven_da == 1.0

    def test_none_for_tier_c(self) -> None:
        """Tier C should have no breakeven DA."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C)

        assert profile.breakeven_da is None


# ---------------------------------------------------------------------------
# Signal-to-noise ratio tests
# ---------------------------------------------------------------------------


class TestSNR:
    """Tests for signal-to-noise ratio via Ridge regression."""

    def test_snr_with_informative_features(self) -> None:
        """Predictable features should yield snr_r2 > snr_r2_noise_baseline."""
        returns = make_deterministic_returns(n=3000, seed=42)
        returns_arr = returns.to_numpy(dtype=np.float64)
        features = make_predictable_features(returns_arr, n_informative=5, n_noise=5, seed=42)

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, features=features)

        assert profile.snr_r2 is not None
        assert profile.snr_r2_noise_baseline is not None
        assert profile.is_predictable_vs_noise is True

    def test_snr_noise_only_features(self) -> None:
        """Random features on random returns should yield snr_r2 near or below noise baseline."""
        returns = make_random_walk_predictability_returns(n=3000, seed=42)
        rng = np.random.default_rng(99)
        noise_features = rng.normal(0, 0.01, size=(3000, 15))

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, features=noise_features)

        assert profile.snr_r2 is not None
        assert profile.snr_r2_noise_baseline is not None
        # With only noise features, R² should be close to noise baseline
        # Allow small margin because of randomness
        assert profile.snr_r2 < profile.snr_r2_noise_baseline + 0.05

    def test_snr_none_without_features(self) -> None:
        """Without features, SNR fields should be None."""
        returns = make_random_walk_predictability_returns(n=3000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.snr_r2 is None
        assert profile.snr_r2_noise_baseline is None
        assert profile.is_predictable_vs_noise is None

    def test_snr_tier_a_only(self) -> None:
        """Tier B should have no SNR fields even with features."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        rng = np.random.default_rng(42)
        features = rng.normal(0, 0.01, size=(2000, 10))

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B, features=features)

        assert profile.snr_r2 is None
        assert profile.snr_r2_noise_baseline is None
        assert profile.is_predictable_vs_noise is None


# ---------------------------------------------------------------------------
# Tier gating tests
# ---------------------------------------------------------------------------


class TestTierGating:
    """Tests for correct tier-based gating of analysis components."""

    def test_tier_a_full(self) -> None:
        """Tier A with features should populate all fields."""
        returns = make_random_walk_predictability_returns(n=3000, seed=42)
        returns_arr = returns.to_numpy(dtype=np.float64)
        features = make_predictable_features(returns_arr, n_informative=3, n_noise=5, seed=42)

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, features=features)

        assert profile.permutation_entropies is not None
        assert profile.n_eff is not None
        assert profile.n_eff_ratio is not None
        assert profile.mde_da is not None
        assert profile.breakeven_da is not None
        assert profile.snr_r2 is not None
        assert profile.snr_r2_noise_baseline is not None
        assert profile.is_predictable_vs_noise is not None

    def test_tier_b_no_snr(self) -> None:
        """Tier B should have PE/N_eff/MDE but no SNR."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        rng = np.random.default_rng(42)
        features = rng.normal(0, 0.01, size=(2000, 10))

        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B, features=features)

        # Populated for Tier B
        assert profile.permutation_entropies is not None
        assert profile.n_eff is not None
        assert profile.mde_da is not None
        assert profile.breakeven_da is not None

        # Not populated for Tier B
        assert profile.snr_r2 is None
        assert profile.snr_r2_noise_baseline is None
        assert profile.is_predictable_vs_noise is None

    def test_tier_c_minimal(self) -> None:
        """Tier C should have all analysis fields as None."""
        returns = make_random_walk_predictability_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C)

        assert profile.permutation_entropies is None
        assert profile.n_eff is None
        assert profile.n_eff_ratio is None
        assert profile.mde_da is None
        assert profile.breakeven_da is None
        assert profile.snr_r2 is None
        assert profile.snr_r2_noise_baseline is None
        assert profile.is_predictable_vs_noise is None


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestPredictabilityConfig:
    """Tests for PredictabilityConfig defaults and immutability."""

    def test_defaults(self) -> None:
        """Verify default config values match specification."""
        config = PredictabilityConfig()

        assert config.pe_dimensions == (3, 4, 5, 6)
        assert config.pe_delay == 1
        assert config.alpha == 0.05
        assert config.power == 0.80
        assert config.round_trip_cost == 0.002
        assert config.snr_holdout_fraction == 0.30
        assert config.snr_ridge_alpha == 1.0
        assert config.snr_n_noise_baselines == 10
        assert config.bartlett_max_lag_fraction == 0.1
        assert config.min_samples_predictability == 100

    def test_immutability(self) -> None:
        """Frozen config should reject attribute assignment."""
        config = PredictabilityConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011
            config.alpha = 0.01  # type: ignore[misc]

    def test_custom_config(self) -> None:
        """Custom configuration values should be accepted."""
        config = PredictabilityConfig(
            pe_dimensions=(3, 5, 7),
            pe_delay=2,
            alpha=0.10,
            power=0.90,
            round_trip_cost=0.003,
            snr_holdout_fraction=0.20,
            snr_ridge_alpha=0.5,
            snr_n_noise_baselines=5,
            bartlett_max_lag_fraction=0.05,
            min_samples_predictability=50,
        )

        assert config.pe_dimensions == (3, 5, 7)
        assert config.pe_delay == 2
        assert config.alpha == 0.10
        assert config.power == 0.90


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and degenerate inputs."""

    def test_short_series(self) -> None:
        """Series shorter than min_samples should return all None."""
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, size=50), name="short")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A)

        assert profile.n_observations == 50
        assert profile.permutation_entropies is None
        assert profile.n_eff is None
        assert profile.mde_da is None
        assert profile.breakeven_da is None
        assert profile.snr_r2 is None

    def test_constant_returns(self) -> None:
        """Constant returns should not raise errors."""
        returns = pd.Series(np.full(500, 0.001, dtype=np.float64), name="constant")
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B)

        # Should complete without raising
        assert profile.n_observations == 500
        assert profile.permutation_entropies is not None
        assert profile.n_eff is not None
        assert profile.breakeven_da is not None
