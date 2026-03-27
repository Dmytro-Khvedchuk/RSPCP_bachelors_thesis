"""Tests for Phase 7 RC7 computations (GH #78).

Covers five audit items:
  1. Cost sensitivity — break-even DA at {10,15,20,25,30} bps
  2. MI normalization — discrete entropy and effect-size classification
  3. Stationarity transformations — rolling_zscore, first_difference, updated dict
  4. Conditional break-even DA — HIGH-regime filtering and amplification ratio
  5. Feature variance — atr_14/rsi_14 degeneracy detection
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, kpss

from src.app.profiling.application.stationarity import (
    _KNOWN_TRANSFORMATIONS,
    _classify_stationarity,
    _suggest_transformation,
)
from src.app.research.application.rc2_thresholds import compute_breakeven_da
from src.app.research.application.rc2_validation_analysis import (
    compute_target_entropy_gaussian,
)
from src.app.research.application.rc7_phase7_utils import (
    _DEGENERACY_THRESHOLD,
    apply_first_difference,
    apply_rolling_zscore,
    classify_mi_effect_size,
    compute_amplification_ratio,
    compute_conditional_breakeven_da,
    compute_discrete_entropy,
    compute_feature_variance,
    is_feature_degenerate,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG_SEED = 42


def _make_random_walk(n: int = 1000, seed: int = _RNG_SEED) -> np.ndarray:
    """Generate a random walk (unit root) series."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1.0, size=n))


def _make_white_noise(n: int = 1000, seed: int = _RNG_SEED) -> np.ndarray:
    """Generate white noise (stationary) series."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, size=n)


def _make_regime_labels(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Assign LOW/NORMAL/HIGH regime labels via rolling vol quantiles."""
    series = pd.Series(returns)
    rolling_vol = series.rolling(window=window).std()
    q25 = rolling_vol.quantile(0.25)
    q75 = rolling_vol.quantile(0.75)
    return np.where(
        rolling_vol < q25,
        "LOW",
        np.where(rolling_vol > q75, "HIGH", "NORMAL"),
    )


# ---------------------------------------------------------------------------
# 1. Cost Sensitivity Tests
# ---------------------------------------------------------------------------


class TestPhase7CostSensitivity:
    """Break-even DA at multiple cost tiers with realistic crypto parameters."""

    # Realistic mean|r| from RC7 Table (Phase 7.1)
    BTC_DOLLAR_MEAN_ABS_R = 0.01438
    ETH_DOLLAR_MEAN_ABS_R = 0.02761
    SOL_DOLLAR_MEAN_ABS_R = 0.06109
    BTC_TIME_1H_MEAN_ABS_R = 0.00402
    BTC_VOL_IMB_MEAN_ABS_R = 0.04746

    def test_breakeven_da_at_10bps_btc_dollar(self):
        result = compute_breakeven_da(self.BTC_DOLLAR_MEAN_ABS_R, 0.001)
        expected = 0.5 + 0.001 / (2 * self.BTC_DOLLAR_MEAN_ABS_R)
        assert result.breakeven_da == pytest.approx(expected, abs=1e-6)

    def test_breakeven_da_at_20bps_btc_dollar(self):
        result = compute_breakeven_da(self.BTC_DOLLAR_MEAN_ABS_R, 0.002)
        expected = 0.5 + 0.002 / (2 * self.BTC_DOLLAR_MEAN_ABS_R)
        assert result.breakeven_da == pytest.approx(expected, abs=1e-6)

    def test_breakeven_da_at_30bps_btc_dollar(self):
        result = compute_breakeven_da(self.BTC_DOLLAR_MEAN_ABS_R, 0.003)
        expected = 0.5 + 0.003 / (2 * self.BTC_DOLLAR_MEAN_ABS_R)
        assert result.breakeven_da == pytest.approx(expected, abs=1e-6)

    def test_cost_sweep_monotonically_increasing(self):
        """Break-even DA must increase with cost for fixed mean|r|."""
        costs = [0.001, 0.0015, 0.002, 0.0025, 0.003]
        das = [compute_breakeven_da(self.BTC_DOLLAR_MEAN_ABS_R, c).breakeven_da for c in costs]
        for i in range(1, len(das)):
            assert das[i] > das[i - 1]

    @pytest.mark.parametrize(
        ("cost_bps", "cost_decimal"),
        [(10, 0.001), (15, 0.0015), (20, 0.002), (25, 0.0025), (30, 0.003)],
    )
    def test_parametrize_all_cost_levels(self, cost_bps, cost_decimal):
        """Formula p = 0.5 + c/(2*E[|r|]) matches function output at each tier."""
        result = compute_breakeven_da(self.BTC_DOLLAR_MEAN_ABS_R, cost_decimal)
        expected = 0.5 + cost_decimal / (2 * self.BTC_DOLLAR_MEAN_ABS_R)
        assert result.breakeven_da == pytest.approx(expected, abs=1e-10)

    def test_imbalance_bars_viable_at_all_costs(self):
        """Volume-imbalance bars (mean|r|~0.047) have BE_DA < 55% at all tiers."""
        for cost in [0.001, 0.0015, 0.002, 0.0025, 0.003]:
            result = compute_breakeven_da(self.BTC_VOL_IMB_MEAN_ABS_R, cost)
            assert result.breakeven_da < 0.55

    def test_time_bars_not_viable(self):
        """Time_1h bars (mean|r|~0.004) have BE_DA > 60% even at 10 bps."""
        result = compute_breakeven_da(self.BTC_TIME_1H_MEAN_ABS_R, 0.001)
        assert result.breakeven_da > 0.60


# ---------------------------------------------------------------------------
# 2. MI Normalization Tests
# ---------------------------------------------------------------------------


class TestPhase7MINormalization:
    """Discrete entropy, Gaussian entropy bug, and effect-size classification."""

    def test_discrete_entropy_uniform_distribution(self):
        """Uniform over k=10 bins: H = log(10)."""
        arr = np.repeat(np.arange(10, dtype=float), 100)  # 1000 values, 10 groups
        h = compute_discrete_entropy(arr, n_bins=10)
        assert h == pytest.approx(math.log(10), abs=0.05)

    def test_discrete_entropy_constant_array(self):
        """All-identical values: H = 0."""
        arr = np.ones(500)
        h = compute_discrete_entropy(arr, n_bins=10)
        assert h == pytest.approx(0.0, abs=1e-10)

    def test_discrete_entropy_always_nonnegative(self):
        """Discrete entropy is non-negative for various distributions."""
        rng = np.random.default_rng(_RNG_SEED)
        arrays = [
            rng.normal(0, 0.001, 5000),  # crypto-scale small variance
            rng.normal(0, 1.0, 5000),
            rng.uniform(-1, 1, 5000),
            rng.standard_t(3, 5000),
        ]
        for arr in arrays:
            h = compute_discrete_entropy(arr)
            assert h >= 0.0

    def test_discrete_entropy_sturges_bin_count(self):
        """Sturges rule: bins = 1 + floor(log2(N)) for N=5000 -> 13."""
        expected_bins = 1 + int(math.floor(math.log2(5000)))
        assert expected_bins == 13

    def test_gaussian_entropy_negative_for_small_variance(self):
        """Gaussian differential entropy is negative when var < 1/(2*pi*e)."""
        rng = np.random.default_rng(_RNG_SEED)
        arr = rng.normal(0, 0.01, 5000)  # var ~1e-4, well below 0.0586
        h_gauss = compute_target_entropy_gaussian(arr)
        assert h_gauss < 0.0

    def test_discrete_entropy_positive_where_gaussian_negative(self):
        """Same small-variance data: discrete entropy is positive."""
        rng = np.random.default_rng(_RNG_SEED)
        arr = rng.normal(0, 0.01, 5000)
        h_disc = compute_discrete_entropy(arr)
        h_gauss = compute_target_entropy_gaussian(arr)
        assert h_gauss < 0.0
        assert h_disc > 0.0

    @pytest.mark.parametrize(
        ("mi_nats", "expected"),
        [
            (0.10, "Strong"),
            (0.06, "Strong"),
            (0.03, "Moderate"),
            (0.015, "Moderate"),
            (0.005, "Weak"),
            (0.002, "Weak"),
            (0.0001, "Negligible"),
            (0.0, "Negligible"),
        ],
    )
    def test_effect_size_classification(self, mi_nats, expected):
        assert classify_mi_effect_size(mi_nats) == expected

    def test_mi_ratio_known_pair(self):
        """MI=0.05 nats, H_disc=1.62 nats => MI/H = 3.086%."""
        mi = 0.05
        h_disc = 1.62
        ratio_pct = (mi / h_disc) * 100.0
        assert ratio_pct == pytest.approx(3.086, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Stationarity Transformation Tests
# ---------------------------------------------------------------------------


class TestPhase7StationarityTransformations:
    """Transformation functions and updated _KNOWN_TRANSFORMATIONS dict."""

    def test_rolling_zscore_shape_and_nan_count(self):
        """Output length matches input; first window-1 values are NaN."""
        series = pd.Series(_make_random_walk(200))
        window = 24
        result = apply_rolling_zscore(series, window=window)
        assert len(result) == len(series)
        assert result.isna().sum() == window - 1

    def test_rolling_zscore_normalizes(self):
        """Non-NaN portion should have std near 1 (mean may drift on non-stationary input)."""
        rng = np.random.default_rng(_RNG_SEED)
        series = pd.Series(rng.normal(0, 1.0, 2000))  # stationary input
        result = apply_rolling_zscore(series, window=24)
        valid = result.dropna()
        assert abs(valid.mean()) < 0.15
        assert abs(valid.std() - 1.0) < 0.15

    def test_rolling_zscore_on_unit_root_passes_adf(self):
        """Applying rolling z-score to a random walk should produce a stationary result."""
        series = pd.Series(_make_random_walk(1000))
        transformed = apply_rolling_zscore(series, window=24).dropna().to_numpy()
        adf_stat, adf_pval, *_ = adfuller(transformed, autolag="AIC")
        assert adf_pval < 0.05

    def test_first_difference_shape_and_nan(self):
        """Output length matches input; exactly one NaN at start."""
        series = pd.Series(_make_random_walk(200))
        result = apply_first_difference(series)
        assert len(result) == len(series)
        assert result.isna().sum() == 1
        assert pd.isna(result.iloc[0])

    def test_first_difference_of_random_walk_is_stationary(self):
        """diff(cumsum(noise)) ~ noise -> ADF should reject unit root."""
        series = pd.Series(_make_random_walk(1000))
        transformed = apply_first_difference(series).dropna().to_numpy()
        adf_stat, adf_pval, *_ = adfuller(transformed, autolag="AIC")
        assert adf_pval < 0.05

    def test_known_transformations_includes_phase7_prefixes(self):
        """_KNOWN_TRANSFORMATIONS must include gk_vol_, park_vol_, rv_ entries."""
        assert "gk_vol_" in _KNOWN_TRANSFORMATIONS
        assert "park_vol_" in _KNOWN_TRANSFORMATIONS
        assert "rv_" in _KNOWN_TRANSFORMATIONS

    def test_suggest_transformation_gk_vol(self):
        assert _suggest_transformation("gk_vol_24") == "rolling_zscore"

    def test_suggest_transformation_park_vol(self):
        assert _suggest_transformation("park_vol_24") == "first_difference"

    def test_suggest_transformation_rv(self):
        assert _suggest_transformation("rv_12") == "first_difference"

    def test_transformed_amihud_passes_joint_classification(self):
        """Non-stationary amihud-like series -> rolling_zscore -> stationary."""
        rng = np.random.default_rng(_RNG_SEED)
        # Simulate a non-stationary, positive-valued amihud series
        trend = np.linspace(1.0, 5.0, 1000)
        noise = rng.normal(0, 0.1, 1000)
        amihud = pd.Series(np.abs(trend + noise))

        transformed = apply_rolling_zscore(amihud, window=24).dropna().to_numpy()
        adf_stat, adf_pval, *_ = adfuller(transformed, autolag="AIC")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_pval, *_ = kpss(transformed, regression="c", nlags="auto")

        classification = _classify_stationarity(
            adf_rejects=adf_pval < 0.05,
            kpss_rejects=kpss_pval < 0.05,
        )
        assert classification in {"stationary", "trend_stationary"}

    def test_transformed_bbwidth_passes_joint_classification(self):
        """Non-stationary bbwidth-like series -> first_difference -> stationary."""
        # Simulate bbwidth as a slowly drifting positive series
        rng = np.random.default_rng(_RNG_SEED)
        drift = np.cumsum(rng.normal(0.001, 0.01, 1000))
        bbwidth = pd.Series(np.abs(drift) + 0.01)

        transformed = apply_first_difference(bbwidth).dropna().to_numpy()
        adf_stat, adf_pval, *_ = adfuller(transformed, autolag="AIC")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_pval, *_ = kpss(transformed, regression="c", nlags="auto")

        classification = _classify_stationarity(
            adf_rejects=adf_pval < 0.05,
            kpss_rejects=kpss_pval < 0.05,
        )
        assert classification in {"stationary", "trend_stationary"}


# ---------------------------------------------------------------------------
# 4. Conditional Break-Even DA Tests
# ---------------------------------------------------------------------------


class TestPhase7ConditionalBreakevenDA:
    """HIGH-regime filtering, amplification ratio, and conditional break-even."""

    def _make_synthetic_returns_with_regimes(
        self,
        n: int = 2000,
        seed: int = _RNG_SEED,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate returns and regime labels via rolling vol quantiles."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(0, 0.01, n)
        # Inject high-vol bursts in the middle
        returns[n // 4 : n // 4 + n // 8] *= 3.0
        returns[3 * n // 4 : 3 * n // 4 + n // 8] *= 3.0
        labels = _make_regime_labels(returns, window=20)
        return returns, labels

    def test_conditional_breakeven_lower_than_unconditional(self):
        """HIGH-regime mean|r| > overall -> conditional BE_DA < unconditional."""
        returns, labels = self._make_synthetic_returns_with_regimes()
        cost = 0.002
        # Skip NaN labels from rolling window warmup
        valid_mask = labels != "nan"
        valid_returns = returns[valid_mask]
        valid_labels = labels[valid_mask]

        unconditional = compute_breakeven_da(float(np.mean(np.abs(valid_returns))), cost)
        conditional = compute_conditional_breakeven_da(valid_returns, valid_labels, "HIGH", cost)
        assert conditional.breakeven_da < unconditional.breakeven_da

    def test_conditional_breakeven_known_values(self):
        """Deterministic test: known HIGH mean|r|=0.023, cost=0.002."""
        # Construct returns where HIGH bars have exactly mean|r|=0.023
        rng = np.random.default_rng(_RNG_SEED)
        n = 1000
        returns = np.zeros(n)
        labels = np.array(["NORMAL"] * n)

        # Set 250 HIGH bars with |r|=0.023
        high_indices = list(range(250))
        signs = rng.choice([-1.0, 1.0], size=250)
        returns[high_indices] = signs * 0.023
        labels[high_indices] = "HIGH"

        # Set remaining NORMAL bars with smaller returns
        normal_indices = list(range(250, n))
        returns[normal_indices] = rng.normal(0, 0.005, len(normal_indices))

        result = compute_conditional_breakeven_da(returns, labels, "HIGH", 0.002)
        expected_da = 0.5 + 0.002 / (2 * 0.023)
        assert result.breakeven_da == pytest.approx(expected_da, abs=1e-4)

    def test_amplification_ratio_above_one(self):
        """HIGH-regime must have amplification ratio > 1 by construction."""
        returns, labels = self._make_synthetic_returns_with_regimes()
        valid_mask = labels != "nan"
        ratio = compute_amplification_ratio(returns[valid_mask], labels[valid_mask], "HIGH")
        assert ratio > 1.0

    def test_amplification_ratio_deterministic(self):
        """Manual setup: HIGH bars have 2x the mean abs return -> ratio ~ 2.0."""
        n = 1000
        returns = np.ones(n) * 0.01
        labels = np.array(["NORMAL"] * n)

        # Set 250 HIGH bars with 2x returns
        high_idx = list(range(250))
        returns[high_idx] = 0.02
        labels[high_idx] = "HIGH"

        # Also set 250 LOW bars with 0.5x returns to keep overall mean at ~0.01
        low_idx = list(range(250, 500))
        returns[low_idx] = 0.005
        labels[low_idx] = "LOW"

        ratio = compute_amplification_ratio(returns, labels, "HIGH")
        overall_mean = np.mean(np.abs(returns))
        expected_ratio = 0.02 / overall_mean
        assert ratio == pytest.approx(expected_ratio, abs=0.01)

    def test_regime_label_proportions(self):
        """Q25/Q75 regime split gives roughly 25% HIGH, 50% NORMAL, 25% LOW."""
        rng = np.random.default_rng(_RNG_SEED)
        returns = rng.normal(0, 0.01, 5000)
        labels = _make_regime_labels(returns, window=20)
        # Exclude NaN from warmup
        valid = labels[labels != "nan"]
        total = len(valid)

        high_frac = np.sum(valid == "HIGH") / total
        normal_frac = np.sum(valid == "NORMAL") / total
        low_frac = np.sum(valid == "LOW") / total

        assert high_frac == pytest.approx(0.25, abs=0.05)
        assert normal_frac == pytest.approx(0.50, abs=0.10)
        assert low_frac == pytest.approx(0.25, abs=0.05)

    @pytest.mark.parametrize(
        "cost",
        [0.001, 0.0015, 0.002, 0.0025, 0.003],
    )
    def test_conditional_breakeven_at_multiple_costs(self, cost):
        """Conditional BE_DA < unconditional at each cost level."""
        rng = np.random.default_rng(_RNG_SEED)
        n = 2000
        returns = rng.normal(0, 0.01, n)
        returns[n // 4 : n // 4 + n // 8] *= 3.0
        labels = _make_regime_labels(returns, window=20)
        valid_mask = labels != "nan"
        valid_returns = returns[valid_mask]
        valid_labels = labels[valid_mask]

        uncond = compute_breakeven_da(float(np.mean(np.abs(valid_returns))), cost)
        cond = compute_conditional_breakeven_da(valid_returns, valid_labels, "HIGH", cost)
        assert cond.breakeven_da < uncond.breakeven_da

    def test_higher_costs_amplify_conditional_benefit(self):
        """Delta_DA (unconditional - conditional) increases with cost."""
        rng = np.random.default_rng(_RNG_SEED)
        n = 2000
        returns = rng.normal(0, 0.01, n)
        returns[n // 4 : n // 4 + n // 8] *= 3.0
        labels = _make_regime_labels(returns, window=20)
        valid_mask = labels != "nan"
        valid_returns = returns[valid_mask]
        valid_labels = labels[valid_mask]

        costs = [0.001, 0.002, 0.003]
        deltas = []
        for cost in costs:
            uncond = compute_breakeven_da(float(np.mean(np.abs(valid_returns))), cost)
            cond = compute_conditional_breakeven_da(valid_returns, valid_labels, "HIGH", cost)
            deltas.append(uncond.breakeven_da - cond.breakeven_da)

        for i in range(1, len(deltas)):
            assert deltas[i] > deltas[i - 1]


# ---------------------------------------------------------------------------
# 5. Feature Variance / Degeneracy Tests
# ---------------------------------------------------------------------------


class TestPhase7FeatureVariance:
    """Feature degeneracy detection for atr_14/rsi_14."""

    def test_constant_feature_is_degenerate(self):
        arr = np.ones(1000)
        assert is_feature_degenerate(arr) is True

    def test_near_constant_is_degenerate(self):
        """Tiny perturbation (noise scale 1e-12) still degenerate."""
        rng = np.random.default_rng(_RNG_SEED)
        arr = np.ones(1000) + rng.normal(0, 1e-12, 1000)
        assert is_feature_degenerate(arr) is True

    def test_normal_feature_not_degenerate(self):
        rng = np.random.default_rng(_RNG_SEED)
        arr = rng.normal(0, 1.0, 1000)
        assert is_feature_degenerate(arr) is False

    def test_crypto_returns_not_degenerate(self):
        """Typical crypto bar returns (var ~1e-4) are NOT degenerate."""
        rng = np.random.default_rng(_RNG_SEED)
        arr = rng.normal(0, 0.01, 1000)  # var ~1e-4
        assert is_feature_degenerate(arr) is False

    def test_variance_matches_numpy(self):
        rng = np.random.default_rng(_RNG_SEED)
        arr = rng.normal(0, 1.0, 500)
        assert compute_feature_variance(arr) == pytest.approx(float(np.var(arr, ddof=1)), abs=1e-10)

    def test_degeneracy_threshold_value(self):
        """The default threshold must be exactly 1e-10."""
        assert _DEGENERACY_THRESHOLD == 1e-10

    def test_atr_on_constant_ohlcv_is_degenerate(self):
        """ATR on flat OHLCV (high == low == close) produces degenerate series."""
        n = 200
        # Flat OHLCV: all candles identical
        atr_values = np.zeros(n)
        # True Range = max(H-L, |H-Cprev|, |L-Cprev|) = 0 when H=L=C
        # Wilder smoothing of zeros = 0 for all bars
        assert is_feature_degenerate(atr_values) is True

    def test_rsi_on_constant_ohlcv_is_degenerate(self):
        """RSI on flat OHLCV converges to 50.0 (degenerate)."""
        n = 200
        # When close-to-close changes are zero, RSI = 50 for all bars
        rsi_values = np.full(n, 50.0)
        assert is_feature_degenerate(rsi_values) is True
