"""Shared fixtures and factory functions for profiling module tests.

Provides synthetic data generators for stationarity testing,
distribution analysis, serial dependence analysis, volatility
modeling, predictability assessment, and reusable configuration fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    AutocorrelationConfig,
    DistributionConfig,
    PredictabilityConfig,
    ProfilingConfig,
    TierConfig,
    VolatilityConfig,
)


def make_stationary_series(
    n: int = 500,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate a stationary (white noise) series.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of i.i.d. standard normal values.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_unit_root_series(
    n: int = 500,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate a unit root (random walk) series.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of cumulative sum of standard normal values.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    steps: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n)
    return np.cumsum(steps)


def make_stationarity_test_df(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a Pandas DataFrame with known stationary and non-stationary features.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame, list of feature column names).
        Contains 'stationary_feat' (white noise) and 'unit_root_feat' (random walk).
    """
    stationary: np.ndarray[tuple[int], np.dtype[np.float64]] = make_stationary_series(n, seed)
    unit_root: np.ndarray[tuple[int], np.dtype[np.float64]] = make_unit_root_series(n, seed + 1)

    df: pd.DataFrame = pd.DataFrame(
        {
            "stationary_feat": stationary,
            "unit_root_feat": unit_root,
        }
    )
    feature_names: list[str] = ["stationary_feat", "unit_root_feat"]
    return df, feature_names


# ---------------------------------------------------------------------------
# Distribution analysis helpers
# ---------------------------------------------------------------------------


def make_normal_returns(
    n: int = 5000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate normally distributed returns with realistic crypto-scale volatility.

    Draws from ``N(0, 0.01)`` (mean=0, std=1% per bar).

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of i.i.d. Normal returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=n)
    return pd.Series(data, dtype=np.float64, name="log_return")


def make_student_t_returns(
    n: int = 5000,
    nu: float = 5.0,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate Student-t distributed returns with given degrees of freedom.

    Draws from ``t(nu)`` scaled to realistic return magnitude (scale=0.01).

    Args:
        n: Number of return observations.
        nu: Degrees of freedom (lower = heavier tails).
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of Student-t returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_t(df=nu, size=n) * 0.01
    return pd.Series(data, dtype=np.float64, name="log_return")


def make_skewed_returns(
    n: int = 5000,
    skew_direction: str = "left",
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate skewed returns for testing skewness detection.

    Uses ``scipy.stats.skewnorm`` to produce left- or right-skewed samples.

    Args:
        n: Number of return observations.
        skew_direction: ``"left"`` for negative skew, ``"right"`` for positive.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of skewed returns.

    Raises:
        ValueError: If *skew_direction* is not ``"left"`` or ``"right"``.
    """
    if skew_direction not in {"left", "right"}:
        msg = f"skew_direction must be 'left' or 'right', got '{skew_direction}'"
        raise ValueError(msg)

    a = -5.0 if skew_direction == "left" else 5.0
    data = stats.skewnorm.rvs(a, size=n, random_state=seed) * 0.01
    return pd.Series(data, dtype=np.float64, name="log_return")


# ---------------------------------------------------------------------------
# Serial dependence analysis helpers
# ---------------------------------------------------------------------------


def make_ar1_returns(
    n: int = 1000,
    phi: float = 0.5,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate an AR(1) return series with parameter *phi*.

    ``r_t = phi * r_{t-1} + epsilon_t`` where ``epsilon ~ N(0, 0.01)``.

    Args:
        n: Number of return observations.
        phi: AR(1) coefficient (|phi| < 1 for stationarity).
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of AR(1) returns.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=0.01, size=n)
    data = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        data[i] = phi * data[i - 1] + noise[i]
    return pd.Series(data, dtype=np.float64, name="ar1_return")


def make_random_walk_returns(
    n: int = 1000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate random walk returns (first differences of a random walk).

    The *returns* are i.i.d. white noise ``N(0, 0.01)`` (since diff of
    random walk = innovations).

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of i.i.d. Normal returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=n)
    return pd.Series(data, dtype=np.float64, name="rw_return")


def make_garch_like_returns(
    n: int = 1000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate returns with GARCH-like volatility clustering.

    Uses a simple approximation: multiply white noise by a rolling
    absolute return to create conditional heteroscedasticity.
    ``sigma_t = 0.005 + 0.8 * |r_{t-1}|``, ``r_t = sigma_t * z_t``.

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of returns exhibiting volatility clustering.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    data = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    sigma[0] = 0.01
    data[0] = sigma[0] * z[0]
    for i in range(1, n):
        sigma[i] = 0.005 + 0.8 * abs(data[i - 1])
        data[i] = sigma[i] * z[i]
    return pd.Series(data, dtype=np.float64, name="garch_return")


def make_causal_pair(
    n: int = 500,
    lag: int = 1,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:  # type: ignore[type-arg]
    """Generate a pair of return series where X Granger-causes Y.

    ``Y_t = alpha * X_{t-lag} + noise``.

    Args:
        n: Number of return observations.
        lag: Lag order of the causal relationship.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(X, Y)`` Pandas Series.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=0.01, size=n)
    noise = rng.normal(loc=0.0, scale=0.005, size=n)
    y = np.zeros(n, dtype=np.float64)
    for i in range(lag, n):
        y[i] = 0.5 * x[i - lag] + noise[i]
    return (
        pd.Series(x, dtype=np.float64, name="X"),
        pd.Series(y, dtype=np.float64, name="Y"),
    )


# ---------------------------------------------------------------------------
# Volatility modeling helpers
# ---------------------------------------------------------------------------


def make_true_garch_returns(
    n: int = 2000,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate returns from a true GARCH(1,1) DGP.

    Args:
        n: Number of return observations.
        omega: Constant term in the conditional variance equation.
        alpha: Coefficient for lagged squared returns.
        beta: Coefficient for lagged conditional variance.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of GARCH(1,1) returns.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sigma2 = np.zeros(n, dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    returns[0] = np.sqrt(sigma2[0]) * z[0]
    for i in range(1, n):
        sigma2[i] = omega + alpha * returns[i - 1] ** 2 + beta * sigma2[i - 1]
        returns[i] = np.sqrt(sigma2[i]) * z[i]
    return pd.Series(returns, dtype=np.float64, name="garch_return")


def make_gjr_garch_returns(  # noqa: PLR0917
    n: int = 2000,
    omega: float = 0.00001,
    alpha: float = 0.05,
    gamma: float = 0.15,
    beta: float = 0.85,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate returns from a GJR-GARCH(1,1) DGP with leverage effect.

    Args:
        n: Number of return observations.
        omega: Constant term in the conditional variance equation.
        alpha: Coefficient for lagged squared returns.
        gamma: Asymmetric leverage coefficient (applied to negative returns).
        beta: Coefficient for lagged conditional variance.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of GJR-GARCH(1,1) returns.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sigma2 = np.zeros(n, dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    returns[0] = np.sqrt(sigma2[0]) * z[0]
    for i in range(1, n):
        leverage = gamma * returns[i - 1] ** 2 * (1 if returns[i - 1] < 0 else 0)
        sigma2[i] = omega + alpha * returns[i - 1] ** 2 + leverage + beta * sigma2[i - 1]
        returns[i] = np.sqrt(sigma2[i]) * z[i]
    return pd.Series(returns, dtype=np.float64, name="gjr_garch_return")


def make_iid_returns(
    n: int = 2000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate i.i.d. N(0, 0.01) returns with no ARCH effects.

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of i.i.d. Normal returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=n)
    return pd.Series(data, dtype=np.float64, name="iid_return")


def make_nonlinear_returns(
    n: int = 2000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate nonlinear (TAR model) returns for BDS rejection.

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of threshold autoregressive returns.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=0.005, size=n)
    data = np.zeros(n, dtype=np.float64)
    data[0] = noise[0]
    for i in range(1, n):
        if data[i - 1] > 0:
            data[i] = 0.8 * data[i - 1] + noise[i]
        else:
            data[i] = -0.5 * data[i - 1] + noise[i]
    return pd.Series(data, dtype=np.float64, name="tar_return")


# ---------------------------------------------------------------------------
# Predictability assessment helpers
# ---------------------------------------------------------------------------


def make_deterministic_returns(
    n: int = 2000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate highly predictable (low entropy) returns.

    Alternating positive/negative pattern with small noise.

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of near-deterministic alternating returns.
    """
    rng = np.random.default_rng(seed)
    pattern = np.array([0.01, -0.01] * (n // 2), dtype=np.float64)
    noise = rng.normal(0, 0.001, size=n)
    return pd.Series(pattern + noise[:n], dtype=np.float64, name="deterministic_return")


def make_random_walk_predictability_returns(
    n: int = 2000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate random walk returns (high entropy, unpredictable).

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of i.i.d. Normal returns simulating random walk increments.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=n)
    return pd.Series(data, dtype=np.float64, name="rw_return")


def make_predictable_features(
    returns: np.ndarray,  # type: ignore[type-arg]
    n_informative: int = 5,
    n_noise: int = 10,
    seed: int = 42,
) -> np.ndarray:  # type: ignore[type-arg]
    """Generate a feature matrix with some informative and some noise features.

    Informative features are lagged copies of returns plus noise.
    Noise features are pure random Gaussian.

    Args:
        returns: 1-D array of return observations.
        n_informative: Number of informative (lagged return) features.
        n_noise: Number of pure noise features.
        seed: Random seed for reproducibility.

    Returns:
        Feature matrix of shape ``(n, n_informative + n_noise)``.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    features = np.zeros((n, n_informative + n_noise), dtype=np.float64)
    for i in range(n_informative):
        lag = i + 1
        features[lag:, i] = returns[:-lag] + rng.normal(0, 0.002, size=n - lag)
    for i in range(n_noise):
        features[:, n_informative + i] = rng.normal(0, 0.01, size=n)
    return features


# ---------------------------------------------------------------------------
# Price-level random walk (for VR = 1 testing)
# ---------------------------------------------------------------------------


def make_random_walk(
    n: int = 1000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate a price-level random walk (cumulative sum of white noise).

    Suitable for variance-ratio tests that expect VR(q) ≈ 1.0 at all horizons,
    since the log-price series satisfies the martingale difference hypothesis.

    Args:
        n: Number of observations (levels, not returns).
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of cumulative sum of i.i.d. N(0, 0.01) increments.
    """
    rng = np.random.default_rng(seed)
    increments: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.normal(loc=0.0, scale=0.01, size=n)
    levels: np.ndarray[tuple[int], np.dtype[np.float64]] = np.cumsum(increments)
    return pd.Series(levels, dtype=np.float64, name="random_walk")


# ---------------------------------------------------------------------------
# ProfilingConfig factory with fast defaults for testing
# ---------------------------------------------------------------------------


def make_profiling_config() -> ProfilingConfig:
    """Return a ProfilingConfig with reduced parameters for fast test execution.

    Reduces computation by:
    - Cutting Ljung-Box lags to (5, 10) instead of (5, 10, 20, 40).
    - Using only a single VR horizon.
    - Fitting only the Normal distribution for GARCH (skips t + skewt).
    - Lowering min_samples_garch so small synthetic datasets qualify.
    - Using only one PE dimension.

    Returns:
        Frozen ``ProfilingConfig`` suitable for fast unit and integration tests.
    """
    fast_autocorr: AutocorrelationConfig = AutocorrelationConfig(
        ljung_box_lags=(5, 10),
        vr_calendar_horizons_days=(1.0, 3.0),
        granger_max_lags=(1, 2),
    )
    fast_vol: VolatilityConfig = VolatilityConfig(
        innovation_distributions=("normal",),
        min_samples_garch=200,
        bds_max_dim=3,
        arch_lm_nlags=5,
    )
    fast_pred: PredictabilityConfig = PredictabilityConfig(
        pe_dimensions=(3, 4),
        snr_n_noise_baselines=3,
        min_samples_predictability=50,
    )
    fast_dist: DistributionConfig = DistributionConfig()
    fast_tier: TierConfig = TierConfig(tier_a_threshold=2000, tier_b_threshold=500)
    return ProfilingConfig(
        distribution=fast_dist,
        autocorrelation=fast_autocorr,
        volatility=fast_vol,
        predictability=fast_pred,
        tier=fast_tier,
    )
