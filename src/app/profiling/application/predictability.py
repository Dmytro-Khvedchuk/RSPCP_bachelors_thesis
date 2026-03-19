"""Predictability assessment via permutation entropy, effective sample size, and signal-to-noise analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import acf  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    PermutationEntropyResult,
    PredictabilityConfig,
    PredictabilityProfile,
    SampleTier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SAMPLES_ACF: int = 2
"""Minimum observations required for ACF-based N_eff computation."""

_MIN_SPLIT_SAMPLES: int = 2
"""Minimum observations in train or test partition for Ridge SNR."""

_MIN_FEATURES: int = 1
"""Minimum number of features for Ridge SNR computation."""


# ---------------------------------------------------------------------------
# Internal computation helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(p: np.ndarray) -> float:  # type: ignore[type-arg]
    """Compute Shannon entropy H(p) in natural logarithm (nats).

    Ignores zero-probability entries to avoid log(0).

    Args:
        p: Probability distribution array (must sum to 1).

    Returns:
        Shannon entropy value (>= 0).
    """
    mask: np.ndarray = p > 0  # type: ignore[type-arg]
    return -float(np.sum(p[mask] * np.log(p[mask])))


def _compute_permutation_entropy(  # noqa: PLR0914
    data: np.ndarray,  # type: ignore[type-arg]
    d: int,
    tau: int,
) -> tuple[float, float]:
    """Compute normalized permutation entropy and Jensen-Shannon complexity.

    Implements the Bandt & Pompe (2002) ordinal pattern method.

    Args:
        data: 1-D array of observations (e.g. log returns).
        d: Embedding dimension (number of elements per ordinal pattern).
        tau: Time delay between consecutive elements in patterns.

    Returns:
        Tuple of ``(normalized_entropy, js_complexity)``.
    """
    n: int = len(data)
    n_factorial: int = math.factorial(d)
    max_entropy: float = math.log(n_factorial)

    # Number of valid patterns
    pattern_length: int = (d - 1) * tau + 1
    n_patterns: int = n - pattern_length + 1

    if n_patterns <= 0 or max_entropy == 0.0:
        return 1.0, 0.0

    # Extract ordinal patterns: rank order via argsort of argsort
    # Build index array for subsequences with delay tau
    indices: np.ndarray = np.arange(d)[np.newaxis, :] * tau + np.arange(n_patterns)[:, np.newaxis]  # type: ignore[type-arg]
    subsequences: np.ndarray = data[indices]  # type: ignore[type-arg]

    # Convert each subsequence to its ordinal rank pattern
    ranks: np.ndarray = np.argsort(np.argsort(subsequences, axis=1), axis=1)  # type: ignore[type-arg]

    # Convert rank tuples to unique integers for counting
    # Use a base-(d) positional encoding: sum(rank[i] * d^i)
    multipliers: np.ndarray = np.array([d**i for i in range(d)], dtype=np.int64)  # type: ignore[type-arg]
    pattern_ids: np.ndarray = ranks @ multipliers  # type: ignore[type-arg]

    # Count occurrences of each pattern
    unique_ids: np.ndarray  # type: ignore[type-arg]
    counts: np.ndarray  # type: ignore[type-arg]
    unique_ids, counts = np.unique(pattern_ids, return_counts=True)

    # Probability distribution over observed patterns
    p: np.ndarray = counts.astype(np.float64) / n_patterns  # type: ignore[type-arg]

    # Shannon entropy of observed distribution
    h: float = _shannon_entropy(p)
    h_norm: float = h / max_entropy if max_entropy > 0 else 1.0
    h_norm = max(0.0, min(1.0, h_norm))

    # Jensen-Shannon complexity
    uniform: np.ndarray = np.full(n_factorial, 1.0 / n_factorial, dtype=np.float64)  # type: ignore[type-arg]

    # Pad observed distribution to full d! support
    p_full: np.ndarray = np.zeros(n_factorial, dtype=np.float64)  # type: ignore[type-arg]
    # Map unique_ids back to indices in the full distribution
    # Since pattern_ids are unique integers, use them directly
    # But they may exceed n_factorial range; we need a mapping
    # Actually the pattern_ids encode ordinal ranks, and each is unique among d! permutations
    # We can use a lookup for all d! permutations
    all_perms: np.ndarray = np.array(  # type: ignore[type-arg]
        [np.argsort(np.argsort(perm)) @ multipliers for perm in _all_permutations(d)],
        dtype=np.int64,
    )
    perm_to_idx: dict[int, int] = {int(pid): idx for idx, pid in enumerate(all_perms)}
    for uid, cnt in zip(unique_ids, counts, strict=True):
        idx: int = perm_to_idx.get(int(uid), 0)
        p_full[idx] = float(cnt) / n_patterns

    # JS divergence: D_JS(p || u) = H((p+u)/2) - (H(p) + H(u))/2
    m: np.ndarray = (p_full + uniform) / 2.0  # type: ignore[type-arg]
    h_uniform: float = math.log(n_factorial)
    h_p_full: float = _shannon_entropy(p_full)
    h_m: float = _shannon_entropy(m)
    js_div: float = h_m - (h_p_full + h_uniform) / 2.0
    js_div = max(0.0, js_div)

    # Maximum JS divergence: between delta distribution and uniform
    delta: np.ndarray = np.zeros(n_factorial, dtype=np.float64)  # type: ignore[type-arg]
    delta[0] = 1.0
    m0: np.ndarray = (delta + uniform) / 2.0  # type: ignore[type-arg]
    h_delta: float = 0.0  # entropy of delta is 0
    h_m0: float = _shannon_entropy(m0)
    js_div_max: float = h_m0 - (h_delta + h_uniform) / 2.0
    js_div_max = max(js_div_max, 1e-15)  # avoid division by zero

    # Statistical complexity: C = (js_div / js_div_max) * H_norm
    js_complexity: float = (js_div / js_div_max) * h_norm
    js_complexity = max(0.0, js_complexity)

    return h_norm, js_complexity


def _all_permutations(d: int) -> list[list[int]]:
    """Generate all permutations of range(d).

    Args:
        d: Number of elements to permute.

    Returns:
        List of all d! permutations, each as a list of integers.
    """
    if d <= 1:
        return [[0]] if d == 1 else [[]]
    result: list[list[int]] = []
    _permute_helper(list(range(d)), 0, d, result)
    return result


def _permute_helper(
    arr: list[int],
    start: int,
    n: int,
    result: list[list[int]],
) -> None:
    """Recursive helper for generating permutations in-place.

    Args:
        arr: Current permutation being built.
        start: Index from which to start swapping.
        n: Total number of elements.
        result: Accumulator list for completed permutations.
    """
    if start == n:
        result.append(arr[:])
        return
    for i in range(start, n):
        arr[start], arr[i] = arr[i], arr[start]
        _permute_helper(arr, start + 1, n, result)
        arr[start], arr[i] = arr[i], arr[start]


def _compute_kish_neff(
    returns: np.ndarray,  # type: ignore[type-arg]
    max_lag_fraction: float,
) -> tuple[float, float]:
    """Compute Kish effective sample size from autocorrelation structure.

    Uses the Bartlett bandwidth truncation: sum stops when
    ``|rho_k| < 1.96 / sqrt(N)`` (insignificant at 5% level).

    Args:
        returns: 1-D array of return observations.
        max_lag_fraction: Maximum lag as fraction of N.

    Returns:
        Tuple of ``(n_eff, n_eff_ratio)`` where n_eff_ratio = n_eff / N.
    """
    n: int = len(returns)
    if n < _MIN_SAMPLES_ACF:
        return float(n), 1.0

    max_lag: int = max(1, int(n * max_lag_fraction))
    # Cap at n//2 - 1 to keep ACF well-defined
    max_lag = min(max_lag, n // 2 - 1)
    if max_lag < 1:
        return float(n), 1.0

    # Compute ACF (statsmodels returns values only when alpha is not specified)
    acf_values: np.ndarray = acf(returns, nlags=max_lag, fft=True, alpha=None)  # type: ignore[type-arg]

    # Bartlett significance threshold
    bartlett_threshold: float = 1.96 / math.sqrt(n)

    # Sum autocorrelations with Bartlett truncation
    acf_sum: float = 0.0
    for k in range(1, max_lag + 1):
        rho_k: float = float(acf_values[k])
        if abs(rho_k) < bartlett_threshold:
            break
        acf_sum += rho_k

    # Kish formula: N_eff = N / (1 + 2 * sum(rho_k))
    denominator: float = 1.0 + 2.0 * acf_sum
    if denominator <= 0:
        denominator = 1.0

    n_eff: float = float(n) / denominator
    # Clamp to [1, N]
    n_eff = max(1.0, min(float(n), n_eff))
    n_eff_ratio: float = n_eff / float(n)

    return n_eff, n_eff_ratio


def _compute_mde_da(
    n_eff: float,
    alpha: float,
    power: float,
) -> float:
    """Compute minimum detectable directional accuracy above 0.50.

    Uses normal approximation to the binomial distribution for
    a one-sided test.

    Args:
        n_eff: Effective sample size.
        alpha: Significance level.
        power: Statistical power.

    Returns:
        Minimum detectable DA (in [0.5, 1.0]).
    """
    if n_eff <= 0:
        return 1.0

    z_alpha: float = float(norm.ppf(1.0 - alpha))
    z_beta: float = float(norm.ppf(power))
    mde: float = (z_alpha + z_beta) / (2.0 * math.sqrt(n_eff))
    result: float = 0.5 + mde

    # Clamp to [0.5, 1.0]
    return max(0.5, min(1.0, result))


def _compute_breakeven_da(
    mean_abs_return: float,
    round_trip_cost: float,
) -> float:
    """Compute break-even directional accuracy from transaction costs.

    Solves: ``(2p - 1) * mean(|r_t|) = round_trip_cost`` for p.

    Args:
        mean_abs_return: Mean absolute return per bar.
        round_trip_cost: Round-trip transaction cost.

    Returns:
        Break-even DA (in [0.5, 1.0]).
    """
    if mean_abs_return <= 0:
        return 1.0

    p: float = 0.5 + round_trip_cost / (2.0 * mean_abs_return)

    # Clamp to [0.5, 1.0]
    return max(0.5, min(1.0, p))


def _compute_snr_r2(  # noqa: PLR0913, PLR0917
    features: np.ndarray,  # type: ignore[type-arg]
    target: np.ndarray,  # type: ignore[type-arg]
    holdout_fraction: float,
    ridge_alpha: float,
    n_noise_baselines: int,
    seed: int,
) -> tuple[float, float]:
    """Compute adjusted R-squared from Ridge regression on temporal holdout.

    Fits Ridge on the first ``(1 - holdout_fraction)`` of the data,
    evaluates on the last ``holdout_fraction``.  Compares against
    random Gaussian features as a noise baseline.

    Args:
        features: Feature matrix of shape ``(n_samples, n_features)``.
        target: Target array of shape ``(n_samples,)``.
        holdout_fraction: Fraction of data for temporal holdout.
        ridge_alpha: Ridge regularization parameter.
        n_noise_baselines: Number of random-feature baseline runs.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(adjusted_r2, mean_noise_r2)``.
    """
    n: int = len(target)
    n_features: int = features.shape[1]
    split_idx: int = int(n * (1.0 - holdout_fraction))

    if split_idx < _MIN_SPLIT_SAMPLES or (n - split_idx) < _MIN_SPLIT_SAMPLES or n_features < _MIN_FEATURES:
        return 0.0, 0.0

    # Temporal split
    x_train: np.ndarray = features[:split_idx]  # type: ignore[type-arg]
    x_test: np.ndarray = features[split_idx:]  # type: ignore[type-arg]
    y_train: np.ndarray = target[:split_idx]  # type: ignore[type-arg]
    y_test: np.ndarray = target[split_idx:]  # type: ignore[type-arg]

    n_test: int = len(y_test)

    # Fit real features
    adj_r2: float = _fit_ridge_adj_r2(x_train, y_train, x_test, y_test, ridge_alpha, n_features, n_test)

    # Noise baselines
    rng: np.random.Generator = np.random.default_rng(seed)
    noise_r2_values: list[float] = []
    for _ in range(n_noise_baselines):
        noise_train: np.ndarray = rng.normal(0, 1, size=(split_idx, n_features))  # type: ignore[type-arg]
        noise_test: np.ndarray = rng.normal(0, 1, size=(n_test, n_features))  # type: ignore[type-arg]
        noise_r2: float = _fit_ridge_adj_r2(noise_train, y_train, noise_test, y_test, ridge_alpha, n_features, n_test)
        noise_r2_values.append(noise_r2)

    mean_noise_r2: float = float(np.mean(noise_r2_values))

    return adj_r2, mean_noise_r2


def _fit_ridge_adj_r2(  # noqa: PLR0913, PLR0917
    x_train: np.ndarray,  # type: ignore[type-arg]
    y_train: np.ndarray,  # type: ignore[type-arg]
    x_test: np.ndarray,  # type: ignore[type-arg]
    y_test: np.ndarray,  # type: ignore[type-arg]
    ridge_alpha: float,
    n_features: int,
    n_test: int,
) -> float:
    """Fit Ridge regression and compute adjusted R-squared on test set.

    Args:
        x_train: Training feature matrix.
        y_train: Training target array.
        x_test: Test feature matrix.
        y_test: Test target array.
        ridge_alpha: Ridge regularization parameter.
        n_features: Number of features (for adjusted R² denominator).
        n_test: Number of test observations.

    Returns:
        Adjusted R-squared value (can be negative on holdout).
    """
    model: Ridge = Ridge(alpha=ridge_alpha)
    model.fit(x_train, y_train)

    y_pred: np.ndarray = model.predict(x_test)  # type: ignore[type-arg]

    ss_res: float = float(np.sum((y_test - y_pred) ** 2))
    ss_tot: float = float(np.sum((y_test - np.mean(y_test)) ** 2))

    if ss_tot == 0:
        return 0.0

    r2: float = 1.0 - ss_res / ss_tot

    # Adjusted R²: penalize for number of features
    denominator: int = n_test - n_features - 1
    if denominator <= 0:
        return r2

    adj_r2: float = 1.0 - (1.0 - r2) * (n_test - 1) / denominator

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, adj_r2))


# ---------------------------------------------------------------------------
# Public analyzer class
# ---------------------------------------------------------------------------


class PredictabilityAnalyzer:
    """Stateless service for predictability assessment.

    Computes permutation entropy (Bandt & Pompe), Kish effective sample
    size, minimum detectable effect for directional accuracy, break-even DA
    from transaction costs, and signal-to-noise ratio via Ridge regression.

    Tier gating:
        - **Tier A/B:** Permutation entropy, Kish N_eff, MDE DA, breakeven DA.
        - **Tier A only (+ features):** SNR R-squared from Ridge.
        - **Tier C:** All analysis fields are None.
    """

    def analyze(  # noqa: PLR6301, PLR0913, PLR0917, PLR0914
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: PredictabilityConfig | None = None,
        features: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> PredictabilityProfile:
        """Compute a full predictability assessment profile.

        Args:
            returns: Pandas Series of log returns (NaN-free).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``).
            tier: Sample-size tier controlling analysis depth.
            config: Predictability analysis configuration.  Uses defaults
                when ``None``.
            features: Optional feature matrix ``(n_samples, n_features)``
                for SNR R-squared computation (Tier A only).

        Returns:
            Frozen ``PredictabilityProfile`` value object.
        """
        if config is None:
            config = PredictabilityConfig()

        n_obs: int = len(returns)
        logger.debug(
            "Analysing predictability: asset={}, bar_type={}, tier={}, n={}",
            asset,
            bar_type,
            tier.value,
            n_obs,
        )

        # Insufficient data — return minimal profile
        if n_obs < config.min_samples_predictability:
            logger.debug(
                "Insufficient samples ({} < {}), returning minimal profile",
                n_obs,
                config.min_samples_predictability,
            )
            return PredictabilityProfile(
                asset=asset,
                bar_type=bar_type,
                tier=tier,
                n_observations=n_obs,
            )

        # Tier C — no analysis
        if tier == SampleTier.C:
            return PredictabilityProfile(
                asset=asset,
                bar_type=bar_type,
                tier=tier,
                n_observations=n_obs,
            )

        returns_arr: np.ndarray = returns.dropna().to_numpy(dtype=np.float64)  # type: ignore[type-arg]
        n_clean: int = len(returns_arr)

        if n_clean < config.min_samples_predictability:
            return PredictabilityProfile(
                asset=asset,
                bar_type=bar_type,
                tier=tier,
                n_observations=n_obs,
            )

        # Permutation entropy (Tier A/B)
        pe_results: list[PermutationEntropyResult] = []
        for d in config.pe_dimensions:
            h_norm: float
            js_c: float
            h_norm, js_c = _compute_permutation_entropy(returns_arr, d, config.pe_delay)
            pe_results.append(
                PermutationEntropyResult(
                    dimension=d,
                    normalized_entropy=h_norm,
                    js_complexity=js_c,
                )
            )
        logger.debug("Permutation entropy computed for {} dimensions", len(pe_results))

        # Kish effective sample size (Tier A/B)
        n_eff: float
        n_eff_ratio: float
        n_eff, n_eff_ratio = _compute_kish_neff(returns_arr, config.bartlett_max_lag_fraction)
        logger.debug("Kish N_eff={:.1f}, ratio={:.3f}", n_eff, n_eff_ratio)

        # Minimum detectable effect for directional accuracy (Tier A/B)
        mde_da: float = _compute_mde_da(n_eff, config.alpha, config.power)
        logger.debug("MDE DA={:.4f}", mde_da)

        # Break-even directional accuracy (Tier A/B)
        mean_abs_ret: float = float(np.mean(np.abs(returns_arr)))
        breakeven_da: float = _compute_breakeven_da(mean_abs_ret, config.round_trip_cost)
        logger.debug("Break-even DA={:.4f}", breakeven_da)

        # Signal-to-noise ratio (Tier A only, requires features)
        snr_r2: float | None = None
        snr_r2_noise: float | None = None
        is_predictable: bool | None = None

        if tier == SampleTier.A and features is not None and len(features) >= n_clean:
            # Align features with cleaned returns
            features_aligned: np.ndarray = features[:n_clean]  # type: ignore[type-arg]
            if features_aligned.shape[0] >= config.min_samples_predictability and features_aligned.shape[1] >= 1:
                snr_r2, snr_r2_noise = _compute_snr_r2(
                    features_aligned,
                    returns_arr,
                    config.snr_holdout_fraction,
                    config.snr_ridge_alpha,
                    config.snr_n_noise_baselines,
                    seed=42,
                )
                is_predictable = snr_r2 > snr_r2_noise
                logger.debug(
                    "SNR R²={:.4f}, noise baseline={:.4f}, predictable={}",
                    snr_r2,
                    snr_r2_noise,
                    is_predictable,
                )

        return PredictabilityProfile(
            asset=asset,
            bar_type=bar_type,
            tier=tier,
            n_observations=n_obs,
            permutation_entropies=tuple(pe_results),
            n_eff=n_eff,
            n_eff_ratio=n_eff_ratio,
            mde_da=mde_da,
            breakeven_da=breakeven_da,
            snr_r2=snr_r2,
            snr_r2_noise_baseline=snr_r2_noise,
            is_predictable_vs_noise=is_predictable,
        )
