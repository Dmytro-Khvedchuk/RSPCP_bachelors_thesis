"""RC2 pre-registration threshold utilities -- quantitative feasibility analysis.

Pure functions for computing the statistical and economic thresholds that
underpin the RC2 pre-registration document.  Every threshold is derived
from first principles with explicit formulas, not magic numbers.

Key concepts:
    - **Break-even DA**: The minimum directional accuracy needed to cover
      round-trip transaction costs.
    - **MDE DA**: The minimum detectable directional accuracy above 0.50
      given effective sample size and desired power.
    - **Feasibility gap**: The relationship between MDE DA and break-even
      DA -- are we powered to detect a profitable signal?
    - **Harvey threshold**: Multiple-testing-adjusted t-statistic required
      for feature significance (Harvey, Liu & Zhu 2016).
    - **Deflated Sharpe Ratio**: Trial-count-adjusted Sharpe significance
      (Bailey & Lopez de Prado 2014).

References:
    - Harvey, C., Liu, Y., Zhu, H. (2016). "...and the Cross-Section of
      Expected Returns." Review of Financial Studies, 29(1), 5-68.
    - Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe
      Ratio: Correcting for Selection Bias, Backtest Overfitting, and
      Non-Normality." Journal of Portfolio Management, 40(5), 94-107.
"""

from __future__ import annotations

import math

from pydantic import BaseModel
from scipy.stats import norm  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Value objects for threshold results
# ---------------------------------------------------------------------------


class BreakevenDAResult(BaseModel, frozen=True):
    """Result of break-even directional accuracy computation.

    Attributes:
        mean_abs_return: Mean absolute return per bar.
        round_trip_cost: Round-trip transaction cost.
        breakeven_da: Minimum DA to cover costs.
        required_edge_pp: Edge over 50% in percentage points.
    """

    mean_abs_return: float
    round_trip_cost: float
    breakeven_da: float
    required_edge_pp: float


class MDEResult(BaseModel, frozen=True):
    """Result of minimum detectable effect computation.

    Attributes:
        n_eff: Effective sample size used.
        alpha: Significance level.
        power: Statistical power.
        mde_da: Minimum detectable DA above 0.50.
        detectable_edge_pp: Detectable edge in percentage points.
    """

    n_eff: float
    alpha: float
    power: float
    mde_da: float
    detectable_edge_pp: float


class FeasibilityGapResult(BaseModel, frozen=True):
    """Assessment of the gap between statistical detectability and economic viability.

    Attributes:
        mde_da: Minimum detectable DA.
        breakeven_da: Break-even DA from costs.
        gap_pp: breakeven_da - mde_da in percentage points (positive = good).
        is_feasible: Whether MDE is below break-even (we can detect profitability).
        interpretation: Human-readable assessment.
    """

    mde_da: float
    breakeven_da: float
    gap_pp: float
    is_feasible: bool
    interpretation: str


class HarveyThresholdResult(BaseModel, frozen=True):
    """Multiple-testing-adjusted significance threshold.

    Attributes:
        n_tests: Number of simultaneous tests.
        alpha: Family-wise error rate.
        bonferroni_alpha: Per-test alpha after Bonferroni correction.
        bonferroni_t: Corresponding t-statistic threshold.
        holm_bonferroni_t: Holm-Bonferroni step-down threshold for the most significant test.
        harvey_t: Harvey et al. (2016) recommended threshold (t > 3.0).
    """

    n_tests: int
    alpha: float
    bonferroni_alpha: float
    bonferroni_t: float
    holm_bonferroni_t: float
    harvey_t: float


class DSRResult(BaseModel, frozen=True):
    """Deflated Sharpe Ratio result.

    Attributes:
        observed_sharpe: The observed (annualized) Sharpe ratio.
        n_trials: Number of strategy/parameter trials conducted.
        n_observations: Number of return observations.
        skewness: Skewness of returns (0 for normal).
        kurtosis: Excess kurtosis of returns (0 for normal).
        expected_max_sharpe: E[max(SR)] under the null across n_trials.
        dsr_pvalue: P-value for the deflated Sharpe ratio test.
        is_significant: Whether DSR p-value < 0.05.
    """

    observed_sharpe: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    expected_max_sharpe: float
    dsr_pvalue: float
    is_significant: bool


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_breakeven_da(
    mean_abs_return: float,
    round_trip_cost: float,
) -> BreakevenDAResult:
    """Compute break-even directional accuracy from transaction costs.

    The expected P&L per trade under a directional strategy is:
        E[PnL] = (2p - 1) * E[|r_t|] - c

    Setting E[PnL] = 0 and solving for p:
        p = 0.5 + c / (2 * E[|r_t|])

    Args:
        mean_abs_return: Mean absolute return per bar (e.g. 0.008).
        round_trip_cost: Round-trip transaction cost (e.g. 0.002 for 20bps).

    Returns:
        BreakevenDAResult with the break-even DA and required edge.
    """
    if mean_abs_return <= 0.0:
        return BreakevenDAResult(
            mean_abs_return=mean_abs_return,
            round_trip_cost=round_trip_cost,
            breakeven_da=1.0,
            required_edge_pp=50.0,
        )

    p: float = 0.5 + round_trip_cost / (2.0 * mean_abs_return)
    p = max(0.5, min(1.0, p))
    edge_pp: float = (p - 0.5) * 100.0

    return BreakevenDAResult(
        mean_abs_return=mean_abs_return,
        round_trip_cost=round_trip_cost,
        breakeven_da=p,
        required_edge_pp=edge_pp,
    )


def compute_mde_da(
    n_eff: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> MDEResult:
    """Compute minimum detectable directional accuracy above 0.50.

    Uses normal approximation to the binomial for a one-sided test
    (H0: p = 0.5 vs H1: p > 0.5):

        MDE = (z_alpha + z_beta) / (2 * sqrt(N_eff))
        MDE_DA = 0.5 + MDE

    Args:
        n_eff: Kish effective sample size.
        alpha: Significance level (one-sided).
        power: Desired statistical power (1 - beta).

    Returns:
        MDEResult with the minimum detectable DA.
    """
    if n_eff <= 0.0:
        return MDEResult(
            n_eff=n_eff,
            alpha=alpha,
            power=power,
            mde_da=1.0,
            detectable_edge_pp=50.0,
        )

    z_alpha: float = float(norm.ppf(1.0 - alpha))
    z_beta: float = float(norm.ppf(power))
    mde: float = (z_alpha + z_beta) / (2.0 * math.sqrt(n_eff))
    mde_da: float = 0.5 + mde
    mde_da = max(0.5, min(1.0, mde_da))
    edge_pp: float = mde * 100.0

    return MDEResult(
        n_eff=n_eff,
        alpha=alpha,
        power=power,
        mde_da=mde_da,
        detectable_edge_pp=edge_pp,
    )


def compute_feasibility_gap(
    mde_da: float,
    breakeven_da: float,
) -> FeasibilityGapResult:
    """Assess the gap between statistical detectability and economic viability.

    If breakeven_da > mde_da: we are well-powered.  We can detect an effect
    smaller than what is needed for profitability.

    If breakeven_da < mde_da: we are underpowered for economic significance.
    Even if a profitable signal exists, we may not have enough data to detect it.

    If breakeven_da ~= mde_da: marginal -- the smallest detectable effect is
    approximately the smallest profitable effect.

    Args:
        mde_da: Minimum detectable DA (from compute_mde_da).
        breakeven_da: Break-even DA (from compute_breakeven_da).

    Returns:
        FeasibilityGapResult with the gap analysis.
    """
    gap_pp: float = (breakeven_da - mde_da) * 100.0
    is_feasible: bool = mde_da < breakeven_da

    _margin_pp: float = 1.0  # 1 percentage point margin for "marginal"
    if gap_pp > _margin_pp:
        interpretation: str = (
            f"Well-powered: can detect {(mde_da - 0.5) * 100:.1f}pp edge but need "
            f"{(breakeven_da - 0.5) * 100:.1f}pp for profitability. "
            f"Gap of {gap_pp:.1f}pp provides comfortable margin."
        )
    elif gap_pp > -_margin_pp:
        interpretation = (
            f"Marginal: MDE ({(mde_da - 0.5) * 100:.1f}pp) is close to "
            f"break-even ({(breakeven_da - 0.5) * 100:.1f}pp). "
            f"Borderline statistical power for economic significance."
        )
    else:
        interpretation = (
            f"Underpowered: need {(breakeven_da - 0.5) * 100:.1f}pp edge for "
            f"profitability but can only detect {(mde_da - 0.5) * 100:.1f}pp. "
            f"Deficit of {abs(gap_pp):.1f}pp — need more data or larger effect."
        )

    return FeasibilityGapResult(
        mde_da=mde_da,
        breakeven_da=breakeven_da,
        gap_pp=gap_pp,
        is_feasible=is_feasible,
        interpretation=interpretation,
    )


def compute_harvey_threshold(
    n_tests: int,
    alpha: float = 0.05,
) -> HarveyThresholdResult:
    """Compute multiple-testing-adjusted significance thresholds.

    Provides three perspectives on the required t-statistic:
    1. Bonferroni: alpha_adj = alpha / n_tests (most conservative).
    2. Holm-Bonferroni: step-down procedure; threshold for the most
       significant test is alpha / n_tests (same as Bonferroni for rank 1).
    3. Harvey et al. (2016): empirical recommendation of t > 3.0 for
       published factors, accounting for data-snooping in the literature.

    Args:
        n_tests: Total number of simultaneous hypothesis tests.
        alpha: Family-wise error rate.

    Returns:
        HarveyThresholdResult with all three thresholds.
    """
    if n_tests <= 0:
        n_tests = 1

    bonferroni_alpha: float = alpha / n_tests
    # Clamp to avoid ppf(1.0) = inf
    bonferroni_alpha = max(bonferroni_alpha, 1e-15)
    bonferroni_t: float = float(norm.ppf(1.0 - bonferroni_alpha / 2.0))

    # Holm-Bonferroni for rank-1 test is identical to Bonferroni
    holm_alpha_rank1: float = alpha / n_tests
    holm_alpha_rank1 = max(holm_alpha_rank1, 1e-15)
    holm_t: float = float(norm.ppf(1.0 - holm_alpha_rank1 / 2.0))

    # Harvey et al. (2016) empirical threshold
    harvey_t: float = 3.0

    return HarveyThresholdResult(
        n_tests=n_tests,
        alpha=alpha,
        bonferroni_alpha=bonferroni_alpha,
        bonferroni_t=bonferroni_t,
        holm_bonferroni_t=holm_t,
        harvey_t=harvey_t,
    )


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio p-value (Bailey & Lopez de Prado 2014).

    The DSR accounts for:
    1. Multiple testing via the expected maximum Sharpe across n_trials.
    2. Non-normality of returns (skewness and kurtosis).
    3. Finite sample length n_observations.

    The expected maximum Sharpe ratio under the null (all trials have SR=0)
    uses the approximation from Euler-Mascheroni:
        E[max(SR)] ~= sqrt(V[SR]) * (z_alpha + gamma / z_alpha)

    where z_alpha = Phi^{-1}(1 - 1/n_trials) and gamma ~= 0.5772 (Euler).

    The variance of the Sharpe ratio estimator (Lo 2002):
        V[SR] = (1 + 0.5 * SR^2 - skew * SR + (kurt/4) * SR^2) / T

    Under the null (SR=0):
        V[SR|H0] = 1/T

    Args:
        observed_sharpe: Best observed (annualized) Sharpe ratio.
        n_trials: Number of strategy configurations tested.
        n_observations: Number of return observations (T in the formulas).
        skewness: Sample skewness of returns.
        kurtosis: Sample excess kurtosis of returns.

    Returns:
        DSRResult with the deflated p-value.
    """
    n_trials = max(n_trials, 1)
    _min_obs: int = 2
    if n_observations < _min_obs:
        return DSRResult(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
            expected_max_sharpe=0.0,
            dsr_pvalue=1.0,
            is_significant=False,
        )

    # Expected max SR under the null (all SR = 0) -- Euler-Mascheroni approx
    # Using E[max(Z_1, ..., Z_n)] for standard normals
    t: int = n_observations
    euler_mascheroni: float = 0.5772156649
    if n_trials == 1:
        expected_max_sr: float = 0.0
    else:
        z_n: float = float(norm.ppf(1.0 - 1.0 / n_trials))
        expected_max_sr = math.sqrt(1.0 / t) * (z_n + euler_mascheroni / z_n) if z_n > 0 else 0.0

    # Variance of the SR estimator (Lo 2002, accounting for non-normality)
    sr: float = observed_sharpe
    sr_var_numerator: float = 1.0 + 0.5 * sr**2 - skewness * sr + (kurtosis / 4.0) * sr**2
    sr_variance: float = sr_var_numerator / t
    sr_std: float = math.sqrt(max(sr_variance, 1e-30))

    # DSR test statistic
    dsr_statistic: float = (sr - expected_max_sr) / sr_std

    # One-sided p-value (H1: SR > E[max(SR)])
    dsr_pvalue: float = 1.0 - float(norm.cdf(dsr_statistic))
    dsr_pvalue = max(0.0, min(1.0, dsr_pvalue))

    _dsr_alpha: float = 0.05

    return DSRResult(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        expected_max_sharpe=expected_max_sr,
        dsr_pvalue=dsr_pvalue,
        is_significant=dsr_pvalue < _dsr_alpha,
    )


def compute_required_n_eff_for_breakeven(
    breakeven_da: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the N_eff required to detect the break-even DA with desired power.

    Inverts the MDE formula:
        N_eff = ((z_alpha + z_beta) / (2 * (breakeven_da - 0.5)))^2

    This answers: "How many effective samples do I need to have an 80%
    chance of detecting a signal that is just barely profitable?"

    Args:
        breakeven_da: Break-even directional accuracy (from compute_breakeven_da).
        alpha: Significance level.
        power: Desired power.

    Returns:
        Required effective sample size. Returns inf if breakeven_da <= 0.5.
    """
    edge: float = breakeven_da - 0.5
    if edge <= 0.0:
        return float("inf")

    z_alpha: float = float(norm.ppf(1.0 - alpha))
    z_beta: float = float(norm.ppf(power))
    n_required: float = ((z_alpha + z_beta) / (2.0 * edge)) ** 2

    return n_required


def compute_mi_significance_threshold(
    target_entropy: float,
    min_fraction: float = 0.01,
) -> float:
    """Compute minimum MI for economic significance as a fraction of target entropy.

    A feature's MI with the target should exceed some fraction of H(target)
    to be considered informative.  This is more meaningful than an absolute
    MI threshold because it scales with target uncertainty.

    For binary direction targets, H(target) = log(2) ~= 0.693 nats when
    balanced.  1% of that is ~0.007 nats.

    Args:
        target_entropy: Shannon entropy of the target variable (nats).
        min_fraction: Minimum MI/H(target) ratio to declare significance.

    Returns:
        Minimum MI threshold (in nats).
    """
    if target_entropy <= 0.0:
        return 0.0

    return target_entropy * min_fraction


def compute_vif_threshold(
    n_samples: int,
    n_features: int,
    conservative: bool = True,
) -> float:
    """Determine the appropriate VIF threshold given sample and feature counts.

    The standard VIF > 10 threshold (Belsley, Kuh & Welsch 1980) assumes
    large samples.  For smaller samples or many features relative to N,
    a more conservative VIF > 5 is advisable to prevent unstable coefficient
    estimates.

    Rule of thumb: use VIF > 5 when N/p < 50 (conservative) or VIF > 10
    when N/p >= 50 (standard).

    Args:
        n_samples: Number of observations.
        n_features: Number of features.
        conservative: If True, always use VIF > 5.

    Returns:
        VIF threshold (5.0 or 10.0).
    """
    if conservative:
        return 5.0

    if n_features <= 0:
        return 10.0

    ratio: float = n_samples / n_features
    _n_to_p_standard_threshold: float = 50.0
    if ratio < _n_to_p_standard_threshold:
        return 5.0

    return 10.0


def compute_stability_threshold(
    n_windows: int,
    min_fraction: float = 0.50,
) -> int:
    """Compute minimum number of temporal windows a feature must be significant in.

    A feature that is only significant in 1 out of 4 temporal windows is
    likely an artifact.  We require significance in at least ``min_fraction``
    of windows.

    Args:
        n_windows: Total number of temporal validation windows.
        min_fraction: Minimum fraction of windows requiring significance.

    Returns:
        Minimum number of significant windows (at least 1).
    """
    if n_windows <= 0:
        return 1

    threshold: int = max(1, math.ceil(n_windows * min_fraction))
    return threshold


# ---------------------------------------------------------------------------
# Composite analysis: run all thresholds for a given scenario
# ---------------------------------------------------------------------------


class RC2ThresholdSummary(BaseModel, frozen=True):
    """Complete RC2 pre-registration threshold analysis for one asset-bar combination.

    Attributes:
        asset: Trading pair symbol.
        bar_type: Bar aggregation type.
        n_bars: Raw bar count.
        n_eff: Effective sample size.
        mean_abs_return: Mean absolute return per bar.
        breakeven: Break-even DA result.
        mde: MDE DA result.
        feasibility: Feasibility gap assessment.
        harvey: Harvey multiple-testing threshold.
        vif_threshold: Selected VIF threshold.
        stability_min_windows: Minimum significant temporal windows.
        n_eff_required_for_breakeven: N_eff needed to detect break-even DA.
    """

    asset: str
    bar_type: str
    n_bars: int
    n_eff: float
    mean_abs_return: float
    breakeven: BreakevenDAResult
    mde: MDEResult
    feasibility: FeasibilityGapResult
    harvey: HarveyThresholdResult
    vif_threshold: float
    stability_min_windows: int
    n_eff_required_for_breakeven: float


def compute_rc2_thresholds(  # noqa: PLR0913, PLR0917
    asset: str,
    bar_type: str,
    n_bars: int,
    n_eff: float,
    mean_abs_return: float,
    round_trip_cost: float = 0.002,
    n_features: int = 23,
    n_bar_types: int = 5,
    n_horizons: int = 3,
    alpha: float = 0.05,
    power: float = 0.80,
    n_temporal_windows: int = 4,
) -> RC2ThresholdSummary:
    """Compute all RC2 pre-registration thresholds for one asset-bar combination.

    This is the main entry point that orchestrates all individual threshold
    computations and packages them into a single summary.

    Args:
        asset: Trading pair symbol (e.g. "BTCUSDT").
        bar_type: Bar aggregation type (e.g. "dollar").
        n_bars: Raw bar count.
        n_eff: Kish effective sample size.
        mean_abs_return: Mean absolute log return per bar.
        round_trip_cost: Round-trip transaction cost.
        n_features: Number of features being tested.
        n_bar_types: Number of bar types.
        n_horizons: Number of forecast horizons.
        alpha: Significance level.
        power: Desired statistical power.
        n_temporal_windows: Number of temporal validation windows.

    Returns:
        RC2ThresholdSummary with all computed thresholds.
    """
    breakeven: BreakevenDAResult = compute_breakeven_da(mean_abs_return, round_trip_cost)
    mde: MDEResult = compute_mde_da(n_eff, alpha, power)
    feasibility: FeasibilityGapResult = compute_feasibility_gap(mde.mde_da, breakeven.breakeven_da)

    n_tests: int = n_features * n_bar_types * n_horizons
    harvey: HarveyThresholdResult = compute_harvey_threshold(n_tests, alpha)

    vif_threshold: float = compute_vif_threshold(n_bars, n_features, conservative=True)
    stability_min_windows: int = compute_stability_threshold(n_temporal_windows)

    n_eff_required: float = compute_required_n_eff_for_breakeven(breakeven.breakeven_da, alpha, power)

    return RC2ThresholdSummary(
        asset=asset,
        bar_type=bar_type,
        n_bars=n_bars,
        n_eff=n_eff,
        mean_abs_return=mean_abs_return,
        breakeven=breakeven,
        mde=mde,
        feasibility=feasibility,
        harvey=harvey,
        vif_threshold=vif_threshold,
        stability_min_windows=stability_min_windows,
        n_eff_required_for_breakeven=n_eff_required,
    )
