"""RC2 research checkpoint value objects -- feature rationale, stationarity summary, and section models.

Provides domain models for RC2 notebook sections that are NOT part of the profiling
module (which owns stationarity testing infrastructure).  These value objects capture
the *thesis-level* interpretation layer: why each feature was chosen, how stationarity
results map to downstream validity, and cross-section summary statistics.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator


class StationarityExpectation(StrEnum):
    """A priori stationarity expectation for a feature.

    Assigned BEFORE running ADF/KPSS tests, based on the feature's
    mathematical construction.  Used to interpret surprises in Section 2.

    Attributes:
        STATIONARY: Expected stationary by construction (log ratios, z-scores, bounded oscillators).
        NON_STATIONARY: Expected non-stationary (absolute price units, cumulative sums).
        BORDERLINE: Stationarity depends on data characteristics (bar type, volume trends).
    """

    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    BORDERLINE = "borderline"


class FeatureRationale(BaseModel, frozen=True):
    """A priori rationale for a single feature in the RC2 feature rationale table.

    Documents the economic justification, stationarity expectation, and
    transformation plan for each feature BEFORE observing any RC2 results.
    This is the core of the Section 3 "examiner-proofing" strategy.

    Attributes:
        feature_name: Column name as produced by ``compute_all_indicators``
            (e.g. ``"logret_1"``, ``"atr_14"``).
        group: Feature group label (returns, volatility, momentum, volume, statistical).
        formula_summary: One-line description of the computation
            (e.g. ``"ln(C_t / C_{t-1})"``).
        economic_rationale: Why this feature should predict returns or volatility.
            Must be stated a priori, not derived from results.
        stationarity_expectation: Whether we EXPECT the feature to be stationary
            before running any tests.
        stationarity_reasoning: Brief mathematical argument for the expectation.
        transformation_if_nonstationary: Pre-registered transformation to apply
            if the feature fails stationarity screening.  ``None`` for features
            expected to be stationary.
        literature_reference: Optional reference supporting the feature's use
            (e.g. ``"Amihud (2002)"``).
    """

    feature_name: str
    group: str
    formula_summary: str
    economic_rationale: str
    stationarity_expectation: StationarityExpectation
    stationarity_reasoning: str
    transformation_if_nonstationary: str | None = None
    literature_reference: str | None = None


class StationaritySurprise(BaseModel, frozen=True):
    """A feature whose observed stationarity classification disagrees with the a priori expectation.

    Surprises are documented in Section 2 to demonstrate intellectual honesty
    and trigger the decision tree from the pre-registration.

    Attributes:
        feature_name: Column name of the surprising feature.
        expected: What we predicted before testing.
        observed_classification: What ADF+KPSS actually reported.
        asset: Asset where the surprise was observed.
        bar_type: Bar type where the surprise was observed.
        explanation: Post-hoc explanation (if any) for the discrepancy.
        action_taken: What was done in response (transform, drop, keep with caveat).
        is_post_hoc: Whether the action was pre-registered (False) or post-hoc (True).
    """

    feature_name: str
    expected: StationarityExpectation
    observed_classification: str
    asset: str
    bar_type: str
    explanation: str
    action_taken: str
    is_post_hoc: bool


class StationaritySectionSummary(BaseModel, frozen=True):
    """Aggregate summary for RC2 Section 2 (Stationarity Report).

    Provides the numbers needed for the "Therefore" paragraph and for the
    Go/No-Go decision table in Section 8.

    Attributes:
        n_features_total: Total features tested.
        n_stationary: Features classified as stationary across all combinations.
        n_trend_stationary: Features classified as trend-stationary.
        n_unit_root: Features classified as having a unit root.
        n_inconclusive: Features with inconclusive ADF+KPSS results.
        n_expected_nonstationary: Features expected non-stationary a priori.
        n_surprises: Features where observed != expected.
        surprises: Detailed records of each surprise.
        pass_rate: Fraction of features that are stationary (n_stationary / n_features_total).
        all_expected_nonstationary_confirmed: Whether every expected-non-stationary feature
            was indeed non-stationary (no false expectations).
        downstream_validity: Whether stationarity results support valid MI/Ridge analysis.
    """

    n_features_total: Annotated[int, PydanticField(ge=0)]
    n_stationary: Annotated[int, PydanticField(ge=0)]
    n_trend_stationary: Annotated[int, PydanticField(ge=0)]
    n_unit_root: Annotated[int, PydanticField(ge=0)]
    n_inconclusive: Annotated[int, PydanticField(ge=0)]
    n_expected_nonstationary: Annotated[int, PydanticField(ge=0)]
    n_surprises: Annotated[int, PydanticField(ge=0)]
    surprises: tuple[StationaritySurprise, ...]
    pass_rate: Annotated[float, PydanticField(ge=0, le=1)]
    all_expected_nonstationary_confirmed: bool
    downstream_validity: bool

    @model_validator(mode="after")
    def _counts_sum_to_total(self) -> Self:
        """Ensure classification counts sum to total features tested.

        Returns:
            Validated instance.

        Raises:
            ValueError: If the counts do not sum to ``n_features_total``.
        """
        total_from_counts: int = self.n_stationary + self.n_trend_stationary + self.n_unit_root + self.n_inconclusive
        if total_from_counts != self.n_features_total:
            msg: str = (
                f"Classification counts ({total_from_counts}) must sum to n_features_total ({self.n_features_total})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _surprise_count_matches(self) -> Self:
        """Ensure n_surprises matches the length of the surprises tuple.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``n_surprises`` does not equal ``len(surprises)``.
        """
        if self.n_surprises != len(self.surprises):
            msg: str = f"n_surprises ({self.n_surprises}) must equal len(surprises) ({len(self.surprises)})"
            raise ValueError(msg)
        return self


def build_default_feature_rationales() -> tuple[FeatureRationale, ...]:
    """Build the complete feature rationale table for the default 23-feature set.

    Each rationale is derived from the feature's mathematical definition in
    ``src/app/features/application/indicators.py`` and the financial literature.
    This function is called at RC2 notebook initialization, BEFORE any
    stationarity or MI results are computed.

    Returns:
        Tuple of 23 FeatureRationale objects covering all indicator groups.
    """
    rationales: list[FeatureRationale] = [
        # ----- Returns (4) -----
        FeatureRationale(
            feature_name="logret_1",
            group="returns",
            formula_summary="ln(C_t / C_{t-1})",
            economic_rationale=(
                "1-bar momentum: captures immediate price direction. "
                "Serial correlation in short-horizon returns is the basis "
                "for momentum strategies."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="First difference of I(1) log prices is I(0) by definition.",
        ),
        FeatureRationale(
            feature_name="logret_4",
            group="returns",
            formula_summary="ln(C_t / C_{t-4})",
            economic_rationale=(
                "4-bar momentum: captures multi-bar trend direction. "
                "Longer horizon reduces noise relative to logret_1."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Sum of 4 stationary 1-bar returns. Overlapping induces "
                "MA(3) structure but does not affect stationarity."
            ),
        ),
        FeatureRationale(
            feature_name="logret_12",
            group="returns",
            formula_summary="ln(C_t / C_{t-12})",
            economic_rationale=(
                "12-bar momentum: captures medium-term trend. On dollar bars (~2-3/day), this spans roughly 4-6 days."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Same argument as logret_4 with MA(11) structure.",
        ),
        FeatureRationale(
            feature_name="logret_24",
            group="returns",
            formula_summary="ln(C_t / C_{t-24})",
            economic_rationale=(
                "24-bar momentum: captures longer-term trend strength. Spans roughly 8-12 days on dollar bars."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Same argument with MA(23) structure.",
        ),
        # ----- Volatility (6) -----
        FeatureRationale(
            feature_name="rv_12",
            group="volatility",
            formula_summary="rolling_std(logret_1, window=12)",
            economic_rationale=(
                "Short-window realized volatility: captures recent volatility regime. "
                "High volatility regimes may precede mean-reversion opportunities."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Rolling std of stationary returns. Bounded below by 0, "
                "no unit root. May have long memory but is I(0)."
            ),
        ),
        FeatureRationale(
            feature_name="rv_24",
            group="volatility",
            formula_summary="rolling_std(logret_1, window=24)",
            economic_rationale=(
                "Medium-window realized volatility: smoother volatility estimate spanning ~8-12 days on dollar bars."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Same as rv_12 with wider window.",
        ),
        FeatureRationale(
            feature_name="rv_48",
            group="volatility",
            formula_summary="rolling_std(logret_1, window=48)",
            economic_rationale=(
                "Long-window realized volatility: captures the background volatility regime over ~16-24 days."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Same as rv_12 with wider window.",
        ),
        FeatureRationale(
            feature_name="gk_vol_24",
            group="volatility",
            formula_summary="sqrt(rolling_mean(GK_var, 24)); GK_var uses ln(H/L) and ln(C/O)",
            economic_rationale=(
                "Garman-Klass volatility: more efficient than close-to-close vol "
                "because it uses full OHLC information. Detects intrabar volatility "
                "that realized vol misses."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "GK variance uses log ratios (ln(H/L), ln(C/O)) which are "
                "scale-free. Rolling mean of stationary quantities is stationary."
            ),
            literature_reference="Garman & Klass (1980)",
        ),
        FeatureRationale(
            feature_name="park_vol_24",
            group="volatility",
            formula_summary="sqrt(rolling_mean(ln(H/L)^2 / 4ln2, 24))",
            economic_rationale=(
                "Parkinson volatility: uses high-low range which captures "
                "intrabar extremes. Complements GK for a more robust vol estimate."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Uses only ln(H/L) -- a log ratio, hence stationary.",
            literature_reference="Parkinson (1980)",
        ),
        FeatureRationale(
            feature_name="atr_14",
            group="volatility",
            formula_summary="ewm_mean(TR, alpha=1/14); TR = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)",
            economic_rationale=(
                "Average True Range: measures typical price range per bar in "
                "absolute units. Used for position sizing and stop-loss calibration."
            ),
            stationarity_expectation=StationarityExpectation.NON_STATIONARY,
            stationarity_reasoning=(
                "TR is in absolute price units (dollars). Scales linearly "
                "with price level. When BTC moves from $20K to $60K, ATR triples."
            ),
            transformation_if_nonstationary="pct_atr (ATR / close)",
        ),
        # ----- Momentum (5) -----
        FeatureRationale(
            feature_name="ema_xover_8_21",
            group="momentum",
            formula_summary="(EMA_8(close) - EMA_21(close)) / (ATR_14 + eps)",
            economic_rationale=(
                "EMA crossover normalized by ATR: captures trend direction "
                "and strength in a scale-free way. Positive = uptrend, "
                "negative = downtrend."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Numerator: mean-reverting spread of EMAs. Denominator: ATR "
                "normalizes for scale. Result is dimensionless and bounded."
            ),
        ),
        FeatureRationale(
            feature_name="rsi_14",
            group="momentum",
            formula_summary="100 - 100 / (1 + ewm(gains) / ewm(losses))",
            economic_rationale=(
                "RSI: measures overbought/oversold conditions. Extreme RSI "
                "values may signal mean-reversion opportunities."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Bounded oscillator [0, 100]. Cannot have a unit root.",
            literature_reference="Wilder (1978)",
        ),
        FeatureRationale(
            feature_name="roc_1",
            group="momentum",
            formula_summary="C_t / C_{t-1} - 1",
            economic_rationale=(
                "1-bar rate of change: simple percentage return. "
                "Approximates logret_1 for small returns but captures "
                "the asymmetry of arithmetic returns."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=("Equivalent to exp(logret_1) - 1. Stationary by same argument as log returns."),
        ),
        FeatureRationale(
            feature_name="roc_4",
            group="momentum",
            formula_summary="C_t / C_{t-4} - 1",
            economic_rationale="4-bar rate of change: medium-term momentum signal.",
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Multi-period ratio minus 1. Stationary.",
        ),
        FeatureRationale(
            feature_name="roc_12",
            group="momentum",
            formula_summary="C_t / C_{t-12} - 1",
            economic_rationale="12-bar rate of change: longer-term momentum signal.",
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Same as roc_1/roc_4 at longer horizon.",
        ),
        # ----- Volume (3) -----
        FeatureRationale(
            feature_name="vol_zscore_24",
            group="volume",
            formula_summary="(V_t - mean(V, 24)) / (std(V, 24) + eps)",
            economic_rationale=(
                "Volume z-score: detects unusual volume activity relative "
                "to recent history. Volume spikes often precede or confirm "
                "directional moves."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Z-score is self-normalizing. Even if raw volume trends, "
                "the z-score measures local deviation. Oscillates around 0."
            ),
        ),
        FeatureRationale(
            feature_name="amihud_24",
            group="volume",
            formula_summary="rolling_mean(|r_t| / (C_t * V_t + eps), 24)",
            economic_rationale=(
                "Amihud illiquidity ratio: measures price impact per unit "
                "of trading volume. Higher values indicate less liquidity "
                "and potentially larger adverse selection risk."
            ),
            stationarity_expectation=StationarityExpectation.BORDERLINE,
            stationarity_reasoning=(
                "Denominator is dollar volume (C_t * V_t). On dollar bars, "
                "dollar volume per bar is approximately constant -> Amihud "
                "is approximately |r_t|/const -> stationary. On time bars, "
                "dollar volume trends with price -> non-stationary."
            ),
            transformation_if_nonstationary="rolling_zscore",
            literature_reference="Amihud (2002)",
        ),
        FeatureRationale(
            feature_name="obv_slope_14",
            group="volume",
            formula_summary="OLS_slope(cumsum(sign(delta_C) * V), window=14)",
            economic_rationale=(
                "OBV slope: captures the trend in buying/selling pressure. "
                "Rising OBV slope = accumulation; falling = distribution."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "OBV is cumulative (non-stationary) but the slope over a "
                "fixed window is stationary -- equivalent to differencing."
            ),
        ),
        # ----- Statistical (5) -----
        FeatureRationale(
            feature_name="ret_zscore_24",
            group="statistical",
            formula_summary="(logret_1_t - mean(logret_1, 24)) / (std(logret_1, 24) + eps)",
            economic_rationale=(
                "Return z-score: identifies extreme returns relative to "
                "recent volatility. Can signal mean-reversion or trend "
                "continuation depending on magnitude."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning="Z-score of stationary series is stationary.",
        ),
        FeatureRationale(
            feature_name="bbpctb_20_2.0",
            group="statistical",
            formula_summary="(C - Lower) / (Upper - Lower + eps)",
            economic_rationale=(
                "Bollinger %B: measures where price sits within the "
                "Bollinger Bands. Near 0 = oversold; near 1 = overbought."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=("Bounded oscillator approximately [0, 1]. Same argument as RSI."),
        ),
        FeatureRationale(
            feature_name="bbwidth_20_2.0",
            group="statistical",
            formula_summary="2 * k * std(C, 20) / (mean(C, 20) + eps)",
            economic_rationale=(
                "Bollinger Bands Width: measures volatility expansion and "
                "contraction. Narrow bands often precede breakouts (Bollinger squeeze)."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Formula is effectively the coefficient of variation (CV) "
                "of close prices. CV is scale-free and stationary if the "
                "return distribution is stationary."
            ),
        ),
        FeatureRationale(
            feature_name="slope_14",
            group="statistical",
            formula_summary="OLS_slope(close, window=14)",
            economic_rationale=(
                "Price slope: measures the linear trend direction and rate of price change over recent bars."
            ),
            stationarity_expectation=StationarityExpectation.NON_STATIONARY,
            stationarity_reasoning=(
                "Fits OLS to raw close prices. Slope has units of $/bar and scales linearly with price level."
            ),
            transformation_if_nonstationary="normalized_slope (slope / close)",
        ),
        FeatureRationale(
            feature_name="hurst_100",
            group="statistical",
            formula_summary="R/S regression slope; H in [0, 1]",
            economic_rationale=(
                "Hurst exponent: detects trending (H > 0.5), mean-reverting "
                "(H < 0.5), or random walk (H ~ 0.5) behavior. Informs "
                "whether momentum or mean-reversion strategies are appropriate."
            ),
            stationarity_expectation=StationarityExpectation.STATIONARY,
            stationarity_reasoning=(
                "Bounded [0, 1] by construction (clamped in code). "
                "R/S analysis is internally scale-invariant (R/S ratio)."
            ),
        ),
    ]

    return tuple(rationales)


def count_expected_nonstationary(rationales: tuple[FeatureRationale, ...]) -> int:
    """Count features with non-stationary or borderline stationarity expectation.

    Args:
        rationales: Feature rationale table.

    Returns:
        Number of features expected to be non-stationary or borderline.
    """
    count: int = sum(
        1
        for r in rationales
        if r.stationarity_expectation in {StationarityExpectation.NON_STATIONARY, StationarityExpectation.BORDERLINE}
    )
    return count


def get_features_needing_transformation(
    rationales: tuple[FeatureRationale, ...],
) -> tuple[FeatureRationale, ...]:
    """Filter to features with pre-registered transformations.

    Args:
        rationales: Feature rationale table.

    Returns:
        Subset of rationales where ``transformation_if_nonstationary`` is not None.
    """
    return tuple(r for r in rationales if r.transformation_if_nonstationary is not None)
