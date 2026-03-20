"""RC2 Section 3 feature rationale table -- economic justifications for all 23 features.

Provides a pure function that returns a complete feature rationale DataFrame
for the RC2 pre-registration notebook.  Every feature has:

- Economic intuition (WHY it should predict future returns)
- Literature reference (WHO established this relationship)
- Expected sign vs forward returns (direction of the relationship)
- Feature group classification
- Stationarity expectation (whether the raw feature is expected to be I(0))
- VIF expectation (collinearity cluster it belongs to)

This module lives in the ML-research path (Pandas) per CLAUDE.md.

The rationale table serves three thesis purposes:
    1. **Pre-registration defense**: every feature has an a priori economic
       justification documented BEFORE seeing results (Nosek et al., 2018).
    2. **Examiner defense**: addresses the "data-mined features" attack by
       showing each feature has grounding in market microstructure or
       financial econometrics literature.
    3. **Feature elimination rationale**: when features are dropped by the
       three-gate validation, the rationale table explains what economic
       hypothesis was tested and falsified.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


class FeatureRationale(BaseModel, frozen=True):
    """Economic rationale for a single feature.

    Documents the a priori justification, expected relationship with
    the target, and metadata for collinearity / stationarity analysis.

    Attributes:
        feature_name: Exact column name produced by IndicatorConfig defaults.
        group: Feature group (returns, volatility, momentum, volume, statistical).
        economic_intuition: One-sentence explanation of WHY this feature
            should predict forward returns.
        literature_ref: Short citation establishing the economic relationship.
        expected_sign: Expected sign of correlation with forward log returns
            ("positive", "negative", "ambiguous", or "unsigned").
        sign_rationale: Brief explanation of the expected sign direction.
        stationarity_expectation: Whether the raw feature is expected to be
            stationary ("stationary", "likely_stationary", "likely_non_stationary").
        vif_cluster: Name of the collinearity cluster this feature belongs to
            (features in the same cluster are expected to have high mutual VIF).
        is_transformation_based: Whether the feature is already a transform
            of prices (returns, z-scores) vs a level-based indicator.
    """

    feature_name: str
    group: str
    economic_intuition: str
    literature_ref: str
    expected_sign: str
    sign_rationale: str
    stationarity_expectation: str
    vif_cluster: str
    is_transformation_based: bool


# ---------------------------------------------------------------------------
# Feature rationale definitions
# ---------------------------------------------------------------------------


def _build_rationale_entries() -> tuple[FeatureRationale, ...]:
    """Construct the complete set of feature rationales for all 23 features.

    Returns:
        Tuple of FeatureRationale entries, one per feature, in the order
        they are produced by the default IndicatorConfig.
    """
    return (
        # === RETURNS ===
        FeatureRationale(
            feature_name="logret_1",
            group="returns",
            economic_intuition=(
                "1-bar micro-momentum captures short-term serial correlation "
                "in returns; positive autocorrelation implies continuation, "
                "negative implies mean reversion."
            ),
            literature_ref="Jegadeesh & Titman (1993); Lo & MacKinlay (1988)",
            expected_sign="ambiguous",
            sign_rationale=(
                "Positive if momentum dominates (trend continuation), "
                "negative if mean reversion dominates. Crypto empirically "
                "shows weak short-term momentum (Baur & Dimpfl, 2021)."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_short",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="logret_4",
            group="returns",
            economic_intuition=(
                "4-bar momentum captures intraday trend persistence; "
                "for dollar bars producing ~2-3 bars/day, this spans "
                "roughly 1-2 days of price action."
            ),
            literature_ref="Jegadeesh & Titman (1993); Moskowitz et al. (2012)",
            expected_sign="ambiguous",
            sign_rationale=(
                "Positive under momentum hypothesis (winners keep winning), "
                "negative under short-term reversal. Time-scale dependent."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_short",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="logret_12",
            group="returns",
            economic_intuition=(
                "12-bar momentum captures multi-day trend; at ~2-3 bars/day "
                "this spans roughly 4-6 days, the timescale where crypto "
                "trend-following strategies typically operate."
            ),
            literature_ref="Moskowitz et al. (2012); Bianchi et al. (2020)",
            expected_sign="positive",
            sign_rationale=(
                "Medium-term crypto returns show weak positive autocorrelation "
                "(trend continuation) per time-series momentum literature."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_medium",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="logret_24",
            group="returns",
            economic_intuition=(
                "24-bar daily trend captures the full daily return cycle; "
                "at ~2-3 bars/day this spans roughly 8-12 days. Tests "
                "whether longer-horizon momentum predicts next-bar direction."
            ),
            literature_ref="Asness et al. (2013); Liu et al. (2022)",
            expected_sign="positive",
            sign_rationale=(
                "Longer-horizon momentum is more robust in equities; "
                "crypto evidence is mixed but slightly positive "
                "(Bianchi et al., 2020)."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_medium",
            is_transformation_based=True,
        ),
        # === VOLATILITY ===
        FeatureRationale(
            feature_name="rv_12",
            group="volatility",
            economic_intuition=(
                "Short-window realized volatility captures recent turbulence; "
                "high RV often precedes both large moves (vol clustering) "
                "and mean reversion (volatility-adjusted position sizing)."
            ),
            literature_ref="Andersen et al. (2003); Bollerslev et al. (2018)",
            expected_sign="unsigned",
            sign_rationale=(
                "RV predicts magnitude (|fwd_logret|) not direction. "
                "Correlation with signed returns is ambiguous. "
                "Useful for SIZE track, not SIDE track directly."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="rv_24",
            group="volatility",
            economic_intuition=(
                "Medium-window realized volatility smooths the RV estimate; "
                "more stable than rv_12 and captures the daily volatility cycle."
            ),
            literature_ref="Andersen et al. (2003); Corsi (2009) HAR-RV model",
            expected_sign="unsigned",
            sign_rationale=(
                "Same as rv_12: predicts magnitude, not direction. "
                "The HAR-RV model uses multiple RV windows as complementary "
                "predictors of future volatility."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="rv_48",
            group="volatility",
            economic_intuition=(
                "Longer-window realized volatility captures the weekly "
                "volatility regime; less noisy than rv_12/rv_24 but "
                "slower to react to regime changes."
            ),
            literature_ref="Corsi (2009) HAR-RV; Bollerslev et al. (2018)",
            expected_sign="unsigned",
            sign_rationale=(
                "Same as rv_12/rv_24. The three RV windows together span the HAR-RV decomposition (fast/medium/slow)."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="gk_vol_24",
            group="volatility",
            economic_intuition=(
                "Garman-Klass volatility uses OHLC data (not just close) "
                "for a more efficient volatility estimate; captures "
                "intrabar price excursions that close-to-close misses."
            ),
            literature_ref="Garman & Klass (1980)",
            expected_sign="unsigned",
            sign_rationale=(
                "More efficient estimator of true volatility than "
                "close-to-close RV. Predicts magnitude, not direction."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="park_vol_24",
            group="volatility",
            economic_intuition=(
                "Parkinson volatility uses only high-low range; more "
                "robust to bid-ask bounce than close-based estimators "
                "and captures the true range of price discovery."
            ),
            literature_ref="Parkinson (1980)",
            expected_sign="unsigned",
            sign_rationale=(
                "High-low range estimator. Predicts magnitude, not "
                "direction. Expected to be highly correlated with "
                "gk_vol_24 (same OHLC information, different formula)."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="atr_14",
            group="volatility",
            economic_intuition=(
                "Average True Range measures absolute price movement "
                "per bar including gaps; widely used in practice for "
                "stop-loss sizing and volatility-adjusted position sizing."
            ),
            literature_ref="Wilder (1978); widespread practitioner use",
            expected_sign="unsigned",
            sign_rationale=(
                "ATR is in price units (not returns), so its level "
                "depends on the asset price. Predicts volatility regime "
                "but not direction. Note: ATR is NOT scale-invariant "
                "and may have stationarity issues for trending assets."
            ),
            stationarity_expectation="likely_non_stationary",
            vif_cluster="volatility",
            is_transformation_based=False,
        ),
        # === MOMENTUM ===
        FeatureRationale(
            feature_name="ema_xover_8_21",
            group="momentum",
            economic_intuition=(
                "ATR-normalized EMA crossover captures trend direction "
                "and strength, scaled by current volatility. Positive "
                "values indicate uptrend (fast EMA > slow EMA)."
            ),
            literature_ref=("Appel (1979) MACD concept; Zakamulin (2014) on optimality of moving average rules"),
            expected_sign="positive",
            sign_rationale=(
                "Positive crossover (uptrend) should predict positive "
                "forward returns under the momentum hypothesis. "
                "ATR normalization makes this comparable across regimes."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="momentum",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="rsi_14",
            group="momentum",
            economic_intuition=(
                "RSI measures the ratio of recent gains to losses; "
                "extreme values (>70 or <30) signal overbought/oversold "
                "conditions that may precede mean reversion."
            ),
            literature_ref="Wilder (1978); widespread practitioner use",
            expected_sign="negative",
            sign_rationale=(
                "High RSI -> overbought -> expect negative forward "
                "returns (mean reversion). However, in strong trends, "
                "high RSI can persist (trend continuation). The net "
                "effect depends on regime."
            ),
            stationarity_expectation="stationary",
            vif_cluster="momentum",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="roc_1",
            group="momentum",
            economic_intuition=(
                "1-bar rate of change is nearly identical to logret_1 "
                "for small returns; included for completeness but expected "
                "to be highly collinear with logret_1."
            ),
            literature_ref="Jegadeesh & Titman (1993)",
            expected_sign="ambiguous",
            sign_rationale=(
                "Same dynamics as logret_1. ROC = P_t/P_{t-1} - 1, "
                "while logret_1 = ln(P_t/P_{t-1}). For small returns "
                "these are nearly identical (Taylor expansion)."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_short",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="roc_4",
            group="momentum",
            economic_intuition=(
                "4-bar rate of change captures short-term price momentum; "
                "slightly different from logret_4 due to linear vs log "
                "scaling, but expected to carry similar information."
            ),
            literature_ref="Moskowitz et al. (2012)",
            expected_sign="ambiguous",
            sign_rationale=(
                "Same dynamics as logret_4. The difference between "
                "ROC and log return grows for larger price changes, "
                "which may capture asymmetry in crypto moves."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_short",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="roc_12",
            group="momentum",
            economic_intuition=(
                "12-bar rate of change captures medium-term trend; "
                "the non-log version emphasizes large absolute price "
                "changes more than logret_12."
            ),
            literature_ref="Asness et al. (2013)",
            expected_sign="positive",
            sign_rationale=(
                "Same positive momentum expectation as logret_12, with slightly different emphasis on large moves."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_medium",
            is_transformation_based=True,
        ),
        # === VOLUME ===
        FeatureRationale(
            feature_name="vol_zscore_24",
            group="volume",
            economic_intuition=(
                "Volume z-score measures whether current volume is "
                "abnormally high or low relative to its recent history. "
                "High volume confirms trend moves; low volume signals "
                "potential trend exhaustion."
            ),
            literature_ref="Karpoff (1987); Llorente et al. (2002)",
            expected_sign="ambiguous",
            sign_rationale=(
                "High volume + positive return -> continuation (informed "
                "buying). High volume + negative return -> continuation "
                "(informed selling). The sign depends on the interaction "
                "with return direction, not volume alone."
            ),
            stationarity_expectation="stationary",
            vif_cluster="volume",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="obv_slope_14",
            group="volume",
            economic_intuition=(
                "OBV slope captures the trend in accumulation/distribution; "
                "positive slope means net buying pressure is increasing, "
                "suggesting price is being supported by volume."
            ),
            literature_ref="Granville (1963); Lo & Wang (2000)",
            expected_sign="positive",
            sign_rationale=(
                "Positive OBV slope -> net accumulation -> expect "
                "positive forward returns. This is the classic "
                "'volume leads price' hypothesis."
            ),
            stationarity_expectation="likely_non_stationary",
            vif_cluster="volume_trend",
            is_transformation_based=False,
        ),
        FeatureRationale(
            feature_name="amihud_24",
            group="volume",
            economic_intuition=(
                "Amihud illiquidity ratio measures price impact per "
                "unit of dollar volume; higher values indicate less "
                "liquid conditions where moves are larger per trade."
            ),
            literature_ref="Amihud (2002)",
            expected_sign="unsigned",
            sign_rationale=(
                "High illiquidity predicts larger absolute moves "
                "(illiquidity premium) but not direction. In crypto, "
                "high Amihud during sell-offs can predict continued "
                "downside (liquidity spiral)."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="liquidity",
            is_transformation_based=True,
        ),
        # === STATISTICAL ===
        FeatureRationale(
            feature_name="ret_zscore_24",
            group="statistical",
            economic_intuition=(
                "Return z-score measures how extreme the current return "
                "is relative to recent history; extreme z-scores suggest "
                "overreaction, which may trigger mean reversion."
            ),
            literature_ref="DeBondt & Thaler (1985); Jegadeesh (1990)",
            expected_sign="negative",
            sign_rationale=(
                "High positive z-score -> overreaction -> expect "
                "negative forward returns (mean reversion). This is "
                "the short-term reversal effect, well-documented in "
                "equities and increasingly in crypto."
            ),
            stationarity_expectation="stationary",
            vif_cluster="returns_short",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="bbpctb_20_2.0",
            group="statistical",
            economic_intuition=(
                "Bollinger %B shows where price sits within the "
                "Bollinger Bands (0 = lower band, 1 = upper band). "
                "Values near 0 or 1 indicate potential reversal zones."
            ),
            literature_ref="Bollinger (2002); Leung & Straus (2024)",
            expected_sign="negative",
            sign_rationale=(
                "High %B (near upper band) -> overbought -> expect "
                "negative forward returns. Classic mean-reversion "
                "signal. Effectiveness depends on whether the market "
                "is ranging or trending."
            ),
            stationarity_expectation="stationary",
            vif_cluster="mean_reversion",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="bbwidth_20_2.0",
            group="statistical",
            economic_intuition=(
                "Bollinger Band Width measures normalized price "
                "volatility; narrow bands (squeeze) often precede "
                "explosive moves, wide bands indicate extended volatility."
            ),
            literature_ref="Bollinger (2002)",
            expected_sign="unsigned",
            sign_rationale=(
                "BBWidth predicts MAGNITUDE of future moves (volatility "
                "clustering) but not direction. Narrow bands -> expect "
                "larger moves in either direction. Useful for SIZE track."
            ),
            stationarity_expectation="likely_stationary",
            vif_cluster="volatility",
            is_transformation_based=True,
        ),
        FeatureRationale(
            feature_name="slope_14",
            group="statistical",
            economic_intuition=(
                "Rolling linear regression slope of close prices captures "
                "the local price trend direction and magnitude; positive "
                "slope indicates uptrend."
            ),
            literature_ref="Brown et al. (1998); Zakamulin (2014)",
            expected_sign="positive",
            sign_rationale=(
                "Positive slope -> uptrend -> expect positive forward "
                "returns under momentum hypothesis. Note: slope is in "
                "price units and may be non-stationary for trending assets."
            ),
            stationarity_expectation="likely_non_stationary",
            vif_cluster="momentum",
            is_transformation_based=False,
        ),
        FeatureRationale(
            feature_name="hurst_100",
            group="statistical",
            economic_intuition=(
                "Hurst exponent via R/S analysis measures the degree "
                "of trending vs mean-reverting behavior over the "
                "estimation window. H>0.5 indicates persistence (trend), "
                "H<0.5 indicates anti-persistence (mean reversion)."
            ),
            literature_ref=("Hurst (1951); Mandelbrot (1971); Cont (2001) on stylized facts of financial returns"),
            expected_sign="ambiguous",
            sign_rationale=(
                "Hurst does not predict return DIRECTION but informs "
                "which STRATEGY is appropriate: momentum for H>0.5, "
                "mean reversion for H<0.5. A meta-feature for the "
                "recommendation system (regime indicator)."
            ),
            stationarity_expectation="stationary",
            vif_cluster="regime",
            is_transformation_based=True,
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_rationale_table() -> pd.DataFrame:
    """Build the complete feature rationale table for RC2 Section 3.

    Returns a Pandas DataFrame with one row per feature (23 features total,
    matching the default ``IndicatorConfig``).  The DataFrame is ready for
    notebook rendering via ``display()`` or Bokeh/matplotlib tables.

    Columns:
        - feature_name: Exact column name from indicators.py
        - group: Feature group (returns, volatility, momentum, volume, statistical)
        - economic_intuition: A priori economic justification
        - literature_ref: Citation(s) establishing the relationship
        - expected_sign: Expected correlation sign with forward returns
        - sign_rationale: Explanation of the expected sign
        - stationarity_expectation: Expected stationarity classification
        - vif_cluster: Collinearity cluster for VIF interpretation
        - is_transformation_based: Whether feature is a transform of prices

    Returns:
        Pandas DataFrame with 23 rows and 9 columns, indexed by feature_name.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    rows: list[dict[str, object]] = [entry.model_dump() for entry in entries]
    df: pd.DataFrame = pd.DataFrame(rows)
    return df.set_index("feature_name", drop=False)


def get_feature_rationale(feature_name: str) -> FeatureRationale | None:
    """Look up the rationale for a single feature by name.

    Args:
        feature_name: Exact column name (e.g. "logret_1", "rv_24").

    Returns:
        The matching ``FeatureRationale``, or ``None`` if the feature
        is not in the pre-registered set.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    for entry in entries:
        if entry.feature_name == feature_name:
            return entry
    return None


def get_group_rationales(group: str) -> list[FeatureRationale]:
    """Return all rationales for features in a given group.

    Args:
        group: Group name (one of: returns, volatility, momentum,
            volume, statistical).

    Returns:
        List of matching ``FeatureRationale`` entries (empty if no match).
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    return [entry for entry in entries if entry.group == group]


def get_expected_vif_clusters() -> dict[str, list[str]]:
    """Return the expected VIF collinearity clusters.

    Features within the same cluster are expected to have high
    mutual VIF scores.  This provides an a priori expectation
    against which observed VIF results can be compared.

    Returns:
        Mapping from cluster name to list of feature names.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    clusters: dict[str, list[str]] = {}
    for entry in entries:
        if entry.vif_cluster not in clusters:
            clusters[entry.vif_cluster] = []
        clusters[entry.vif_cluster].append(entry.feature_name)
    return clusters


def get_all_feature_names() -> tuple[str, ...]:
    """Return all 23 feature names in canonical order.

    Returns:
        Tuple of feature names matching the default IndicatorConfig output.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    return tuple(entry.feature_name for entry in entries)


def build_sign_expectation_summary() -> pd.DataFrame:
    """Build a summary of expected signs for thesis narrative.

    Groups features by their expected sign and counts them.
    Useful for the "Therefore" paragraph in Section 3.

    Returns:
        DataFrame with columns: expected_sign, count, features.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    sign_groups: dict[str, list[str]] = {}
    for entry in entries:
        if entry.expected_sign not in sign_groups:
            sign_groups[entry.expected_sign] = []
        sign_groups[entry.expected_sign].append(entry.feature_name)

    rows: list[dict[str, object]] = [
        {
            "expected_sign": sign,
            "count": len(features),
            "features": ", ".join(features),
        }
        for sign, features in sorted(sign_groups.items())
    ]
    return pd.DataFrame(rows)


def build_vif_expectation_table() -> pd.DataFrame:
    """Build a table of expected VIF clusters for thesis interpretation.

    Returns:
        DataFrame with columns: vif_cluster, n_features, features,
        expected_high_vif (whether the cluster is expected to show VIF > 5).
    """
    clusters: dict[str, list[str]] = get_expected_vif_clusters()

    rows: list[dict[str, object]] = [
        {
            "vif_cluster": cluster_name,
            "n_features": len(features),
            "features": ", ".join(features),
            "expected_high_vif": len(features) > 1,
        }
        for cluster_name, features in sorted(clusters.items())
    ]
    return pd.DataFrame(rows)


def generate_section3_therefore() -> str:
    """Generate the programmatic 'Therefore' conclusion for RC2 Section 3.

    This paragraph connects the feature rationale to the thesis argument.
    It is structured as:
    1. State what the features collectively test.
    2. Acknowledge the DA gap between single-feature and break-even.
    3. Argue why ML combination (Phase 9-12) is needed.
    4. Connect to the recommender system value proposition.

    Returns:
        Multi-line markdown string for the Section 3 conclusion.
    """
    entries: tuple[FeatureRationale, ...] = _build_rationale_entries()
    n_total: int = len(entries)

    n_positive: int = sum(1 for e in entries if e.expected_sign == "positive")
    n_negative: int = sum(1 for e in entries if e.expected_sign == "negative")
    n_ambiguous: int = sum(1 for e in entries if e.expected_sign == "ambiguous")
    n_unsigned: int = sum(1 for e in entries if e.expected_sign == "unsigned")

    n_stationary: int = sum(1 for e in entries if e.stationarity_expectation == "stationary")
    n_transform: int = sum(1 for e in entries if not e.is_transformation_based)

    clusters: dict[str, list[str]] = get_expected_vif_clusters()
    n_multicol_clusters: int = sum(1 for c in clusters.values() if len(c) > 1)

    lines: list[str] = [
        "**Therefore (Section 3 -- Feature Exploration):**\n",
        f"The {n_total} pre-registered features test five distinct economic hypotheses "
        "about crypto returns: momentum (returns and momentum indicators), "
        "volatility clustering (6 volatility measures), volume confirmation "
        "(3 volume features), statistical structure (Bollinger, Hurst), and "
        "mean reversion (z-scores, %B).\n",
        f"Of {n_total} features, {n_positive} have positive expected sign with forward "
        f"returns (momentum/trend), {n_negative} have negative expected sign "
        f"(mean reversion), {n_ambiguous} are ambiguous (regime-dependent), "
        f"and {n_unsigned} are unsigned (predict magnitude, not direction). "
        "This mix of directional hypotheses is by design: the recommendation "
        "system must recognize WHEN momentum works and WHEN mean reversion "
        "works, rather than betting on one regime.\n",
        f"Stationarity: {n_stationary}/{n_total} features are expected to be stationary "
        f"a priori (returns, z-scores, bounded oscillators). {n_transform} features "
        "are level-based (ATR, OBV slope, price slope) and may require "
        "transformation. Section 2 results will confirm or refute these "
        "expectations.\n",
        f"Collinearity: {n_multicol_clusters} VIF clusters are expected to show high "
        "internal collinearity (notably: 6 volatility measures, 4 short-term "
        "return variants). Per the pre-registration, VIF is diagnostic only -- "
        "Ridge regularization handles collinearity, and we do NOT drop features "
        "based on VIF alone.\n",
        "**The critical insight for the thesis:** No single feature is expected "
        "to achieve directional accuracy above the break-even threshold "
        "(~56-63% for dollar bars). The typical single-feature DA in crypto is "
        "51-53%. This gap between statistical significance and economic "
        "significance is the fundamental motivation for the ML recommendation "
        "system: by COMBINING features that capture different market conditions, "
        "the ensemble can achieve DA above break-even even when no individual "
        "feature can.\n",
        "This connects Section 3 to Phases 9-12: the feature rationale table "
        "defines WHAT hypotheses the models test, the validation pipeline "
        "(Phase 4D) determines WHICH features pass, and the recommendation "
        "system learns WHEN to weight each feature based on the current regime.",
    ]

    return "\n".join(lines)
