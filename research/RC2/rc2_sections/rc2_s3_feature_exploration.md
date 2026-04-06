# RC2 Section 3: Feature Exploration & Validation

## Methodology

Section 3 has two parts:

**Part 1 (Exploration):** Display all 23 features with economic rationale,
correlation heatmap, VIF analysis, violin plots, scatter grids, and per-feature
validation details. Kept/dropped status is shown as a color overlay, not a filter
(preventing survivorship bias).

**Part 2 (Validation):** MI permutation tests, Ridge DA tables, temporal stability
heatmap, cross-bar-type MI comparison, holdout preview, and multi-horizon analysis.

## Part 1: Feature Characteristics

### Feature Groups (23 features)

| Group | N | Features |
|-------|---|----------|
| Returns | 4 | logret_1, logret_4, logret_12, logret_24 |
| Volatility | 6 | rv_12, rv_24, rv_48, gk_vol_24, park_vol_24, atr_14 |
| Momentum | 5 | ema_xover_8_21, rsi_14, roc_1, roc_4, roc_12 |
| Volume | 3 | vol_zscore_24, amihud_24, obv_slope_14 |
| Statistical | 5 | ret_zscore_24, bbpctb_20_2.0, bbwidth_20_2.0, slope_14, hurst_100 |

### VIF Analysis (Rule F3)

VIF (Variance Inflation Factor) measures multicollinearity. Per Rule F3, features
with VIF > 10 are flagged but **not dropped** -- Ridge regression handles collinearity,
and dropping features post-hoc would inflate the trial count.

**Expected high-VIF features:**
- rv_12, rv_24, rv_48 (realized volatility at different windows -- highly correlated by construction)
- logret_1, roc_1 (nearly identical: roc_1 = exp(logret_1) - 1)
- gk_vol_24, park_vol_24 (both range-based volatility estimators)

**Rationale for keeping collinear features:** In Ridge regression, collinear features
share coefficient weight without destabilizing the model. Removing them post-hoc based
on VIF would introduce a researcher degree of freedom (counted as a DSR trial). The
pre-registered rule is: diagnose but do not drop.

### Feature Distributions (Violin Plots)

Violin plots grouped by kept/dropped status show the distributional properties of all
23 features. Volatility features (rv_*, gk_vol, park_vol) show right-skewed
distributions consistent with the known positive skewness of volatility. Returns
features are approximately symmetric with fat tails (kurtosis > 3).

### Feature-Target Scatter Grid

Scatter plots of kept features vs fwd_logret_1 show no visible linear relationship --
consistent with the extremely weak MI values. Any predictive signal is buried in noise
and requires statistical tests to detect.

## Part 2: Validation Results

### Three-Gate Validation (BTCUSDT/dollar)

**Feature matrix:** 5,164 rows x 23 features + 5 targets.

| Gate | Test | Passing | Rate |
|------|------|---------|------|
| Gate 1 (MI) | MI permutation, 1,000 perms, BH correction | 8/23 | 34.8% |
| Gate 2 (DA) | Ridge DA, 500 perms, BH correction | 0/23 | 0.0% |
| Gate 3 (Stability) | MI sig in >= 50% of 4 windows | 7/23 | 30.4% |
| **All gates** | MI AND DA AND Stability | **0/23** | **0.0%** |

**Critical finding:** Zero features pass all three gates. The DA gate is the
bottleneck -- no feature's Ridge DA significantly exceeds the permutation null after
BH correction. The fallback mechanism (Rule F2) keeps the top 5 by composite score.

**Kept features (fallback):** amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48

### MI Results

**Significance:** 8/23 features after BH correction (34.8%).

**Effect size issue:** The MI/H(target) normalization produced extreme values because
the target's differential entropy is negative (H(target) = -2.5371 nats). This is
mathematically correct for a narrowly distributed continuous variable (differential
entropy, unlike discrete entropy, can be negative), but makes the percentage
interpretation meaningless. The raw MI values and p-values remain valid.

### Ridge DA Results

**Key numbers:**
- DA beats null: 0/23
- DA above break-even: 0/23
- Best: ret_zscore_24 at DA = 51.81% (+1.81 pp over random, -5.42 pp below break-even)
- Break-even DA: 57.23% (mean |r| = 0.013829, round-trip cost = 20 bps)

**Break-even DA formula:**
```
break_even_DA = 0.5 + round_trip_cost / (2 * mean(|r_t|))
             = 0.5 + 0.002 / (2 * 0.013829)
             = 0.5 + 0.0723
             = 0.5723
```

The high break-even DA is driven by the small mean absolute return per dollar bar
(1.38%). Each bar represents ~7-10 hours of price movement, and the 20 bps cost is
a significant fraction of the expected per-bar profit.

### Temporal Stability

**Per-window results:**

| Window | N Rows | MI-significant | Note |
|--------|--------|---------------|------|
| 2020-2021 | 188 | 0/23 | Very few dollar bars in this period |
| 2021-2022 | 1,155 | 0/23 | Bull market, low volatility returns |
| 2022-2023 | 1,196 | 8/23 | Bear/recovery market, high volatility |
| 2023-2024 | 912 | 0/23 | Consolidation period |

**All features classified UNSTABLE** (stability score 0.00-0.25).

**Interpretation:** MI significance is concentrated in the 2022-2023 window. This
is the period of the crypto bear market and early recovery, characterized by high
volatility and regime transitions. The absence of MI significance in other windows
means:

1. Signal is **regime-conditional** -- present during turbulent periods, absent during
   trending or quiet periods.
2. Any model trained on the full period will be diluted by noise from the
   non-informative windows.
3. Regime detection is essential for the recommendation system.

### Cross-Bar-Type MI Comparison

All bar types trigger the fallback mechanism for BTCUSDT. This confirms that the
weak signal is not an artifact of dollar-bar sampling -- it persists across all
bar types.

### Multi-Horizon Analysis

**Per-horizon kept features (all via fallback):**

| Horizon | Features | Economic Interpretation |
|---------|----------|----------------------|
| fwd_logret_1 (~8-12h) | amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48 | Pure volatility/liquidity |
| fwd_logret_4 (~1-2d) | amihud_24, gk_vol_24, logret_12, park_vol_24, roc_12 | Mixed: vol + medium-term momentum |
| fwd_logret_24 (~8-12d) | amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48 | Volatility + range-based estimators |

**Robustly informative (>= 2/3 horizons):** amihud_24, bbwidth_20_2.0, park_vol_24,
rv_24, rv_48.

**Key insight:** The medium-term horizon (fwd_logret_4) uniquely includes momentum
features (logret_12, roc_12), suggesting that directional information, while weak, is
concentrated at the 1-2 day time scale. Short-term and long-term horizons are
dominated purely by volatility features.

## Connection to Lopez de Prado

Lopez de Prado (2018, Ch. 8) emphasizes that feature importance should be assessed
with respect to economic significance, not just statistical significance. The RC2
validation framework operationalizes this by separating MI significance (statistical)
from DA vs break-even (economic). The zero-pass rate on the DA gate is exactly the
kind of result that a rigorous pipeline should produce: most features are noise.

The fallback mechanism (F2) is the safety valve that prevents a total feature vacuum.
Lopez de Prado would likely approve of this approach -- it is conservative (5 features
from 23), respects the pre-registration, and the selected features are from the
volatility group where financial theory predicts the strongest signal.

## Recommendations

1. **For Phase 9 (Classification):** Use all 23 features with strong regularization
   rather than the fallback 5. The classifier benefits from dimensionality even if
   individual features are weak.
2. **For Phase 10 (Regression):** Focus on the volatility features as the primary
   signal carriers. Consider predicting future realized volatility as the regression
   target.
3. **Investigate the 2022-2023 window anomaly.** Why is MI significant only during
   this period? Is it high volatility? A specific market event? Understanding this
   informs regime-conditional strategy design.
4. **Consider alternative MI estimators.** The block-permutation MI test may be
   conservative due to the small effective block count on dollar bars (~100 blocks
   of size 50 in 5,164 observations). Parametric or nearest-neighbor MI estimators
   could provide more power.
