---
name: RC2 S3 Deep Feature Analysis
description: Deep analysis of feature exploration results -- economic intuitions, VIF interpretation, MI thresholds, DA gap, cross-bar hypotheses
type: reference
---

# RC2 Section 3: Deep Feature Analysis

## 1. Feature-by-Feature Economic Rationale

### Returns Group (4 features)
| Feature | Horizon | Economic Hypothesis | Expected Sign | Literature |
|---------|---------|---------------------|---------------|------------|
| logret_1 | ~8-12 hrs | Micro-momentum / short-term reversal | ambiguous | Jegadeesh & Titman (1993); Lo & MacKinlay (1988) |
| logret_4 | ~1-2 days | Intraday trend persistence | ambiguous | Moskowitz et al. (2012) |
| logret_12 | ~4-6 days | Medium-term momentum | positive | Bianchi et al. (2020) |
| logret_24 | ~8-12 days | Weekly trend | positive | Asness et al. (2013) |

Short-horizon returns (1, 4) have ambiguous sign because crypto alternates between momentum and mean reversion regimes. Longer horizons (12, 24) lean positive per time-series momentum literature, but the effect is weak in crypto (Baur & Dimpfl, 2021).

### Volatility Group (6 features)
| Feature | Estimator | Unique Information | Expected Sign |
|---------|-----------|-------------------|---------------|
| rv_12 | Close-to-close std | Fast volatility cycle | unsigned |
| rv_24 | Close-to-close std | Daily volatility | unsigned |
| rv_48 | Close-to-close std | Weekly volatility regime | unsigned |
| gk_vol_24 | Garman-Klass (OHLC) | Intrabar excursions | unsigned |
| park_vol_24 | Parkinson (HL) | True range breadth | unsigned |
| atr_14 | Wilder true range | Absolute price movement | unsigned |

All volatility features predict MAGNITUDE not DIRECTION. Relevant for the SIZE track. The HAR-RV decomposition (rv_12/24/48) captures multi-scale volatility persistence (Corsi, 2009). GK and Parkinson are more efficient estimators than close-to-close but highly correlated with each other (same window). ATR is in price units and will likely be non-stationary.

### Momentum Group (5 features)
| Feature | Signal Type | Expected Sign |
|---------|------------|---------------|
| ema_xover_8_21 | Trend direction (ATR-normalized) | positive |
| rsi_14 | Overbought/oversold | negative (mean reversion) |
| roc_1 | Nearly = logret_1 | ambiguous |
| roc_4 | Nearly = logret_4 | ambiguous |
| roc_12 | Nearly = logret_12 | positive |

Key tension: ema_xover and roc suggest momentum (positive sign), while rsi suggests mean reversion (negative sign). This is intentional -- different regimes favor different hypotheses.

roc_1 and roc_4 are nearly collinear with logret_1 and logret_4 (for small returns, ln(1+r) approx r). This redundancy is diagnostic (VIF section) but not a problem for Ridge.

### Volume Group (3 features)
| Feature | Hypothesis | Expected Sign |
|---------|-----------|---------------|
| vol_zscore_24 | Volume confirms moves | ambiguous (interaction with returns) |
| obv_slope_14 | Accumulation/distribution | positive |
| amihud_24 | Illiquidity premium | unsigned |

Volume features are most valuable as interaction terms. vol_zscore alone is ambiguous because high volume confirms both buy and sell moves. The direction depends on the return sign. OBV slope directly captures net buying pressure.

### Statistical Group (5 features)
| Feature | Signal Type | Expected Sign |
|---------|------------|---------------|
| ret_zscore_24 | Overreaction | negative (mean reversion) |
| bbpctb_20_2.0 | Band position | negative (mean reversion) |
| bbwidth_20_2.0 | Volatility squeeze | unsigned (magnitude) |
| slope_14 | Price trend direction | positive |
| hurst_100 | Regime indicator (trending vs mean-reverting) | ambiguous (meta-feature) |

hurst_100 is the most interesting feature conceptually: it does not predict return direction but tells the recommendation system WHICH strategy to deploy. H > 0.5 = momentum regime, H < 0.5 = mean reversion regime. This is the regime detection feature for the recommender.

## 2. VIF Interpretation Framework for Crypto

### Expected Collinearity Clusters
1. **Volatility cluster** (6 features): rv_12, rv_24, rv_48, gk_vol_24, park_vol_24, bbwidth_20_2.0. All measure the same underlying quantity (volatility) over similar windows. Expect VIF > 10 for all pairs.
2. **Short-term returns** (4 features): logret_1, roc_1, logret_4, roc_4, ret_zscore_24. logret_1 and roc_1 will have VIF approaching infinity (nearly identical).
3. **Medium-term returns** (2 features): logret_12, roc_12, logret_24. Moderate collinearity.
4. **Momentum** (2 features): ema_xover_8_21, slope_14. Both capture trend direction but via different methods.

### Why NOT to Drop High-VIF Features
Per the pre-registration: VIF is diagnostic, not a filter. Three reasons:

1. **Ridge handles collinearity**: The L2 penalty in Ridge regression distributes coefficient weight across correlated features, preventing the coefficient explosion that makes OLS unreliable with high VIF.

2. **Information is not redundant**: Even when VIF > 10, the 10% unique variance may contain the signal. gk_vol_24 captures intrabar excursions that rv_24 misses (close-to-close). Dropping one discards information.

3. **Feature elimination changes hypothesis space**: If we drop features based on VIF, we silently change which economic hypotheses the model tests. This adds +1 to the DSR trial count.

### What VIF > 10 Means Economically
VIF = 1/(1-R^2) where R^2 is from regressing feature j on all other features.
- VIF = 10 -> R^2 = 0.90 -> 90% of the feature's variance is explained by others -> only 10% unique information
- VIF = 50 -> R^2 = 0.98 -> only 2% unique information
- VIF = infinity -> perfect collinearity -> zero unique information

For our features: the volatility cluster will have VIF > 50, but this is EXPECTED and UNDERSTOOD. The thesis documents this a priori.

## 3. MI/H(target) Threshold Analysis

### Target Entropy for Crypto Returns
For continuous target fwd_logret_1 (approximately N(0, sigma^2)):
- Differential entropy: H = 0.5 * ln(2*pi*e*sigma^2) (in nats)
- For sigma ~ 0.01 (1% per bar): H ~ 0.5 * ln(2*pi*e*0.0001) ~ -3.18 nats
- Differential entropy is negative for continuous variables -- not directly comparable to MI

For practical purposes, use the DISCRETIZED direction target (sign of fwd_logret_1):
- If roughly balanced (p(+) ~ 0.50): H(direction) = ln(2) ~ 0.693 nats
- MI/H(target) = MI / 0.693

### MI Effect Size Thresholds
| MI/H(target) | Interpretation | Typical Example |
|--------------|---------------|-----------------|
| < 0.5% | Below noise floor of MI estimator | Random features |
| 0.5% - 1% | Marginally informative | Most crypto features |
| 1% - 3% | Weakly informative | Best single features (momentum) |
| 3% - 5% | Moderately informative | Exceptional; verify not spurious |
| > 5% | Strongly informative | Almost certainly leakage or artifact |

For our project: expect most features to be in the 0.5-2% range. MI > 3% should trigger scrutiny for look-ahead bias. The MI permutation test (Phase 4D) provides the statistical significance filter.

## 4. The DA Economic Significance Gap -- THE KEY THESIS INSIGHT

### The Numbers
From rc2_agent2_thresholds.md:
- Break-even DA for BTC dollar bars: ~0.625 (12.5pp edge over coin flip)
- Break-even DA for ETH dollar bars: ~0.600 (10.0pp edge)
- Typical single-feature DA in crypto: 0.51-0.53 (1-3pp edge)

### The Gap
- Best single-feature DA: ~0.53 (3pp edge)
- Required DA for profitability: ~0.60-0.63 (10-12.5pp edge)
- **Gap: ~7-10 percentage points**

### Why This Gap Exists
1. **Weak individual predictability**: Crypto returns are close to random (R5: Brownian noise). Individual features capture at most 1-3% of return variation.
2. **High transaction costs relative to signal**: At 20bps round-trip, the cost eats most of the expected gain from weak directional signals.
3. **Regime instability**: A feature that works in bull markets may anti-predict in bear markets, washing out in the full sample.

### Why This Gap Is THE Thesis Argument
The gap motivates the ENTIRE recommendation system:

"No single feature achieves economically significant directional accuracy. Therefore, the thesis proposes a ML recommendation system that COMBINES features, exploiting the fact that:
(a) different features capture different market conditions (diversification of hypotheses),
(b) the regime indicator (hurst_100) tells the model WHEN each hypothesis is relevant, and
(c) the meta-labeling framework allows the model to ABSTAIN when no feature combination exceeds the break-even threshold."

This is not a weakness -- it is the central finding that justifies the thesis.

### Connection to the "Abstain" Signal
When no feature combination exceeds break-even DA, the recommender produces a NO-TRADE signal. This is its most valuable output: it prevents trading during unprofitable regimes. The recommender's value is not in alpha generation but in LOSS AVOIDANCE.

## 5. Cross-Bar-Type Hypotheses

### Dollar Bars vs Time Bars
**Hypothesis**: Dollar bars show higher MI than time bars for momentum features.
**Rationale**: Dollar bars synchronize sampling with trading activity (Lopez de Prado, 2018). In periods of high activity, time bars undersample (missing structure); in quiet periods, they oversample (adding noise). Dollar bars maintain more consistent information content per bar.
**Expected result**: MI for momentum features (logret_*, roc_*, ema_xover) should be higher on dollar bars than time_1h bars.

### Imbalance Bars
**Hypothesis**: Imbalance bars show higher MI for volume features but insufficient N for reliable inference.
**Rationale**: Volume and dollar imbalance bars trigger when buying/selling pressure becomes asymmetric. They naturally capture order flow imbalance, so volume features should be more informative. However, N ~ 530 (dollar_imbalance) or N ~ 568 (volume_imbalance) is marginal for MI estimation.
**Expected result**: Higher MI but wider confidence intervals. Tier B treatment per pre-registration.

### What "Dollar > Time" Would Mean for the Thesis
If confirmed: "Information-driven sampling (Lopez de Prado, 2018) extracts structure from crypto markets that uniform time sampling misses. This supports the use of dollar bars as the primary sampling method for the recommendation system."

If NOT confirmed: "The sampling method does not significantly affect feature informativeness, suggesting that the signal is robust to the observation frequency. Time bars may be preferred for their simplicity."

Both outcomes are valid thesis findings.

## 6. The "Therefore" Paragraph for Section 3

**Therefore (Section 3 -- Feature Exploration):**

The 23 pre-registered features test five distinct economic hypotheses about crypto returns: momentum (returns and momentum indicators), volatility clustering (6 volatility measures), volume confirmation (3 volume features), statistical structure (Bollinger, Hurst), and mean reversion (z-scores, %B).

MI permutation testing with BH correction will determine which features carry genuine predictive information above the noise floor. VIF analysis will quantify collinearity (expected to be severe in the volatility cluster) but will NOT be used for feature elimination (Ridge handles collinearity; pre-registered decision).

The critical finding: **no single feature is expected to achieve directional accuracy above the break-even threshold (~56-63% for dollar bars)**. This gap between single-feature DA (51-53%) and break-even DA (56-63%) is not a failure -- it is the fundamental motivation for the ML recommendation system. By combining features that capture different market conditions, the ensemble can potentially bridge this gap. The recommendation system's most valuable output may be the NO-TRADE signal during regimes where no feature combination exceeds break-even.

## 7. Connection to Phases 9-12 (Model Design)

### What Section 3 Tells the Models
1. **Feature selection** (which features pass three-gate validation) determines the input dimensionality for Phase 9-10 models.
2. **VIF clusters** suggest which features carry redundant information, informing model regularization strength.
3. **Expected signs** provide sanity checks: if the model assigns a large positive coefficient to rsi_14, something may be wrong (expected negative).
4. **The DA gap** sets the performance target: the model must achieve DA > break-even, which individual features cannot.
5. **hurst_100 as regime indicator** motivates regime-conditional modeling or attention mechanisms in the TFT.

### Model Architecture Implications
- **SIDE track** (classification): Use features with directional expected signs (positive/negative). The model must learn WHEN to apply momentum vs mean reversion.
- **SIZE track** (regression): Use unsigned features (volatility measures, bbwidth, amihud). The model predicts HOW MUCH the price will move.
- **Recommendation system** (Phase 12): Combine SIDE and SIZE predictions. Use hurst_100 as a regime feature. The output is a CONTINUOUS expected return, not a binary bet.

### Feature Subset Strategy
- If only momentum features pass validation: the model is limited to trend-following.
- If only mean-reversion features pass: the model is limited to contrarian strategies.
- If both pass: the model can learn regime-switching, which is the thesis ideal.
- If nothing passes: the negative result protocol (pre-registration) applies -- switch to the recommender as a pure NO-TRADE filter.

## Implementation

Module: `src/app/research/application/rc2_feature_rationale.py`
Tests: `src/tests/research/test_rc2_feature_rationale.py`

The rationale table is a pure function returning a Pandas DataFrame with 23 rows and 9 columns. Each feature has: economic_intuition, literature_ref, expected_sign, sign_rationale, stationarity_expectation, vif_cluster, is_transformation_based.

The `generate_section3_therefore()` function produces a ready-to-paste "Therefore" paragraph.
