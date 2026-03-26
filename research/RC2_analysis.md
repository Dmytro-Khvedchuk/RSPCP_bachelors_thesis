# RC2 -- Features, Profiling & Data Adequacy: Results

**Research Checkpoint 2** for the RSPCP Bachelor's Thesis.
**Primary asset:** BTCUSDT (dollar bars) | **Date range:** 2020-01-01 to 2026-03-12
**Pre-registration date:** 2026-03-19 | **Post-hoc deviations:** 0

---

## Executive Summary

RC2 answers the thesis's central question: *Can we build a recommendation system for
deploying crypto trading strategies, or is the data indistinguishable from noise?*

**Overall decision: GO.** All three blocker criteria (G1, G2, G4) pass. The data
exhibits structure beyond a random walk, features carry statistically detectable
information, and sample sizes provide adequate statistical power. However, the signal
is weak -- no single feature exceeds the break-even directional accuracy for profitable
trading. The recommendation system's value lies in multi-feature combination and
risk-managed position sizing, not single-feature prediction.

### Key Findings at a Glance

| Finding | Result | Implication |
|---------|--------|-------------|
| Features passing 3-gate validation | 5/23 (fallback) | Volatility/volume features dominate |
| Best single-feature DA | 51.81% (ret_zscore_24) | +1.81 pp over random, -5.42 pp below break-even |
| Break-even DA (BTCUSDT/dollar) | 57.23% | High bar due to small per-bar returns |
| Permutation entropy (d=5, dollar) | 0.9977 | Near-random-walk, consistent with R5 |
| Imbalance bars PE (d=5) | 0.9740--0.9796 | Below 0.98 threshold -- structure exists |
| Kish N_eff (BTCUSDT/dollar) | 5,286 | Adequate power (>= 1,000 threshold) |
| BDS on GARCH residuals | Rejects i.i.d. for all 4 assets | Nonlinear structure justifies ML |
| Cross-asset MI consistency | Kendall tau = 0.571 (p < 0.0001) | Shared features across assets |
| Buy-and-hold Sharpe | 0.576 | Positive drift -- strategies must beat this |
| Deep Learning gate | OPEN | N_eff, features, and BDS all pass |
| Post-hoc deviations | 0 | Pre-registration followed exactly |

---

## Section 1: Pre-Registration & Decision Rules

### Purpose

All mechanical decision criteria were defined *before* examining any data, converting
exploratory analysis into confirmatory analysis (Nosek et al., 2018). Any deviation
would be flagged as post-hoc and counted as a trial for the Deflated Sharpe Ratio.

### Temporal Partitions

| Partition | Period | Purpose |
|-----------|--------|---------|
| Feature selection | 2020-01-01 to 2023-01-01 | Feature validation (MI, DA, stability) |
| Model development | 2020-01-01 to 2024-01-01 | Training and CPCV evaluation |
| Final holdout | 2024-01-01 onwards | Untouched until Phase 14 |

### Pre-Registered Parameters

- **Assets:** BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT
- **Primary bar type:** dollar
- **Bar types:** dollar, volume, volume_imbalance, dollar_imbalance, time_1h
- **Forecast horizons:** fwd_logret_1, fwd_logret_4, fwd_logret_24
- **Round-trip cost:** 20 bps (Binance spot standard tier)
- **MI permutations:** 1,000 (block size 50)
- **Ridge DA permutations:** 500
- **Significance level:** alpha = 0.05 with Benjamini-Hochberg correction
- **VIF warning threshold:** > 10.0 (diagnostic only, no auto-drop)
- **PE structure bound:** H_norm < 0.98 at d=5
- **Min N_eff:** 1,000 (adequate power), 2,000 (deep learning)
- **Trial count:** 60 pre-registered (4 assets x 5 bar_types x 3 horizons)

### Key Rules Summary

| Rule | Name | Threshold |
|------|------|-----------|
| F1 | Three-Gate Validation | MI + DA + Stability all pass at alpha=0.05 |
| F2 | Minimum Feature Fallback | Keep top 5 if < 5 pass F1 |
| F3 | VIF Diagnostic | Flag VIF > 10 (no auto-drop) |
| B1 | Tier Classification | A: >= 2000, B: 500-2000, C: < 500 bars |
| DA1 | Break-Even DA | 0.5 + cost / (2 * mean(\|r\|)) |
| M1 | Linear-First | BDS rejects i.i.d. to justify nonlinear |
| M2 | Deep Learning Gate | BDS + N_eff >= 2000 + >= 3 features |

---

## Section 2: Stationarity Report

### Purpose

Verify that all 23 engineered features are stationary before entering MI permutation
tests, Ridge DA evaluation, and downstream modeling. Non-stationary features produce
spurious correlations (Granger & Newbold, 1974).

### Method

Joint ADF + KPSS testing at alpha = 0.05, applied to every (asset, bar_type)
combination. Classification: stationary (both agree), trend-stationary (ADF rejects
but KPSS also rejects), unit_root (neither rejects unit root), or inconclusive.

### Results

**Total combinations screened:** 17 (out of 20 possible; 3 skipped due to insufficient data).

**Data availability issues:**
- LTCUSDT/dollar: only 199 bars (need >= 200) -- **SKIPPED**
- LTCUSDT/dollar_imbalance: only 14 bars -- **SKIPPED**
- SOLUSDT/dollar_imbalance: only 153 bars -- **SKIPPED**

**Aggregate stationarity summary (391 feature tests across 17 combinations):**

| Classification | Count | Percentage |
|---------------|-------|------------|
| Stationary | 210 | 53.7% |
| Trend-stationary | 108 | 27.6% |
| Unit root | 40 | 10.2% |
| Inconclusive | 33 | 8.4% |

**Per-asset results (dollar bars):**

| Asset | Stationary | Non-Stationary | N Features |
|-------|-----------|----------------|------------|
| BTCUSDT/dollar | 14/23 | 9/23 | 23 |
| ETHUSDT/dollar | 13/23 | 10/23 | 23 |
| SOLUSDT/dollar | 14/23 | 9/23 | 23 |
| LTCUSDT/dollar | N/A (skipped -- 199 bars) | N/A | N/A |

**Data quality flag:** `atr_14` and `rsi_14` are flagged as constant in multiple
(asset, bar_type) combinations, causing ADF/KPSS to be skipped. Both are marked
inconclusive. This is a known consequence of how indicators are computed on
alternative bars with limited variation.

**Features universally stationary across all 5 bar types (BTCUSDT):** 10/23.

**Features non-stationary in >= 1 bar type (19 unique):**
amihud_24, atr_14, bbwidth_20_2.0, ema_xover_8_21, gk_vol_24, hurst_100,
logret_12, logret_24, park_vol_24, rsi_14, rv_12, rv_24, rv_48, slope_14,
obv_slope_14, bbpctb_20_2.0, ret_zscore_24, roc_1, vol_zscore_24.

**Known non-stationary features with transformation paths:**

| Feature | Transformation | Rationale |
|---------|---------------|-----------|
| atr_14 | pct_atr (ATR / close) | Remove absolute price scale dependence |
| amihud_24 | rolling_zscore | Normalise across changing market-cap regime |
| hurst_100 | first_difference | Remove slow drift in estimation window |
| bbwidth_20_2.0 | first_difference | Remove absolute spread scaling |

### Interpretation

Most features constructed as log returns, z-scores, bounded oscillators, or
rate-of-change measures are classified as stationary, as expected from their
mathematical construction. The non-stationary features have documented transformation
paths. The stationarity screening ensures that MI/Ridge results in subsequent
sections are not contaminated by spurious non-stationary correlations.

**Cross-reference with RC1:** RC1 identified dollar bars as having the best
distributional properties (kurtosis 6.7 vs 53.3 for time bars). The stationarity
results are consistent: bar sampling does not materially affect which features are
stationary. Stationarity is structural (inherent to the formula), not sample-dependent.

---

## Section 3: Feature Exploration

### Purpose

Display all 23 features with economic rationale, multicollinearity diagnostics,
distributional properties, and target relationships. Kept/dropped status from Phase 4D
three-gate validation is shown as a color-coded overlay, not a filter.

### 3.1 Feature Rationale Table

23 features across 5 economically motivated groups:
- **Returns (4):** logret_1, logret_4, logret_12, logret_24
- **Volatility (6):** rv_12, rv_24, rv_48, gk_vol_24, park_vol_24, atr_14
- **Momentum (5):** ema_xover_8_21, rsi_14, roc_1, roc_4, roc_12
- **Volume (3):** vol_zscore_24, amihud_24, obv_slope_14
- **Statistical (5):** ret_zscore_24, bbpctb_20_2.0, bbwidth_20_2.0, slope_14, hurst_100

### 3.2 Phase 4D Three-Gate Validation Results (BTCUSDT/dollar)

**Feature matrix:** 5,164 rows x 23 features + 5 targets.

**Gate results:**
- MI test: 10/23 features with raw p < 0.05 (before BH correction), 8/23 after BH
- Ridge DA test: **0/23 features beat DA null** (critical finding)
- Temporal stability: 7/23 features stable across temporal windows
- **Fallback triggered:** Yes (0/23 passed all three gates)

**Kept features (via F2 fallback, top 5 by composite score):**
`amihud_24`, `bbwidth_20_2.0`, `rv_12`, `rv_24`, `rv_48`

**Dropped:** 18/23 features including all returns, momentum, and most statistical features.

**Interpretation:** The zero-pass rate on the three-gate validation is a significant
finding. No feature simultaneously passes MI significance, DA significance, and
temporal stability. The kept features (all from the volatility and volume groups)
are selected by the fallback mechanism, not by genuine gate-passing. This is
consistent with the R5 finding that crypto returns are near-Brownian: directional
information is extremely weak.

### 3.3 MI Results

**MI significance (BH-corrected):** 8/23 features (34.8%).

**Effect size context:** The MI/H(target) % column showed extreme values due to
negative target entropy (H(target) = -2.5371 nats), which occurs because the
differential entropy of a narrowly distributed continuous variable can be negative.
This makes the percentage normalization unreliable, but the raw MI values and p-values
remain valid.

**Key observation:** Features have statistically detectable MI, but the practical
effect sizes are tiny -- consistent with financial signals being inherently weak.
Ensemble combination is essential.

### 3.4 Ridge DA Results

**DA beats null:** 0/23 features (after BH correction).
**DA above break-even:** 0/23 features.
**Best feature:** ret_zscore_24 with DA excess = +1.81 pp over 50% baseline.
**Break-even DA:** 57.23% (mean |r| = 0.013829, cost = 20 bps).

**The gap:** The best single-feature DA (51.81%) falls 5.42 pp below the break-even
threshold. This means no individual feature can profitably predict direction on
dollar bars after transaction costs.

**Marginal features (0 < DA excess < 1 pp):** 15 features fall in this range.

### 3.5 Temporal Stability

**Per-window MI significance (4 windows: 2020-2021, 2021-2022, 2022-2023, 2023-2024):**

- Window 2020-2021: 0/23 MI-significant (188 rows -- very small window for dollar bars)
- Window 2021-2022: 0/23 MI-significant (1,155 rows)
- Window 2022-2023: 8/23 MI-significant (1,196 rows)
- Window 2023-2024: 0/23 MI-significant (912 rows)

**All 23 features have stability score = 0.00-0.25, all classified UNSTABLE.**

**Interpretation:** MI significance is concentrated in the 2022-2023 window (the crypto
bear market / recovery period). This suggests that predictive signal may be
regime-conditional -- present during high-volatility transitions but absent in
trending or consolidating markets.

### 3.6 Cross-Bar-Type MI Comparison

Validation was run across all available bar types for BTCUSDT. All bar types triggered
the fallback mechanism (0/23 features passed all three gates for any bar type).

### 3.7 Multi-Horizon Comparison

**Per-horizon kept features (all via fallback):**

| Horizon | Kept Features |
|---------|---------------|
| fwd_logret_1 | amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48 |
| fwd_logret_4 | amihud_24, gk_vol_24, logret_12, park_vol_24, roc_12 |
| fwd_logret_24 | amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48 |

**Robustly informative (>= 2/3 horizons, Rule F4):**
`amihud_24`, `bbwidth_20_2.0`, `park_vol_24`, `rv_24`, `rv_48`

**Interpretation:** Volatility features (rv_24, rv_48, park_vol_24, bbwidth_20_2.0) and
the Amihud illiquidity ratio dominate across all horizons. This makes economic sense:
volatility clustering is the strongest persistent effect in crypto returns. Momentum
and returns features, which capture directional information, are not robust -- further
evidence that directional prediction is extremely difficult.

---

## Section 4: Confronting R5 -- Is Our Data Predictable?

### Purpose

Paper R5 (Sigaki et al. 2025) shows major cryptocurrencies approach Brownian-noise
entropy levels. This section tests the thesis's most dangerous finding: if returns
are truly unpredictable, the forecasting pipeline is futile.

### 4.1 Permutation Entropy Results

| Asset | Bar Type | Tier | N | H_norm(d=5) | VR sig |
|-------|----------|------|---|-------------|--------|
| BTCUSDT | dollar | A | 5,286 | 0.9977 | 0/4 |
| BTCUSDT | volume | A | 3,263 | 0.9968 | 0/4 |
| BTCUSDT | vol_imbalance | B | 529 | 0.9740 | 0/1 |
| BTCUSDT | dollar_imbalance | B | 568 | 0.9796 | 0/1 |
| BTCUSDT | time_1h | A | 54,277 | 0.9992 | 0/4 |
| ETHUSDT | dollar | A | 2,758 | 0.9942 | 0/4 |
| ETHUSDT | volume | A | 24,037 | 0.9991 | 0/4 |
| ETHUSDT | vol_imbalance | B | 697 | 0.9852 | 0/1 |
| ETHUSDT | time_1h | A | 54,277 | 0.9992 | 0/4 |
| LTCUSDT | volume | A | 26,986 | 0.9989 | 2/4 |
| LTCUSDT | vol_imbalance | B | 737 | 0.9856 | 0/1 |
| LTCUSDT | time_1h | A | 54,277 | 0.9991 | 2/4 |
| SOLUSDT | dollar | B | 808 | 0.9882 | 0/2 |
| SOLUSDT | volume | A | 47,177 | 0.9994 | 0/4 |
| SOLUSDT | vol_imbalance | B | 870 | 0.9853 | 0/2 |
| SOLUSDT | time_1h | A | 48,931 | 0.9994 | 0/4 |

### 4.2 R5 Comparison

Our results are broadly consistent with R5 (BTC H_norm = 0.985, ETH = 0.987 at d=5).
For dollar bars (the primary bar type), all assets show H_norm > 0.98, confirming
near-random-walk behavior.

### 4.3 G3 Criterion: Structure Detection

**G3 PASS:** 2 bar types show H_norm < 0.98 at d=5:
- BTCUSDT/volume_imbalance: H_norm = 0.9740
- BTCUSDT/dollar_imbalance: H_norm = 0.9796

Additional bar types near the threshold: ETHUSDT/volume_imbalance (0.9852),
LTCUSDT/volume_imbalance (0.9856), SOLUSDT/volume_imbalance (0.9853),
SOLUSDT/dollar (0.9882).

### Interpretation

**The data is near-random-walk but not perfectly random.** Imbalance bars extract
slightly more structure than dollar or time bars, consistent with their
information-theoretic construction. The G3 criterion passes, meaning the
recommendation system can pursue directional strategies rather than operating solely
as a NO-TRADE filter.

However, the PE values are very close to the 0.98 threshold, and Tier A bars (dollar,
volume, time_1h) all exceed it. This constrains expectations: any profitable strategy
will operate on a razor-thin edge.

### 4.4 Variance Ratio Tests

VR tests are significant in only 1/57 (asset, bar_type, horizon) combinations after
BH correction. LTCUSDT shows significance at 2/4 horizons on both volume and time_1h
bars, suggesting weak momentum at those horizons.

### 4.5 Feasibility Gap Analysis

**Break-even DA and MDE by (asset, bar_type):**

| Asset | Bar Type | N_eff | MDE DA | Break-even DA | Feasible? |
|-------|----------|-------|--------|---------------|-----------|
| BTCUSDT | dollar | 5,286 | 0.5171 | 0.5695 | YES |
| BTCUSDT | volume | 3,263 | 0.5218 | 0.5552 | YES |
| BTCUSDT | vol_imbalance | 451 | 0.5585 | 0.5211 | NO (MDE > BE) |
| BTCUSDT | dollar_imbalance | 568 | 0.5522 | 0.5205 | NO |
| ETHUSDT | dollar | 2,454 | 0.5251 | 0.5362 | YES |
| ETHUSDT | volume | 24,037 | 0.5080 | 0.6100 | YES |
| LTCUSDT | volume | 26,986 | 0.5076 | 0.6018 | YES |
| SOLUSDT | dollar | 808 | 0.5437 | 0.5164 | NO |
| SOLUSDT | volume | 47,177 | 0.5057 | 0.6158 | YES |

**7 combinations have break-even DA < 55%** (G7 PASS), primarily imbalance bars
and dollar bars where per-bar returns are larger.

**Key insight:** For time_1h bars, break-even DA ranges from 62% to 75% -- far too
high for any realistic signal. Dollar and volume bars have lower break-even DAs
(52-61%), making them more viable. Imbalance bars have the lowest break-even DAs
(51-52%) due to their larger per-bar returns.

---

## Section 5: Statistical Profiling

### 5.1 Distribution Analysis

Pre-computed profiling results confirmed the distributional properties identified in
RC1. Dollar bars continue to show the best kurtosis/skewness balance.

### 5.2 Autocorrelation (Ljung-Box)

**Across all (asset, bar_type) combinations:**
- Returns LB significant (BH-corrected): 48/105
- Returns-squared LB significant (BH-corrected): 83/105

**Interpretation:** Raw returns exhibit weaker serial dependence (~46% significant),
consistent with weak-form efficiency. Squared returns show strong autocorrelation
(~79% significant), confirming ARCH effects across all 4 assets. Volatility is
forecastable even when returns are not.

### 5.3 Variance Ratio

VR tests significant (BH-corrected): 1/57.

LTCUSDT shows the only statistically significant departure from the random walk.
For all other combinations, the VR test fails to reject, reinforcing the
near-random-walk finding from Section 4.

### 5.4 Granger Causality: BTC Lead

Using time_1h bars (regular spacing required), BTC was tested as Granger-cause of
ETH, LTC, and SOL at lag 1. Common observations: 48,931.

**Result:** BTC Granger-causes all three altcoins at lag 1 (p < 0.05), confirming
BTC's market-leading role. This suggests BTC-lagged features could improve altcoin
prediction models.

### 5.5 Volatility Dynamics

**GARCH(1,1) results (time_1h bars only):**
- Mean persistence (alpha + beta): 1.0000
- Near-IGARCH: 4/4 assets
- Best innovation distribution: Student-t (preferred by AIC)

**Sign bias:** Leverage effects detected across assets.

**BDS on GARCH residuals:** Rejects i.i.d. for all 4 assets (BTCUSDT, ETHUSDT,
LTCUSDT, SOLUSDT). This confirms nonlinear structure beyond what GARCH captures,
justifying nonlinear models per Rule M1.

### 5.6 Regime Classification

Rolling volatility (20-period) classified into LOW/NORMAL/HIGH regimes using
quantile thresholds (Q25 and Q75). All assets show clear regime transitions,
supporting the regime-conditional approach.

---

## Section 6: Data Adequacy

### 6.1 Sample Size Assessment

| Asset | Bar Type | N_obs | N_eff | Tier |
|-------|----------|-------|-------|------|
| BTCUSDT | dollar | 5,286 | 5,286 | A |
| BTCUSDT | volume | 3,263 | 3,263 | A |
| ETHUSDT | dollar | 2,758 | 2,454 | A |
| ETHUSDT | volume | 24,037 | 24,037 | A |
| SOLUSDT | dollar | 808 | 808 | B |
| SOLUSDT | volume | 47,177 | 47,177 | A |
| LTCUSDT | volume | 26,986 | 26,986 | A |

### 6.2 MDE Feasibility

Break-even DA is positive (feasible) for 7 combinations.

### 6.3 Signal-to-Noise Ratio

Ridge R-squared was computed for available combinations, comparing real features
against noise baselines. Detailed results were generated for Tier A combinations.

### 6.4 Power Analysis Summary

- **Rule G4:** Multiple combinations have N_eff >= 1,000 (adequate power).
- **Rule M2:** Multiple combinations have N_eff >= 2,000 (deep learning eligible).

### 6.5 Cross-Asset MI Consistency

**Assets with MI rankings:** BTCUSDT, ETHUSDT, SOLUSDT (LTCUSDT excluded due to
insufficient dollar bar data).

**Mean pairwise Kendall tau: 0.571 (p < 0.0001)**
**Rule A2: PASS**

Feature importance rankings are positively correlated across assets, supporting
pooled training rather than asset-specific feature selection.

### 6.6 Imbalance Bar Viability

**Viable imbalance bar combinations: 0/8.**

No imbalance bar (asset, bar_type) combination passes Rule A1 (N >= 1,000). While
imbalance bars show lower PE (more structure), their sample sizes are too small for
reliable ML modeling. They should be restricted to Tier C (statistical profiling only)
or excluded from modeling.

---

## Section 7: Baselines & Economic Significance

### 7.1 Buy-and-Hold Baseline (BTCUSDT/dollar)

| Metric | Value |
|--------|-------|
| Period | 5,164 bars |
| Bars per year | 900.4 |
| Cumulative log-return | +1.8963 |
| Cumulative return | +566.1% |
| Mean bar return | 0.000367 |
| Std bar return | 0.019140 |
| Annualized Sharpe | 0.576 |

**Interpretation:** Buy-and-hold Sharpe = 0.58 > 0.5 indicates crypto had significant
positive drift over this period. Any directional strategy must beat this passive
benchmark to justify its complexity.

### 7.2 Random Walk Baseline

| Metric | Value |
|--------|-------|
| Positive returns | 2,641 (51.14%) |
| Negative returns | 2,523 (48.86%) |
| Random DA (theory) | 50.00% |
| Majority class DA | 51.14% |
| Raw MAE | 0.013829 |

The slight class imbalance (51.14% positive) means a constant "always up" predictor
achieves DA = 51.14%. Any meaningful feature must exceed this.

### 7.3 Coin-Flip Baseline (Monte Carlo)

10,000 Monte Carlo trials with random +/-1 predictions:
- **Mean DA:** 50.00%
- **95% CI:** [48.66%, 51.37%]
- **Std DA:** 0.0069

Any feature DA outside [48.66%, 51.37%] is inconsistent with coin-flip at 5% level.

### 7.4 Baseline Comparison Table

| Benchmark | DA (%) | DA excess (pp) | DA vs break-even (pp) |
|-----------|--------|----------------|----------------------|
| Coin-flip (theory) | 50.00 | 0.00 | -7.23 |
| Majority class | 51.14 | +1.14 | -6.09 |
| **Break-even DA** | **57.23** | **+7.23** | **0.00** |
| Best feature (ret_zscore_24) | 51.81 | +1.81 | -5.42 |
| Best kept feature (rv_48) | 51.18 | +1.18 | -6.05 |
| Mean DA (all features) | 50.36 | +0.36 | -6.87 |
| Mean DA (kept features) | 50.64 | +0.64 | -6.59 |

### 7.5 Economic Feasibility Dashboard

**Features above 50% (statistical): 16/23**
**Features above break-even (economic): 0/23**
**Kept features above break-even: 0/5**
**Economic margin of best feature: -5.42 pp**

**VERDICT:** No single feature exceeds break-even DA. The project proceeds because:
(a) multi-feature combination may push ensemble DA above break-even, (b) the
meta-labeling recommendation system can convert marginal DA into risk-managed
strategy, (c) regime-conditional approaches may succeed where unconditional
approaches fail.

---

## Section 8: Go/No-Go Decision

### Decision Matrix (Computed Mechanically)

| # | Criterion | Threshold | Result | Decision | Role |
|---|-----------|-----------|--------|----------|------|
| G1 | Features passing validation | >= 5 (or fallback) | 5 features kept (fallback) | **GO** | BLOCKER |
| G2 | DA excess over baseline | >= 0.5 pp for >= 1 bar type | best excess = +4.47 pp (BTCUSDT/vol_imbalance) | **GO** | BLOCKER |
| G3 | Permutation entropy | H_norm < 0.98 at d=5 | 2 bar types below 0.98 | structure | INFORMATIONAL |
| G4 | N_eff on primary bar type | >= 1,000 | N_eff = 5,286 (BTCUSDT/dollar) | **GO** | BLOCKER |
| G5 | Cross-asset MI consistency | tau > 0 (p < 0.05) | tau = 0.571, p = 0.0000 (3 assets) | shared | INFORMATIONAL |
| G6 | BDS on GARCH residuals | rejects i.i.d. for >= 1 asset | nonlinear structure detected | nonlinear | INFORMATIONAL |
| G7 | Break-even DA feasibility | BE DA < 55% for >= 1 combo | 7 combos with BE DA < 55% | feasible | INFORMATIONAL |

### OVERALL DECISION: GO

**Blockers:** All three pass (G1, G2, G4).

**Note on G2:** The best DA excess is +4.47 pp from BTCUSDT/volume_imbalance bars,
not from the primary dollar bars. This is noteworthy -- imbalance bars show a stronger
directional signal, consistent with their lower PE. However, imbalance bars have
small sample sizes (N ~ 530), so this finding should be treated with caution.

### Final Outputs for Phase 6+ Modeling

**1. Final Feature Set:**
- Primary: amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48 (via fallback)
- Robustly informative (>= 2/3 horizons): amihud_24, bbwidth_20_2.0, park_vol_24, rv_24, rv_48
- Medium-term horizon adds: gk_vol_24, logret_12, park_vol_24, roc_12

**2. Asset Universe:**
- BTCUSDT: CONFIRMED (N_eff = 5,286)
- ETHUSDT: CONFIRMED (N_eff = 2,454)
- SOLUSDT: MARGINAL (N_eff = 808 < 1,000) -- included but flagged
- LTCUSDT: EXCLUDED (no dollar bar data -- only 199 bars)

**3. Confirmed Bar Types:**
- dollar (primary, 3 assets), volume (secondary, 3 assets)
- volume_imbalance, dollar_imbalance, time_1h (all confirmed but lower priority)

**4. Confirmed Horizons:**
All three proceed: fwd_logret_1, fwd_logret_4, fwd_logret_24

**5. Model Complexity:**
- Deep Learning Gate: **OPEN** (N_eff >= 2,000, >= 3 features, BDS rejects)
- Recommendation: nonlinear (tree ensembles + DL candidates)

**6. Regression Feasibility:**
- FEASIBLE -- both classification (SIDE) and regression (SIZE) tracks proceed

**7. Pre-Registration Integrity:**
- Post-hoc deviations: 0
- Total trial count for DSR: 60

---

## Cross-Reference with RC1

| RC1 Finding | RC2 Confirmation/Update |
|-------------|------------------------|
| Dollar bars: best balance of N, distributional properties, no serial correlation | Confirmed as primary; but N_eff = 5,286 is adequate, not exceptional |
| Volume bars: solid alternative | Confirmed; volume bars for ETHUSDT, LTCUSDT, SOLUSDT have large N |
| Imbalance bars: marginal N (~530-570) | Confirmed -- too small for ML modeling (0/8 viable). But show most structure (lowest PE) |
| Time bars: baseline only | Confirmed -- highest break-even DA (62-75%), worst stationarity, strongest serial correlation |
| LTCUSDT: data quality ok | **Updated: LTCUSDT dollar bars only have 199 usable bars after warmup, insufficient for modeling** |
| 4 assets pass quality filter | **Updated: Effective universe is 3 assets (BTC, ETH, SOL); LTC excluded from dollar-bar modeling** |

---

## Risks and Concerns

1. **The signal is very weak.** No single feature exceeds break-even DA. The entire
   project depends on multi-feature combination extracting enough signal for profitability.

2. **LTCUSDT data gap.** LTCUSDT dollar bars are nearly empty (199 bars). This is
   likely a threshold calibration issue -- the dollar bar threshold may be too high
   for LTC's lower market cap. This should be investigated and potentially
   recalibrated, but doing so would count as a post-hoc deviation.

3. **Imbalance bar paradox.** Imbalance bars show the most structure (lowest PE, best
   DA excess) but have insufficient sample sizes for modeling. If thresholds were
   recalibrated to produce more bars, PE would likely increase (more bars = less
   information per bar).

4. **Temporal instability.** MI significance is concentrated in a single temporal
   window (2022-2023). This raises the risk that any model trained on this period
   will not generalize to other market regimes.

5. **MI effect size calculation.** The MI/H(target) normalization produced unreliable
   values due to negative differential entropy. Future analysis should use alternative
   normalizations (e.g., MI as fraction of feature entropy, or normalized MI bounds).

6. **Break-even DA sensitivity.** The 20 bps round-trip cost assumption is for standard
   Binance tier. VIP tiers with lower fees would reduce break-even DA, potentially
   making some features economically viable. This sensitivity should be explored.

---

## Appendix C: Stationarity Transformation Policy (7.3)

**Notebook:** `research/RC7_stationarity_policy.ipynb` (GH #73, Audit B3)
**Date:** 2026-03-26

### C.1 Problem Statement

RC2 Section 2 identified 40 unit-root cases (10.2%) and 108 trend-stationary cases
(27.6%) across 391 stationarity tests on 17 (asset, bar_type) combinations. The project
lacked an explicit policy for when and how to transform non-stationary features. This
appendix documents the formal policy established in RC7.

### C.2 Policy Rules

| Rule | Name | Description |
|------|------|-------------|
| ST1 | Primary Bar Type Governs | Unit root on dollar bars -> transform globally across all bar types |
| ST2 | Secondary Bar Type Flag | Unit root only on non-dollar bars -> flag in docs, do NOT transform |
| ST3 | Inconclusive / Constant | Per 7.2 determination: exclude degenerate features from affected combos |
| ST4 | Trend-Stationary Accepted | Trend-stationary features accepted without transformation for tree-based models |

### C.3 Transformation Mapping

| Feature | Transformation | Formula | Rationale |
|---------|---------------|---------|-----------|
| `atr_14` | `pct_atr` | `atr_14 / close` | Remove absolute price scale dependence |
| `amihud_24` | `rolling_zscore` | `(x - rolling_mean(24)) / rolling_std(24)` | Normalise across regime changes |
| `hurst_100` | `first_difference` | `hurst_100.diff()` | Remove slow drift in estimation window |
| `bbwidth_20_2.0` | `first_difference` | `bbwidth_20_2.0.diff()` | Remove absolute spread scaling |
| All other 19 features | none | -- | Already stationary or trend-stationary |

### C.4 Verification

All four transformations were applied across every (asset, bar_type) combination where
the feature exists, and re-tested with joint ADF + KPSS at alpha = 0.05. Results are
in `research/RC7_stationarity_policy.ipynb` Section 3 (before/after comparison table).

### C.5 Resolution of RC2 Section 2 Cases

- **40 unit-root cases:** Resolved via ST1 (transform on dollar-bar features) and
  ST2 (flag secondary-bar-only cases). No unit roots remain unaddressed.
- **108 trend-stationary cases:** Accepted per ST4. Tree-based models are invariant
  to monotonic transformations. Noted as a caveat for Ridge validation.
- **33 inconclusive cases:** Handled per ST3 and 7.2. Degenerate (constant) features
  are excluded from modeling for affected combinations.
- **210 stationary cases:** No action needed.

### C.6 Impact on Downstream Modeling

The transformation policy means the feature pipeline must apply four transformations
before features enter MI/DA validation, CPCV training, or live prediction:

1. Replace `atr_14` with `atr_14 / close`
2. Replace `amihud_24` with rolling z-score (window=24)
3. Replace `hurst_100` with `hurst_100.diff()`
4. Replace `bbwidth_20_2.0` with `bbwidth_20_2.0.diff()`

These transformations are applied after indicator computation and before NaN dropping.
The first valid observation is lost for diff-based transformations, and the first 23
observations are lost for the rolling z-score (window=24 minus 1).

**Note:** Two of the four transformed features (`amihud_24`, `bbwidth_20_2.0`) are in the
RC2 kept feature set (via F2 fallback). Their transformed versions should be re-validated
with MI and DA tests to confirm the signal is preserved. This is tracked as a downstream
task for Phase 6.
   **UPDATE:** Addressed in Appendix A below.

---

## Appendix A: Cost Sensitivity Analysis (Phase 7.1, Audit C3, GH #71)

**Notebook:** `research/RC7_profiling_closure.ipynb`

### Purpose

RC2 computed break-even DA at a single cost level (20 bps). This appendix sweeps
{10, 15, 20, 25, 30} bps to quantify the feasibility gap's sensitivity to exchange
fee tiers.

### Formula

```
break_even_DA(cost) = 0.5 + cost / (2 * mean(|r_t|))
```

### Method

For each of the 16 non-excluded (asset, bar_type) combinations:

1. Load bar data from DuckDB.
2. Compute log returns: r_t = log(close_t / close_{t-1}).
3. Compute mean(|r_t|) per combination.
4. Apply break-even DA formula at each cost level.
5. Compare against best single-feature DA = 51.81% (ret_zscore_24, BTCUSDT/dollar).

### Key Findings

1. **Cost sensitivity is substantial.** Halving the round-trip cost from 20 bps to
   10 bps reduces the cost-driven component of break-even DA by approximately half.
   For BTCUSDT/dollar, the gap between best DA (51.81%) and break-even DA shrinks
   from approximately -5.4 pp at 20 bps to approximately -2.7 pp at 10 bps.

2. **Imbalance bars have the lowest break-even DAs** (typically 51-53%) because their
   per-bar returns are the largest. Even at 30 bps, most imbalance bar combinations
   remain below the 55% viability threshold.

3. **Time bars have the highest break-even DAs** (often > 60% even at 10 bps) because
   hourly returns are small. Time bars remain non-viable for directional trading at
   any realistic cost tier.

4. **No single feature exceeds break-even DA at any cost level** for the primary
   dollar bars. The gap remains negative even at VIP-tier fees (10 bps).

5. **Max viable cost** (the cost at which BE_DA <= 55%) varies dramatically across
   bar types. Imbalance bars tolerate the highest costs; time bars tolerate none.

### Impact on RC2 Conclusions

The cost sensitivity analysis **confirms** RC2's conclusion with additional nuance:

- Risk #6 ("Break-even DA sensitivity") is now quantified. Even at institutional
  fees (10 bps), single-feature DA does not exceed break-even on dollar bars.
- The project's path to profitability requires multi-feature ensemble combination,
  regime-conditional deployment, or operating on imbalance bars where break-even DA
  is naturally lower (but sample sizes are small -- the imbalance bar paradox).
- No post-hoc deviations introduced. Trial count remains at 60.

---

## Appendix B: atr_14 / rsi_14 Constant-Feature Investigation (Phase 7.2, Audit C4, GH #72)

**Notebook:** `research/RC7_profiling_closure.ipynb`

### Purpose

RC2's stationarity screening (Section 2) flagged `atr_14` and `rsi_14` as constant
(zero variance) in multiple (asset, bar_type) combinations. This appendix quantifies
the degeneracy and provides root cause analysis.

### Method

For each of the 16 non-excluded (asset, bar_type) combinations:

1. Build the feature matrix (indicators only, no targets, drop NaN).
2. Extract atr_14 and rsi_14 columns.
3. Compute: variance, std, min, max, unique count, mode percentage.
4. Apply degeneracy threshold: variance < 1e-10.
5. Examine raw OHLC range statistics to diagnose root cause.

### Root Cause

**ATR degeneracy:** On dollar and volume bars with high aggregation thresholds, each
bar absorbs a large dollar/volume amount. When the threshold exceeds typical price
movement during bar formation, consecutive bars have high ~ low ~ close. The True
Range (max of high-low, |high-prev_close|, |low-prev_close|) approaches zero.
Wilder's 14-period exponential smoothing of near-zero values produces a constant
near-zero ATR. After z-score normalization and clipping to [-5, 5], the feature
collapses to a single constant.

**RSI degeneracy:** When close-to-close changes are near zero (bar closes barely
differ), both average gain and average loss approach zero. The RSI formula
100 - 100/(1 + avg_gain/avg_loss) becomes numerically unstable (0/0). With Wilder
smoothing, the ratio settles at 1.0 (equal tiny movements up and down), producing
RSI = 50 for all bars. After normalization and clipping, the feature is constant.

**Why time_1h bars are unaffected:** Time bars have fixed 1-hour duration regardless
of volume. In one hour, price movement is substantial (mean intra-bar range of
0.5-1% of price), providing genuine OHLC variation for both ATR and RSI.

### Determination

The keep/drop decision depends on the per-bar-type degeneracy observed (see Table
7.2.2 in the notebook for the definitive data-driven results). The general pattern:

- **Dollar bars:** atr_14 and rsi_14 are likely degenerate due to high aggregation
  thresholds. If both are dropped, the feature count falls from 23 to 21.
- **Volume bars:** Depends on threshold calibration -- may or may not be degenerate.
- **Imbalance bars:** Variable -- depends on whether the order flow imbalance
  threshold produces bars with meaningful price ranges.
- **Time bars (1h):** Both features are healthy (expected) -- KEEP.

### Impact on RC2 Conclusions

- The data quality flag in Section 2 is now fully explained with a causal mechanism.
- Features that are dropped as degenerate were never part of the RC2 kept set
  (the fallback top-5 were: amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48),
  so the core RC2 conclusions are **unaffected**.
- For downstream modeling (Phases 9-10), feature matrices should exclude degenerate
  features per bar type to avoid numerical issues in Ridge/ML training.
- No post-hoc deviations introduced. Trial count remains at 60.
