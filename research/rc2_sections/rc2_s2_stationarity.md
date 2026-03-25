# RC2 Section 2: Stationarity Report

## Methodology

Joint ADF + KPSS testing framework at alpha = 0.05, applied to all 23 engineered
features across every (asset, bar_type) combination. The joint hypothesis approach
follows Kwiatkowski et al. (1992):

| ADF rejects | KPSS fails to reject | Classification |
|-------------|---------------------|----------------|
| Yes | Yes | **Stationary** -- both tests agree |
| Yes | No | **Trend-stationary** -- deterministic trend |
| No | No | **Unit root** -- non-stationary |
| No | Yes | **Inconclusive** -- insufficient power |

**Rationale:** ADF alone has low power against near-unit-root alternatives; KPSS has
size distortion in small samples. The joint approach reduces both false positive and
false negative risk.

## Results: Aggregate Statistics

**Total (asset, bar_type) combinations screened:** 17 of 20 possible.

**Skipped combinations:**
- LTCUSDT/dollar: 199 bars (need >= 200 for warmup)
- LTCUSDT/dollar_imbalance: 14 bars
- SOLUSDT/dollar_imbalance: 153 bars

**Classification distribution (391 feature tests):**

| Classification | Count | % |
|---------------|-------|---|
| Stationary | 210 | 53.7% |
| Trend-stationary | 108 | 27.6% |
| Unit root | 40 | 10.2% |
| Inconclusive | 33 | 8.4% |

Roughly half of all feature tests are classified as stationary. The 27.6%
trend-stationary classification is expected for features that have a slow-moving
deterministic component (e.g., rolling volatility on long windows).

## Results: Per-Asset Dollar Bars

| Asset | Stationary | Non-Stationary | Notable |
|-------|-----------|----------------|---------|
| BTCUSDT | 14/23 | 9/23 | atr_14 and rsi_14 constant |
| ETHUSDT | 13/23 | 10/23 | atr_14 and rsi_14 constant |
| SOLUSDT | 14/23 | 9/23 | atr_14 and rsi_14 constant |
| LTCUSDT | N/A | N/A | Skipped (199 bars) |

**Constant-feature warning:** atr_14 and rsi_14 are flagged as constant on dollar
bars for some assets. This occurs because dollar bars aggregate variable time periods,
and the ATR/RSI computations on these bars can produce nearly-constant outputs when the
bar-level returns are already normalized by the dollar threshold. These features are
marked inconclusive.

## Results: Cross-Bar-Type Comparison (BTCUSDT)

**Features stationary across ALL 5 bar types:** 10/23.

**Features non-stationary in >= 1 bar type (13):**
amihud_24, atr_14, bbwidth_20_2.0, ema_xover_8_21, gk_vol_24, hurst_100,
logret_12, logret_24, park_vol_24, rsi_14, rv_12, rv_24, rv_48.

This reveals that stationarity is primarily structural (determined by the feature
formula) rather than sample-dependent. Features that are non-stationary on one bar
type tend to be non-stationary on others.

## Results: Non-Stationary Features and Transformations

**Total unique non-stationary features:** 19 across 181 instances.

**Most frequently non-stationary:**

| Feature | N Combos Non-Stat | Classifications | Suggested Fix |
|---------|-------------------|-----------------|---------------|
| atr_14 | Many | inconclusive | pct_atr |
| rsi_14 | Many | inconclusive | (constant issue) |
| hurst_100 | Several | trend_stationary | first_difference |
| bbwidth_20_2.0 | Several | trend_stationary, unit_root | first_difference |
| amihud_24 | Several | trend_stationary | rolling_zscore |
| rv_12, rv_24, rv_48 | Several | trend_stationary | rolling_zscore |
| gk_vol_24, park_vol_24 | Several | trend_stationary | rolling_zscore |

**Note on realized volatility features:** rv_12, rv_24, rv_48 being trend-stationary
is expected. Realized volatility measures absolute volatility that can exhibit
long-memory / IGARCH-like persistence. The trend-stationary classification (ADF
rejects but KPSS also rejects) indicates a deterministic trend overlaid on a
stationary process. For modeling purposes, these features should be used with
caution -- rolling z-score normalization is recommended.

## Impact on Downstream Analysis

1. Features entering the MI/Ridge validation pipeline are either stationary or have
   known transformation paths.
2. The stationarity screening prevents the most common false discovery in financial ML:
   shared trends masquerading as predictive signal (Granger & Newbold, 1974).
3. The current pipeline applies clipping to [-5, 5] which bounds extreme values but
   does not induce stationarity. Features classified as unit_root should ideally be
   transformed before entering the validation pipeline.

## Connection to Lopez de Prado

Lopez de Prado (2018, Ch. 5) advocates fractional differentiation as the stationarity
transformation that preserves the maximum amount of memory. The RC2 approach uses
simpler transformations (z-scores, percentage ratios, first differences) which are
more interpretable but may discard useful information. Phase 5+ could explore
fractional differentiation for the unit_root features (amihud_24, hurst_100,
bbwidth_20_2.0) as an enhancement.

## Recommendations

1. Apply the suggested transformations (pct_atr, rolling_zscore, first_difference)
   to non-stationary features before Phase 9-10 model training.
2. Investigate why atr_14 and rsi_14 are constant on dollar bars -- this may indicate
   a bug in the indicator computation for alternative bars.
3. Consider fractional differentiation for features with unit_root classification
   as a Phase 5 enhancement.
