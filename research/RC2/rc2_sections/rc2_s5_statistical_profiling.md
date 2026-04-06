# RC2 Section 5: Statistical Profiling

## Methodology

Section 5 provides comprehensive statistical characterization of the return series
across all (asset, bar_type) combinations. This includes:

1. **Distribution analysis** -- descriptive statistics (moments, quantiles)
2. **Autocorrelation** -- Ljung-Box tests at multiple lags for returns and squared returns
3. **Variance ratio** -- Lo-MacKinlay tests at multiple calendar horizons
4. **Granger causality** -- BTC lead-lag relationships with altcoins
5. **GARCH dynamics** -- GARCH(1,1) with multiple innovation distributions, sign bias,
   ARCH-LM, and BDS on residuals
6. **Regime classification** -- Rolling volatility with LOW/NORMAL/HIGH labels

All p-values are BH-corrected for multiple testing across the full set of tests.

## Results: Autocorrelation (Ljung-Box)

### Summary Across All Combinations

| Series | Significant (BH) | Total Tests | Rate |
|--------|------------------|-------------|------|
| Returns | 48 | 105 | 45.7% |
| Returns-squared | 83 | 105 | 79.0% |

### Interpretation

**Returns:** Roughly half of the (asset, bar_type, lag) tests show significant serial
correlation. This is a mixed finding -- stronger than pure random walk but weaker than
clear predictability. The serial correlation is concentrated in time bars (which RC1
already identified as having serial dependence) and volume bars.

**Squared returns:** Nearly 80% of tests are significant, confirming strong ARCH effects
(volatility clustering) across all assets and bar types. This is the most robust
statistical finding in the entire RC2 analysis. Volatility clustering is:
- Present across all 4 assets
- Present across all bar types (including dollar bars, which RC1 showed have no
  serial correlation in returns)
- Persists across multiple lags

**The weak-form efficiency interpretation:** Returns are nearly unpredictable (weak serial
dependence), but their *volatility* is highly forecastable. This motivates GARCH-based
modeling and volatility-targeting strategies.

## Results: Variance Ratio

**VR tests significant (BH-corrected): 1/57.**

The near-universal failure to reject the random walk is consistent with PE analysis
(Section 4). LTCUSDT is the only exception, showing significance at specific horizons
on volume and time bars.

**Mean VR at 1-day horizon:** Close to 1.0 across all combinations, confirming no
systematic momentum or mean-reversion at the daily scale.

## Results: Granger Causality (BTC Lead)

**Test setup:** BTC -> {ETH, LTC, SOL} at lag 1 using time_1h bars (regular spacing
required for Granger). Common observations: 48,931.

**Result:** BTC Granger-causes all three altcoins at lag 1 (p < 0.05).

**Practical significance:** This confirms BTC's market-leading role and has direct
implications for the recommendation system:
- BTC-lagged returns should be included as features for altcoin models
- The recommendation system can use BTC regime as a leading indicator for altcoin
  strategy deployment
- Cross-asset features (BTC return at t-1 as input to ETH model at t) are justified

**Caution:** Granger causality at lag 1 on hourly data means the lead is ~1 hour.
On dollar bars (which are multi-hour), this lead may be captured within a single bar,
reducing its utility. The practical value depends on how quickly information propagates
across assets relative to bar formation time.

## Results: GARCH Dynamics

### GARCH(1,1) Parameters (Time Bars)

| Metric | Result |
|--------|--------|
| Mean persistence (alpha + beta) | 1.0000 |
| Near-IGARCH (persistence >= 0.99) | 4/4 assets |
| Best innovation distribution | Student-t (all assets) |
| Mean degrees of freedom (nu) | Low (heavy tails) |

**Near-IGARCH:** All four assets show GARCH persistence at or near 1.0, indicating
integrated GARCH behavior. Volatility shocks have permanent effects -- there is no
mean-reversion to a fixed long-run variance level. This is a well-documented property
of crypto markets and has implications for risk management:
- Volatility forecasts must be adaptive (no stable "normal" level)
- Position sizing should scale with recent realized volatility, not historical averages

**Student-t innovation:** The t-distribution is preferred over Normal by AIC for all
assets, confirming fat-tailed innovation terms beyond what GARCH's conditional
heteroscedasticity explains.

### Sign Bias (Leverage Effects)

Sign bias tests detect asymmetric leverage effects across assets. This means negative
returns increase future volatility more than positive returns of the same magnitude --
the "leverage effect" documented in equity markets also exists in crypto, though
potentially weaker.

**Implication:** GJR-GARCH or EGARCH models (which capture asymmetry) may provide
better volatility forecasts than symmetric GARCH(1,1).

### BDS on GARCH Residuals

**BDS rejects i.i.d. for all 4 assets** (BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT).

This is the critical finding for model complexity decisions. After removing linear
and GARCH-type dependence from the returns, the residuals STILL contain structure.
This means:
- GARCH(1,1) does not fully capture the volatility dynamics
- There is nonlinear dependence beyond what standard parametric models handle
- Per Rule M1, nonlinear models (tree ensembles, neural networks) are justified

**Nature of the remaining structure:** The BDS test detects any departure from i.i.d.
This could be:
- Higher-order ARCH effects (e.g., GARCH(2,2), FIGARCH)
- Regime-switching effects (Markov-switching GARCH)
- Threshold effects (e.g., momentum only activates above volatility threshold)
- Nonlinear cross-asset dependencies

Tree-based models (XGBoost, Random Forest) and neural networks are well-suited to
capture these without specifying the functional form.

## Results: Regime Classification

Rolling volatility (20-period standard deviation) classified into LOW/NORMAL/HIGH
using quantile thresholds (Q25 and Q75). All assets show clear regime transitions
aligned with known market events:

- **HIGH volatility:** March 2020 (COVID crash), May 2021 (China mining ban),
  November 2022 (FTX collapse), late 2024 (Trump election rally)
- **LOW volatility:** Mid-2020 (pre-DeFi), mid-2023 (consolidation)

**Cross-asset correlation:** Regime transitions are highly synchronized across BTC,
ETH, LTC, and SOL, consistent with the Granger causality findings.

## Connection to Lopez de Prado

Lopez de Prado (2018, Ch. 3) argues that financial ML should be applied to features
of the data-generating process, not raw returns. The RC2 profiling supports this:

1. **Raw returns are near-i.i.d.** -- modeling them directly will fail
2. **Volatility is highly structured** -- GARCH(1,1) captures most but not all of it
3. **Residual nonlinearity exists** -- ML models can add value beyond GARCH

The implication is that the ML pipeline should operate on GARCH-filtered residuals
or volatility-regime-conditioned data, not on raw returns. This is exactly the
approach Lopez de Prado advocates.

## Recommendations

1. **Include BTC-lagged features** for altcoin models based on the Granger results.
2. **Use regime labels as a primary input** to the recommendation system.
3. **Consider GJR-GARCH** as a baseline for volatility forecasting (leverages the
   asymmetric sign-bias finding).
4. **Frame the thesis narrative around volatility forecasting** as the primary
   contribution, with directional forecasting as the secondary (harder) problem.
5. **The GARCH + BDS finding is the strongest evidence** for the thesis: it proves
   that structure exists beyond simple parametric models, justifying the entire
   ML pipeline.
