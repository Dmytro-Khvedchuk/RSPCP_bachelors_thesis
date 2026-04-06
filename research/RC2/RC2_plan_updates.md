# RC2 Plan Update Proposal

**Based on:** RC2 Features, Profiling & Data Adequacy results (2026-03-24)
**Status:** Overall GO with constraints

---

## 1. What RC2 Revealed That Affects Future Phases

### 1.1 The Signal-Strength Reality

The most consequential RC2 finding is that **no single feature exceeds the break-even
directional accuracy** on the primary bar type (dollar bars, BTCUSDT). The best
individual feature (ret_zscore_24) achieves DA = 51.81%, which is 5.42 percentage
points below the break-even DA of 57.23%.

This does not invalidate the project, but it fundamentally constrains the modeling
approach. Single-feature linear models will not produce profitable strategies.
The path to profitability runs through:

1. **Multi-feature ensemble combination** (the recommendation system's raison d'etre)
2. **Regime-conditional deployment** (avoiding noise-dominated periods)
3. **Volatility-targeting** (leveraging the strong vol-clustering signal)
4. **Alternative bar types** where break-even DA is lower (imbalance bars: 52-53%)

### 1.2 The LTCUSDT Data Gap

LTCUSDT dollar bars produced only 199 usable bars -- insufficient for any analysis.
This is a threshold calibration problem: the dollar bar threshold (set globally) is
too aggressive for LTC's lower market cap and dollar volume. LTCUSDT volume bars
(26,986 bars) and time bars are fine.

**Impact:** LTCUSDT drops from the dollar-bar pipeline. It remains viable for
volume-bar modeling and serves as a useful negative-control comparison.

### 1.3 Imbalance Bar Paradox

Imbalance bars show the most promising statistical properties:
- Lowest permutation entropy (H_norm = 0.9740 for vol_imbalance vs 0.9977 for dollar)
- Best DA excess (G2: +4.47 pp on vol_imbalance, the best across all bar types)
- Lowest break-even DA (51-52% vs 57% for dollar bars)

But they fail viability (0/8 pass Rule A1: N >= 1,000). They are Tier B or C,
restricting them to simple models or profiling only.

**Impact:** Imbalance bars should be treated as a "most promising but undersampled"
finding. The thesis should discuss this as a data-limitation insight and recommend
threshold recalibration as future work.

### 1.4 Volatility Features Dominate

All 5 features retained by the fallback mechanism are from the volatility/volume
groups: rv_12, rv_24, rv_48, bbwidth_20_2.0, amihud_24. No returns-based or
momentum-based feature passed. This suggests:

- **Volatility is predictable** (GARCH effects, ARCH clustering confirmed)
- **Direction is nearly unpredictable** (consistent with R5)
- The forecasting pipeline should heavily weight volatility-based features
- The regression arm (SIZE prediction) may be more feasible than the classification
  arm (SIDE prediction)

### 1.5 Deep Learning Gate Is Open

Despite weak directional signal, the DL gate passes all three criteria:
- N_eff >= 2,000 (BTCUSDT: 5,286)
- >= 3 features (5 kept)
- BDS rejects i.i.d. (all 4 assets)

This means Phases 9-10 can explore tree ensembles and neural architectures.
However, given the weak signal, regularization must be aggressive to prevent
overfitting to the small edge.

### 1.6 Cross-Asset Feature Generalization

Kendall tau = 0.571 (p < 0.0001) across BTCUSDT, ETHUSDT, SOLUSDT means feature
importance rankings are shared. This supports:
- Pooled training across assets (larger effective sample)
- Transfer learning from BTC to altcoins
- The recommendation system comparing strategy performance across correlated assets

### 1.7 BTC Granger-Causes All Altcoins

BTC -> {ETH, LTC, SOL} at lag 1 is significant. This directly supports the Phase 12
recommendation system design: BTC-lagged features should be included as cross-asset
predictors for altcoin models.

---

## 2. Specific Changes Needed to Phase 5+

### Phase 5: Statistical Profiling (Current)

**Status:** Largely complete -- RC2 incorporated Phase 5 profiling into Sections 4-6.

**Recommended changes:**
- Add LTCUSDT volume-bar profiling to compensate for dollar-bar exclusion
- Add sensitivity analysis on the round-trip cost parameter (10, 20, 30 bps)
  to show how break-even DA varies with fee tier
- Document the LTCUSDT dollar-bar threshold issue as a known limitation

### Phase 6: Walk-Forward Framework

**No changes needed.** The temporal split structure (feature selection 2020-2023,
model development 2020-2024, holdout 2024+) was pre-registered and validated.

### Phase 7: Backtest Engine

**Recommended changes:**
- Add a **cost-sensitivity parameter** to the backtest engine (not just 20 bps)
- Implement **regime-conditional position sizing**: reduce exposure during
  HIGH-entropy periods and increase during LOW-entropy (PE-guided)
- The backtest must handle the fact that buy-and-hold Sharpe = 0.576 is the
  hurdle rate; any strategy below this has negative alpha

### Phase 8: Base Trading Strategies

**Recommended changes:**
- **Deprioritize pure directional strategies.** Given DA < break-even, strategies
  that rely solely on direction prediction will lose money after costs.
- **Add volatility-targeting strategies.** The strong volatility-clustering signal
  (rv_24, rv_48 are the top features) suggests strategies that size positions
  inversely to volatility could be effective even without directional edge.
- **Add a "NO-TRADE" strategy** that stands aside when PE exceeds 0.98 or
  volatility is in the LOW regime. This is the recommendation system's most
  basic value-add.

### Phase 9: Classification (Direction)

**Recommended changes:**
- **Lower expectations explicitly.** The pre-registration shows 0/23 features beat
  the DA null on dollar bars. Classify this as a "hard problem" and set the
  success criterion as: classifier DA > majority class DA (51.14%) with
  statistical significance, not classifier DA > break-even DA.
- **Try imbalance bars as primary for classification** despite small N. The +4.47 pp
  DA excess on vol_imbalance is the strongest directional signal. Use strong
  regularization (Ridge, early stopping) to mitigate small-sample overfitting.
- **Add BTC-lagged features** for altcoin classifiers (motivated by Granger results).
- **Use all 23 features, not just the fallback 5.** The fallback mechanism is
  conservative; the classifier benefits from higher dimensionality even if
  individual features are weak, provided regularization is adequate.
- **Ensemble over horizons:** Different features are informative at different
  horizons (logret_12 and roc_12 appear only for fwd_logret_4). The classifier
  should consider multi-horizon ensembling.

### Phase 10: Regression (Magnitude)

**Recommended changes:**
- **Focus on volatility forecasting as the regression target.** Since volatility
  features dominate, predicting future realized volatility (not just return
  magnitude) is more feasible and directly useful for position sizing.
- **Direction-conditional regression remains viable** but should be framed as
  exploratory. Report DC-MAE alongside unconditional MAE.
- **Consider the regression arm as SIZE = expected volatility** rather than
  SIZE = expected return magnitude. This reframes the recommendation system:
  instead of "how much will this trade make?", it answers "how risky is this
  trade?" -- which is more defensible given the weak directional signal.

### Phase 11: Model Comparison (RC3)

**Recommended changes:**
- Set the benchmark as buy-and-hold (Sharpe = 0.576), not random walk
- The Model Confidence Set (MCS) threshold should account for the fact that
  even the best single feature has negative economic margin
- Consider adding "feature DA > majority class DA" as a pass criterion rather
  than "feature DA > break-even DA" -- the latter is too stringent for single
  features

### Phase 12: Recommendation System

**Recommended changes:**
- **Regime-conditional activation is essential**, not optional. PE = 0.9977 on
  dollar bars means the system should default to NO-TRADE and activate only
  during detected regime transitions where PE temporarily drops.
- **Volatility-regime metadata** should be a first-class input to the recommender.
  The thesis already plans this, but RC2 makes it clear this is load-bearing.
- **Cross-asset features** (BTC-lagged returns for altcoins, Kendall tau rank
  correlations) should be included in the recommender's feature set.
- **Position sizing should scale with (DA - break_even_DA)** -- negative values
  mean the recommender outputs zero size.

### Phase 13-14: Evaluation (RC4)

**Recommended changes:**
- **Deflated Sharpe Ratio** uses 60 trials (pre-registered, 0 deviations). This is a
  strong position. Keep the trial counter at 60 unless post-hoc changes are made.
- **Monte Carlo validation on GBM** is essential. The near-random-walk PE values
  mean that any profitable strategy on real data MUST show zero or negative profit
  on synthetic random walks. If it doesn't, the strategy is fitting noise.
- **Add permutation entropy tracking** to the evaluation dashboard: show how PE
  varies over the backtest period and correlate with strategy PnL.

---

## 3. Bar Type and Feature Set Recommendations

### Bar Types to Proceed

| Bar Type | Priority | Justification | Modeling Tier |
|----------|----------|---------------|---------------|
| **dollar** | Primary | Best balance of N, stationarity, no serial corr | A (>= 2,000 bars) |
| **volume** | Secondary | Large N for all assets, good distributional properties | A |
| **volume_imbalance** | Exploratory | Lowest PE, best DA excess, but small N | B (500-2,000) |
| **dollar_imbalance** | Exploratory | Low PE, small N | B/C |
| **time_1h** | Baseline only | Highest break-even DA, strongest serial corr | A (for comparison) |

### Feature Set to Proceed

**Robustly informative (use for all horizons):**
amihud_24, bbwidth_20_2.0, rv_24, rv_48, park_vol_24

**Horizon-specific additions:**
- fwd_logret_1: rv_12
- fwd_logret_4: gk_vol_24, logret_12, roc_12
- fwd_logret_24: (same as robust set)

**Recommended approach:** Use the full 23-feature set for model training with
regularization (Ridge alpha selection via CPCV). The fallback-5 set defines the
minimum feature set; the full-23 set allows the model to discover weak but
combinatorially useful interactions.

---

## 4. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Multi-feature DA still below break-even | High | Project reports negative result (valid per N2) | Frame as "information boundary" finding; recommender pivots to risk filter |
| Overfitting to 2022-2023 MI spike | Medium | Model fails on holdout | CPCV with embargo; explicit window-stability checks |
| LTCUSDT exclusion weakens universality claim | Medium | Thesis covers 3/4 assets | Document as threshold calibration issue; show LTC works on volume bars |
| Imbalance bar signal vanishes with more data | Medium | Best DA excess unreliable | Report as exploratory; do not base primary conclusions on imbalance bars |
| BDS nonlinearity is from GARCH residuals, not returns | Low | DL doesn't help | GARCH-residual-based modeling already accounts for this |

---

## 5. Updated Priority Ordering

Original Phase order: 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14

**Recommended reordering:**

1. **Phase 5 (complete)**: Close out with LTCUSDT volume-bar profiling and cost sensitivity
2. **Phase 7**: Backtest engine (add cost sensitivity, regime awareness)
3. **Phase 8**: Base strategies -- add volatility-targeting strategy alongside directional
4. **Phase 10 (moved up)**: Regression -- start with volatility forecasting (strongest signal)
5. **Phase 9**: Classification -- temper expectations, use strong regularization
6. **Phase 12**: Recommendation system -- regime-conditional activation is essential
7. **Phase 11 (RC3)**: Model comparison
8. **Phase 13-14 (RC4)**: Evaluation with Monte Carlo, DSR, and PE tracking

**Rationale for reordering:** Moving regression (Phase 10) before classification (Phase 9)
leverages the strongest signal first (volatility features) rather than the weakest
(directional features). This produces useful intermediate results and de-risks the
project timeline.

---

## 6. Summary of Key Statistics Supporting Recommendations

| Statistic | Value | Source | Decision it supports |
|-----------|-------|--------|---------------------|
| Best DA (any bar type) | 54.47% (vol_imbalance) | Section 8 G2 | GO -- but on small-N bar type |
| Best DA (dollar bars) | 51.81% | Section 7 | No single feature is profitable |
| Break-even DA (dollar) | 57.23% | Section 4 | High bar for directional trading |
| Break-even DA (vol_imbalance) | 52.11% | Section 4 | Lower bar, but small N |
| PE (dollar, d=5) | 0.9977 | Section 4 | Near-random-walk on primary bars |
| PE (vol_imbalance, d=5) | 0.9740 | Section 4 | Structure exists on imbalance bars |
| BDS rejects i.i.d. | 4/4 assets | Section 5 | Nonlinear models justified |
| GARCH persistence | 1.000 | Section 5 | Strong vol memory -> vol forecasting viable |
| Kendall tau (cross-asset) | 0.571 | Section 6 | Shared features -> pooled training |
| Buy-and-hold Sharpe | 0.576 | Section 7 | High hurdle for active strategies |
| Post-hoc deviations | 0 | Section 8 | Pre-registration integrity maintained |
