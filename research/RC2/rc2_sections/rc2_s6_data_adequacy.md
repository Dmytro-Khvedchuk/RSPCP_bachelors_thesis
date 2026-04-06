# RC2 Section 6: Data Adequacy

## Methodology

Section 6 assesses whether the available data is sufficient for valid statistical
inference and ML modeling. It covers:

1. **Sample size classification** -- Tier A/B/C per Rule B1
2. **MDE feasibility** -- Is the minimum detectable effect smaller than the minimum
   economically meaningful effect?
3. **Signal-to-noise ratio** -- Ridge R-squared on real vs noise features
4. **Power analysis** -- Kish N_eff and statistical power at alpha=0.05, beta=0.80
5. **Cross-asset consistency** -- Kendall tau of MI rankings across assets
6. **Imbalance bar viability** -- Final viability verdict for small-sample bar types

## Results: Sample Size Classification

| Asset | Bar Type | N_obs | N_eff | Tier | DL Eligible |
|-------|----------|-------|-------|------|-------------|
| BTCUSDT | dollar | 5,286 | 5,286 | A | Yes (>= 2,000) |
| BTCUSDT | volume | 3,263 | 3,263 | A | Yes |
| BTCUSDT | vol_imbalance | 529 | 451 | B | No |
| BTCUSDT | dollar_imbalance | 568 | 568 | B | No |
| BTCUSDT | time_1h | 54,277 | 54,277 | A | Yes |
| ETHUSDT | dollar | 2,758 | 2,454 | A | Yes |
| ETHUSDT | volume | 24,037 | 24,037 | A | Yes |
| ETHUSDT | vol_imbalance | 697 | 697 | B | No |
| ETHUSDT | time_1h | 54,277 | 54,277 | A | Yes |
| LTCUSDT | volume | 26,986 | 26,986 | A | Yes |
| LTCUSDT | vol_imbalance | 737 | 737 | B | No |
| LTCUSDT | time_1h | 54,277 | 54,277 | A | Yes |
| SOLUSDT | dollar | 808 | 808 | B | No |
| SOLUSDT | volume | 47,177 | 47,177 | A | Yes |
| SOLUSDT | vol_imbalance | 870 | 870 | B | No |
| SOLUSDT | time_1h | 48,931 | 48,931 | A | Yes |

**Key observations:**
- BTCUSDT is the only asset with Tier A dollar bars (N_eff = 5,286)
- ETHUSDT dollar bars are Tier A but with reduced N_eff (2,454 vs 2,758 raw) due
  to autocorrelation-adjusted Kish estimator
- SOLUSDT dollar bars are Tier B (808 bars) -- deep learning not eligible
- Volume bars have much larger sample sizes across all assets
- All time_1h bars are Tier A but are baseline only (not used for primary modeling)

## Results: MDE Feasibility

The MDE DA is computed from the Kish effective sample size:
```
MDE_DA = 0.5 + (z_alpha + z_beta) / (2 * sqrt(N_eff))
       = 0.5 + (1.645 + 0.842) / (2 * sqrt(N_eff))
       = 0.5 + 1.244 / sqrt(N_eff)
```

**Feasibility = break-even DA > MDE DA** (can detect economically meaningful signal).

| Asset | Bar Type | MDE DA | Break-even DA | Gap | Feasible? |
|-------|----------|--------|---------------|-----|-----------|
| BTCUSDT | dollar | 0.5171 | 0.5695 | +5.24 pp | YES |
| BTCUSDT | volume | 0.5218 | 0.5552 | +3.34 pp | YES |
| BTCUSDT | vol_imbalance | 0.5585 | 0.5211 | -3.74 pp | NO |
| BTCUSDT | dollar_imbalance | 0.5522 | 0.5205 | -3.17 pp | NO |
| ETHUSDT | dollar | 0.5251 | 0.5362 | +1.11 pp | YES |
| ETHUSDT | volume | 0.5080 | 0.6100 | +10.20 pp | YES |
| LTCUSDT | volume | 0.5076 | 0.6018 | +9.42 pp | YES |
| SOLUSDT | dollar | 0.5437 | 0.5164 | -2.73 pp | NO |
| SOLUSDT | volume | 0.5057 | 0.6158 | +11.01 pp | YES |

**7 of 17 combinations have positive feasibility gaps.**

**The imbalance bar paradox quantified:** Imbalance bars have the lowest break-even DA
(~52%) due to their large per-bar returns, but their MDE DA is too high (~55-56%)
due to small sample sizes. The feasibility gap is negative -- we *cannot detect* a
signal that would be economically meaningful on these bars.

**Volume bars have the largest feasibility gaps** (9-11 pp) because they combine
large N (low MDE) with moderate break-even DA. However, RC1 showed volume bars have
borderline serial correlation, which could inflate apparent predictability.

## Results: Signal-to-Noise Ratio

Ridge R-squared was computed on holdout data (last 30% of sample) for the primary
(BTCUSDT/dollar) combination. The real-feature R-squared was compared against noise
baselines (10 random Gaussian feature sets, averaged).

Detailed results were generated via the `_compute_snr_r2` function. The key question:
does the real-feature regression explain more variance than random noise?

## Results: Cross-Asset MI Consistency

**Assets with MI rankings:** BTCUSDT, ETHUSDT, SOLUSDT (LTCUSDT excluded due to
insufficient dollar-bar data).

**Pairwise Kendall tau matrix (dollar bars, fwd_logret_1):**

Mean pairwise Kendall tau = 0.571 (Fisher-combined p < 0.0001).

**Rule A2: PASS** -- tau > 0 with strong significance.

**Interpretation:** Feature MI rankings are positively correlated across all three
assets. The same features that are informative for BTC are also informative for ETH
and SOL (to a lesser degree). This supports:

1. **Pooled training:** Models trained on BTC data can transfer to altcoins
2. **Shared feature selection:** The fallback-5 feature set is likely valid across
   assets, not just BTC
3. **The recommendation system can use a single feature pipeline** rather than
   per-asset feature selection

The tau = 0.571 is moderate -- not perfect correlation (which would suggest the signal
is purely market-wide) but strong enough to justify shared modeling.

## Results: Imbalance Bar Viability

**Viable imbalance bar combinations: 0/8.**

| Asset | Bar Type | N_obs | Tier | Rule A1 | Rule G4 | Verdict |
|-------|----------|-------|------|---------|---------|---------|
| BTCUSDT | vol_imbalance | 529 | B | NO | NO | NOT VIABLE |
| BTCUSDT | dollar_imbalance | 568 | B | NO | NO | NOT VIABLE |
| ETHUSDT | vol_imbalance | 697 | B | NO | NO | NOT VIABLE |
| ETHUSDT | dollar_imbalance | 427 | C | NO | NO | NOT VIABLE |
| LTCUSDT | vol_imbalance | 737 | B | NO | NO | NOT VIABLE |
| LTCUSDT | dollar_imbalance | 14 | -- | NO | NO | NOT VIABLE |
| SOLUSDT | vol_imbalance | 870 | B | NO | NO | NOT VIABLE |
| SOLUSDT | dollar_imbalance | 153 | -- | NO | NO | NOT VIABLE |

**None pass Rule A1 (N >= 1,000).** All are Tier B or C.

**This is the definitive answer:** Imbalance bars, despite showing the most promising
statistical properties (lowest PE, best DA excess), cannot be used for ML modeling
with current thresholds. They should be reported as an exploratory finding and
discussed as a data-limitation insight.

## Power Analysis Summary

- **Rule G4:** Multiple combinations have N_eff >= 1,000 (adequate power) --
  primarily dollar and volume bars for BTC and ETH.
- **Rule M2:** Multiple combinations have N_eff >= 2,000 (deep learning eligible) --
  BTCUSDT dollar, ETHUSDT dollar, plus all volume and time_1h bars.

## Connection to Lopez de Prado

Lopez de Prado (2018, Ch. 12) introduces the concept of the "Combinatorial Purged
Cross-Validation" specifically because financial datasets are small relative to the
complexity of the models. The RC2 data adequacy analysis quantifies this: even with
5,286 dollar bars for BTC, the MDE DA is only 1.7% above 50%. Any meaningful signal
must be very subtle.

The cross-asset consistency finding (tau = 0.571) partially mitigates the small-sample
problem: by pooling data across 3 assets with shared feature rankings, the effective
training set is larger. Lopez de Prado's meta-labeling framework was designed precisely
for this situation -- learning when signals are strong enough to act on.

## Recommendations

1. **Prioritize dollar and volume bars** for Tier A modeling. Do not attempt deep
   learning on imbalance bars.
2. **Pool training across BTC, ETH, and SOL** given the shared MI rankings.
3. **Report the imbalance bar paradox** as a thesis contribution: the most
   information-rich bars have insufficient sample sizes, quantifying the
   information-sample-size tradeoff.
4. **Consider expanding the data range** (pre-2020 data for BTC/ETH) to increase
   N_eff. However, market structure changes (DeFi, institutional adoption) may
   make older data less relevant.
5. **The SOL dollar-bar marginal status (N_eff = 808)** means SOL results should
   be flagged with wider confidence intervals. Consider SOL as a robustness check
   rather than a primary asset.
